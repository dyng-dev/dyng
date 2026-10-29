// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_compat.cpp
 * @brief dyng-compat-hsosp: the sessions of MOSP_ESCHER@4b86159's H-SOSP dumps (the golden set
 *        hyper_sssp_mosp_escher, ADR 0081) as dynG files, and their replay through
 *        dyng::hypergraph on slack_csr storage (M11a step 2; ADRs 0085, 0086).
 *
 *     dyng-compat-hsosp to-dgb <dump.hsd[.zst]> <out-prefix>
 *     dyng-compat-hsosp replay <out-prefix>.s<k> [--backend sequential|openmp] [--threads N]
 *
 * to-dgb writes, for the k-th session whose load MOSP_ESCHER (B) accepted (k = 1, 2, ...):
 *  - <prefix>.s<k>.hg: B's loaded rows in B's id order, its virtual source and target rows at
 *    their positions (hyperedge i of the file is dynG id i = B id - 1), with B's weights (the
 *    virtual rows' weight 0 written as 1: dynG's weights are >= 1), 1-based vertices;
 *  - <prefix>.s<k>.dgb: the batches B applied, weighted int32, with the flags expected_ids (B's
 *    finalIds - 1), insert_first and, when a batch has one, incidence_edits (B's vtxDelete /
 *    vtxInsert as (B id - 1, vertex)); the operations naming a virtual row are filtered out (B
 *    skips them) and counted, and so are those a file cannot hold (an id below 1, a negative
 *    vertex: B skips them too);
 *  - <prefix>.s<k>.txt: the vertex count (a .hg file does not hold an isolated last vertex), the
 *    virtual rows' ids and members (the source and target vertices), B's Caps (headroom factor
 *    and extra payload ints: capacity_policy::mosp_escher(extra) with initial_headroom = factor -
 *    1), its CSR knobs (entry headroom, node capacity), and per batch B's skippedOps and the
 *    filtered and unrepresentable operations (B's skippedOps = dynG's ignored deletions + skipped
 *    edits + both).
 * The output depends only on the dump (no time, no path but the dump's file name): converting
 * twice gives the same bytes (the SHA-256 of the converted files are in parity/goldens.toml).
 *
 * replay reads the three files, builds the hypergraph (slack_csr storage without a line graph:
 * the ids do not depend on it; id_reuse::insert_first, capacity_policy::mosp_escher(extra)) and
 * streams the .dgb with
 * io::batch_reader: every batch's inserted ids must equal the expected ids, and the skipped count
 * B's. Exit status: 0 equal, 1 a difference, 2 a usage or input error.
 */
#include "parity/hsosp/hsosp_dump.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace hd = dyng::test::hsosp_dump;
namespace io = dyng::io;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using wbatch = dyng::hyperedge_batch<std::int32_t, std::int32_t>;

int usage() {
  std::fprintf(stderr,
               "usage: dyng-compat-hsosp to-dgb <dump.hsd[.zst]> <out-prefix>\n"
               "       dyng-compat-hsosp replay <out-prefix>.s<k> [--backend sequential|openmp] "
               "[--threads N]\n");
  return 2;
}

std::string real_text(const hd::record& r, const std::string& name, const std::string& fallback) {
  // The dump's reals are text fields (their shortest round-trip form).
  if (!r.has(name)) return fallback;
  const hd::field& f = r.fields.at(name);
  return f.type == hd::field::kind::str ? f.text : std::to_string(f.value);
}

// --- to-dgb ------------------------------------------------------------------------------------

/// One batch B applied, in dynG's terms.
struct converted_batch {
  wbatch batch;
  std::vector<std::int64_t> expected;
  std::int64_t b_skipped = 0, filtered = 0, unrepresentable = 0;
};

/// One session B loaded.
struct converted_session {
  std::int64_t id = 0, num_vertices = 0, source = 0, target = 0, extra = 0, max_nodes = 0;
  std::string factor = "1.3", entry_headroom = "1.6";
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> members;
  std::vector<std::int32_t> weights;
  std::vector<converted_batch> batches;
  std::int64_t rejected_batches = 0;
  bool loaded = false, convertible = true;
  std::string why;
};

int to_dgb(const std::string& dump, const std::string& prefix) {
  const std::vector<hd::record> recs = hd::read_dump(dump);
  std::vector<converted_session> sessions;
  converted_session* s = nullptr;
  converted_batch pending;
  bool has_pending = false;
  const auto is_virtual = [&](std::int64_t b_id) { return b_id == s->source || b_id == s->target; };
  for (const hd::record& r : recs) {
    if (r.tag == "session") {
      sessions.emplace_back();
      s = &sessions.back();
      s->id = r.i64("session");
      s->num_vertices = r.i64("num_vertices");
      s->factor = real_text(r, "caps.headroom_factor", "1.3");
      s->extra = r.i64("caps.extra_payload_ints");
      has_pending = false;
      continue;
    }
    if (s == nullptr) continue;
    if (r.tag == "load" && r.str("phase") == "enter") {
      s->offsets = r.longs("rows.offsets");
      s->members = r.ints("rows.values");
      s->source = r.i64("source_he");
      s->target = r.i64("target_he");
      const auto& w = r.longs("weights");
      s->weights.clear();
      for (std::size_t i = 0; i < w.size(); ++i) {
        const auto b_id = static_cast<std::int64_t>(i) + 1;
        if (is_virtual(b_id) && w[i] == 0) {
          s->weights.push_back(1);  // B's virtual rows weigh 0; dynG's weights are >= 1
        } else if (w[i] < 1 || w[i] > INT_MAX) {
          s->convertible = false;
          s->why = "weight " + std::to_string(w[i]) + " of B id " + std::to_string(b_id);
          s->weights.push_back(1);
        } else {
          s->weights.push_back(static_cast<std::int32_t>(w[i]));
        }
      }
    } else if (r.tag == "load" && r.str("phase") == "exit") {
      s->loaded = r.str("exception").empty();
    } else if (r.tag == "device_build" && r.str("exception").empty()) {
      s->entry_headroom = real_text(r, "entry_headroom", "1.6");
      s->max_nodes = r.i64("max_nodes");
    } else if (r.tag == "begin_batch" && r.str("phase") == "enter") {
      pending = converted_batch{};
      has_pending = true;
      const auto keep_id = [&](std::int64_t b_id) {
        if (is_virtual(b_id)) {
          ++pending.filtered;
          return false;
        }
        if (b_id < 1) {
          ++pending.unrepresentable;  // B skips it; a file holds ids >= 0
          return false;
        }
        return true;
      };
      for (const std::int32_t id : r.ints("he_delete"))
        if (keep_id(id)) pending.batch.delete_hyperedge(static_cast<std::int64_t>(id) - 1);
      const auto edits = [&](const char* he, const char* vx, bool add) {
        const auto& h = r.ints(he);
        const auto& v = r.ints(vx);
        for (std::size_t i = 0; i < h.size(); ++i) {
          if (!keep_id(h[i])) continue;
          if (v[i] < 0) {
            ++pending.unrepresentable;
            continue;
          }
          if (add) {
            pending.batch.insert_incidence(static_cast<std::int64_t>(h[i]) - 1, v[i]);
          } else {
            pending.batch.delete_incidence(static_cast<std::int64_t>(h[i]) - 1, v[i]);
          }
        }
      };
      edits("vtx_delete.he", "vtx_delete.vertex", false);
      edits("vtx_insert.he", "vtx_insert.vertex", true);
      const auto& off = r.longs("he_insert.offsets");
      const auto& val = r.ints("he_insert.values");
      const auto& w = r.longs("he_insert.weight");
      for (std::size_t i = 0; i + 1 < off.size(); ++i) {
        const std::vector<std::int32_t> m(val.begin() + off[i], val.begin() + off[i + 1]);
        if (w[i] < 1 || w[i] > INT_MAX || m.empty()) {
          has_pending = false;  // B rejects it (its validation); the exit says so
          break;
        }
        pending.batch.insert_hyperedge(dyng::host_view(m), static_cast<std::int32_t>(w[i]));
      }
    } else if (r.tag == "begin_batch" && r.str("phase") == "exit") {
      if (!r.str("exception").empty()) {
        ++s->rejected_batches;  // B's validation rejected it before anything changed
        has_pending = false;
        continue;
      }
      if (!has_pending) {
        s->convertible = false;
        s->why = "a batch B applied that a .dgb file cannot hold";
        continue;
      }
      for (const std::int32_t id : r.ints("final_ids"))
        pending.expected.push_back(static_cast<std::int64_t>(id) - 1);
      pending.b_skipped = r.i64("skipped_ops");
      s->batches.push_back(std::move(pending));
      has_pending = false;
    }
  }
  std::int64_t k = 0, written = 0;
  const std::string name = std::filesystem::path(dump).filename().string();
  for (const converted_session& c : sessions) {
    if (!c.loaded) continue;
    ++k;
    const std::string p = prefix + ".s" + std::to_string(k);
    if (!c.convertible) {
      std::fprintf(stderr, "dyng-compat-hsosp: session %lld of %s is not convertible: %s\n",
                   static_cast<long long>(c.id), dump.c_str(), c.why.c_str());
      return 2;
    }
    dyng::hyperedge_list<std::int32_t, std::int32_t> list;
    list.num_vertices = c.num_vertices;
    list.offsets = c.offsets;
    list.members = c.members;
    list.weights = c.weights;
    io::write_hypergraph(p + ".hg", list.view(), 1);
    std::uint64_t flags = io::batch_file_flags::expected_ids | io::batch_file_flags::insert_first;
    for (const auto& b : c.batches)
      if (!b.batch.delete_incidence_hyperedge().empty() ||
          !b.batch.insert_incidence_hyperedge().empty())
        flags |= io::batch_file_flags::incidence_edits;
    io::batch_writer out(p + ".dgb",
                         io::hyperedge_batch_file_header<std::int32_t, std::int32_t>(flags));
    for (const auto& b : c.batches) {
      io::hyperedge_batch_annotations<std::int32_t> a;
      a.expected_ids = dyng::host_view(b.expected);
      out.write(b.batch.view(), a);
    }
    out.close();
    std::ofstream txt(p + ".txt");
    txt << "# dyng-compat-hsosp to-dgb: session " << c.id << " of " << name
        << " (MOSP_ESCHER@4b86159's H-SOSP)\n";
    const auto row = [&](std::int64_t b_id) {
      std::string out_row;
      if (b_id < 1 || static_cast<std::size_t>(b_id) >= c.offsets.size()) return out_row;
      for (auto i = c.offsets[static_cast<std::size_t>(b_id) - 1];
           i < c.offsets[static_cast<std::size_t>(b_id)]; ++i)
        out_row +=
            (out_row.empty() ? "" : " ") + std::to_string(c.members[static_cast<std::size_t>(i)]);
      return out_row;
    };
    txt << "session " << c.id << "\nvertices " << c.num_vertices << "\nhyperedges "
        << c.weights.size() << "\nsource_id " << c.source - 1 << "\ntarget_id " << c.target - 1
        << "\nsource_vertices " << row(c.source) << "\ntarget_vertices " << row(c.target)
        << "\nheadroom_factor " << c.factor << "\nextra_payload_ints " << c.extra
        << "\nentry_headroom " << c.entry_headroom << "\nmax_nodes " << c.max_nodes << "\nbatches "
        << c.batches.size() << "\nrejected_batches " << c.rejected_batches << "\n";
    for (std::size_t i = 0; i < c.batches.size(); ++i) {
      const auto& b = c.batches[i];
      txt << "batch " << i << " skipped_ops " << b.b_skipped << " filtered " << b.filtered
          << " unrepresentable " << b.unrepresentable << "\n";
    }
    if (!txt) {
      std::fprintf(stderr, "dyng-compat-hsosp: cannot write %s.txt\n", p.c_str());
      return 2;
    }
    ++written;
  }
  std::printf("%s: %lld sessions -> %s.s1 .. .s%lld (.hg, .dgb, .txt)\n", dump.c_str(),
              static_cast<long long>(written), prefix.c_str(), static_cast<long long>(written));
  return 0;
}

// --- replay ------------------------------------------------------------------------------------

int replay(const std::string& prefix, const std::string& backend, int threads) {
  const dyng::resources res =
      backend == "openmp" ? dyng::resources::openmp(threads) : dyng::resources::sequential();
  std::map<std::string, std::string> info;
  std::vector<std::int64_t> b_skipped, filtered;
  {
    std::ifstream txt(prefix + ".txt");
    if (!txt) {
      std::fprintf(stderr, "dyng-compat-hsosp: cannot read %s.txt\n", prefix.c_str());
      return 2;
    }
    for (std::string line; std::getline(txt, line);) {
      if (line.empty() || line[0] == '#') continue;
      std::istringstream ls(line);
      std::string key;
      ls >> key;
      if (key == "batch") {
        std::int64_t i = 0, skipped = 0, f = 0, u = 0;
        std::string a, b, c;
        ls >> i >> a >> skipped >> b >> f >> c >> u;
        b_skipped.push_back(skipped);
        filtered.push_back(f + u);
        continue;
      }
      std::string rest;
      std::getline(ls, rest);
      info[key] = rest.empty() ? rest : rest.substr(1);
    }
  }
  io::hypergraph_read_options ro;
  ro.num_vertices = std::stoll(info["vertices"]);
  const auto list = io::read_hypergraph<std::int32_t, std::int32_t>(prefix + ".hg", ro);
  dyng::hypergraph_properties props;
  props.storage = dyng::hypergraph_storage::slack_csr;
  props.line_graph = dyng::line_graph_policy::none;  // the ids do not depend on it
  props.reuse = dyng::id_reuse::insert_first;
  props.capacity = dyng::capacity_policy::mosp_escher(std::stoll(info["extra_payload_ints"]));
  const double factor = std::stod(info["headroom_factor"]);
  props.capacity.initial_headroom = factor == 1.3 ? 0.3 : factor - 1.0;
  whgraph h = whgraph::from_hyperedges(res, list.view(), props);
  io::batch_reader in(prefix + ".dgb");
  std::int64_t batches = 0, ids = 0, differences = 0;
  while (in.next()) {
    const auto v = in.hyperedge_batch<std::int32_t, std::int32_t>();
    const auto a = in.annotations<std::int32_t>();
    const std::vector<std::int64_t> want(a.expected_ids.begin(), a.expected_ids.end());
    const auto s = h.apply(res, v);
    const auto i = static_cast<std::size_t>(batches);
    if (s.inserted_ids != want) {
      std::fprintf(stderr, "batch %lld: the inserted ids differ from the expected ids\n",
                   static_cast<long long>(batches));
      ++differences;
    }
    // B's skippedOps: dynG's ignored deletions and skipped edits, and the filtered operations.
    const std::int64_t skipped = s.skipped_ops - s.dropped_members + filtered[i];
    if (i < b_skipped.size() && skipped != b_skipped[i]) {
      std::fprintf(stderr, "batch %lld: %lld skipped, B %lld\n", static_cast<long long>(batches),
                   static_cast<long long>(skipped), static_cast<long long>(b_skipped[i]));
      ++differences;
    }
    ids += static_cast<std::int64_t>(want.size());
    ++batches;
  }
  h.check_integrity(res);
  std::printf("%s %s: %lld batches, %lld ids, %lld differences\n",
              differences == 0 ? "ok" : "DIFFERENT", prefix.c_str(),
              static_cast<long long>(batches), static_cast<long long>(ids),
              static_cast<long long>(differences));
  return differences == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string cmd = argv[1];
  try {
    if (cmd == "to-dgb") {
      if (argc != 4) return usage();
      return to_dgb(argv[2], argv[3]);
    }
    if (cmd == "replay") {
      std::string backend = "sequential";
      int threads = 0;
      for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
          backend = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
          threads = std::atoi(argv[++i]);
        } else {
          return usage();
        }
      }
      if (backend != "sequential" && backend != "openmp") return usage();
      return replay(argv[2], backend, threads);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng-compat-hsosp: %s\n", e.what());
    return 2;
  }
  return usage();
}
