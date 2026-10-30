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
 * replay reads the three files, builds the hypergraph (slack_csr storage; id_reuse::insert_first,
 * capacity_policy::mosp_escher(extra); with --line-graph the slack CSR line graph with B's knobs)
 * and streams the .dgb with io::batch_reader: every batch's inserted ids must equal the expected
 * ids, and the skipped count B's; with --line-graph also the batch's pair delta must equal
 * testing::line_graph_delta of the two states, and check_integrity() pass (the CUDA store: the
 * incidence mirror and the device CSR against the host model). --backend cuda runs it on the
 * CUDA store (M11a step 3, ADR 0089).
 *
 *     dyng-compat-hsosp replay <dump.hsd[.zst]> --algorithm by_id|by_id_public|by_key
 *         [--backend sequential|cuda] [--device D] [--check-compute] [--json OUT]
 *
 * replay --algorithm (M11a step 4, criterion 14) replays every session of a dump through
 * hyper_sssp and compares every update and recompute MOSP_ESCHER recorded, in the three parity
 * modes of HD2 (cpp/tests/parity/hsosp/hsosp_results.hpp; the per-call UpdateConfig of the dump).
 *
 *     dyng-compat-hsosp bench --hg <file.hg> --kind hyperedge|vertex --batch N [--batches 3]
 *         [--del 50] [--seed 20260725] [--maxcard 25] [--maxiter 4096] [--work-budget 1]
 *         [--verify none|first|all] --golden <dump.hsd[.zst]> [--out <csv>] [--timing <file.json>]
 *         [--maintain-only] [--public] [--backend cuda|sequential]
 *         [--cuda-engine automatic|fused|operators] [--sync-stages] [--digests] [--device D]
 *         [--threads T] [--check-load <loads/<dataset>.hsd.zst>]
 *
 * bench is the port side of hyper_sssp's timed regions (parity/timed_regions/hyper_sssp.toml; HD3,
 * HD4, HD5): hsospBench --hg with B's options on the cuda backend. It loads the .hg file as B's
 * loader does (io::read_hypergraph with hypergraph_read_options::hsosp(seed, maxcard)) and adds
 * B's virtual source row {s} (first: s the vertex of largest degree, the first of them) and
 * target row {t} (last: t drawn uniformly from std::mt19937_64(seed) before the weights), weight 1
 * in the container; plans B's capacities (hsospBench.cu:201, 209-218, 496-511: planned = batch x
 * batches for hyperedge batches + 65536, maxHyperedges = rows + planned + 1024, headroom factor
 * 1.3, extraPayloadInts = planned x (maxcard + 2 x 128 + 16) + 8 Mi, the line graph's and the
 * mirror's entry headroom 1.6); then replays the batches B exported (the golden dump's
 * begin_batch records, HD5), each converted to a host batch before its timer. --maintain-only (M11a
 * step 3, criterion 12) times hypergraph::apply alone (the maintain row: the stage
 * hypergraph.apply). Without it (M11a step 4) the bench is hsospBench's: the initial solve into
 * result A in the load (hyper_sssp's virtual-endpoint mode from B's source row; --public: the
 * public API from B's source vertex), result B allocated (B's state B), then per batch
 * hyper_sssp::update of A (update_ms: the dynamic_comparable and dynamic rows) and the static
 * recompute into B (static_ms; stage hyper_sssp.compute), each after a synchronization; untimed
 * after them update == static (mismatch_static), with --verify first|all the oracle
 * (testing::check_hyper_sssp_tree: mismatch_oracle, parent_errors), and the digests. --timing
 * writes the profiler's stages per batch and the driver timers (load_ms from the start of main to
 * the end of the initial solve and the reachable fraction, maintain_ms or update_ms and static_ms
 * per batch, dynamic_ms and static_sum_ms, total_ms); --out hsospBench's CSV (the 44 columns of
 * hsospBench.cu:155-165, in its default number format) with the port's values, except
 * timestamp, dev_mem_mb (cudaMemGetInfo; the memory gate reads nsys instead) and escher_mb (dynG
 * keeps no CBST), which are left empty (the merged results/data CSVs of B add a leading `run`
 * column; B writes seedCount to both `seeds` and `invalidated`, and so does the port: stats
 * invalidated); stderr gets hsospBench's [e2e] line and, after the load and after every batch,
 * "[memory] load|batch <k> graph_scope_bytes <bytes>" (B's graph_mb scope: the store, its mirror and
 * both results; untimed); --nvtx makes the profiler's stages NVTX ranges (the memory reading's
 * ranges). --digests prints after every batch
 * (untimed) the SHA-256 of its canonical pairs, its pair count and its CSR rebuilds, and without
 * --maintain-only the digests of A's distances and parents, its fallback decision and
 * mismatch_static (the per-round guards of parity/hsosp_perf.py); --check-load compares the
 * loaded rows and weights with B's dumped ones once. Exit status: 0, 1 a check failed, 2 a usage
 * or input error.
 */
#include "algorithms/hyper_sssp/virtual_endpoints.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "parity/hsosp/hsosp_dump.hpp"
#include "parity/hsosp/hsosp_results.hpp"
#include "support/sha256.hpp"
#include "util/rng.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <optional>
#include <random>
#include <sstream>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace hd = dyng::test::hsosp_dump;
namespace io = dyng::io;
using whgraph = dyng::hypergraph<std::int32_t, std::int32_t>;
using wbatch = dyng::hyperedge_batch<std::int32_t, std::int32_t>;

using clock_type = std::chrono::steady_clock;
namespace access = dyng::detail;

int usage() {
  std::fprintf(stderr,
               "usage: dyng-compat-hsosp to-dgb <dump.hsd[.zst]> <out-prefix>\n"
               "       dyng-compat-hsosp replay <out-prefix>.s<k> "
               "[--backend sequential|openmp|cuda] [--threads N] [--device D] [--line-graph]\n"
               "       dyng-compat-hsosp replay <dump> --algorithm by_id|by_id_public|by_key "
               "[--backend sequential|cuda] [--device D] [--check-compute] [--json OUT]\n"
               "       dyng-compat-hsosp bench --hg <file.hg> --kind hyperedge|vertex --batch N "
               "[--batches 3] [--del 50] [--seed S] [--maxcard 25] [--maxiter 4096] "
               "[--work-budget 1] [--verify none|first|all] --golden <dump> [--out CSV] "
               "[--timing JSON] [--maintain-only] [--public] [--backend cuda|sequential] "
               "[--cuda-engine automatic|fused|operators] [--sync-stages] [--digests] [--nvtx] "
               "[--device D] [--threads T] [--check-load <load dump>]\n");
  return 2;
}

double ms_since(clock_type::time_point t0) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - t0).count();
}

/// The resources of a backend (cuda: the device, with `threads` host threads).
dyng::resources resources_of(const std::string& backend, int threads, int device) {
  if (backend == "cuda") return dyng::resources::cuda(device, {}, threads);
  if (backend == "openmp") return dyng::resources::openmp(threads);
  return dyng::resources::sequential();
}

/// The SHA-256 of a batch's canonical pairs (each list sorted, int32 little-endian, the deleted
/// list first; parity/hsosp_dump.py canonical_pairs).
std::string pairs_digest(const dyng::detail::hyper::line_graph_delta& d) {
  dyng::test::sha256 h;
  for (int list = 0; list < 2; ++list) {
    const auto& x = list == 0 ? d.deleted_a : d.inserted_a;
    const auto& y = list == 0 ? d.deleted_b : d.inserted_b;
    std::vector<std::pair<std::int64_t, std::int64_t>> sorted;
    sorted.reserve(x.size());
    for (std::size_t i = 0; i < x.size(); ++i) sorted.emplace_back(x[i], y[i]);
    std::sort(sorted.begin(), sorted.end());
    std::vector<std::int32_t> flat;
    flat.reserve(2 * sorted.size());
    for (const auto& [a, b] : sorted) {
      flat.push_back(static_cast<std::int32_t>(a));
      flat.push_back(static_cast<std::int32_t>(b));
    }
    h.update_le(flat);
  }
  return h.hex();
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

/// The flat rows of a hypergraph by id (the oracle's input).
std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> flat_rows(
    const dyng::resources& res, const whgraph& h) {
  const auto v = h.h2v();
  return {dyng::to_vector(res, v.row_ptr), dyng::to_vector(res, v.col_ind)};
}

int replay(const std::string& prefix, const std::string& backend, int threads, int device,
           bool line_graph) {
  const dyng::resources res = resources_of(backend, threads, device);
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
  // Without --line-graph none: the ids do not depend on it.
  props.line_graph =
      line_graph ? dyng::line_graph_policy::slack_csr : dyng::line_graph_policy::none;
  props.reuse = dyng::id_reuse::insert_first;
  props.capacity = dyng::capacity_policy::mosp_escher(std::stoll(info["extra_payload_ints"]));
  const double factor = std::stod(info["headroom_factor"]);
  props.capacity.initial_headroom = factor == 1.3 ? 0.3 : factor - 1.0;
  dyng::detail::hyper::slack::knobs knobs;
  knobs.entry_headroom = std::stod(info["entry_headroom"]);
  if (std::stoll(info["max_nodes"]) > 0) knobs.node_capacity = std::stoll(info["max_nodes"]);
  whgraph h = access::hypergraph_access::build(res, list.view(), props, knobs);
  io::batch_reader in(prefix + ".dgb");
  std::int64_t batches = 0, ids = 0, differences = 0, pairs = 0;
  while (in.next()) {
    const auto v = in.hyperedge_batch<std::int32_t, std::int32_t>();
    const auto a = in.annotations<std::int32_t>();
    const std::vector<std::int64_t> want(a.expected_ids.begin(), a.expected_ids.end());
    std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> pre;
    if (line_graph) pre = flat_rows(res, h);
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
    if (line_graph) {
      // The pair delta against the oracle's (testing::line_graph_delta of the two states).
      const auto post = flat_rows(res, h);
      const auto want_delta = dyng::testing::line_graph_delta(pre.first, pre.second, post.first,
                                                              post.second, h.num_vertices());
      const auto& d = access::hypergraph_access::pairs(h);
      std::vector<std::pair<std::int64_t, std::int64_t>> del, ins;
      for (std::size_t k = 0; k < d.deleted_a.size(); ++k)
        del.emplace_back(d.deleted_a[k], d.deleted_b[k]);
      for (std::size_t k = 0; k < d.inserted_a.size(); ++k)
        ins.emplace_back(d.inserted_a[k], d.inserted_b[k]);
      if (del != want_delta.first || ins != want_delta.second) {
        std::fprintf(stderr, "batch %lld: the pair delta differs from testing::line_graph_delta\n",
                     static_cast<long long>(batches));
        ++differences;
      }
      pairs += static_cast<std::int64_t>(del.size() + ins.size());
      h.check_integrity(res);
    }
    ids += static_cast<std::int64_t>(want.size());
    ++batches;
  }
  h.check_integrity(res);
  std::printf("%s %s (%s): %lld batches, %lld ids, %lld pairs, %lld differences\n",
              differences == 0 ? "ok" : "DIFFERENT", prefix.c_str(), backend.c_str(),
              static_cast<long long>(batches), static_cast<long long>(ids),
              static_cast<long long>(pairs), static_cast<long long>(differences));
  return differences == 0 ? 0 : 1;
}

// --- bench -------------------------------------------------------------------------------------

struct bench_options {
  std::string hg, kind = "hyperedge", verify = "first", golden, out, timing, check_load;
  std::string backend = "cuda", cuda_engine = "automatic";
  std::int64_t batch = 50000, batches = 3, maxcard = 25, max_iterations = 4096;
  double del = 50.0, work_budget = 1.0;
  std::uint64_t seed = 20260725U;
  bool maintain_only = false, sync_stages = false, digests = false, public_source = false;
  bool nvtx = false;
  int device = 0, threads = 0;
  std::vector<std::string> argv;  // as given (the --timing file's meta)
};

/// The driver's timers and the profiler's samples, per batch (0: the load), written as the
/// --timing JSON of the other compat drivers ({meta, stages[{stage, batch, ms}]}).
class timing_log {
 public:
  explicit timing_log(const dyng::profiler& prof) : prof_(prof) {}
  void timer(const std::string& name, std::int64_t batch, double ms) {
    records_.push_back({name, batch, ms});
  }
  /// The profiler samples completed since the last call belong to `batch`.
  void stages(std::int64_t batch) {
    const auto& s = prof_.samples();
    for (; seen_ < s.size(); ++seen_) records_.push_back({s[seen_].name, batch, s[seen_].host_ms});
  }
  [[nodiscard]] double sum(const std::string& name, std::int64_t batch) const {
    double t = 0;
    for (const auto& r : records_)
      if (r.name == name && r.batch == batch) t += r.ms;
    return t;
  }
  void write(const std::string& path, const bench_options& o) const {
    std::ofstream out(path);
    out << std::fixed << std::setprecision(3);
    out << "{\n  \"meta\": {\"driver\": \"dyng-compat-hsosp bench\", \"command\": \"";
    for (std::size_t i = 0; i < o.argv.size(); ++i) out << (i ? " " : "") << o.argv[i];
    out << "\", \"seed\": \"" << o.seed << "\", \"batches\": \"" << o.batches
        << "\", \"maintain_only\": \"" << (o.maintain_only ? 1 : 0) << "\", \"sync_stages\": \""
        << (o.sync_stages ? 1 : 0) << "\", \"backend\": \"" << o.backend
        << "\"},\n  \"stages\": [\n";
    for (std::size_t i = 0; i < records_.size(); ++i) {
      out << "    {\"stage\": \"" << records_[i].name << "\", \"batch\": " << records_[i].batch
          << ", \"ms\": " << records_[i].ms << "}" << (i + 1 < records_.size() ? "," : "") << "\n";
    }
    out << "  ]\n}\n";
    if (!out) throw std::runtime_error("cannot write the --timing file " + path);
  }

 private:
  struct record {
    std::string name;
    std::int64_t batch;
    double ms;
  };
  const dyng::profiler& prof_;
  std::size_t seen_ = 0;
  std::vector<record> records_;
};

/// One batch of the golden dump, converted (the virtual rows' operations filtered, as B skips
/// them), with what B recorded of it.
struct golden_batch {
  wbatch batch;
  double entry_headroom = 1.6;
  std::int64_t b_pairs = 0;  // h2hPairCount after it
};

/// The SHA-256 of distances (int64 little-endian) and parents (int32) by id: the golden's
/// canonical encodings of a result (ADR 0080).
std::pair<std::string, std::string> result_digests(
    const dyng::resources& res, const dyng::hyper_sssp::result<std::int32_t>& r) {
  const std::vector<std::int64_t> d = dyng::to_vector(res, r.hyperedge_distances());
  const std::vector<std::int32_t> p = dyng::to_vector(res, r.hyperedge_parents());
  dyng::test::sha256 hd64;
  for (const std::int64_t v : d) {
    const auto u = static_cast<std::uint64_t>(v);
    unsigned char b[8];
    for (int k = 0; k < 8; ++k) b[k] = static_cast<unsigned char>(u >> (8 * k));
    hd64.update(b, 8);
  }
  dyng::test::sha256 hp;
  hp.update_le(p);
  return {hd64.hex(), hp.hex()};
}

/// The engine of --cuda-engine.
dyng::engine engine_of(const std::string& e) {
  if (e == "fused") return dyng::engine::fused;
  if (e == "operators") return dyng::engine::operators;
  return dyng::engine::automatic;
}

int bench(const bench_options& o, clock_type::time_point start) {
  const dyng::resources res = resources_of(o.backend, o.threads, o.device);
  dyng::profiler_options popt;
  popt.sync_stages = o.sync_stages;
  popt.nvtx = o.nvtx;  // the memory reading's ranges (parity/hsosp_perf.py memory)
  dyng::profiler prof(popt);
  dyng::resources r = res;
  r.attach_profiler(&prof);
  timing_log log(prof);
  r.warm_up();
  // ---- load (from the start of main: B's runRealDataset reads, builds and uploads) ----
  io::hypergraph_file_info file_info;
  const auto read = io::read_hypergraph<std::int32_t, std::int32_t>(
      o.hg, io::hypergraph_read_options::hsosp(o.seed, o.maxcard), &file_info);
  // B's virtual source row {s} (s: the largest degree, the first of them) and target row {t}
  // (t: uniform from std::mt19937_64(seed), drawn before the weights).
  const std::int32_t n = read.num_vertices;
  std::vector<std::int64_t> degree(static_cast<std::size_t>(n), 0);
  for (const std::int32_t v : read.members) ++degree[static_cast<std::size_t>(v)];
  const auto s_vertex =
      static_cast<std::int32_t>(std::max_element(degree.begin(), degree.end()) - degree.begin());
  std::mt19937_64 rng(o.seed);
  const auto t_vertex = static_cast<std::int32_t>(
      dyng::detail::legacy_uniform_int64(rng, 0, static_cast<std::uint64_t>(n) - 1));
  dyng::hyperedge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.offsets.clear();
  list.members.clear();
  list.weights.clear();
  list.offsets.reserve(read.offsets.size() + 2);
  list.members.reserve(read.members.size() + 2);
  list.weights.reserve(read.weights.size() + 2);
  list.offsets.push_back(0);
  list.members.push_back(s_vertex);
  list.offsets.push_back(1);
  list.weights.push_back(1);  // B's weight 0: the container's weights are >= 1
  for (std::size_t i = 1; i < read.offsets.size(); ++i) list.offsets.push_back(read.offsets[i] + 1);
  list.members.insert(list.members.end(), read.members.begin(), read.members.end());
  list.weights.insert(list.weights.end(), read.weights.begin(), read.weights.end());
  list.members.push_back(t_vertex);
  list.offsets.push_back(static_cast<std::int64_t>(list.members.size()));
  list.weights.push_back(1);
  const auto rows = static_cast<std::int64_t>(list.weights.size());
  // B's capacities (hsospBench.cu:496-511, 209-218).
  const std::int64_t planned = (o.kind == "hyperedge" ? o.batch : 0) * o.batches + 65536;
  dyng::hypergraph_properties props;
  props.storage = dyng::hypergraph_storage::slack_csr;
  props.line_graph = dyng::line_graph_policy::slack_csr;
  props.reuse = dyng::id_reuse::insert_first;
  props.capacity =
      dyng::capacity_policy::mosp_escher(planned * (o.maxcard + 2 * 128 + 16) + (8LL << 20));
  props.capacity.initial_headroom = 0.3;  // headroomFactor 1.3 (M8b deviation 2)
  dyng::detail::hyper::slack::knobs knobs;
  knobs.entry_headroom = 1.6;  // kEntryHeadroom
  knobs.node_capacity = rows + planned + 1024;
  whgraph h = access::hypergraph_access::build(r, list.view(), props, knobs);
  // The two results, as B's two states: A computed (the initial solve), B allocated (the static
  // recompute of every batch writes it; MOSP_ESCHER's stateB, ADR 0080's scope rule).
  dyng::hyper_sssp::options sopt;
  sopt.max_iterations = o.max_iterations;
  sopt.work_budget = o.work_budget;
  sopt.cuda_engine = engine_of(o.cuda_engine);
  dyng::detail::hyper_sssp_endpoints ends;
  ends.source_hyperedge = 0;  // B's source row (B id 1)
  ends.zero_weight_a = 0;
  ends.zero_weight_b = rows - 1;  // B's target row (B id m + 2)
  std::optional<dyng::hyper_sssp::result<std::int32_t>> state_a, state_b;
  double reachable = 0;
  if (!o.maintain_only) {
    if (o.public_source) {
      state_a.emplace(dyng::hyper_sssp::compute(r, h, s_vertex, sopt));
      state_b.emplace(dyng::hyper_sssp::compute(r, h, s_vertex, sopt));
    } else {
      state_a.emplace(dyng::detail::hyper_sssp_virtual::compute(r, h, ends, sopt));
      state_b.emplace(dyng::detail::hyper_sssp_virtual::allocate(r, h, ends, sopt));
    }
    // The reachable fraction from the initial solve (B downloads it inside its load).
    const std::vector<std::int64_t> d = dyng::to_vector(r, state_a->hyperedge_distances());
    std::int64_t reach = 0;
    for (const std::int64_t x : d) reach += x < INT64_MAX / 8 ? 1 : 0;
    reachable = h.num_hyperedges() ? static_cast<double>(reach) / h.num_hyperedges() : 0.0;
  }
  r.synchronize();
  log.timer("load_ms", 0, ms_since(start));
  log.stages(0);
  int status = 0;
  if (!o.check_load.empty()) {
    // Once, outside every timer: the loaded rows and weights are B's (its load record).
    const auto recs = hd::read_dump(o.check_load);
    for (const hd::record& rec : recs) {
      if (rec.tag != "load" || rec.str("phase") != "enter") continue;
      std::vector<std::int64_t> b_weights = rec.longs("weights");
      b_weights.front() = b_weights.front() == 0 ? 1 : b_weights.front();
      b_weights.back() = b_weights.back() == 0 ? 1 : b_weights.back();
      const std::vector<std::int64_t> w(list.weights.begin(), list.weights.end());
      const bool same = rec.longs("rows.offsets") == list.offsets &&
                        rec.ints("rows.values") == list.members && b_weights == w &&
                        rec.i64("source_he") == 1 && rec.i64("target_he") == rows;
      std::printf("[check-load] %s: %lld rows, the rows and weights %s B's\n", o.hg.c_str(),
                  static_cast<long long>(rows), same ? "equal" : "DIFFER from");
      if (!same) status = 1;
      break;
    }
  }
  // ---- the golden batches (read before the timers; HD5) ----
  std::vector<golden_batch> batches;
  {
    const auto recs = hd::read_dump(o.golden);
    golden_batch pending;
    bool has_pending = false;
    std::int64_t source = 1, target = rows;
    for (const hd::record& rec : recs) {
      if (rec.tag == "load" && rec.str("phase") == "enter") {
        source = rec.i64("source_he");
        target = rec.i64("target_he");
      } else if (rec.tag == "begin_batch" && rec.str("phase") == "enter") {
        pending = golden_batch{};
        has_pending = true;
        const auto keep = [&](std::int64_t b_id) {
          return b_id >= 1 && b_id != source && b_id != target;
        };
        for (const std::int32_t id : rec.ints("he_delete"))
          if (keep(id)) pending.batch.delete_hyperedge(static_cast<std::int64_t>(id) - 1);
        const auto& dh = rec.ints("vtx_delete.he");
        const auto& dv = rec.ints("vtx_delete.vertex");
        for (std::size_t i = 0; i < dh.size(); ++i)
          if (keep(dh[i]))
            pending.batch.delete_incidence(static_cast<std::int64_t>(dh[i]) - 1, dv[i]);
        const auto& ih = rec.ints("vtx_insert.he");
        const auto& iv = rec.ints("vtx_insert.vertex");
        for (std::size_t i = 0; i < ih.size(); ++i)
          if (keep(ih[i]))
            pending.batch.insert_incidence(static_cast<std::int64_t>(ih[i]) - 1, iv[i]);
        const auto& off = rec.longs("he_insert.offsets");
        const auto& val = rec.ints("he_insert.values");
        const auto& w = rec.longs("he_insert.weight");
        for (std::size_t i = 0; i + 1 < off.size(); ++i) {
          const std::vector<std::int32_t> m(val.begin() + off[i], val.begin() + off[i + 1]);
          pending.batch.insert_hyperedge(dyng::host_view(m), static_cast<std::int32_t>(w[i]));
        }
      } else if (rec.tag == "apply_batch" && has_pending) {
        pending.entry_headroom =
            rec.has("entry_headroom")
                ? std::strtod(real_text(rec, "entry_headroom", "1.6").c_str(), nullptr)
                : 1.6;
        pending.b_pairs = rec.i64("h2h_pair_count");
        batches.push_back(std::move(pending));
        has_pending = false;
      }
    }
  }
  if (static_cast<std::int64_t>(batches.size()) != o.batches) {
    std::fprintf(stderr, "dyng-compat-hsosp: the golden %s has %zu batches, not %lld\n",
                 o.golden.c_str(), batches.size(), static_cast<long long>(o.batches));
    return 2;
  }
  std::ofstream csv;
  if (!o.out.empty()) {
    csv.open(o.out);
    if (o.maintain_only) {
      csv << "suite,experiment,dataset,m_hyperedges,n_vertices,c_min,c_max,batch_kind,batch_size,"
             "del_pct,rep,seed,t_delta_ms,t_csr_apply_ms,h2h_pairs,overflow_rebuilds,graph_mb\n";
    } else {
      // hsospBench's 44 columns (hsospBench.cu:155-165); see the file comment for the columns
      // the port does not fill.
      csv << "timestamp,suite,experiment,dataset,m_hyperedges,n_vertices,c_min,c_max,pool_size,"
             "bridge_frac,avg_h2h_deg,h2h_pairs,batch_kind,batch_size,del_pct,placement,rep,seed,"
             "t_escher_ms,t_delta_ms,t_csr_apply_ms,t_sosp_update_ms,t_dynamic_total_ms,"
             "t_static_ms,speedup,iters,fallback,seeds,max_frontier,overflow_rebuilds,"
             "dev_mem_mb,escher_mb,graph_mb,reachable_frac,verified,correct,oracle,"
             "mismatch_static,mismatch_oracle,parent_errors,fallback_iters,invalidated,"
             "update_work,static_iters\n";
    }
  }
  auto* slack = access::hypergraph_access::slack(h);
  // B's graph_mb scope in bytes (DeviceH2H + its mirror + the two states; hsospBench.cu:396-398):
  // the slack_csr store, its mirror and the two results, after the load and after every batch
  // (stderr, untimed; the memory gate's accounting rows, parity/hsosp_perf.py memory).
  const auto graph_scope_bytes = [&]() -> long long {
    return static_cast<long long>(
        slack->graph_scope_bytes() +
        (state_a ? dyng::detail::hyper_sssp_virtual::state_bytes(*state_a) : 0) +
        (state_b ? dyng::detail::hyper_sssp_virtual::state_bytes(*state_b) : 0));
  };
  std::fprintf(stderr, "[memory] load graph_scope_bytes %lld\n", graph_scope_bytes());
  std::string name = o.hg;
  if (name.find_last_of('/') != std::string::npos) name = name.substr(name.find_last_of('/') + 1);
  double dynamic_sum = 0, static_sum = 0;
  for (std::size_t b = 0; b < batches.size(); ++b) {
    slack->set_entry_headroom(batches[b].entry_headroom);
    const auto batch = static_cast<std::int64_t>(b) + 1;
    r.synchronize();
    dyng::hyper_sssp::stats st;
    double update = 0, statik = 0;
    if (o.maintain_only) {
      const auto t0 = clock_type::now();
      (void)h.apply(r, batches[b].batch.view());
      r.synchronize();
      log.timer("maintain_ms", batch, ms_since(t0));
    } else {
      auto t0 = clock_type::now();
      st = dyng::hyper_sssp::update(r, h, batches[b].batch.view(), *state_a);
      r.synchronize();
      update = ms_since(t0);
      log.timer("update_ms", batch, update);
      // The static baseline: a recompute into the second, pre-allocated result.
      t0 = clock_type::now();
      dyng::detail::hyper_sssp_virtual::recompute(r, h, *state_b);
      r.synchronize();
      statik = ms_since(t0);
      log.timer("static_ms", batch, statik);
      dynamic_sum += update;
      static_sum += statik;
    }
    log.stages(batch);
    std::fprintf(stderr, "[memory] batch %lld graph_scope_bytes %lld\n",
                 static_cast<long long>(batch), graph_scope_bytes());
    const auto rows_now = slack->rows();
    const auto counters = slack->counters();
    // Untimed from here: update == static, the oracle (--verify), the round guards' digests.
    std::int64_t mismatch_static = 0, mismatch_oracle = -1, parent_errors = -1;
    std::string dist_sha, par_sha;
    if (!o.maintain_only) {
      const std::vector<std::int64_t> da = dyng::to_vector(r, state_a->hyperedge_distances());
      const std::vector<std::int64_t> db = dyng::to_vector(r, state_b->hyperedge_distances());
      for (std::size_t k = 0; k < da.size() && k < db.size(); ++k)
        mismatch_static += da[k] != db[k] ? 1 : 0;
      const bool verify = o.verify == "all" || (o.verify == "first" && b == 0);
      if (verify) {
        const auto v = h.h2v();
        const std::vector<std::int64_t> off = dyng::to_vector(r, v.row_ptr);
        const std::vector<std::int32_t> mem = dyng::to_vector(r, v.col_ind);
        const auto hw = h.weights();
        std::vector<std::int64_t> w(hw.begin(), hw.end());
        const std::vector<std::int32_t> pa = dyng::to_vector(r, state_a->hyperedge_parents());
        std::pair<std::int64_t, std::int64_t> bad;
        if (o.public_source) {
          bad = dyng::testing::check_hyper_sssp_tree(off, mem, h.num_vertices(), w, s_vertex, true,
                                                     da, pa);
        } else {
          w.front() = 0;
          w[static_cast<std::size_t>(rows - 1)] = 0;
          bad =
              dyng::testing::check_hyper_sssp_tree(off, mem, h.num_vertices(), w, 0, false, da, pa);
        }
        mismatch_oracle = bad.first;
        parent_errors = bad.second;
      }
      if (o.digests) std::tie(dist_sha, par_sha) = result_digests(r, *state_a);
      if (mismatch_static != 0 || mismatch_oracle > 0 || parent_errors > 0) status = 1;
    }
    if (o.digests) {
      // Untimed: the round's guard reads these (parity/hsosp_perf.py).
      const auto& d = access::hypergraph_access::pairs(h);
      std::printf("[batch %lld] pairs %s deleted %zu inserted %zu h2h_pairs %lld csr_rebuilds %lld",
                  static_cast<long long>(batch), pairs_digest(d).c_str(), d.deleted_a.size(),
                  d.inserted_a.size(), static_cast<long long>(rows_now.pair_count),
                  static_cast<long long>(counters.csr_rebuilds));
      if (!o.maintain_only) {
        std::printf(" distances %s parents %s fallback %d mismatch_static %lld", dist_sha.c_str(),
                    par_sha.c_str(), st.fallback_used ? 1 : 0,
                    static_cast<long long>(mismatch_static));
      }
      std::printf("\n");
      std::fflush(stdout);
    }
    if (csv.is_open()) {
      const double delta = log.sum("hypergraph.apply.normalize", batch) +
                           log.sum("hypergraph.apply.ids", batch) +
                           log.sum("hypergraph.apply.incidence", batch) +
                           log.sum("hypergraph.apply.translate", batch);
      const double graph_mb = static_cast<double>(graph_scope_bytes()) / (1024.0 * 1024.0);
      if (o.maintain_only) {
        csv << std::fixed << std::setprecision(4) << "real,paper," << name << "," << (rows - 2)
            << "," << n << ",1," << o.maxcard << "," << o.kind << "," << o.batch << "," << o.del
            << "," << b << "," << o.seed << "," << delta << ","
            << log.sum("hypergraph.apply.line_graph", batch) << "," << rows_now.pair_count << ","
            << counters.csr_rebuilds << "," << graph_mb << "\n";
      } else {
        const auto& c = dyng::detail::hyper_sssp_virtual::last_counters(*state_a);
        const double escher = log.sum("hypergraph.apply.ids", batch);
        const double csr = log.sum("hypergraph.apply.line_graph", batch);
        const double sosp = log.sum("hyper_sssp.identify_affected", batch) +
                            log.sum("hyper_sssp.seed", batch) + log.sum("hyper_sssp.loop", batch) +
                            log.sum("hyper_sssp.recompute", batch) +
                            log.sum("hyper_sssp.finalize", batch);
        const double avg_deg = h.num_hyperedges() ? 2.0 * static_cast<double>(rows_now.pair_count) /
                                                        static_cast<double>(h.num_hyperedges())
                                                  : 0.0;
        const bool verified = mismatch_oracle == 0 && parent_errors == 0;
        const bool correct = mismatch_static == 0 && mismatch_oracle <= 0 && parent_errors <= 0;
        // hsospBench's default stream format (its deterministic columns compare as text);
        // dev_mem_mb (cudaMemGetInfo) and escher_mb are not filled (see the file comment).
        csv << std::defaultfloat << std::setprecision(6) << "," << "real,paper," << name << ","
            << (rows - 2) << "," << n << ",1," << o.maxcard << ",0,0," << avg_deg << ","
            << rows_now.pair_count << "," << o.kind << "," << o.batch << "," << o.del << ",random,"
            << b << "," << o.seed << "," << escher << "," << delta << "," << csr << "," << sosp
            << "," << update << "," << statik << "," << (update > 0 ? statik / update : 0.0) << ","
            << c.iterations << "," << (st.fallback_used ? 1 : 0) << "," << st.invalidated << ","
            << c.max_frontier << "," << counters.csr_rebuilds << ",,," << graph_mb << ","
            << reachable << "," << (verified ? 1 : 0) << "," << (correct ? 1 : 0) << ","
            << (mismatch_oracle >= 0 ? "host" : "none") << "," << mismatch_static << ","
            << mismatch_oracle << "," << parent_errors << "," << c.fallback_iterations << ","
            << st.invalidated << "," << st.relaxations << ","
            << dyng::detail::hyper_sssp_virtual::last_counters(*state_b).iterations << "\n";
      }
    }
  }
  const double total = ms_since(start);
  log.timer("total_ms", 0, total);
  if (o.maintain_only) {
    double maintain_sum = 0;
    for (std::size_t b = 0; b < batches.size(); ++b)
      maintain_sum += log.sum("maintain_ms", static_cast<std::int64_t>(b) + 1);
    std::fprintf(stderr, "[e2e] %s: load %.0f ms, %lld batches: maintain %.0f ms, wall %.0f ms\n",
                 name.c_str(), log.sum("load_ms", 0), static_cast<long long>(batches.size()),
                 maintain_sum, total);
  } else {
    log.timer("dynamic_ms", 0, dynamic_sum);
    log.timer("static_sum_ms", 0, static_sum);
    // hsospBench's [e2e] line (hsospBench.cu:532-536).
    std::fprintf(stderr,
                 "[e2e] %s: load %.0f ms, %lld batches: dynamic %.0f ms, static %.0f ms, wall "
                 "%.0f ms\n",
                 name.c_str(), log.sum("load_ms", 0), static_cast<long long>(batches.size()),
                 dynamic_sum, static_sum, total);
  }
  if (!o.timing.empty()) log.write(o.timing, o);
  return status;
}

// --- replay --algorithm --------------------------------------------------------------------------

int replay_results(const std::string& dump, const std::string& mode, const std::string& backend,
                   int device, bool check_compute, const std::string& json) {
  dyng::test::hsosp_results::options opt;
  if (mode == "by_id") {
    opt.mode = dyng::test::hsosp_results::mode::by_id;
  } else if (mode == "by_id_public") {
    opt.mode = dyng::test::hsosp_results::mode::by_id_public;
  } else if (mode == "by_key") {
    opt.mode = dyng::test::hsosp_results::mode::by_key;
  } else {
    return usage();
  }
  opt.check_compute = check_compute;
  if (backend != "sequential" && backend != "cuda") return usage();
  const auto rep = dyng::test::hsosp_results::replay(hd::read_dump(dump),
                                                     resources_of(backend, 0, device), opt, dump);
  std::printf(
      "%s %s %s: %lld sessions, %lld batches, %lld updates, %lld recomputes, %lld ids, "
      "%lld mismatches%s\n",
      rep.ok() ? "ok" : "MISMATCH", rep.mode.c_str(), dump.c_str(),
      static_cast<long long>(rep.sessions), static_cast<long long>(rep.batches),
      static_cast<long long>(rep.updates), static_cast<long long>(rep.recomputes),
      static_cast<long long>(rep.ids_compared), static_cast<long long>(rep.mismatch_count),
      rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
  if (!json.empty()) {
    std::ofstream out(json);
    out << "[\n" << rep.to_json() << "\n]\n";
  }
  return rep.ok() ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const auto start = clock_type::now();
  if (argc < 3) return usage();
  const std::string cmd = argv[1];
  try {
    if (cmd == "to-dgb") {
      if (argc != 4) return usage();
      return to_dgb(argv[2], argv[3]);
    }
    if (cmd == "replay") {
      std::string backend = "sequential", algorithm, json;
      int threads = 0, device = 0;
      bool line_graph = false, check_compute = false;
      for (int i = 3; i < argc; ++i) {
        const std::string a = argv[i];
        if (a == "--backend" && i + 1 < argc) {
          backend = argv[++i];
        } else if (a == "--threads" && i + 1 < argc) {
          threads = std::atoi(argv[++i]);
        } else if (a == "--device" && i + 1 < argc) {
          device = std::atoi(argv[++i]);
        } else if (a == "--line-graph") {
          line_graph = true;
        } else if (a == "--algorithm" && i + 1 < argc) {
          algorithm = argv[++i];
        } else if (a == "--check-compute") {
          check_compute = true;
        } else if (a == "--json" && i + 1 < argc) {
          json = argv[++i];
        } else {
          return usage();
        }
      }
      if (!algorithm.empty())
        return replay_results(argv[2], algorithm, backend, device, check_compute, json);
      if (backend != "sequential" && backend != "openmp" && backend != "cuda") return usage();
      return replay(argv[2], backend, threads, device, line_graph);
    }
    if (cmd == "bench") {
      bench_options o;
      for (int i = 1; i < argc; ++i) o.argv.emplace_back(argv[i]);
      for (int i = 2; i < argc; ++i) {
        const std::string a = argv[i];
        const auto value = [&]() -> std::string {
          if (i + 1 >= argc) throw std::invalid_argument("missing value for " + a);
          return argv[++i];
        };
        if (a == "--hg")
          o.hg = value();
        else if (a == "--kind")
          o.kind = value();
        else if (a == "--batch")
          o.batch = std::stoll(value());
        else if (a == "--batches")
          o.batches = std::stoll(value());
        else if (a == "--del")
          o.del = std::stod(value());
        else if (a == "--seed")
          o.seed = std::stoull(value());
        else if (a == "--maxcard")
          o.maxcard = std::stoll(value());
        else if (a == "--verify")
          o.verify = value();
        else if (a == "--golden")
          o.golden = value();
        else if (a == "--out")
          o.out = value();
        else if (a == "--timing")
          o.timing = value();
        else if (a == "--check-load")
          o.check_load = value();
        else if (a == "--device")
          o.device = std::stoi(value());
        else if (a == "--threads")
          o.threads = std::stoi(value());
        else if (a == "--maintain-only")
          o.maintain_only = true;
        else if (a == "--sync-stages")
          o.sync_stages = true;
        else if (a == "--digests")
          o.digests = true;
        else if (a == "--nvtx")
          o.nvtx = true;
        else if (a == "--maxiter")
          o.max_iterations = std::stoll(value());
        else if (a == "--work-budget")
          o.work_budget = std::stod(value());
        else if (a == "--public")
          o.public_source = true;
        else if (a == "--backend")
          o.backend = value();
        else if (a == "--cuda-engine")
          o.cuda_engine = value();
        else
          return usage();
      }
      if (o.hg.empty() || o.golden.empty() || (o.kind != "hyperedge" && o.kind != "vertex") ||
          o.batch < 1 || o.batches < 0 || o.maxcard < 1)
        return usage();
      if (o.verify != "none" && o.verify != "first" && o.verify != "all") return usage();
      if (o.backend != "cuda" && o.backend != "sequential") return usage();
      if (o.cuda_engine != "automatic" && o.cuda_engine != "fused" && o.cuda_engine != "operators")
        return usage();
      return bench(o, start);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng-compat-hsosp: %s\n", e.what());
    return 2;
  }
  return usage();
}
