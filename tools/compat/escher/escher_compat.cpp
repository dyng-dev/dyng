// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file escher_compat.cpp
 * @brief dyng-compat-escher: ESCHER-GPU@abcf9b2's batch dumps (the batch records of M8a's CBST
 *        traces) as dynG files, and their replay through dyng::hypergraph (M8b, criterion 14; for
 *        the triad_count parity of M9).
 *
 *     dyng-compat-escher to-dgb <trace.cbt[.zst]> <out-prefix> [--tmp DIR]
 *     dyng-compat-escher replay <prefix> [--backend sequential|openmp|cuda] [--threads N]
 *
 * to-dgb writes <prefix>.hg (A's loaded hypergraph in A's ids, read back by
 * io::read_hypergraph() with hypergraph_read_options::escher_gpu(): hyperedge i is A's key i + 1),
 * <prefix>.dgb (A's batches: deletions by id = key - 1, insertions as generated; the flags
 * deleted_members and expected_ids carry A's deletedRows and insertedKeys - 1) and <prefix>.txt
 * (A's command line, its vertex count maxVertexId, which a .hg file does not hold when the largest
 * vertex is isolated, and the payload capacity of its H2V tree, which capacity_policy::
 * escher_gpu() takes).
 *
 * replay builds the hypergraph from <prefix>.hg under {id_reuse::erase_first,
 * capacity_policy::escher_gpu(capacity)} and streams <prefix>.dgb with io::batch_reader (through
 * pinned buffers to the device on cuda) into two hypergraphs: one deletes by id and must give the
 * expected ids, the other deletes by content (escher_dump::by_content_resolver: each deleted
 * member set resolved to the smallest live id with that set not taken yet in the batch) and must
 * hold the same live member sets after every batch. Exit status: 0 equal, 1 a difference, 2 a
 * usage or input error.
 */
#include "parity/escher/cbst_trace.hpp"
#include "parity/escher/escher_dump.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

namespace tr = dyng::test::cbst_trace;
namespace ed = dyng::test::escher_dump;
namespace io = dyng::io;
using rows = ed::rows;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

int usage() {
  std::fprintf(stderr,
               "usage: dyng-compat-escher to-dgb <trace.cbt[.zst]> <out-prefix> [--tmp DIR]\n"
               "       dyng-compat-escher replay <prefix> [--backend sequential|openmp|cuda] "
               "[--threads N]\n");
  return 2;
}

// --- to-dgb ------------------------------------------------------------------------------------

int to_dgb(const std::string& trace, const std::string& prefix, const std::string& tmp) {
  tr::trace_source source(trace, tmp);
  ed::decoder d;
  tr::record rec;
  std::map<std::string, std::int64_t> capacity;  // payload capacity per tree name
  std::map<std::uint32_t, std::string> tree_name;
  std::unique_ptr<io::batch_writer> out;
  std::int64_t batches = 0;
  std::int64_t hyperedges = 0;
  while (source.reader().next(rec)) {
    if (rec.kind == tr::record_kind::tree) {
      capacity.emplace(rec.text("dataset_name"), rec.integer("payload_capacity"));
      continue;
    }
    switch (d.feed(rec, source.reader())) {
      case ed::decoder::event::header:
        if (d.repo() != "ESCHER-GPU") {
          std::fprintf(stderr, "dyng-compat-escher: %s is a trace of %s, not ESCHER-GPU\n",
                       trace.c_str(), d.repo().c_str());
          return 2;
        }
        break;
      case ed::decoder::event::exception:
        std::fprintf(stderr, "dyng-compat-escher: the original threw %s\n", d.exception().c_str());
        return 2;
      case ed::decoder::event::build: {
        if (out) {
          std::fprintf(stderr, "dyng-compat-escher: the trace has more than one build\n");
          return 2;
        }
        const ed::build_record& b = d.build();
        dyng::hyperedge_list<std::int32_t> list;
        list.num_vertices = b.num_vertices;
        for (const auto& r : b.loaded) list.add_hyperedge(dyng::host_view(r));
        io::write_hypergraph(prefix + ".hg", list.view(), 1);
        hyperedges = list.num_hyperedges();
        out = std::make_unique<io::batch_writer>(
            prefix + ".dgb",
            io::hyperedge_batch_file_header<std::int32_t, dyng::unweighted>(
                io::batch_file_flags::deleted_members | io::batch_file_flags::expected_ids));
        break;
      }
      case ed::decoder::event::batch: {
        if (!out) {
          std::fprintf(stderr, "dyng-compat-escher: a batch before the build\n");
          return 2;
        }
        const ed::batch_record& b = d.batch();
        dyng::hyperedge_batch<std::int32_t, dyng::unweighted> batch;
        for (const std::int64_t id : b.deleted_ids) batch.delete_hyperedge(id);
        for (const auto& r : b.inserted_rows) batch.insert_hyperedge(dyng::host_view(r));
        std::vector<std::int64_t> offsets{0};
        std::vector<std::int32_t> members;
        for (const auto& r : b.deleted_rows) {
          members.insert(members.end(), r.begin(), r.end());
          offsets.push_back(static_cast<std::int64_t>(members.size()));
        }
        io::hyperedge_batch_annotations<std::int32_t> a;
        a.deleted_offsets = dyng::host_view(offsets);
        a.deleted_members = dyng::host_view(members);
        a.expected_ids = dyng::host_view(b.inserted_ids);
        out->write(batch.view(), a);
        ++batches;
        break;
      }
      default:
        break;
    }
  }
  if (!out) {
    std::fprintf(stderr, "dyng-compat-escher: the trace has no build record\n");
    return 2;
  }
  out->close();
  const ed::escher_options a = ed::escher_options::parse(d.argv());
  // ESCHER-GPU's payloadCapacity: max(requested, size + size / 8 + 1024) for its H2V tree; the
  // requested capacity is the larger traced one (as dyng_hypergraph_ids recovers it).
  std::int64_t flat = 0;
  for (const auto& r : d.build().loaded) flat += (static_cast<std::int64_t>(r.size()) + 4) / 4 * 4;
  const std::int64_t formula = flat + flat / 8 + 1024;
  const std::int64_t traced = capacity.count("H2V") != 0 ? capacity["H2V"] : formula;
  std::ofstream txt(prefix + ".txt");
  txt << "# dyng-compat-escher to-dgb: the run of ESCHER-GPU@abcf9b2 behind " << prefix
      << ".hg / .dgb\n";
  txt << "argv";
  for (std::size_t i = 1; i < d.argv().size(); ++i) txt << ' ' << d.argv()[i];
  txt << "\nseed " << a.seed << "\npayload_capacity " << (traced > formula ? traced : 0)
      << "\nvertices " << d.build().num_vertices << "\nhyperedges " << hyperedges << "\nbatches "
      << batches << "\n";
  if (!txt) {
    std::fprintf(stderr, "dyng-compat-escher: cannot write %s.txt\n", prefix.c_str());
    return 2;
  }
  std::printf("%s: %lld hyperedges, %lld batches -> %s.hg, %s.dgb, %s.txt\n", trace.c_str(),
              static_cast<long long>(hyperedges), static_cast<long long>(batches), prefix.c_str(),
              prefix.c_str(), prefix.c_str());
  return 0;
}

// --- replay ------------------------------------------------------------------------------------

// The live member sets of a hypergraph, sorted (by content: ids do not matter).
rows live_sets(const dyng::resources& res, const hgraph& h) {
  const auto v = h.h2v();
  const std::vector<std::int64_t> ptr = dyng::to_vector(res, v.row_ptr);
  const std::vector<std::int32_t> col = dyng::to_vector(res, v.col_ind);
  rows out;
  for (std::size_t i = 0; i + 1 < ptr.size(); ++i) {
    if (ptr[i + 1] > ptr[i]) out.emplace_back(col.begin() + ptr[i], col.begin() + ptr[i + 1]);
  }
  std::sort(out.begin(), out.end());
  return out;
}

int replay(const std::string& prefix, const std::string& backend, int threads) {
  dyng::resources res = backend == "cuda"     ? dyng::resources::cuda(0)
                        : backend == "openmp" ? dyng::resources::openmp(threads)
                                              : dyng::resources::sequential();
  std::int64_t capacity = 0;
  std::int64_t vertices = -1;
  {
    std::ifstream txt(prefix + ".txt");
    for (std::string line; std::getline(txt, line);) {
      std::istringstream s(line);
      std::string key;
      s >> key;
      if (key == "payload_capacity") s >> capacity;
      if (key == "vertices") s >> vertices;
    }
  }
  io::hypergraph_read_options ro = io::hypergraph_read_options::escher_gpu();
  ro.num_vertices = vertices;
  const auto list = io::read_hypergraph<std::int32_t, dyng::unweighted>(prefix + ".hg", ro);
  dyng::hypergraph_properties props;
  props.reuse = dyng::id_reuse::erase_first;
  props.capacity = dyng::capacity_policy::escher_gpu(capacity);
  props.line_graph = dyng::line_graph_policy::none;
  hgraph by_id = hgraph::from_hyperedges(res, list.view(), props);
  hgraph by_content = hgraph::from_hyperedges(res, list.view(), props);
  ed::by_content_resolver resolver;
  {
    rows loaded;
    for (std::int64_t i = 0; i < list.num_hyperedges(); ++i) {
      loaded.emplace_back(list.members.begin() + list.offsets[static_cast<std::size_t>(i)],
                          list.members.begin() + list.offsets[static_cast<std::size_t>(i) + 1]);
    }
    resolver.reset(loaded);
  }
  io::batch_reader in(res, prefix + ".dgb", io::batch_reader_options{list.num_vertices});
  const io::batch_file_header& h = in.header();
  if ((h.flags & io::batch_file_flags::expected_ids) == 0 ||
      (h.flags & io::batch_file_flags::deleted_members) == 0) {
    std::fprintf(stderr, "dyng-compat-escher: %s.dgb lacks the expected ids or deleted members\n",
                 prefix.c_str());
    return 2;
  }
  std::int64_t batches = 0, ids = 0, ambiguous = 0, differences = 0;
  while (in.next()) {
    ++batches;
    const auto batch = in.hyperedge_batch<std::int32_t, dyng::unweighted>();
    const auto ann = in.annotations<std::int32_t>();
    // By id: the batch as it is (its views are in the hypergraph's memory space).
    const auto s = by_id.apply(res, batch);
    const std::vector<std::int64_t> expected = dyng::to_vector(res, ann.expected_ids);
    ids += static_cast<std::int64_t>(expected.size());
    if (s.inserted_ids != expected) {
      ++differences;
      std::fprintf(stderr, "batch %lld: the inserted ids differ from the expected ids\n",
                   static_cast<long long>(batches));
    }
    // By content: the deleted member sets resolved in the by-content hypergraph.
    const std::vector<std::int64_t> off = dyng::to_vector(res, ann.deleted_offsets);
    const std::vector<std::int32_t> mem = dyng::to_vector(res, ann.deleted_members);
    rows deleted_sets;
    for (std::size_t j = 0; j + 1 < off.size(); ++j) {
      deleted_sets.emplace_back(mem.begin() + off[j], mem.begin() + off[j + 1]);
    }
    const std::vector<std::int64_t> by_key = dyng::to_vector(res, batch.delete_hyperedges);
    std::vector<bool> amb;
    const std::vector<std::int64_t> resolved = resolver.resolve(deleted_sets, &amb);
    dyng::hyperedge_batch<std::int32_t, dyng::unweighted> content_batch;
    for (std::size_t j = 0; j < resolved.size(); ++j) {
      if (resolved[j] < 0) {
        ++differences;
        std::fprintf(stderr, "batch %lld: deleted set %zu has no live hyperedge\n",
                     static_cast<long long>(batches), j);
        continue;
      }
      if (amb[j] && resolved[j] != by_key[j]) ++ambiguous;
      content_batch.delete_hyperedge(resolved[j]);
    }
    const std::vector<std::int64_t> ins_off = dyng::to_vector(res, batch.insert_offsets);
    const std::vector<std::int32_t> ins_mem = dyng::to_vector(res, batch.insert_members);
    rows inserted;
    for (std::size_t i = 0; i + 1 < ins_off.size(); ++i) {
      inserted.emplace_back(ins_mem.begin() + ins_off[i], ins_mem.begin() + ins_off[i + 1]);
      content_batch.insert_hyperedge(dyng::host_view(inserted.back()));
    }
    const auto c = by_content.apply(res, content_batch.view());
    resolver.commit(content_batch.delete_hyperedges(), inserted, c.inserted_ids);
    if (live_sets(res, by_id) != live_sets(res, by_content)) {
      ++differences;
      std::fprintf(stderr, "batch %lld: by content and by id hold other member sets\n",
                   static_cast<long long>(batches));
    }
  }
  by_id.check_integrity(res);
  by_content.check_integrity(res);
  std::printf(
      "%s %s: %lld batches, %lld ids as expected by id, by content the same member sets "
      "(%lld ambiguous resolutions), %lld differences\n",
      differences == 0 ? "ok" : "DIFFERENT", prefix.c_str(), static_cast<long long>(batches),
      static_cast<long long>(ids), static_cast<long long>(ambiguous),
      static_cast<long long>(differences));
  return differences == 0 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 3) return usage();
  const std::string command = argv[1];
  std::string tmp = std::filesystem::temp_directory_path().string();
  std::string backend = "sequential";
  int threads = 0;
  std::vector<std::string> args;
  for (int i = 2; i < argc; ++i) {
    const std::string a = argv[i];
    if ((a == "--tmp" || a == "--backend" || a == "--threads") && i + 1 >= argc) return usage();
    if (a == "--tmp") {
      tmp = argv[++i];
    } else if (a == "--backend") {
      backend = argv[++i];
    } else if (a == "--threads") {
      threads = std::atoi(argv[++i]);
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      args.push_back(a);
    }
  }
  try {
    if (command == "to-dgb" && args.size() == 2) return to_dgb(args[0], args[1], tmp);
    if (command == "replay" && args.size() == 1) return replay(args[0], backend, threads);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng-compat-escher: %s\n", e.what());
    return 2;
  }
  return usage();
}
