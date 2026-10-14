// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_ids_main.cpp
 * @brief dyng_hypergraph_ids: container-level id parity of dyng::hypergraph against the batch
 *        records of ESCHER-GPU@abcf9b2's and MOSP_ESCHER@4b86159's CBST traces (M8b, criterion 4;
 *        ADR 0041, section 11). parity/compare.py hypergraph_ids drives it over the golden sets.
 *
 *     dyng_hypergraph_ids [--backend sequential|openmp|cuda] [--threads N]
 *                         [--mode by_id|by_content|both] [--no-model] [--no-integrity]
 *                         [--line-graph] [--apply-only] [--timing FILE] [--sync-stages] [--nvtx]
 *                         [--legacy-generator] [--hg FILE] [--json OUT] [--tmp DIR] <trace>...
 *
 * --line-graph (ESCHER-GPU's traces): the by-id container keeps the line graph
 * (line_graph_policy::sorted_merge), its H2H is compared with ESCHER-GPU's after the build and
 * every batch (M8b criterion 7), and its device bytes with A's DynamicHypergraph::deviceBytes()
 * (the memory gate, criterion 8). --apply-only applies the batches and records the counters
 * without comparing rows (the memory and timing runs; implies --no-model --no-integrity).
 * --timing FILE (one trace) attaches a profiler and writes the container's stages per batch in
 * ESCHER-GPU's --timing format ({"meta", "stages": [{"stage", "batch", "ms"}]}; batch 0 is the
 * build); --sync-stages synchronizes the stream at the stage boundaries (as A's ScopedTimer does),
 * --nvtx emits an NVTX range per stage (Nsight Systems).
 *
 * --legacy-generator (ESCHER-GPU's traces, M8b criterion 16): generators::legacy draws A's
 * synthetic hypergraph and every batch from the by-id container's live state with one
 * std::mt19937_64(--seed of the trace's command line); each must equal A's. --hg FILE (criterion
 * 13): FILE, the hypergraph the run loaded, is read with the original's option set
 * (io::hypergraph_read_options::escher_gpu(), hsosp(--seed, --maxcard)) and must give its loaded
 * rows (and, for MOSP_ESCHER, its vertex count and weights).
 *
 * A trace is a `.cbt` file or a `.cbt.zst` (as dyng_cbst_replay reads them). Prints one line per
 * trace; --json writes a JSON array of the reports. Exit status: 0 every trace matched, 1 a
 * mismatch, 2 a usage or input error.
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/hypergraph_ids.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <fstream>
#include <string>
#include <vector>

namespace {

int usage() {
  std::fprintf(stderr,
               "usage: dyng_hypergraph_ids [--backend sequential|openmp|cuda] [--threads N] "
               "[--mode by_id|by_content|both] [--no-model] [--no-integrity] [--line-graph] "
               "[--apply-only] [--timing FILE] [--sync-stages] [--nvtx] [--legacy-generator] "
               "[--hg FILE] [--json OUT] [--tmp DIR] "
               "<trace>...\n");
  return 2;
}

}  // namespace

int main(int argc, char** argv) {
  std::string backend = "sequential", json_path, tmp_dir, timing_path;
  dyng::profiler_options prof_options;
  int threads = 0;
  dyng::test::hypergraph_ids::options opt;
  std::vector<std::string> traces;
  for (int i = 1; i < argc; ++i) {
    const std::string a = argv[i];
    auto value = [&]() -> std::string {
      if (i + 1 >= argc) {
        std::fprintf(stderr, "%s needs a value\n", a.c_str());
        std::exit(2);
      }
      return argv[++i];
    };
    if (a == "--backend") {
      backend = value();
    } else if (a == "--threads") {
      threads = std::atoi(value().c_str());
    } else if (a == "--mode") {
      opt.mode = value();
      if (opt.mode != "by_id" && opt.mode != "by_content" && opt.mode != "both") return usage();
    } else if (a == "--no-model") {
      opt.model = false;
    } else if (a == "--no-integrity") {
      opt.integrity = false;
    } else if (a == "--line-graph") {
      opt.line_graph = true;
    } else if (a == "--apply-only") {
      opt.compare = false;
      opt.model = false;
      opt.integrity = false;
    } else if (a == "--legacy-generator") {
      opt.legacy_generator = true;
    } else if (a == "--hg") {
      opt.hg_file = value();
    } else if (a == "--timing") {
      timing_path = value();
    } else if (a == "--sync-stages") {
      prof_options.sync_stages = true;
    } else if (a == "--nvtx") {
      prof_options.nvtx = true;
    } else if (a == "--json") {
      json_path = value();
    } else if (a == "--tmp") {
      tmp_dir = value();
    } else if (!a.empty() && a[0] == '-') {
      return usage();
    } else {
      traces.push_back(a);
    }
  }
  if (traces.empty() || (!timing_path.empty() && traces.size() != 1)) return usage();
  dyng::resources res;
  try {
    if (backend == "sequential") {
      res = dyng::resources::sequential();
    } else if (backend == "openmp") {
      res = dyng::resources::openmp(threads);
    } else if (backend == "cuda") {
      res = dyng::resources::cuda(0);
    } else {
      return usage();
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "dyng_hypergraph_ids: %s\n", e.what());
    return 2;
  }
  dyng::profiler prof(prof_options);
  if (!timing_path.empty() || prof_options.nvtx) res.attach_profiler(&prof);
  bool all_ok = true;
  std::string json = "[";
  for (std::size_t t = 0; t < traces.size(); ++t) {
    try {
      dyng::test::cbst_compare::trace_source source(traces[t], tmp_dir);
      const auto rep = dyng::test::hypergraph_ids::replay(source.reader(), res, opt, traces[t]);
      std::printf(
          "%s %s %s: %s, %lld sessions, %lld batches, %lld ids, %lld rows, %lld mismatches%s\n",
          rep.ok() ? "ok" : "MISMATCH", rep.backend.c_str(), rep.mode.c_str(), traces[t].c_str(),
          static_cast<long long>(rep.sessions), static_cast<long long>(rep.batches),
          static_cast<long long>(rep.ids_compared), static_cast<long long>(rep.rows_compared),
          static_cast<long long>(rep.mismatch_count),
          rep.mismatches.empty() ? "" : (" (first: " + rep.mismatches[0] + ")").c_str());
      all_ok = all_ok && rep.ok();
      json += (t ? ",\n" : "\n") + rep.to_json();
      if (!timing_path.empty()) {
        std::ofstream out(timing_path);
        out << "{\n  \"meta\": {\"trace\": \"" << traces[t] << "\", \"backend\": \"" << backend
            << "\"},\n  \"stages\": [\n";
        for (std::size_t i = 0; i < rep.stage_times.size(); ++i) {
          const auto& st = rep.stage_times[i];
          char ms[64];
          std::snprintf(ms, sizeof ms, "%.3f", st.ms);
          out << "    {\"stage\": \"" << st.stage << "\", \"batch\": " << st.batch
              << ", \"ms\": " << ms << "}" << (i + 1 < rep.stage_times.size() ? "," : "") << "\n";
        }
        out << "  ]\n}\n";
        if (!out) {
          std::fprintf(stderr, "dyng_hypergraph_ids: cannot write %s\n", timing_path.c_str());
          return 2;
        }
      }
    } catch (const std::exception& e) {
      std::fprintf(stderr, "dyng_hypergraph_ids: %s: %s\n", traces[t].c_str(), e.what());
      return 2;
    }
  }
  json += "\n]\n";
  if (!json_path.empty()) {
    std::ofstream out(json_path);
    out << json;
    if (!out) {
      std::fprintf(stderr, "dyng_hypergraph_ids: cannot write %s\n", json_path.c_str());
      return 2;
    }
  }
  return all_ok ? 0 : 1;
}
