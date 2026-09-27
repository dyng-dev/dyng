// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/mosp.cpp (the `mosp` driver: options, concurrent reads
// and writes, report lines) and src/mospPrep.cpp (`mospPrep init`)
/**
 * @file mosp_compat.cpp
 * @brief dyng-compat-mosp: a drop-in clone of the SOSP part of MOSP-OpenMP's `mosp` driver and of
 *        `mospPrep init`, built on dynG, for the parity harness (PLAN Section 5.6, tools/compat).
 *
 * Usage:
 *   dyng-compat-mosp --graph <csrPrefix> --changes <dir> --init <dir> [options]
 *   dyng-compat-mosp init <csrPrefix> <outDir> [--source s] [-k K] [--backend b] [--threads t]
 *
 * Update mode reads the inputs of `mosp` (the CSR text files <prefix>{RowPtr,ColInd,Values}.txt,
 * <changes>/insert.txt and delete.txt, and the initial trees <init>/obj<k>/distancesOriginal.txt
 * and SSSPTreeOriginal.txt), applies the batch once to a graph with
 * graph_properties::mosp_compatible() and updates the K trees with dyng::update_each(), then
 * writes <out>/obj<k>/distancesUpdated.txt and SSSPTreeUpdated.txt, byte-compatible with `mosp`.
 * The combined graph of the MOSP update (Steps 2-3) belongs to the mosp algorithm (0.2) and is not
 * written. Init mode writes <outDir>/obj<k>/distancesOriginal.txt and SSSPTreeOriginal.txt with
 * sssp::compute(), byte-compatible with `mospPrep init` (Dijkstra with lowest-id ties).
 *
 * Options (update mode; the `mosp` options keep their names):
 *   -k <K>               objectives to use (default: all of the graph; required without edges)
 *   --source <s>         source vertex (default 0)
 *   --delta <D>          near-far bucket width (default: automatic per objective)
 *   --canonicalize       apply the lowest-id tie rule to the initial trees
 *   --no-validate-inputs skip the O(n) checks of the initial trees (dynG checks by default)
 *   --out <dir>          output directory (default mosp-output)
 *   --no-output          do not write the result files
 *   --write-graph <p>    also write the updated graph as <p>{RowPtr,ColInd,Values}.txt (the format
 *                        of MOSP's writeCsrGraph; for the parity harness, not timed)
 *   --timing <csv>       write the profiler stages and counters (kind,name,value)
 *   --quiet              print only the summary line
 *   --backend <b>        sequential | openmp (default openmp if built)
 *   --threads <t>        OpenMP threads (default: the OpenMP default)
 *
 * Summary line (stable format):
 *   RESULT sosp_ms=<a> apply_ms=<b> end_to_end_ms=<c> threads=<t>
 * where (a) is the sum over the objectives of the stages that correspond to MOSP's
 * `obj<k>/sosp_update_compute` region (sospUpdateCpu): sssp.identify_affected + sssp.seed +
 * sssp.loop + sssp.finalize (parity/timed_regions/sssp.toml), (b) the commit (batch applied once,
 * with the transposition), and (c) the whole run, from reading the inputs to writing the outputs.
 */
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using vertex_t = std::int32_t;  // the originals' types
using edge_t = std::int32_t;
using weight_t = std::int32_t;
using graph_t = dyng::graph<vertex_t, edge_t, weight_t>;
using result_t = dyng::sssp::result<vertex_t>;
using clock_type = std::chrono::steady_clock;

struct options {
  std::string graph;
  std::string changes;
  std::string init;
  std::string out = "mosp-output";
  std::string timing;
  std::string backend;
  std::string write_graph;
  int K = 0;
  int threads = 0;
  vertex_t source = 0;
  std::int64_t delta = 0;
  bool canonicalize = false;
  bool validate = true;
  bool write_output = true;
  bool quiet = false;
};

class usage_error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

void usage() {
  std::cerr << "usage: dyng-compat-mosp --graph <csrPrefix> --changes <dir> --init <dir>\n"
               "                        [-k K] [--source s] [--delta D] [--canonicalize]\n"
               "                        [--no-validate-inputs] [--out dir] [--no-output]\n"
               "                        [--write-graph prefix] [--timing file.csv] [--quiet]\n"
               "                        [--backend sequential|openmp] [--threads t]\n"
               "       dyng-compat-mosp init <csrPrefix> <outDir> [--source s] [-k K]\n"
               "                        [--backend sequential|openmp] [--threads t]\n";
}

/// Strict integer parsing (the CUDA driver's rule: the whole token must be a number in range).
template <typename int_t>
int_t parse_int(std::string_view text, std::string_view flag, int_t lo, int_t hi) {
  int_t value{};
  const auto [end, ec] = std::from_chars(text.data(), text.data() + text.size(), value);
  if (ec != std::errc{} || end != text.data() + text.size() || value < lo || value > hi) {
    throw usage_error(std::string(flag) + ": expected an integer in [" + std::to_string(lo) + ", " +
                      std::to_string(hi) + "], got '" + std::string(text) + "'");
  }
  return value;
}

/// MOSP's runConcurrently(): the jobs run on the threads of an OpenMP parallel region (placed on
/// distinct cores under OMP_PROC_BIND / OMP_PLACES; std::thread workers would inherit the single
/// core the initial thread is pinned to). The first failing job's exception (in job order) is
/// rethrown after all jobs finished.
void run_concurrently(const std::vector<std::function<void()>>& jobs) {
  const auto count = static_cast<int>(jobs.size());
  std::vector<std::exception_ptr> errors(jobs.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(count)
#endif
  for (int i = 0; i < count; ++i) {
    try {
      jobs[static_cast<std::size_t>(i)]();
    } catch (...) {
      errors[static_cast<std::size_t>(i)] = std::current_exception();
    }
  }
  for (const std::exception_ptr& e : errors) {
    if (e) {
      std::rethrow_exception(e);
    }
  }
}

double ms_since(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

dyng::resources make_resources(const options& opt) {
  std::string b = opt.backend;
  if (b.empty()) {
    b = dyng::backend_available(dyng::backend::openmp) ? "openmp" : "sequential";
  }
  if (b == "sequential") {
    return dyng::resources::sequential();
  }
  if (b == "openmp") {
    return dyng::resources::openmp(opt.threads);
  }
  throw usage_error("--backend: expected sequential or openmp, got '" + b + "'");
}

/// Read the CSR; like MOSP, a graph without edges takes K from -k.
dyng::csr<vertex_t, edge_t, weight_t> read_graph(const std::string& prefix, int K) {
  try {
    return dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(prefix);
  } catch (const dyng::io_error&) {
    if (K <= 0) {
      throw;
    }
    dyng::io::csr_triplet_options csr_options;
    csr_options.num_weights = K;
    return dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(prefix, csr_options);
  }
}

int run_init(int argc, char** argv) {
  if (argc < 4) {
    throw usage_error("init needs <csrPrefix> <outDir>");
  }
  options opt;
  const std::string prefix = argv[2];
  const std::string out_dir = argv[3];
  for (int i = 4; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (i + 1 >= argc) {
      throw usage_error(std::string(a) + " needs a value");
    }
    const std::string_view value = argv[++i];
    if (a == "--source") {
      opt.source = parse_int<vertex_t>(value, a, 0, INT32_MAX);
    } else if (a == "-k") {
      opt.K = parse_int<int>(value, a, 1, 1 << 20);
    } else if (a == "--backend") {
      opt.backend = std::string(value);
    } else if (a == "--threads") {
      opt.threads = parse_int<int>(value, a, 1, 1 << 16);
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  const dyng::resources res = make_resources(opt);
  const auto csr = read_graph(prefix, opt.K);
  const graph_t g = graph_t::from_csr(res, csr.view(), dyng::graph_properties::mosp_compatible());
  const int K = opt.K > 0 ? std::min(opt.K, g.num_weights()) : g.num_weights();
  for (int k = 0; k < K; ++k) {
    dyng::sssp::options sssp_options;
    sssp_options.objective = k;
    const auto start = clock_type::now();
    const result_t r = dyng::sssp::compute(res, g, opt.source, sssp_options);
    const double ms = ms_since(start);
    const std::string dir = out_dir + "/obj" + std::to_string(k);
    dyng::io::write_distances(dir + "/distancesOriginal.txt", r.distances());
    dyng::io::write_parents(dir + "/SSSPTreeOriginal.txt", r.parents());
    std::cout << "compute obj" << k << ": " << ms << " ms\n";
  }
  return 0;
}

options parse_update_options(int argc, char** argv) {
  options opt;
  for (int i = 1; i < argc; ++i) {
    const std::string_view a = argv[i];
    auto next = [&]() -> std::string_view {
      if (i + 1 >= argc) {
        throw usage_error(std::string(a) + " needs a value");
      }
      return argv[++i];
    };
    if (a == "--graph") {
      opt.graph = std::string(next());
    } else if (a == "--changes") {
      opt.changes = std::string(next());
    } else if (a == "--init") {
      opt.init = std::string(next());
    } else if (a == "--out") {
      opt.out = std::string(next());
    } else if (a == "--timing") {
      opt.timing = std::string(next());
    } else if (a == "--write-graph") {
      opt.write_graph = std::string(next());
    } else if (a == "--backend") {
      opt.backend = std::string(next());
    } else if (a == "-k") {
      opt.K = parse_int<int>(next(), a, 1, 1 << 20);
    } else if (a == "--threads") {
      opt.threads = parse_int<int>(next(), a, 1, 1 << 16);
    } else if (a == "--source") {
      opt.source = parse_int<vertex_t>(next(), a, 0, INT32_MAX);
    } else if (a == "--delta") {
      opt.delta = parse_int<std::int64_t>(next(), a, 1, INT64_MAX);
    } else if (a == "--canonicalize") {
      opt.canonicalize = true;
    } else if (a == "--no-validate-inputs") {
      opt.validate = false;
    } else if (a == "--no-output") {
      opt.write_output = false;
    } else if (a == "--quiet") {
      opt.quiet = true;
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  if (opt.graph.empty() || opt.changes.empty() || opt.init.empty()) {
    throw usage_error("--graph, --changes and --init are required");
  }
  return opt;
}

int run_update(int argc, char** argv) {
  const options opt = parse_update_options(argc, argv);
  const auto start = clock_type::now();
  dyng::resources res = make_resources(opt);
  dyng::profiler prof;

  // --- Inputs ---------------------------------------------------------------------------------
  auto t = clock_type::now();
  const auto csr = read_graph(opt.graph, opt.K);
  const vertex_t n = csr.num_vertices();
  const int KG = csr.num_weights;
  const int K = opt.K > 0 ? std::min(opt.K, KG) : KG;
  if (K <= 0 || opt.source >= n) {
    throw usage_error("invalid MOSP update input (K = " + std::to_string(K) + ", source " +
                      std::to_string(opt.source) + ", n = " + std::to_string(n) + ")");
  }
  dyng::io::legacy_batch_options batch_options;
  batch_options.num_weights = KG;
  batch_options.num_vertices = n;
  batch_options.mosp_lenient = true;  // `mosp`'s accept/reject decisions (PLAN Section 8.3)
  // Like `mosp` (runConcurrently), the batch and the 2K tree files are read concurrently.
  dyng::edge_batch<vertex_t, weight_t> batch;
  std::vector<std::vector<std::int64_t>> dists(static_cast<std::size_t>(K));
  std::vector<std::vector<vertex_t>> trees(static_cast<std::size_t>(K));
  std::vector<std::function<void()>> reads;
  reads.emplace_back([&] {
    batch = dyng::io::read_legacy_batch<vertex_t, weight_t>(
        opt.changes + "/insert.txt", opt.changes + "/delete.txt", batch_options);
  });
  for (int k = 0; k < K; ++k) {
    const std::string dir = opt.init + "/obj" + std::to_string(k);
    const auto i = static_cast<std::size_t>(k);
    reads.emplace_back([&, dir, i] {
      dists[i] = dyng::io::read_distances<std::int64_t>(dir + "/distancesOriginal.txt", n);
    });
    reads.emplace_back([&, dir, i] {
      trees[i] = dyng::io::read_parents<vertex_t>(dir + "/SSSPTreeOriginal.txt", n);
    });
  }
  run_concurrently(reads);
  graph_t g = graph_t::from_csr(res, csr.view(), dyng::graph_properties::mosp_compatible());
  const double load_ms = ms_since(t);

  // The initial trees become results (checked with validate_inputs; not timed by `mosp`).
  t = clock_type::now();
  // The K results are built concurrently (from_arrays only reads the graph; PLAN Section 4.7.4).
  std::vector<std::optional<result_t>> built(static_cast<std::size_t>(K));
  std::vector<std::function<void()>> builds;
  for (int k = 0; k < K; ++k) {
    builds.emplace_back([&, k] {
      const auto i = static_cast<std::size_t>(k);
      dyng::sssp::options sssp_options;
      sssp_options.objective = k;
      sssp_options.delta = opt.delta;
      sssp_options.validate_inputs = opt.validate;
      built[i].emplace(result_t::from_arrays(res, g, opt.source, dyng::host_view(dists[i]),
                                             dyng::host_view(trees[i]), opt.canonicalize,
                                             sssp_options));
    });
  }
  run_concurrently(builds);
  std::vector<result_t> results;
  results.reserve(static_cast<std::size_t>(K));
  for (auto& r : built) {
    results.push_back(std::move(*r));
  }
  dists.clear();
  trees.clear();
  const double canonicalize_ms = ms_since(t);

  // --- Update (the batch is applied once) -----------------------------------------------------
  std::vector<result_t*> pointers;
  for (result_t& r : results) {
    pointers.push_back(&r);
  }
  res.attach_profiler(&prof);
  const std::vector<dyng::sssp::stats> stats = dyng::update_each(
      res, g, batch.view(), dyng::array_view<result_t* const>(pointers.data(), pointers.size()));
  res.attach_profiler(nullptr);

  // Per-objective times: the i-th call of each stage belongs to objective i.
  std::map<std::string, std::vector<double>> calls;
  for (const dyng::stage_sample& s : prof.samples()) {
    calls[s.name].push_back(s.host_ms);
  }
  std::vector<double> sosp_ms(static_cast<std::size_t>(K), 0.0);
  for (const char* stage : {"sssp.identify_affected", "sssp.seed", "sssp.loop", "sssp.finalize"}) {
    const std::vector<double>& samples = calls[stage];
    for (std::size_t k = 0; k < samples.size() && k < sosp_ms.size(); ++k) {
      sosp_ms[k] += samples[k];
    }
  }
  double sosp_total = 0;
  for (int k = 0; k < K; ++k) {
    sosp_total += sosp_ms[static_cast<std::size_t>(k)];
    const std::string obj = "obj" + std::to_string(k);
    const dyng::sssp::stats& st = stats[static_cast<std::size_t>(k)];
    prof.add_counter("sssp.invalidated." + obj, st.invalidated);
    prof.add_counter("sssp.iterations." + obj, st.iterations);
    prof.add_counter("sssp.epochs." + obj, st.epochs);
    prof.add_counter("sssp.pushes." + obj, st.pushes);
  }
  const double apply_ms = prof.total_host_ms("update.commit");

  // --- Outputs --------------------------------------------------------------------------------
  double write_ms = 0;
  if (opt.write_output) {
    t = clock_type::now();
    std::vector<std::function<void()>> writes;  // concurrently, like `mosp`
    for (int k = 0; k < K; ++k) {
      const std::string dir = opt.out + "/obj" + std::to_string(k);
      const result_t& r = results[static_cast<std::size_t>(k)];
      writes.emplace_back(
          [dir, &r] { dyng::io::write_distances(dir + "/distancesUpdated.txt", r.distances()); });
      writes.emplace_back(
          [dir, &r] { dyng::io::write_parents(dir + "/SSSPTreeUpdated.txt", r.parents()); });
    }
    run_concurrently(writes);
    write_ms = ms_since(t);
  }
  const double end_to_end = ms_since(start);
  if (!opt.write_graph.empty()) {
    dyng::io::write_csr_triplet(opt.write_graph, g.view().out);
  }

  // --- Report ---------------------------------------------------------------------------------
  if (!opt.quiet) {
    std::printf("graph  n=%d m=%lld K=%d; batch %zu inserts, %zu deletes\n", n,
                static_cast<long long>(g.num_edges()), K, batch.num_insertions(),
                batch.num_deletions());
    std::printf(
        "host   %s, threads %d, read inputs %.1f ms, canonicalize %.1f ms, apply batch %.1f ms, "
        "write %.1f ms\n",
        std::string(dyng::to_string(res.get_backend())).c_str(), res.num_threads(), load_ms,
        canonicalize_ms, apply_ms, write_ms);
    for (int k = 0; k < K; ++k) {
      const dyng::sssp::stats& s = stats[static_cast<std::size_t>(k)];
      std::printf(
          "obj%d   SOSP update %.3f ms (invalidated %lld, iterations %lld, epochs %lld, pushes "
          "%lld)\n",
          k, sosp_ms[static_cast<std::size_t>(k)], static_cast<long long>(s.invalidated),
          static_cast<long long>(s.iterations), static_cast<long long>(s.epochs),
          static_cast<long long>(s.pushes));
    }
  }
  bool timing_written = true;
  if (!opt.timing.empty()) {
    std::ofstream csv(opt.timing);
    prof.write_csv(csv);
    csv << "stage,total.end_to_end," << end_to_end << '\n';
    timing_written = static_cast<bool>(csv);
    if (!timing_written) {
      std::cerr << "cannot write " << opt.timing << "\n";
    }
  }
  std::printf("RESULT sosp_ms=%.3f apply_ms=%.3f end_to_end_ms=%.3f threads=%d\n", sosp_total,
              apply_ms, end_to_end, res.num_threads());
  std::fflush(stdout);
  return timing_written ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    if (argc >= 2 && std::string_view(argv[1]) == "init") {
      return run_init(argc, argv);
    }
    return run_update(argc, argv);
  } catch (const usage_error& e) {
    std::cerr << "dyng-compat-mosp: " << e.what() << "\n";
    usage();
    return 2;
  } catch (const std::exception& e) {
    std::cerr << e.what() << "\n";
    return 1;
  }
}
