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
 *   dyng-compat-mosp changes <csrPrefix> <outDir> [the options of `mospPrep changes`]
 *
 * Update mode reads the inputs of `mosp` (the CSR text files <prefix>{RowPtr,ColInd,Values}.txt,
 * <changes>/insert.txt and delete.txt, and the initial trees <init>/obj<k>/distancesOriginal.txt
 * and SSSPTreeOriginal.txt), applies the batch once to a graph with
 * graph_properties::mosp_compatible() and updates the K trees with dyng::update_each(), then
 * writes <out>/obj<k>/distancesUpdated.txt and SSSPTreeUpdated.txt, byte-compatible with `mosp`.
 * The combined graph of the MOSP update (Steps 2-3) belongs to the mosp algorithm (0.2) and is not
 * written. Init mode writes <outDir>/obj<k>/distancesOriginal.txt and SSSPTreeOriginal.txt with
 * sssp::compute(), byte-compatible with `mospPrep init` (Dijkstra with lowest-id ties). Changes
 * mode is `mospPrep changes` (--changes N --ins PCT --mode M --local HOPS --safe --seed S
 * --source s --wmin a --wmax b) on generators::legacy::mosp_changes(): the same insert.txt,
 * delete.txt and report line.
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
 *   --backend <b>        sequential | openmp | cuda (default openmp if built)
 *   --threads <t>        OpenMP threads (default: the OpenMP default)
 *   --device <d>         CUDA device of --backend cuda (default 0)
 *
 * Report lines (without --quiet): the inputs, `obj<k>   SOSP update <ms> (invalidated ...)`,
 * `graph  read graph <ms>, read changes and trees <ms>, build <ms>` and `obj<k>   result import <ms>, workspace <ms>, validate <ms>` (the
 * from_arrays stages; parity/timed_regions/sssp.toml maps import and workspace to parts of the
 * original's "prepare").
 *
 * Summary line (stable format):
 *   RESULT sosp_ms=<a> apply_ms=<b> end_to_end_ms=<c> threads=<t>
 * where (a) is the sum over the objectives of the stages that correspond to MOSP's per-objective
 * SOSP region (parity/timed_regions/sssp.toml): on the host backends MOSP-OpenMP's
 * `obj<k>/sosp_update_compute` (sospUpdateCpu) = sssp.identify_affected + sssp.seed + sssp.loop +
 * sssp.finalize; on cuda MOSP-CUDA's `obj<k>/sosp_update_gpu` (sospUpdateGpu, host wall time up to
 * the synchronization that ends it) = sssp.enact_fused. (b) is the commit (the batch applied once,
 * with the transposition of the updated graph, or on cuda its upload and the in-edges built on the
 * device), and (c) the whole run, from reading the inputs to writing the outputs.
 *
 * CUDA backend (MOSP-CUDA's `mosp`): the device is initialized first with resources::warm_up()
 * (context and eager kernel loading; MOSP-CUDA sets CUDA_MODULE_LOADING=EAGER and times
 * "cuda_context_init"), the stages are timed with CUDA events as well (profiler_options::
 * cuda_events), and the --timing CSV gets one `device,<stage>,<ms>` row per stage sample with a
 * device time, next to the `stage` rows. The per-objective line then reads
 * `obj<k>   SOSP update <host ms> ms (invalidated ..., device <ms> ms)`. The results are copied to
 * the host (to_vector; MOSP-CUDA's "download") before they are written.
 */
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/legacy.hpp>
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
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <map>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
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
  int device = 0;
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
               "                        [--backend sequential|openmp|cuda] [--threads t]\n"
               "                        [--device d]\n"
               "       dyng-compat-mosp init <csrPrefix> <outDir> [--source s] [-k K]\n"
               "                        [--backend sequential|openmp|cuda] [--threads t]\n"
               "                        [--device d]\n"
               "       dyng-compat-mosp changes <csrPrefix> <outDir> [--changes N] [--ins PCT]\n"
               "                        [--mode uniform|targeted|reweight|increase]\n"
               "                        [--local HOPS] [--safe] [--seed S] [--source s]\n"
               "                        [--wmin a] [--wmax b]\n";
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
  if (b == "cuda") {
    dyng::resources res = dyng::resources::cuda(opt.device);
    res.warm_up();  // context and eager kernel loading (MOSP-CUDA: CUDA_MODULE_LOADING=EAGER)
    return res;
  }
  throw usage_error("--backend: expected sequential, openmp or cuda, got '" + b + "'");
}

/// A host copy of a result array (device arrays of the cuda backend are downloaded).
template <typename value_t>
std::vector<value_t> host_array(const dyng::resources& res, dyng::array_view<const value_t> v) {
  if (dyng::is_host_accessible(v.space())) {
    return std::vector<value_t>(v.begin(), v.end());
  }
  return dyng::to_vector(res, v);
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
    } else if (a == "--device") {
      opt.device = parse_int<int>(value, a, 0, 1 << 10);
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  const dyng::resources res = make_resources(opt);
  auto csr = read_graph(prefix, opt.K);
  const graph_t g =
      graph_t::from_csr(res, std::move(csr), dyng::graph_properties::mosp_compatible());
  const int K = opt.K > 0 ? std::min(opt.K, g.num_weights()) : g.num_weights();
  for (int k = 0; k < K; ++k) {
    dyng::sssp::options sssp_options;
    sssp_options.objective = k;
    const auto start = clock_type::now();
    const result_t r = dyng::sssp::compute(res, g, opt.source, sssp_options);
    const double ms = ms_since(start);
    const std::string dir = out_dir + "/obj" + std::to_string(k);
    dyng::io::write_distances(dir + "/distancesOriginal.txt",
                              dyng::host_view(host_array(res, r.distances())));
    dyng::io::write_parents(dir + "/SSSPTreeOriginal.txt",
                            dyng::host_view(host_array(res, r.parents())));
    std::cout << "compute obj" << k << ": " << ms << " ms\n";
  }
  return 0;
}

/// `mospPrep changes <csrPrefix> <outDir> [options]`: generators::legacy::mosp_changes(), written
/// as <outDir>/insert.txt and delete.txt with the report line "changes: ..." (the same flags and
/// output as the original's tool).
int run_changes(int argc, char** argv) {
  if (argc < 4) {
    throw usage_error("changes needs <csrPrefix> <outDir>");
  }
  const std::string prefix = argv[2];
  const std::string out_dir = argv[3];
  dyng::generators::legacy::mosp_change_options opt;
  for (int i = 4; i < argc; ++i) {
    const std::string_view a = argv[i];
    if (a == "--safe") {
      opt.safe_deletions = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw usage_error(std::string(a) + " needs a value");
    }
    const std::string_view value = argv[++i];
    if (a == "--changes") {
      opt.num_changes = parse_int<std::int64_t>(value, a, 0, INT32_MAX);
    } else if (a == "--ins") {
      opt.insertion_percentage = std::stod(std::string(value));
    } else if (a == "--mode") {
      using mode = dyng::generators::legacy::mosp_change_mode;
      if (value == "uniform") {
        opt.mode = mode::uniform;
      } else if (value == "targeted") {
        opt.mode = mode::targeted;
      } else if (value == "reweight") {
        opt.mode = mode::reweight;
      } else if (value == "increase") {
        opt.mode = mode::increase;
      } else {
        throw usage_error("--mode: expected uniform, targeted, reweight or increase");
      }
    } else if (a == "--local") {
      opt.local_hops = parse_int<std::int64_t>(value, a, 0, INT32_MAX);
    } else if (a == "--seed") {
      opt.seed = parse_int<std::uint32_t>(value, a, 0, UINT32_MAX);
    } else if (a == "--source") {
      opt.source = parse_int<std::int64_t>(value, a, 0, INT32_MAX);
    } else if (a == "--wmin") {
      opt.weight_min = parse_int<std::int32_t>(value, a, 1, INT32_MAX);
    } else if (a == "--wmax") {
      opt.weight_max = parse_int<std::int32_t>(value, a, 1, INT32_MAX);
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  const auto csr = read_graph(prefix, 0);
  dyng::generators::legacy::mosp_change_report report;
  const auto batch = dyng::generators::legacy::mosp_changes(csr.view(), opt, &report);
  std::filesystem::create_directories(out_dir);
  dyng::io::write_legacy_batch(out_dir + "/insert.txt", out_dir + "/delete.txt", batch.view());
  std::cout << "changes: " << report.summary() << "\n";
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
    } else if (a == "--device") {
      opt.device = parse_int<int>(next(), a, 0, 1 << 10);
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
  auto t = clock_type::now();
  dyng::resources res = make_resources(opt);
  const double context_ms = ms_since(t);
  const bool cuda = res.get_backend() == dyng::backend::cuda;
  dyng::profiler_options profiling;
  profiling.cuda_events = cuda;  // device times of the update's stages (CUDA events)
  dyng::profiler prof(profiling);

  // --- Inputs ---------------------------------------------------------------------------------
  // The setup (graph and result builds) is recorded by its own profiler, the update by `prof`.
  dyng::profiler setup_prof;
  res.attach_profiler(&setup_prof);
  t = clock_type::now();
  auto csr = read_graph(opt.graph, opt.K);
  const double read_graph_ms = ms_since(t);
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
  auto t_reads = clock_type::now();
  run_concurrently(reads);
  const double read_rest_ms = ms_since(t_reads);
  // The graph takes over the CSR's arrays (a MOSP CSR is already in its final form); its in-edges
  // are built on first use, i.e. for the updated graph inside the commit, as `mosp` builds its
  // reverse graph once in "prepare".
  graph_t g = graph_t::from_csr(res, std::move(csr), dyng::graph_properties::mosp_compatible());
  const double load_ms = ms_since(t);

  // The initial trees become results (checked with validate_inputs; not timed by `mosp`). The
  // builds run one after another, each parallel inside, and share the scratch memory of `res`
  // (sized by the first one; ADR 0015). Their stages (sssp.import, sssp.workspace,
  // sssp.validate, ...) are recorded per objective, in objective order, for the timed-region map
  // (parity/timed_regions/sssp.toml).
  t = clock_type::now();
  std::vector<result_t> results;
  results.reserve(static_cast<std::size_t>(K));
  for (int k = 0; k < K; ++k) {
    const auto i = static_cast<std::size_t>(k);
    dyng::sssp::options sssp_options;
    sssp_options.objective = k;
    sssp_options.delta = opt.delta;
    sssp_options.validate_inputs = opt.validate;
    results.push_back(result_t::from_arrays(res, g, opt.source, dyng::host_view(dists[i]),
                                            dyng::host_view(trees[i]), opt.canonicalize,
                                            sssp_options));
    dists[i] = {};
    trees[i] = {};
  }
  res.attach_profiler(nullptr);
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
  std::vector<double> sosp_device_ms(static_cast<std::size_t>(K), 0.0);
  const std::vector<const char*> sosp_stages =
      cuda ? std::vector<const char*>{"sssp.enact_fused"}
           : std::vector<const char*>{"sssp.identify_affected", "sssp.seed", "sssp.loop",
                                      "sssp.finalize"};
  for (const char* stage : sosp_stages) {
    const std::vector<double>& samples = calls[stage];
    for (std::size_t k = 0; k < samples.size() && k < sosp_ms.size(); ++k) {
      sosp_ms[k] += samples[k];
    }
  }
  if (cuda) {
    std::size_t k = 0;
    for (const dyng::stage_sample& s : prof.samples()) {
      if (s.name == "sssp.enact_fused" && k < sosp_device_ms.size()) {
        sosp_device_ms[k++] = s.device_ms;
      }
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
  // The results in host memory (on cuda: MOSP-CUDA's "download", which it does with or without
  // --no-output).
  double download_ms = 0;
  std::vector<std::vector<std::int64_t>> downloaded_d(static_cast<std::size_t>(K));
  std::vector<std::vector<vertex_t>> downloaded_p(static_cast<std::size_t>(K));
  if (cuda) {
    t = clock_type::now();
    for (int k = 0; k < K; ++k) {
      const result_t& r = results[static_cast<std::size_t>(k)];
      downloaded_d[static_cast<std::size_t>(k)] = dyng::to_vector(res, r.distances());
      downloaded_p[static_cast<std::size_t>(k)] = dyng::to_vector(res, r.parents());
    }
    download_ms = ms_since(t);
  }
  const auto distances_of = [&](int k) {
    const auto i = static_cast<std::size_t>(k);
    return cuda ? dyng::host_view(std::as_const(downloaded_d[i])) : results[i].distances();
  };
  const auto parents_of = [&](int k) {
    const auto i = static_cast<std::size_t>(k);
    return cuda ? dyng::host_view(std::as_const(downloaded_p[i])) : results[i].parents();
  };
  double write_ms = 0;
  if (opt.write_output) {
    t = clock_type::now();
    std::vector<std::function<void()>> writes;  // concurrently, like `mosp`
    for (int k = 0; k < K; ++k) {
      const std::string dir = opt.out + "/obj" + std::to_string(k);
      const dyng::array_view<const std::int64_t> d = distances_of(k);
      const dyng::array_view<const vertex_t> p = parents_of(k);
      writes.emplace_back(
          [dir, d] { dyng::io::write_distances(dir + "/distancesUpdated.txt", d); });
      writes.emplace_back([dir, p] { dyng::io::write_parents(dir + "/SSSPTreeUpdated.txt", p); });
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
        "host   %s, threads %d, context %.1f ms, read inputs %.1f ms, canonicalize %.1f ms, "
        "apply batch %.1f ms, download %.1f ms, write %.1f ms\n",
        std::string(dyng::to_string(res.get_backend())).c_str(), res.num_threads(), context_ms,
        load_ms, canonicalize_ms, apply_ms, download_ms, write_ms);
    for (int k = 0; k < K; ++k) {
      const dyng::sssp::stats& s = stats[static_cast<std::size_t>(k)];
      char device[64] = "";
      if (cuda) {
        std::snprintf(device, sizeof(device), ", device %.3f ms",
                      sosp_device_ms[static_cast<std::size_t>(k)]);
      }
      std::printf(
          "obj%d   SOSP update %.3f ms (invalidated %lld, iterations %lld, epochs %lld, pushes "
          "%lld, affected %lld, packed %d%s)\n",
          k, sosp_ms[static_cast<std::size_t>(k)], static_cast<long long>(s.invalidated),
          static_cast<long long>(s.iterations), static_cast<long long>(s.epochs),
          static_cast<long long>(s.pushes), static_cast<long long>(s.affected),
          s.packed_parents ? 1 : 0, device);
    }
    std::map<std::string, std::vector<double>> setup;
    for (const dyng::stage_sample& s : setup_prof.samples()) {
      setup[s.name].push_back(s.host_ms);
    }
    const auto sample = [&setup](const char* name, int k) {
      const std::vector<double>& v = setup[name];
      return static_cast<std::size_t>(k) < v.size() ? v[static_cast<std::size_t>(k)] : 0.0;
    };
    std::printf("graph  read graph %.1f ms, read changes and trees %.1f ms, build %.3f ms\n",
                read_graph_ms, read_rest_ms, setup_prof.total_host_ms("graph.build"));
    for (int k = 0; k < K; ++k) {
      std::printf("obj%d   result import %.3f ms, workspace %.3f ms, validate %.3f ms\n", k,
                  sample("sssp.import", k), sample("sssp.workspace", k),
                  sample("sssp.validate", k));
    }
  }
  bool timing_written = true;
  if (!opt.timing.empty()) {
    std::ofstream csv(opt.timing);
    prof.write_csv(csv);  // the update: one sample per stage and objective, in objective order
    // The setup: the graph build and the result builds (from_arrays), objective by objective,
    // then the whole run.
    for (const dyng::stage_sample& s : setup_prof.samples()) {
      csv << "stage," << s.name << ',' << s.host_ms << '\n';
    }
    csv << "stage,total.end_to_end," << end_to_end << '\n';
    csv << "stage,total.context," << context_ms << '\n';
    csv << "stage,total.download," << download_ms << '\n';
    // Device times (CUDA events) of the update's stages, one row per sample, in sample order.
    for (const dyng::stage_sample& s : prof.samples()) {
      if (s.device_ms > 0.0) {
        csv << "device," << s.name << ',' << s.device_ms << '\n';
      }
    }
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
    if (argc >= 2 && std::string_view(argv[1]) == "changes") {
      return run_changes(argc, argv);
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
