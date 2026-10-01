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
 *                    [--device d] [--edge-type t]
 *   dyng-compat-mosp changes <csrPrefix> <outDir> [the options of `mospPrep changes`]
 *
 * Update mode reads the inputs of `mosp` (the CSR text files <prefix>{RowPtr,ColInd,Values}.txt,
 * <changes>/insert.txt and delete.txt, and the initial trees <init>/obj<k>/distancesOriginal.txt
 * and SSSPTreeOriginal.txt), applies the batch once to a graph with
 * graph_properties::mosp_compatible() and updates the K trees with dyng::update_each(), then
 * writes <out>/obj<k>/distancesUpdated.txt and SSSPTreeUpdated.txt, byte-compatible with `mosp`.
 * With --mosp it runs the whole MOSP update instead (dyng::mosp: the K trees adopted with
 * mosp::result::from_arrays(), mosp::update(): the K sssp updates, the combined graph, its SOSP
 * tree and the path costs) and also writes <out>/combinedGraph/distancesCsr.txt,
 * SSSPTreeCsr.txt and mospCosts.txt, byte-compatible with `mosp` (the clone of the full `mosp`
 * driver; without --mosp the tool keeps the SOSP part only, as the sssp gates measure it since
 * M1a). Init mode writes <outDir>/obj<k>/distancesOriginal.txt and SSSPTreeOriginal.txt with
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
 *   --edge-type <t>      int32 | int64: the graph's edge-offset type (default int32, the
 *                        originals' type; int64 for the edge_t benchmark of ADR 0009)
 *   --cuda-engine <e>    automatic | fused | operators: sssp::options::cuda_engine (default
 *                        automatic; the operators engine is the multi-kernel engine of M7, compared
 *                        with the fused one by parity/compare.py --configs cuda-operators)
 *   --mosp               the whole MOSP update (dyng::mosp) and the combinedGraph/ outputs
 *   --pref p1,..,pK      with --mosp: mosp::options::preferences (`mosp --pref`; default all 1)
 *   --no-path-costs      with --mosp: mosp::options::compute_path_costs = false (no mospCosts.txt)
 *
 * Report lines (without --quiet): the inputs, `obj<k>   SOSP update <ms> (invalidated ...)`,
 * `graph  read graph <ms>, read changes and trees <ms>, build <ms>` and `obj<k>   result import <ms>, workspace <ms>, validate <ms>` (the
 * from_arrays stages; parity/timed_regions/sssp.toml maps import and workspace to parts of the
 * original's "prepare").
 *
 * With --mosp the report adds `comb   combined graph + SOSP <ms> ms (...)` (mosp.combine +
 * mosp.combined_sssp + mosp.finalize, the original's combinedGraphSosp* region) and the summary
 * line begins with `RESULT compute_ms=<x>`, the K objectives' SOSP regions (a) plus that combined
 * region: the region MOSP times as gpu_compute_ms / compute_ms.
 *
 * Summary line (stable format):
 *   RESULT sosp_ms=<a> apply_ms=<b> end_to_end_ms=<c> threads=<t>
 * where (a) is the sum over the objectives of the stages that correspond to MOSP's per-objective
 * SOSP region (parity/timed_regions/sssp.toml): on the host backends MOSP-OpenMP's
 * `obj<k>/sosp_update_compute` (sospUpdateCpu) = sssp.identify_affected + sssp.seed + sssp.loop +
 * sssp.finalize; on cuda MOSP-CUDA's `obj<k>/sosp_update_gpu` (sospUpdateGpu, host wall time up to
 * the synchronization that ends it) = sssp.enact_fused with the fused engine, and the four Tier A
 * stages with the operators engine (--cuda-engine operators). (b) is the commit (the batch applied once,
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
#include <dyng/core/types.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/mosp_oracle.hpp>
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
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

using vertex_t = std::int32_t;  // the originals' types (edge offsets: --edge-type, default int32)
using weight_t = std::int32_t;
template <typename edge_t>
using graph_type = dyng::graph<vertex_t, edge_t, weight_t>;
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
  bool edge64 = false;  // --edge-type int64: 64-bit edge offsets (the edge_t benchmark, ADR 0009)
  dyng::engine cuda_engine = dyng::engine::automatic;  // --cuda-engine
  vertex_t source = 0;
  std::int64_t delta = 0;
  bool canonicalize = false;
  bool validate = true;
  bool write_output = true;
  bool quiet = false;
  bool mosp = false;               // --mosp: the whole MOSP update
  bool path_costs = true;          // --no-path-costs
  std::vector<std::int32_t> pref;  // --pref
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
               "                        [--device d] [--edge-type int32|int64]\n"
               "                        [--cuda-engine automatic|fused|operators]\n"
               "                        [--mosp [--pref p1,..,pK] [--no-path-costs]]\n"
               "       dyng-compat-mosp init <csrPrefix> <outDir> [--source s] [-k K]\n"
               "                        [--backend sequential|openmp|cuda] [--threads t]\n"
               "                        [--device d] [--edge-type int32|int64]\n"
               "                        [--cuda-engine automatic|fused|operators]\n"
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

/// `--edge-type int32|int64`: the edge-offset type of the graph (ADR 0009).
bool parse_edge_type(std::string_view value) {
  if (value == "int32") {
    return false;
  }
  if (value == "int64") {
    return true;
  }
  throw usage_error("--edge-type: expected int32 or int64, got '" + std::string(value) + "'");
}

/// `--cuda-engine automatic|fused|operators`: sssp::options::cuda_engine.
dyng::engine parse_engine(std::string_view value) {
  for (const dyng::engine e :
       {dyng::engine::automatic, dyng::engine::fused, dyng::engine::operators}) {
    if (value == dyng::to_string(e)) {
      return e;
    }
  }
  throw usage_error("--cuda-engine: expected automatic, fused or operators, got '" +
                    std::string(value) + "'");
}

/// Read the CSR; like MOSP, a graph without edges takes K from -k.
template <typename edge_t>
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

template <typename edge_t>
int run_init_typed(const options& opt, const std::string& prefix, const std::string& out_dir);

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
    } else if (a == "--edge-type") {
      opt.edge64 = parse_edge_type(value);
    } else if (a == "--cuda-engine") {
      opt.cuda_engine = parse_engine(value);
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  return opt.edge64 ? run_init_typed<std::int64_t>(opt, prefix, out_dir)
                    : run_init_typed<std::int32_t>(opt, prefix, out_dir);
}

template <typename edge_t>
int run_init_typed(const options& opt, const std::string& prefix, const std::string& out_dir) {
  using graph_t = graph_type<edge_t>;
  const dyng::resources res = make_resources(opt);
  auto csr = read_graph<edge_t>(prefix, opt.K);
  const graph_t g =
      graph_t::from_csr(res, std::move(csr), dyng::graph_properties::mosp_compatible());
  const int K = opt.K > 0 ? std::min(opt.K, g.num_weights()) : g.num_weights();
  for (int k = 0; k < K; ++k) {
    dyng::sssp::options sssp_options;
    sssp_options.objective = k;
    sssp_options.cuda_engine = opt.cuda_engine;
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
  const auto csr = read_graph<std::int32_t>(prefix, 0);
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
    } else if (a == "--edge-type") {
      opt.edge64 = parse_edge_type(next());
    } else if (a == "--cuda-engine") {
      opt.cuda_engine = parse_engine(next());
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
    } else if (a == "--mosp") {
      opt.mosp = true;
    } else if (a == "--no-path-costs") {
      opt.path_costs = false;
    } else if (a == "--pref") {
      // `mosp --pref p1,..,pK`: integers >= 1 (the CUDA driver's strict parsing).
      std::istringstream list{std::string(next())};
      std::string item;
      opt.pref.clear();
      while (std::getline(list, item, ',')) {
        opt.pref.push_back(parse_int<std::int32_t>(item, a, 1, INT32_MAX));
      }
    } else {
      throw usage_error("unknown option: " + std::string(a));
    }
  }
  if ((!opt.pref.empty() || !opt.path_costs) && !opt.mosp) {
    throw usage_error("--pref and --no-path-costs need --mosp");
  }
  if (opt.graph.empty() || opt.changes.empty() || opt.init.empty()) {
    throw usage_error("--graph, --changes and --init are required");
  }
  return opt;
}

/// What update mode has read and built before the update (shared by the SOSP-only and the MOSP
/// paths).
template <typename edge_t>
struct loaded_inputs {
  graph_type<edge_t>& g;                                    ///< the graph (G_t)
  const dyng::edge_batch<vertex_t, weight_t>& batch;        ///< the batch
  std::vector<std::vector<std::int64_t>>& dists;            ///< the K initial distance arrays
  std::vector<std::vector<vertex_t>>& trees;                ///< the K initial trees
  vertex_t n;                                               ///< vertices
  int K;                                                    ///< objectives used
  int KG;                                                   ///< weight columns of the graph
  clock_type::time_point start;                             ///< the run's start
  double context_ms, read_graph_ms, read_rest_ms, load_ms;  ///< the input stages
};

/// The sum of the samples of `stages` per call index (the k-th sample of a stage is objective k).
std::vector<double> per_objective(const dyng::profiler& prof,
                                  const std::vector<std::string>& stages, int K, bool device) {
  std::vector<double> out(static_cast<std::size_t>(K), 0.0);
  std::map<std::string, std::size_t> seen;
  for (const dyng::stage_sample& s : prof.samples()) {
    if (std::find(stages.begin(), stages.end(), s.name) != stages.end()) {
      const std::size_t k = seen[s.name]++;
      if (k < out.size()) {
        out[k] += device ? s.device_ms : s.host_ms;
      }
    }
  }
  return out;
}

/// `mosp` (the whole MOSP update): the K trees adopted as one mosp result, mosp::update(), the
/// outputs of the original driver including combinedGraph/.
template <typename edge_t>
int run_mosp(const options& opt, dyng::resources& res, bool cuda, dyng::profiler& prof,
             dyng::profiler& setup_prof, loaded_inputs<edge_t>& in) {
  using mosp_result = dyng::mosp::result<vertex_t>;
  const int K = in.K;
  // The initial trees become one mosp result (sssp's import and checks per objective, then the
  // combined graph of the initial trees; not timed by `mosp`).
  auto t = clock_type::now();
  dyng::mosp::options mopt;
  mopt.preferences = opt.pref;
  mopt.delta = opt.delta;
  mopt.cuda_engine = opt.cuda_engine;
  mopt.validate_inputs = opt.validate;
  mopt.compute_path_costs = opt.path_costs && K == in.KG;
  mopt.num_objectives = K;
  std::vector<dyng::array_view<const std::int64_t>> dv;
  std::vector<dyng::array_view<const vertex_t>> pv;
  for (int k = 0; k < K; ++k) {
    dv.push_back(dyng::host_view(std::as_const(in.dists[static_cast<std::size_t>(k)])));
    pv.push_back(dyng::host_view(std::as_const(in.trees[static_cast<std::size_t>(k)])));
  }
  mosp_result r =
      mosp_result::from_arrays(res, in.g, opt.source, dyng::host_view(std::as_const(dv)),
                               dyng::host_view(std::as_const(pv)), opt.canonicalize, mopt);
  for (int k = 0; k < K; ++k) {
    in.dists[static_cast<std::size_t>(k)] = {};
    in.trees[static_cast<std::size_t>(k)] = {};
  }
  res.attach_profiler(nullptr);
  const double canonicalize_ms = ms_since(t);

  // --- Update (the batch is applied once) -----------------------------------------------------
  res.attach_profiler(&prof);
  const dyng::mosp::stats stats = dyng::mosp::update(res, in.g, in.batch.view(), r);
  res.attach_profiler(nullptr);
  const std::vector<std::string> sosp_stages{"sssp.enact_fused", "sssp.identify_affected",
                                             "sssp.seed", "sssp.loop", "sssp.finalize"};
  const std::vector<double> sosp_ms = per_objective(prof, sosp_stages, K, false);
  const std::vector<double> sosp_device_ms = per_objective(prof, sosp_stages, K, true);
  double sosp_total = 0;
  for (int k = 0; k < K; ++k) {
    sosp_total += sosp_ms[static_cast<std::size_t>(k)];
    const std::string obj = "obj" + std::to_string(k);
    const dyng::sssp::stats& st = stats.objectives[static_cast<std::size_t>(k)];
    prof.add_counter("sssp.invalidated." + obj, st.invalidated);
    prof.add_counter("sssp.iterations." + obj, st.iterations);
    prof.add_counter("sssp.epochs." + obj, st.epochs);
    prof.add_counter("sssp.pushes." + obj, st.pushes);
  }
  // The original's combinedGraphSosp* region: the combined graph, its solve and (dynG's addition)
  // the `affected` count.
  const double combined_ms = prof.total_host_ms("mosp.combine") +
                             prof.total_host_ms("mosp.combined_sssp") +
                             prof.total_host_ms("mosp.finalize");
  prof.add_counter("mosp.combined_edges", stats.combined_edges);
  prof.add_counter("mosp.affected", stats.affected);
  const double apply_ms = prof.total_host_ms("mosp.commit");

  // --- Outputs --------------------------------------------------------------------------------
  // The results in host memory (on cuda: MOSP-CUDA's "download" of the K trees and the combined
  // arrays, which it does with or without --no-output).
  double download_ms = 0;
  t = clock_type::now();
  std::vector<std::vector<std::int64_t>> d(static_cast<std::size_t>(K));
  std::vector<std::vector<vertex_t>> p(static_cast<std::size_t>(K));
  std::vector<std::int64_t> combined_d;
  std::vector<vertex_t> combined_p;
  if (cuda) {
    for (int k = 0; k < K; ++k) {
      d[static_cast<std::size_t>(k)] = dyng::to_vector(res, r.distances(k));
      p[static_cast<std::size_t>(k)] = dyng::to_vector(res, r.parents(k));
    }
    combined_d = dyng::to_vector(res, r.combined_distances());
    combined_p = dyng::to_vector(res, r.combined_parents());
    download_ms = ms_since(t);
  }
  const auto distances_of = [&](int k) {
    return cuda ? dyng::host_view(std::as_const(d[static_cast<std::size_t>(k)])) : r.distances(k);
  };
  const auto parents_of = [&](int k) {
    return cuda ? dyng::host_view(std::as_const(p[static_cast<std::size_t>(k)])) : r.parents(k);
  };
  const dyng::array_view<const std::int64_t> combined_distances =
      cuda ? dyng::host_view(std::as_const(combined_d)) : r.combined_distances();
  const dyng::array_view<const vertex_t> combined_parents =
      cuda ? dyng::host_view(std::as_const(combined_p)) : r.combined_parents();
  double write_ms = 0;
  if (opt.write_output) {
    t = clock_type::now();
    std::vector<std::function<void()>> writes;  // concurrently, like `mosp`
    for (int k = 0; k < K; ++k) {
      const std::string dir = opt.out + "/obj" + std::to_string(k);
      const dyng::array_view<const std::int64_t> dk = distances_of(k);
      const dyng::array_view<const vertex_t> pk = parents_of(k);
      writes.emplace_back(
          [dir, dk] { dyng::io::write_distances(dir + "/distancesUpdated.txt", dk); });
      writes.emplace_back([dir, pk] { dyng::io::write_parents(dir + "/SSSPTreeUpdated.txt", pk); });
    }
    const std::string combined_dir = opt.out + "/combinedGraph";
    writes.emplace_back(
        [&] { dyng::io::write_distances(combined_dir + "/distancesCsr.txt", combined_distances); });
    writes.emplace_back(
        [&] { dyng::io::write_parents(combined_dir + "/SSSPTreeCsr.txt", combined_parents); });
    if (opt.path_costs) {
      writes.emplace_back([&] {
        // MOSP's mospPathCosts covers every weight column of the graph: with -k K < KG the other
        // columns' costs come from the host reference along the same tree.
        if (K == in.KG) {
          dyng::io::write_path_costs(combined_dir + "/mospCosts.txt", r.path_costs(), K);
        } else {
          const std::vector<vertex_t> tree(combined_parents.begin(), combined_parents.end());
          const std::vector<std::int64_t> costs =
              dyng::testing::mosp_path_costs_reference(in.g.view().out, tree, opt.source, in.KG);
          dyng::io::write_path_costs(combined_dir + "/mospCosts.txt", dyng::host_view(costs),
                                     in.KG);
        }
      });
    }
    run_concurrently(writes);
    write_ms = ms_since(t);
  }
  const double end_to_end = ms_since(in.start);
  if (!opt.write_graph.empty()) {
    dyng::io::write_csr_triplet(opt.write_graph, in.g.view().out);
  }

  // --- Report ---------------------------------------------------------------------------------
  if (!opt.quiet) {
    std::printf("graph  n=%d m=%lld K=%d; batch %zu inserts, %zu deletes\n", in.n,
                static_cast<long long>(in.g.num_edges()), K, in.batch.num_insertions(),
                in.batch.num_deletions());
    std::printf(
        "host   %s, threads %d, context %.1f ms, read inputs %.1f ms, canonicalize %.1f ms, "
        "apply batch %.1f ms, download %.1f ms, write %.1f ms\n",
        std::string(dyng::to_string(res.get_backend())).c_str(), res.num_threads(), in.context_ms,
        in.load_ms, canonicalize_ms, apply_ms, download_ms, write_ms);
    for (int k = 0; k < K; ++k) {
      const dyng::sssp::stats& s = stats.objectives[static_cast<std::size_t>(k)];
      char device[64] = "";
      if (cuda) {
        std::snprintf(device, sizeof(device), ", device %.3f ms",
                      sosp_device_ms[static_cast<std::size_t>(k)]);
      }
      std::printf(
          "obj%d   SOSP update %.3f ms (invalidated %lld, iterations %lld, epochs %lld, pushes "
          "%lld, affected %lld, packed %d, engine %s%s)\n",
          k, sosp_ms[static_cast<std::size_t>(k)], static_cast<long long>(s.invalidated),
          static_cast<long long>(s.iterations), static_cast<long long>(s.epochs),
          static_cast<long long>(s.pushes), static_cast<long long>(s.affected),
          s.packed_parents ? 1 : 0, std::string(dyng::to_string(s.engine_used)).c_str(), device);
    }
    std::printf(
        "comb   combined graph + SOSP %.3f ms (%lld edges, L=%lld, affected %lld; path costs "
        "%.3f ms)\n",
        combined_ms, static_cast<long long>(stats.combined_edges),
        static_cast<long long>(stats.preference_scale), static_cast<long long>(stats.affected),
        prof.total_host_ms("mosp.path_costs"));
    std::printf("graph  read graph %.1f ms, read changes and trees %.1f ms, build %.3f ms\n",
                in.read_graph_ms, in.read_rest_ms, setup_prof.total_host_ms("graph.build"));
  }
  bool timing_written = true;
  if (!opt.timing.empty()) {
    std::ofstream csv(opt.timing);
    prof.write_csv(csv);
    for (const dyng::stage_sample& s : setup_prof.samples()) {
      csv << "stage," << s.name << ',' << s.host_ms << '\n';
    }
    csv << "stage,total.end_to_end," << end_to_end << '\n';
    csv << "stage,total.context," << in.context_ms << '\n';
    csv << "stage,total.download," << download_ms << '\n';
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
  std::printf("RESULT compute_ms=%.3f sosp_ms=%.3f apply_ms=%.3f end_to_end_ms=%.3f threads=%d\n",
              sosp_total + combined_ms, sosp_total, apply_ms, end_to_end, res.num_threads());
  std::fflush(stdout);
  return timing_written ? 0 : 1;
}

template <typename edge_t>
int run_update_typed(const options& opt) {
  using graph_t = graph_type<edge_t>;
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
  auto csr = read_graph<edge_t>(opt.graph, opt.K);
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
  if (opt.mosp) {
    loaded_inputs<edge_t> in{g,     batch,      dists,         trees,        n,      K, KG,
                             start, context_ms, read_graph_ms, read_rest_ms, load_ms};
    return run_mosp<edge_t>(opt, res, cuda, prof, setup_prof, in);
  }

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
    sssp_options.cuda_engine = opt.cuda_engine;
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
  const std::vector<dyng::sssp::stats> stats =
      dyng::update_each(res, g, batch.view(), dyng::host_view(pointers));
  res.attach_profiler(nullptr);

  // Per-objective times: the i-th call of each stage belongs to objective i.
  std::map<std::string, std::vector<double>> calls;
  for (const dyng::stage_sample& s : prof.samples()) {
    calls[s.name].push_back(s.host_ms);
  }
  // The engine's stages: sssp.enact_fused (the fused CUDA engine) or the four Tier A hooks (the
  // host engines and the CUDA operators engine); a run has one or the other.
  std::vector<double> sosp_ms(static_cast<std::size_t>(K), 0.0);
  std::vector<double> sosp_device_ms(static_cast<std::size_t>(K), 0.0);
  const std::vector<std::string> sosp_stages{"sssp.enact_fused", "sssp.identify_affected",
                                             "sssp.seed", "sssp.loop", "sssp.finalize"};
  for (const std::string& stage : sosp_stages) {
    const std::vector<double>& samples = calls[stage];
    for (std::size_t k = 0; k < samples.size() && k < sosp_ms.size(); ++k) {
      sosp_ms[k] += samples[k];
    }
  }
  if (cuda) {
    std::map<std::string, std::size_t> seen;  // the k-th sample of a stage is objective k
    for (const dyng::stage_sample& s : prof.samples()) {
      if (std::find(sosp_stages.begin(), sosp_stages.end(), s.name) != sosp_stages.end()) {
        const std::size_t k = seen[s.name]++;
        if (k < sosp_device_ms.size()) {
          sosp_device_ms[k] += s.device_ms;
        }
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
          "%lld, affected %lld, packed %d, engine %s%s)\n",
          k, sosp_ms[static_cast<std::size_t>(k)], static_cast<long long>(s.invalidated),
          static_cast<long long>(s.iterations), static_cast<long long>(s.epochs),
          static_cast<long long>(s.pushes), static_cast<long long>(s.affected),
          s.packed_parents ? 1 : 0, std::string(dyng::to_string(s.engine_used)).c_str(), device);
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

int run_update(int argc, char** argv) {
  const options opt = parse_update_options(argc, argv);
  return opt.edge64 ? run_update_typed<std::int64_t>(opt) : run_update_typed<std::int32_t>(opt);
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
