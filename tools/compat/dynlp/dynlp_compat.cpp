// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/main.cu (the run), src/utils.cu (parseArguments,
// printConfig, writeLabelsToFile, writeResultsCSV) and src/dynLP.cu (computeAccuracy,
// classAgreement)
/**
 * @file dynlp_compat.cpp
 * @brief dyng-compat-dynlp: LabelPropagation-CUDA@a276a3a's `dynLP` command line on dynG (the
 *        parity and gate driver of label_propagation; ADR 0073).
 *
 * It takes the original's camelCase arguments with their atoi / atof semantics and defaults
 * (verbose unless --quiet; --validate on by default), draws the original's stream with
 * generators::legacy::dynlp_vertex_arrival_stream(), and runs the published setup: a graph from
 * graph::with_capacity(stream.row_capacity) with slotted rows and batch_semantics::net_effect(),
 * label_propagation::compute() on the empty graph and result::reserve() for the largest batch
 * (untimed), then one label_propagation::update() per batch (timed), each followed by the
 * evaluation (the labels on the host, the accuracy over the active vertices without a seed label,
 * as the original's computeAccuracy) and, with --validate, IrLP on the snapshot (compute()). The
 * outputs are the original's files, byte for byte where the computation is: results.csv,
 * dynlp_labels.txt, irlp_labels.txt (--validate) and graph/csr{RowPtr,ColInd,Values,TrueLabels}.txt
 * (unless --noGraphFiles).
 *
 * The comparator rules (sheet C8, decided by the author on 2026-10-28): a negative --tau keeps the
 * original's automatic threshold (the published one, stream.similarity_threshold, which is the
 * original's computeAvgEdgeWeight); --tau 0, a NaN --tau and --maxIter <= 0, which the original
 * accepts unchecked, are refused with an error message and exit status 1.
 *
 * dynG's own flags: --backend sequential|cuda (default cuda when it is built, else sequential),
 * --cuda-engine automatic|fused|operators, --label-dump DIR (the float32 bits of every label after
 * every batch, written after the timer to DIR/labels.f32, one block of n floats per batch; with
 * --validate also IrLP's of every snapshot to DIR/irlp.f32),
 * --trace DIR (the stage trace of the recorder's format, written by the library:
 * DYNG_DYNLP_TRACE_DIR=DIR; ADR 0073), --sync-stages (the profiler's synchronized stages, a
 * diagnostic run), and --check (dyng::testing's oracles after every batch, untimed: the harmonic
 * solution's mean error at most 2000 delta, no PCG skip, seeds exact and labels in [0, 1]; exit
 * status 1 if one fails). The per-batch times are also written to dyng_times.csv (time_ms, the
 * update's time: CUDA events on the resources' stream on cuda, host wall time on sequential, and
 * host_ms, the host wall time of the call).
 *
 * The tool includes installed public headers only (dyng/<name>.hpp, dyng/generators/<name>,
 * dyng/testing/<name>); a test checks its includes and CI builds it against an install tree.
 */
#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/testing/harmonic_solution.hpp>

#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <system_error>
#include <vector>

namespace {

namespace fs = std::filesystem;
namespace lp = dyng::label_propagation;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;

/// The original's DynLPConfig with its defaults (headers/graphTypes.h), plus dynG's flags.
struct config {
  int total_nodes = 50000;
  int avg_degree = 5;
  int num_batches = 10;
  unsigned int seed = 42;
  float tau = -1.0F;  // -1: auto
  float delta = 0.0001F;
  int max_iterations = 100000;
  bool damped = false;
  float ground_truth_fraction = 0.01F;
  float unlabeled_fraction = 0.90F;
  float deleted_fraction = 0.09F;
  std::string output_dir = "output/";
  bool validate = true;
  bool write_graph_files = true;
  bool verbose = true;
  bool profile = false;
  bool check = false;
  // dynG's own
  std::string backend;
  std::string cuda_engine = "automatic";
  std::string label_dump;
  std::string trace;
  bool sync_stages = false;
};

void usage() {
  std::cout
      << "Usage: dyng-compat-dynlp [options]   (LabelPropagation-CUDA's dynLP on dynG)\n"
      << "\nOptions of the original:\n"
      << "  --totalNodes N    Total vertices across all batches (default: 50000)\n"
      << "  --numBatches N    Number of dynamic batches (default: 10)\n"
      << "  --avgDegree  N    Average degree for graph gen (default: 5)\n"
      << "  --tau        F    Sparsification threshold (default: auto; negative: auto;\n"
      << "                    0 and NaN are refused)\n"
      << "  --delta      F    Convergence threshold (default: 0.0001)\n"
      << "  --maxIter    N    Max propagation iterations (default: 100000; > 0)\n"
      << "  --update     R    Label update rule: inplace (default) or damped\n"
      << "  --gtFraction F    Ground-truth fraction (default: 0.01)\n"
      << "  --seed       N    RNG seed (default: 42)\n"
      << "  --output     DIR  Output directory (default: output/)\n"
      << "  --validate        Run IrLP baseline comparison (default)\n"
      << "  --noValidate      Skip IrLP baseline\n"
      << "  --noGraphFiles    Do not write the generated graph to <output>/graph/\n"
      << "  --verbose         Print per-batch details (default)\n"
      << "  --quiet           Suppress per-batch details\n"
      << "  --profile         Write dynG's profiler stages to <output>/stages.csv\n"
      << "  --check           Run dyng::testing's oracles after every batch (untimed)\n"
      << "  --help, -h        Show this help message\n"
      << "\ndynG's own options:\n"
      << "  --backend B       sequential or cuda (default: cuda when built)\n"
      << "  --cuda-engine E   automatic, fused or operators (default: automatic)\n"
      << "  --label-dump DIR  The float32 bits of the labels after every batch\n"
      << "                    (DIR/labels.f32; written after the timer)\n"
      << "  --trace DIR       The library's stage trace (recorder format; DIR/<program>.dlt)\n"
      << "  --sync-stages     Synchronize at the profiler's stage boundaries (diagnostics)\n";
}

/// parseArguments (utils.cu 25-137) with the comparator rules of sheet C8.
bool parse(int argc, char** argv, config& c) {
  for (int i = 1; i < argc; ++i) {
    const std::string arg = argv[i];
    const bool has_value = i + 1 < argc;
    if (arg == "--totalNodes" && has_value) {
      c.total_nodes = std::atoi(argv[++i]);
      if (c.total_nodes <= 0) {
        std::cout << "Error: --totalNodes must be positive.\n";
        return false;
      }
    } else if (arg == "--numBatches" && has_value) {
      c.num_batches = std::atoi(argv[++i]);
      if (c.num_batches <= 0) {
        std::cout << "Error: --numBatches must be positive.\n";
        return false;
      }
    } else if (arg == "--avgDegree" && has_value) {
      c.avg_degree = std::atoi(argv[++i]);
      if (c.avg_degree <= 0) {
        std::cout << "Error: --avgDegree must be positive.\n";
        return false;
      }
    } else if (arg == "--tau" && has_value) {
      c.tau = static_cast<float>(std::atof(argv[++i]));
      if (std::isnan(c.tau) || c.tau == 0.0F) {
        // C8: the original accepts these unchecked (tau = 0 keeps every positive weight in G';
        // NaN keeps none); dyng-compat-dynlp refuses them, a documented difference.
        std::cout << "Error: --tau must be positive (or negative for the automatic threshold); "
                     "0 and NaN are refused.\n";
        return false;
      }
    } else if (arg == "--delta" && has_value) {
      c.delta = static_cast<float>(std::atof(argv[++i]));
      if (c.delta <= 0.0F) {
        std::cout << "Error: --delta must be positive.\n";
        return false;
      }
    } else if (arg == "--maxIter" && has_value) {
      c.max_iterations = std::atoi(argv[++i]);
      if (c.max_iterations <= 0) {
        std::cout << "Error: --maxIter must be positive.\n";  // C8: the original accepts it
        return false;
      }
    } else if (arg == "--update" && has_value) {
      const std::string rule = argv[++i];
      if (rule == "inplace") {
        c.damped = false;
      } else if (rule == "damped") {
        c.damped = true;
      } else {
        std::cout << "Error: --update must be 'inplace' or 'damped'.\n";
        return false;
      }
    } else if (arg == "--gtFraction" && has_value) {
      c.ground_truth_fraction = static_cast<float>(std::atof(argv[++i]));
      if (c.ground_truth_fraction <= 0.0F || c.ground_truth_fraction >= 1.0F) {
        std::cout << "Error: --gtFraction must be in (0, 1).\n";
        return false;
      }
    } else if (arg == "--seed" && has_value) {
      c.seed = static_cast<unsigned int>(std::atoi(argv[++i]));
    } else if (arg == "--output" && has_value) {
      c.output_dir = argv[++i];
    } else if (arg == "--validate") {
      c.validate = true;
    } else if (arg == "--noValidate") {
      c.validate = false;
    } else if (arg == "--noGraphFiles") {
      c.write_graph_files = false;
    } else if (arg == "--verbose") {
      c.verbose = true;
    } else if (arg == "--quiet") {
      c.verbose = false;
    } else if (arg == "--profile") {
      c.profile = true;
    } else if (arg == "--check") {
      c.check = true;
    } else if (arg == "--backend" && has_value) {
      c.backend = argv[++i];
      if (c.backend != "sequential" && c.backend != "cuda") {
        std::cout << "Error: --backend must be 'sequential' or 'cuda'.\n";
        return false;
      }
    } else if (arg == "--cuda-engine" && has_value) {
      c.cuda_engine = argv[++i];
      if (c.cuda_engine != "automatic" && c.cuda_engine != "fused" &&
          c.cuda_engine != "operators") {
        std::cout << "Error: --cuda-engine must be 'automatic', 'fused' or 'operators'.\n";
        return false;
      }
    } else if (arg == "--label-dump" && has_value) {
      c.label_dump = argv[++i];
    } else if (arg == "--trace" && has_value) {
      c.trace = argv[++i];
    } else if (arg == "--sync-stages") {
      c.sync_stages = true;
    } else if (arg == "--help" || arg == "-h") {
      usage();
      return false;
    } else {
      std::cout << "Error: Unknown argument: " << arg << "\n";
      std::cout << "Run with --help for usage information.\n";
      return false;
    }
  }
  const float total = c.ground_truth_fraction + c.unlabeled_fraction + c.deleted_fraction;
  if (total > 1.001F) {
    std::cout << "Error: Batch fractions (GT + unlabeled + deleted) exceed 1.0.\n";
    return false;
  }
  return true;
}

void print_config(const config& c, const std::string& backend) {
  std::cout << "\n"
            << "============================================================\n"
            << " DynLP Configuration (dyng-compat-dynlp)\n"
            << "============================================================\n"
            << "  Total nodes:        " << c.total_nodes << "\n"
            << "  Number of batches:  " << c.num_batches << "\n"
            << "  Average degree:     " << c.avg_degree << "\n"
            << "  Tau (sparsify):     "
            << (c.tau < 0 ? "auto (avg edge weight)" : std::to_string(c.tau)) << "\n"
            << "  Delta (converge):   " << c.delta << "\n"
            << "  Max iterations:     " << c.max_iterations << "\n"
            << "  Update rule:        " << (c.damped ? "damped (1/2-Jacobi)" : "inplace") << "\n"
            << "  GT fraction:        " << c.ground_truth_fraction << "\n"
            << "  Deleted fraction:   " << c.deleted_fraction << "\n"
            << "  Seed:               " << c.seed << "\n"
            << "  Output directory:   " << c.output_dir << "\n"
            << "  Validate (IrLP):    " << (c.validate ? "yes" : "no") << "\n"
            << "  Graph text files:   " << (c.write_graph_files ? "yes" : "no") << "\n"
            << "  Backend:            " << backend << "\n"
            << "============================================================\n\n";
}

/// The original's TextWriter (utils.cu): a buffered writer of integers and of floats as fixed
/// decimals with 6 digits (std::to_chars of the float widened to double).
class text_writer {
 public:
  explicit text_writer(const std::string& path)
      : file_(std::fopen(path.c_str(), "w")), buffer_(std::size_t{1} << 22) {}
  text_writer(const text_writer&) = delete;
  text_writer& operator=(const text_writer&) = delete;
  ~text_writer() {
    (void)close();
  }
  [[nodiscard]] bool ok() const {
    return file_ != nullptr;
  }
  void put_int(long long x) {
    reserve(24);
    pos_ = static_cast<std::size_t>(
        std::to_chars(buffer_.data() + pos_, buffer_.data() + buffer_.size(), x).ptr -
        buffer_.data());
  }
  void put_fixed6(float x) {
    reserve(64);
    pos_ = static_cast<std::size_t>(
        std::to_chars(buffer_.data() + pos_, buffer_.data() + buffer_.size(),
                      static_cast<double>(x), std::chars_format::fixed, 6)
            .ptr -
        buffer_.data());
  }
  void put_char(char c) {
    reserve(1);
    buffer_[pos_++] = c;
  }
  bool close() {
    if (file_ == nullptr) {
      return false;
    }
    flush();
    good_ = (std::fclose(file_) == 0) && good_;
    file_ = nullptr;
    return good_;
  }

 private:
  void reserve(std::size_t n) {
    if (pos_ + n > buffer_.size()) {
      flush();
    }
  }
  void flush() {
    if (pos_ > 0 && std::fwrite(buffer_.data(), 1, pos_, file_) != pos_) {
      good_ = false;
    }
    pos_ = 0;
  }
  std::FILE* file_;
  std::vector<char> buffer_;
  std::size_t pos_ = 0;
  bool good_ = true;
};

bool ensure_directory(const std::string& dir) {
  std::error_code ec;
  if (!fs::exists(dir, ec)) {
    fs::create_directories(dir, ec);
  }
  if (ec) {
    std::cout << "Error: Could not create directory " << dir << ": " << ec.message() << "\n";
    return false;
  }
  return true;
}

/// writeLabelsToFile: "i label" with the label in fixed, 6 decimals.
bool write_labels(const std::string& path, const float* labels, std::size_t n) {
  text_writer file(path);
  if (!file.ok()) {
    std::cout << "Error: Could not open file for writing: " << path << "\n";
    return false;
  }
  for (std::size_t i = 0; i < n; ++i) {
    file.put_int(static_cast<long long>(i));
    file.put_char(' ');
    file.put_fixed6(labels[i]);
    file.put_char('\n');
  }
  return file.close();
}

/// generateSparseGraph's text files (writeGraphFiles).
bool write_graph_files(
    const std::string& prefix,
    const dyng::generators::vertex_arrival_stream<std::int32_t, std::int32_t>& s) {
  {
    text_writer out(prefix + "RowPtr.txt");
    for (const std::int32_t x : s.graph.row_ptr) {
      out.put_int(x);
      out.put_char('\n');
    }
    if (!out.close()) {
      return false;
    }
  }
  {
    text_writer out(prefix + "ColInd.txt");
    for (const std::int32_t x : s.graph.col_ind) {
      out.put_int(x);
      out.put_char('\n');
    }
    if (!out.close()) {
      return false;
    }
  }
  {
    text_writer out(prefix + "Values.txt");
    for (const float w : s.graph.weights) {
      out.put_fixed6(w);
      out.put_char('\n');
    }
    if (!out.close()) {
      return false;
    }
  }
  text_writer out(prefix + "TrueLabels.txt");
  for (std::size_t i = 0; i < s.true_labels.size(); ++i) {
    out.put_int(static_cast<long long>(i));
    out.put_char(' ');
    out.put_int(s.true_labels[i] >= 1 ? 1 : 0);
    out.put_char('\n');
  }
  return out.close();
}

/// One batch's row of results.csv (the original's BatchResult).
struct batch_result {
  float time_ms = 0.0F;
  double host_ms = 0.0;
  std::int64_t iterations = 0;
  std::int64_t frontier_visits = 0;
  std::int64_t components = 0;
  float accuracy = 0.0F;
  float eval_ms = 0.0F;
  float irlp_time_ms = 0.0F;
  std::int64_t irlp_iterations = 0;
  float irlp_accuracy = 0.0F;
  float accuracy_vs_irlp = 0.0F;
};

/// writeResultsCSV (utils.cu): the original's format and precisions.
bool write_results(const std::string& path, const std::vector<batch_result>& results,
                   bool validate) {
  std::ofstream file(path);
  if (!file.is_open()) {
    std::cout << "Error: Could not open CSV file for writing: " << path << "\n";
    return false;
  }
  file << "batch,time_ms,iterations,frontier_visits,components,accuracy,eval_ms";
  if (validate) {
    file << ",irlp_time_ms,irlp_iterations,speedup,accuracy_vs_irlp,irlp_accuracy";
  }
  file << "\n";
  for (std::size_t i = 0; i < results.size(); ++i) {
    const batch_result& r = results[i];
    file << i << "," << std::fixed << std::setprecision(3) << r.time_ms << "," << r.iterations
         << "," << r.frontier_visits << "," << r.components << "," << std::setprecision(6)
         << r.accuracy << "," << std::setprecision(3) << r.eval_ms;
    if (validate) {
      file << "," << std::setprecision(3) << r.irlp_time_ms << "," << r.irlp_iterations << ","
           << (r.time_ms > 0.0F ? r.irlp_time_ms / r.time_ms : 0.0F) << "," << std::setprecision(6)
           << r.accuracy_vs_irlp << "," << r.irlp_accuracy;
    }
    file << "\n";
  }
  return static_cast<bool>(file);
}

/// computeAccuracy: the active vertices without a seed label (the original's VERTEX_ACTIVE).
float accuracy(const float* labels, const std::vector<std::int8_t>& truth,
               dyng::array_view<const dyng::vertex_status> status,
               dyng::array_view<const std::int8_t> seeds) {
  long long correct = 0;
  long long total = 0;
  for (std::size_t i = 0; i < truth.size(); ++i) {
    if (status[i] != dyng::vertex_status::active || seeds[i] >= 0) {
      continue;
    }
    const int predicted = labels[i] >= 0.5F ? 1 : 0;
    const int actual = truth[i] >= 1 ? 1 : 0;
    correct += predicted == actual ? 1 : 0;
    ++total;
  }
  return total > 0 ? static_cast<float>(correct) / static_cast<float>(total) : 1.0F;
}

/// classAgreement: the class agreement of two label vectors on the same vertices.
float class_agreement(const float* a, const float* b,
                      dyng::array_view<const dyng::vertex_status> status,
                      dyng::array_view<const std::int8_t> seeds) {
  long long same = 0;
  long long total = 0;
  for (std::size_t i = 0; i < status.size(); ++i) {
    if (status[i] != dyng::vertex_status::active || seeds[i] >= 0) {
      continue;
    }
    same += (a[i] >= 0.5F) == (b[i] >= 0.5F) ? 1 : 0;
    ++total;
  }
  return total > 0 ? static_cast<float>(same) / static_cast<float>(total) : 1.0F;
}

/// The profiler counter of IrLP's iterations (label_propagation::compute).
constexpr const char* irlp_counter = "label_propagation.irlp_iterations";

/// --check: dyng::testing's oracles on the labels after batch `b` (untimed): the harmonic
/// solution of the current graph, mean |F - F*| <= 2000 delta (AD7), no PCG skip, the seeds' labels
/// exact and every label in [0, 1].
bool check_batch(const config& c,
                 const dyng::generators::vertex_arrival_stream<std::int32_t, std::int32_t>& s,
                 const lp_graph& g, const lp::result<std::int32_t>& r,
                 const std::vector<float>& labels, std::size_t b) {
  const auto view = g.view();
  const std::size_t n = labels.size();
  std::vector<std::uint8_t> present(n);
  for (std::size_t v = 0; v < n; ++v) {
    present[v] = view.status[v] == dyng::vertex_status::active ? 1 : 0;
  }
  const auto seed_view = r.seed_labels();
  const std::vector<std::int8_t> seeds(seed_view.begin(), seed_view.end());
  const dyng::testing::harmonic_result f = dyng::testing::harmonic_solution(
      s.graph.row_ptr, s.graph.col_ind, s.graph.weights, present, seeds);
  const dyng::testing::label_error err =
      dyng::testing::label_error_against(labels, f, present, seeds);
  bool ok = true;
  const double bound = 2000.0 * static_cast<double>(c.delta);
  std::cout << "[check] batch " << b << ": labels vs harmonic optimum (PCG, " << f.iterations
            << " it, residual " << std::scientific << std::setprecision(1) << f.relative_residual
            << "): " << err.compared << " vertices, mean |F-F*| " << std::setprecision(2)
            << err.mean << " (tol " << std::setprecision(1) << bound << "), max "
            << std::setprecision(2) << err.max << ", class flips " << err.flips << "; "
            << err.excluded << " in GT-free components excluded" << std::defaultfloat << "\n";
  if (!f.converged) {
    std::cout << "[check] batch " << b << ": FAILED: the PCG did not converge (skip)\n";
    ok = false;
  }
  if (err.mean > bound) {
    std::cout << "[check] batch " << b << ": FAILED: mean |F-F*| above 2000 delta\n";
    ok = false;
  }
  for (std::size_t v = 0; v < n; ++v) {
    const float x = labels[v];
    if (!(x >= 0.0F && x <= 1.0F) || (seeds[v] == 0 && x != 0.0F) || (seeds[v] == 1 && x != 1.0F)) {
      std::cout << "[check] batch " << b << ": FAILED: vertex " << v << " has the label " << x
                << " (seed " << static_cast<int>(seeds[v]) << ")\n";
      ok = false;
      break;
    }
  }
  return ok;
}

double ms_since(std::chrono::steady_clock::time_point t0) {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
}

/// The profiler's last sample of `stage` (its host and device times).
const dyng::stage_sample* last_sample(const dyng::profiler& p, const std::string& stage) {
  const auto& samples = p.samples();
  for (auto it = samples.rbegin(); it != samples.rend(); ++it) {
    if (it->name == stage) {
      return &*it;
    }
  }
  return nullptr;
}

int run(const config& c) {
  const bool cuda_built = dyng::backend_available(dyng::backend::cuda);
  const std::string backend = c.backend.empty() ? (cuda_built ? "cuda" : "sequential") : c.backend;
  print_config(c, backend);
  if (!c.trace.empty()) {
    // The library writes its stage trace when this variable is set (ADR 0073).
    if (::setenv("DYNG_DYNLP_TRACE_DIR", c.trace.c_str(), 1) != 0) {
      std::cout << "Error: could not set DYNG_DYNLP_TRACE_DIR.\n";
      return 1;
    }
  }
  dyng::resources res =
      backend == "cuda" ? dyng::resources::cuda(0) : dyng::resources::sequential();
  const dyng::resources host = dyng::backend_available(dyng::backend::openmp)
                                   ? dyng::resources::openmp()
                                   : dyng::resources::sequential();
  dyng::profiler_options popt;
  popt.sync_stages = c.sync_stages;
  popt.cuda_events = backend == "cuda";
  dyng::profiler profiler(popt);
  res.attach_profiler(&profiler);
  if (!ensure_directory(c.output_dir)) {
    return 1;
  }
  if (c.write_graph_files && !ensure_directory(c.output_dir + "/graph")) {
    return 1;
  }
  // --- The stream (generateSparseGraph, generateBatches) ---
  std::cout << "Generating sparse graph...\n";
  dyng::generators::vertex_arrival_options va;
  va.total_vertices = c.total_nodes;
  va.average_degree = c.avg_degree;
  va.num_batches = c.num_batches;
  va.seed = c.seed;
  va.ground_truth_fraction = c.ground_truth_fraction;
  va.deleted_fraction = c.deleted_fraction;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(host, va);
  if (c.write_graph_files && !write_graph_files(c.output_dir + "/graph/csr", stream)) {
    std::cout << "Error: Graph generation failed.\n";
    return 1;
  }
  std::cout << "Generating batches...\n";
  const auto n = static_cast<std::size_t>(c.total_nodes);
  // --- The setup (untimed): the slotted graph, compute() on it, the reservation ---
  dyng::graph_properties props;
  props.directed = false;
  props.layout = dyng::row_layout::slotted;
  props.order = dyng::row_order::append;
  props.parallel_edges = dyng::multi_edges::allow;
  props.semantics = dyng::batch_semantics::net_effect();
  lp_graph g = lp_graph::with_capacity(res, static_cast<std::int32_t>(n),
                                       dyng::host_view(stream.row_capacity), props);
  const float tau = c.tau < 0.0F ? stream.similarity_threshold : c.tau;
  if (c.tau < 0.0F) {
    std::cout << "Auto tau (average edge weight): " << tau << "\n";
  }
  lp::options opt;
  opt.similarity_threshold = tau;
  opt.tolerance = c.delta;
  opt.max_iterations = c.max_iterations;
  opt.update_schedule = c.damped ? lp::schedule::damped : lp::schedule::in_place;
  opt.on_iteration_limit = lp::on_limit::report;
  opt.cuda_engine = c.cuda_engine == "fused"       ? dyng::engine::fused
                    : c.cuda_engine == "operators" ? dyng::engine::operators
                                                   : dyng::engine::automatic;
  auto labels = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                            dyng::array_view<const std::int8_t>(), opt);
  labels.reserve(res, stream.max_batch_vertices, stream.max_batch_edges);
  std::cout << "Running DynLP...\n\n";
  std::vector<batch_result> results(stream.batches.size());
  std::vector<float> host_labels(n, 0.5F);
  std::vector<float> irlp_labels(n, 0.5F);
  std::FILE* dump = nullptr;
  std::FILE* irlp_dump = nullptr;
  if (!c.label_dump.empty()) {
    if (!ensure_directory(c.label_dump)) {
      return 1;
    }
    dump = std::fopen((c.label_dump + "/labels.f32").c_str(), "wb");
    irlp_dump = c.validate ? std::fopen((c.label_dump + "/irlp.f32").c_str(), "wb") : nullptr;
    if (dump == nullptr || (c.validate && irlp_dump == nullptr)) {
      std::cout << "Error: could not create the label dump in " << c.label_dump << "\n";
      return 1;
    }
  }
  bool checks_passed = true;
  for (std::size_t b = 0; b < stream.batches.size(); ++b) {
    const auto& batch = stream.batches[b];
    if (c.verbose) {
      std::size_t gt = 0;
      for (const std::int8_t l : batch.insert_vertex_labels) {
        gt += l >= 0 ? 1 : 0;
      }
      std::cout << "--- Batch " << b << " ---\n"
                << "  Inserting " << batch.insert_vertices.size() << " vertices (" << gt
                << " GT), deleting " << batch.delete_vertices.size() << "\n";
    }
    batch_result& r = results[b];
    // --- Timed: one update() ---
    const auto t0 = std::chrono::steady_clock::now();
    const lp::stats st = lp::update(res, g, batch.view(), labels);
    res.synchronize();
    r.host_ms = ms_since(t0);
    const dyng::stage_sample* sample = last_sample(profiler, "label_propagation.update");
    r.time_ms = backend == "cuda" && sample != nullptr ? static_cast<float>(sample->device_ms)
                                                       : static_cast<float>(r.host_ms);
    r.iterations = st.iterations;
    r.frontier_visits = st.frontier_visits;
    r.components = st.components;
    // --- Evaluation (after the timer) ---
    const auto e0 = std::chrono::steady_clock::now();
    const dyng::array_view<const float> view = labels.labels();
    std::memcpy(host_labels.data(), view.data(), n * sizeof(float));
    const auto gv = g.view();
    r.accuracy = accuracy(host_labels.data(), stream.true_labels, gv.status, labels.seed_labels());
    r.eval_ms = static_cast<float>(ms_since(e0));
    if (dump != nullptr && std::fwrite(host_labels.data(), sizeof(float), n, dump) != n) {
      std::cout << "Error: could not write the label dump\n";
      return 1;
    }
    if (c.validate) {
      std::vector<std::int32_t> seed_vertices;
      std::vector<std::int8_t> seed_classes;
      const auto seeds = labels.seed_labels();
      for (std::size_t v = 0; v < n; ++v) {
        if (gv.status[v] == dyng::vertex_status::active && seeds[v] >= 0) {
          seed_vertices.push_back(static_cast<std::int32_t>(v));
          seed_classes.push_back(seeds[v]);
        }
      }
      const std::int64_t counted = profiler.counter(irlp_counter);
      const auto i0 = std::chrono::steady_clock::now();
      const std::vector<std::int32_t>& sv = seed_vertices;
      const std::vector<std::int8_t>& sc = seed_classes;
      const auto irlp = lp::compute(res, g, dyng::host_view(sv), dyng::host_view(sc), opt);
      res.synchronize();
      r.irlp_time_ms = static_cast<float>(ms_since(i0));
      const auto il = irlp.labels();
      std::memcpy(irlp_labels.data(), il.data(), n * sizeof(float));
      r.irlp_iterations = profiler.counter(irlp_counter) - counted;
      if (irlp_dump != nullptr &&
          std::fwrite(irlp_labels.data(), sizeof(float), n, irlp_dump) != n) {
        std::cout << "Error: could not write the IrLP label dump\n";
        return 1;
      }
      r.irlp_accuracy = accuracy(irlp_labels.data(), stream.true_labels, gv.status, seeds);
      r.accuracy_vs_irlp =
          class_agreement(host_labels.data(), irlp_labels.data(), gv.status, seeds);
    }
    if (c.check) {
      checks_passed = check_batch(c, stream, g, labels, host_labels, b) && checks_passed;
    }
    if (c.verbose) {
      std::cout << "  Step 1: " << st.components << " connected components, " << st.sparsified_edges
                << " subgraph edges (tau=" << tau << ")\n"
                << "  Step 3: Converged in " << st.iterations << " iterations"
                << (st.converged ? "" : " (cap reached)") << "\n"
                << "  Accuracy: " << (r.accuracy * 100.0F) << "%, Time: " << r.time_ms << " ms\n\n";
    }
  }
  if (dump != nullptr) {
    std::fclose(dump);
  }
  if (irlp_dump != nullptr) {
    std::fclose(irlp_dump);
  }
  // --- The outputs ---
  if (!write_labels(c.output_dir + "/dynlp_labels.txt", host_labels.data(), n)) {
    std::cout << "Error: Could not write DynLP labels.\n";
    return 1;
  }
  if (!write_results(c.output_dir + "/results.csv", results, c.validate)) {
    std::cout << "Error: Could not write results CSV.\n";
    return 1;
  }
  if (c.validate && !write_labels(c.output_dir + "/irlp_labels.txt", irlp_labels.data(), n)) {
    std::cout << "Error: Could not write IrLP labels.\n";
    return 1;
  }
  {
    std::ofstream times(c.output_dir + "/dyng_times.csv");
    times << "batch,time_ms,host_ms\n";
    for (std::size_t i = 0; i < results.size(); ++i) {
      times << i << "," << std::fixed << std::setprecision(6) << results[i].time_ms << ","
            << results[i].host_ms << "\n";
    }
  }
  if (c.profile) {
    std::ofstream stages(c.output_dir + "/stages.csv");
    profiler.write_csv(stages);
  }
  double total = 0.0;
  std::int64_t iterations = 0;
  for (const batch_result& r : results) {
    total += r.time_ms;
    iterations += r.iterations;
  }
  std::cout << "============================================================\n"
            << " DynLP Results Summary (dyng-compat-dynlp)\n"
            << "============================================================\n"
            << "  Total DynLP time:     " << std::fixed << std::setprecision(2) << total << " ms\n"
            << "  Propagation iters:    " << iterations << "\n"
            << "  Final batch accuracy: " << (results.back().accuracy * 100.0F) << "%\n"
            << "============================================================\n"
            << "\nOutput files written to: " << c.output_dir << "/\n";
  if (c.check) {
    std::cout << (checks_passed ? "[check] all oracle checks passed\n"
                                : "Error: oracle checks failed (see [check] lines above).\n");
  }
  return checks_passed ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  config c;
  if (!parse(argc, argv, c)) {
    return 1;
  }
  try {
    return run(c);
  } catch (const std::exception& e) {
    std::cout << "Error: " << e.what() << "\n";
    return 1;
  }
}
