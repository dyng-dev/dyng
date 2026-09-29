// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/cli/cycle_enum_main.cpp (the `cycle-enum` driver:
// options, their validation, run_update, the output)
/**
 * @file cycle_enum_compat.cpp
 * @brief dyng-compat-cycle-enum: a drop-in clone of CycleEnumeration-GPU's `cycle-enum` for the
 *        static simple-cycle count and the incremental update on the sequential, OpenMP and CUDA
 *        backends, built on dynG, for the parity harness (PLAN Section 5.6, tools/compat).
 *
 * Usage (the original's flags and their aliases):
 *   dyng-compat-cycle-enum --input <path> [--backend sequential|openmp|cuda] [--openmp-threads t]
 *       [--cuda-device d] [--cuda-scheduler naive|work-queue] [--cuda-work-items
 *       auto|roots|edges|two-hop] [--report-timing] [--max-cycle-length k] [--task count|update]
 *       [--deletes d --inserts i --batch-seed s [--batch-locality w]] [--compare-recompute]
 *       [--timing <csv>] [--write-batch <path>] [--scope original|resident]
 *       [--edge-type int32|int64] [--chain n]
 *
 * Standard output is the original's histogram CSV ("# cycle_size, num_of_cycles", "len, count"
 * per non-zero length, "Total, N"), byte for byte. The input is read with io::read_edge_list
 * (the original's parser), the graph is built under graph_properties::cycle_enum_compatible()
 * (its DirectedGraph), `--task count` runs cycle_count::compute() on the chosen backend, and
 * `--task update` follows run_update(): the prior histogram with the static counter of the same
 * backend (not timed), the batch of generators::legacy::cycle_enum_batch() (generate_batch), then
 * the timed cycle_count::update() (update_histogram), whose time is printed as the original's
 * `update_seconds=`. `--compare-recompute` recomputes the post-batch graph with the same backend
 * and prints `recompute_seconds=` and `match=yes|no`. Diagnostics go to standard error; errors
 * exit with status 1 as the original's.
 *
 * Not ported: --algorithm read-tarjan and brute-force, --mode simple-time-window and temporal (0.4)
 * exit with status 1 and a message.
 *
 * The cuda backend (M2b): `--cuda-device`, `--cuda-scheduler` and `--cuda-work-items` map to
 * resources::cuda(d), options::scheduler and options::work_items; `--report-timing` (count task)
 * prints the original's lines "vertices:", "edges:", "kernel_ms:", "memcpy_ms:" and "total_ms:" on
 * standard error, measured with CUDA events (profiler_options::cuda_events): kernel_ms is the
 * stage cycle_count.count (building the work items and counting, the original's kernel region),
 * memcpy_ms the upload of the graph (graph.upload) plus the copy of the histogram
 * (cycle_count.finalize), total_ms the stage cycle_count.compute. The CUDA context is created and
 * every kernel loaded before any timed region (resources::warm_up(), the original's
 * initialize_device and occupancy queries). Two scopes (dynG's addition, PLAN 6.4.3):
 * `--scope original` (the default) times what the original times, with the graph uploaded inside
 * the timed call (count: the first compute() of the graph; update: the graph is moved to a fresh
 * copy without a device copy before the timed update(), as the original uploads G_t per call);
 * `--scope resident` first makes the graph resident on the device (an untimed compute() of the
 * 2-cycles for the count task; the prior's upload for the update task), so the timed call reads
 * the resident graph.
 * `--edge-type` selects the graph's edge-offset type: the default is int32 on cuda (the original's
 * 32-bit device CSR) and int64 on the host backends.
 * `--chain n` (dynG, update task; M2b review): n batches are generated from the graph before the
 * first update (seeds s, s + 1, ..., s + n - 1; a later batch may delete edges already deleted and
 * insert edges already present, which set semantics ignores) and applied by n chained update()
 * calls on the same graph and result, each timed with the same clock as update_ms. Standard output
 * and update_ms are those of the first update (the original's single update); the RESULT line adds
 * chain_ms=<ms of each update, comma separated> and chain_match=<yes|no> (the final histogram
 * against an untimed recompute with the same backend). It measures the resident graph across
 * batches (PLAN 6.4.3): the updates after the first read a graph that a device apply produced.
 *
 * dynG additions: --timing <csv> writes the profiler stages (kind,name,value); --write-batch <path>
 * (update task) writes the generated batch before the update, one change per line ("- u v" for
 * each deletion, then "+ u v" for each insertion, in the generator's order: the text of the
 * exporter's `generate` command, for the golden corpus of parity/cycle_count_goldens.py); and a
 * summary line on standard error:
 *   RESULT task=<count|update> read_ms=<> build_ms=<> prior_ms=<> generate_ms=<> compute_ms=<>
 *          update_ms=<> end_to_end_ms=<> threads=<> backend=<> scope=<> kernel_ms=<> memcpy_ms=<>
 *          total_ms=<> update_device_ms=<>
 * where compute_ms is the static count (task count) and update_ms the timed update (the region of
 * the original's update_seconds; parity/timed_regions/cycle_count.toml); kernel_ms, memcpy_ms and
 * total_ms are the CUDA-event times of --report-timing (0 otherwise) and update_device_ms the
 * CUDA-event time of the stage cycle_count.update (cuda with --report-timing; 0 otherwise).
 */
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/result_io.hpp>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using clock_type = std::chrono::steady_clock;

enum class task_kind { count, update };

struct cli_config {
  std::string input_path;
  std::string algorithm = "johnson";
  std::string backend = "sequential";
  std::string mode = "simple";
  std::optional<std::int64_t> time_window;
  std::optional<std::size_t> max_cycle_length;
  int openmp_threads = 1;
  int cuda_device_id = 0;
  task_kind task = task_kind::count;
  dyng::generators::legacy::cycle_enum_batch_options batch;
  bool compare_recompute = false;
  bool show_help = false;
  bool report_timing = false;
  std::string timing;
  std::string write_batch;
  dyng::cycle_count::cuda_scheduler scheduler = dyng::cycle_count::cuda_scheduler::work_queue;
  dyng::cycle_count::cuda_work_items work_items = dyng::cycle_count::cuda_work_items::automatic;
  std::string scope = "original";
  std::string edge_type;  // empty: int32 on cuda, int64 otherwise
  std::size_t chain = 1;  // chained updates (--chain; 1: the original's single update)
};

void print_usage(std::ostream& out) {
  out << "Usage: dyng-compat-cycle-enum --input <path> [options]\n\n"
      << "Options:\n"
      << "  --algorithm <johnson>\n"
      << "  --backend <sequential|openmp|cuda>\n"
      << "  --mode <simple>\n"
      << "  --max-cycle-length <integer >= 2>\n"
      << "  --openmp-threads <positive integer>\n"
      << "  --cuda-device <non-negative integer>\n"
      << "  --cuda-scheduler <naive|work-queue>\n"
      << "  --cuda-work-items <auto|roots|edges|two-hop>  (work-queue items)\n"
      << "  --task <count|update>\n"
      << "  --deletes <count>  --inserts <count>  (update task)\n"
      << "  --batch-seed <integer>  --batch-locality <window>  (update task)\n"
      << "  --compare-recompute  (update task: verify against a full recompute\n"
      << "                        with the same backend and time it)\n"
      << "  --report-timing  (cuda count: print kernel/memcpy/total ms to stderr)\n"
      << "  --timing <csv>  (dynG: write the profiler stages)\n"
      << "  --scope <original|resident>  (dynG, cuda: the graph's upload inside the timed call\n"
      << "                                or resident before it)\n"
      << "  --edge-type <int32|int64>  (dynG: edge offsets; default int32 on cuda, else int64)\n"
      << "  --write-batch <path>  (dynG, update task: write the generated batch)\n"
      << "  --chain <n>  (dynG, update task: n chained updates, batches of seeds s..s+n-1)\n"
      << "  --help\n";
}

bool is_option(std::string_view value) noexcept {
  return value.rfind("--", 0) == 0;
}

std::optional<std::string_view> next_value(const std::vector<std::string_view>& args,
                                           std::size_t& index, std::string_view option,
                                           std::ostream& err) {
  if (index + 1 >= args.size() || is_option(args[index + 1])) {
    err << option << " requires a value\n";
    return std::nullopt;
  }
  ++index;
  return args[index];
}

template <typename integer_t>
std::optional<integer_t> parse_integer(std::string_view value, std::string_view option,
                                       std::ostream& err) {
  integer_t parsed{};
  const char* const begin = value.data();
  const char* const end = value.data() + value.size();
  const auto [position, error] = std::from_chars(begin, end, parsed);
  if (error != std::errc{} || position != end) {
    err << option << " expects an integer value\n";
    return std::nullopt;
  }
  return parsed;
}

std::optional<std::string> parse_algorithm(std::string_view value) {
  if (value == "johnson") {
    return "johnson";
  }
  if (value == "read-tarjan" || value == "read_tarjan" || value == "readtarjan") {
    return "read-tarjan";
  }
  if (value == "brute-force" || value == "bruteforce" || value == "brute_force") {
    return "brute-force";
  }
  return std::nullopt;
}

std::optional<std::string> parse_execution(std::string_view value) {
  if (value == "sequential" || value == "seq" || value == "cpu") {
    return "sequential";
  }
  if (value == "openmp" || value == "omp") {
    return "openmp";
  }
  if (value == "cuda" || value == "gpu") {
    return "cuda";
  }
  return std::nullopt;
}

std::optional<std::string> parse_mode(std::string_view value) {
  if (value == "simple") {
    return "simple";
  }
  if (value == "simple-time-window" || value == "simple_time_window" || value == "time-window" ||
      value == "window") {
    return "simple-time-window";
  }
  if (value == "temporal") {
    return "temporal";
  }
  return std::nullopt;
}

/// The original's parse_args() (the same options, aliases, messages and checks).
std::optional<cli_config> parse_args(int argc, char** argv, std::ostream& err) {
  cli_config config;
  std::vector<std::string_view> args(argv, argv + argc);
  for (std::size_t index = 1; index < args.size(); ++index) {
    const std::string_view option = args[index];
    if (option == "--help" || option == "-h") {
      config.show_help = true;
      return config;
    }
    // Options with a value.
    const auto take = [&]() { return next_value(args, index, option, err); };
    if (option == "--input" || option == "-i") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      config.input_path = std::string(*value);
    } else if (option == "--algorithm" || option == "-a") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      const auto algorithm = parse_algorithm(*value);
      if (!algorithm) {
        err << "unknown algorithm: " << *value << '\n';
        return std::nullopt;
      }
      config.algorithm = *algorithm;
    } else if (option == "--backend" || option == "--execution" || option == "--policy") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      const auto execution = parse_execution(*value);
      if (!execution) {
        err << "unknown backend: " << *value << '\n';
        return std::nullopt;
      }
      config.backend = *execution;
    } else if (option == "--mode" || option == "-m") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      const auto mode = parse_mode(*value);
      if (!mode) {
        err << "unknown mode: " << *value << '\n';
        return std::nullopt;
      }
      config.mode = *mode;
    } else if (option == "--openmp-threads" || option == "--threads") {
      const auto value = take();
      const auto parsed = value ? parse_integer<int>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      config.openmp_threads = *parsed;
    } else if (option == "--cuda-device" || option == "--device") {
      const auto value = take();
      const auto parsed = value ? parse_integer<int>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      config.cuda_device_id = *parsed;
    } else if (option == "--cuda-scheduler" || option == "--scheduler") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      if (*value == "naive") {
        config.scheduler = dyng::cycle_count::cuda_scheduler::naive;
      } else if (*value == "work-queue" || *value == "work_queue" || *value == "workqueue" ||
                 *value == "queue") {
        config.scheduler = dyng::cycle_count::cuda_scheduler::work_queue;
      } else {
        err << "unknown cuda scheduler: " << *value << '\n';
        return std::nullopt;
      }
    } else if (option == "--cuda-work-items") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      using items = dyng::cycle_count::cuda_work_items;
      if (*value == "auto") {
        config.work_items = items::automatic;
      } else if (*value == "roots" || *value == "root") {
        config.work_items = items::roots;
      } else if (*value == "edges" || *value == "edge") {
        config.work_items = items::edges;
      } else if (*value == "two-hop" || *value == "two_hop" || *value == "twohop") {
        config.work_items = items::two_hop;
      } else {
        err << "unknown cuda work items: " << *value << '\n';
        return std::nullopt;
      }
    } else if (option == "--task") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      if (*value == "count" || *value == "recompute") {
        config.task = task_kind::count;
      } else if (*value == "update" || *value == "incremental") {
        config.task = task_kind::update;
      } else {
        err << "unknown task: " << *value << '\n';
        return std::nullopt;
      }
    } else if (option == "--deletes" || option == "--deletions" || option == "--inserts" ||
               option == "--insertions" || option == "--batch-locality") {
      const auto value = take();
      const auto parsed = value ? parse_integer<std::size_t>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      if (option == "--deletes" || option == "--deletions") {
        config.batch.num_deletions = static_cast<std::int64_t>(*parsed);
      } else if (option == "--batch-locality") {
        config.batch.locality_window = static_cast<std::int64_t>(*parsed);
      } else {
        config.batch.num_insertions = static_cast<std::int64_t>(*parsed);
      }
    } else if (option == "--batch-seed") {
      const auto value = take();
      const auto parsed = value ? parse_integer<std::uint64_t>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      config.batch.seed = *parsed;
    } else if (option == "--compare-recompute") {
      config.compare_recompute = true;
    } else if (option == "--report-timing") {
      config.report_timing = true;  // cuda count task only, as the original's
    } else if (option == "--scope") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      if (*value != "original" && *value != "resident") {
        err << "--scope: expected original or resident, got " << *value << '\n';
        return std::nullopt;
      }
      config.scope = std::string(*value);
    } else if (option == "--edge-type") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      if (*value != "int32" && *value != "int64") {
        err << "--edge-type: expected int32 or int64, got " << *value << '\n';
        return std::nullopt;
      }
      config.edge_type = std::string(*value);
    } else if (option == "--time-window" || option == "--window") {
      const auto value = take();
      const auto parsed = value ? parse_integer<std::int64_t>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      config.time_window = *parsed;
    } else if (option == "--max-cycle-length" || option == "--max-length") {
      const auto value = take();
      const auto parsed = value ? parse_integer<std::size_t>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      config.max_cycle_length = *parsed;
    } else if (option == "--timing") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      config.timing = std::string(*value);
    } else if (option == "--chain") {
      const auto value = take();
      const auto parsed = value ? parse_integer<std::size_t>(*value, option, err) : std::nullopt;
      if (!parsed) {
        return std::nullopt;
      }
      if (*parsed < 1) {
        err << "--chain: expected at least 1\n";
        return std::nullopt;
      }
      config.chain = *parsed;
    } else if (option == "--write-batch") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      config.write_batch = std::string(*value);
    } else {
      err << "unknown option: " << option << '\n';
      return std::nullopt;
    }
  }

  if (config.input_path.empty()) {
    err << "--input is required\n";
    return std::nullopt;
  }
  // validate_options()
  std::vector<std::string> errors;
  const bool windowed = config.mode != "simple";
  if (windowed) {
    if (!config.time_window) {
      errors.emplace_back("time_window is required for the selected cycle mode");
    } else if (*config.time_window <= 0) {
      errors.emplace_back("time_window must be positive");
    }
  }
  if (config.time_window && *config.time_window <= 0) {
    errors.emplace_back("time_window must be positive when provided");
  }
  if (config.max_cycle_length && *config.max_cycle_length < 2) {
    errors.emplace_back("max_cycle_length must be at least 2 when provided");
  }
  if (config.openmp_threads <= 0) {
    errors.emplace_back("openmp_threads must be positive");
  }
  if (config.cuda_device_id < 0) {
    errors.emplace_back("cuda_device_id must be non-negative");
  }
  // validate_cli_supported()
  if (config.backend == "cuda") {
    if (config.algorithm != "johnson") {
      errors.emplace_back("cuda backend currently supports only johnson");
    }
    if (!config.max_cycle_length) {
      errors.emplace_back("cuda backend requires --max-cycle-length for bounded device stacks");
    }
  }
  if (config.backend == "openmp") {
    if (config.algorithm == "brute-force") {
      errors.emplace_back("brute-force is only available with sequential backend");
    }
    if (config.mode == "simple-time-window") {
      errors.emplace_back("openmp backend does not implement simple-time-window mode yet");
    }
  }
  if (!errors.empty()) {
    for (const std::string& e : errors) {
      err << e << '\n';
    }
    return std::nullopt;
  }
  if (config.chain > 1 && config.task != task_kind::update) {
    err << "--chain needs --task update\n";
    return std::nullopt;
  }
  if (config.task == task_kind::update) {
    bool ok = true;
    if (config.mode != "simple") {
      err << "update task currently supports only --mode simple\n";
      ok = false;
    }
    if (!config.max_cycle_length) {
      err << "update task requires --max-cycle-length\n";
      ok = false;
    }
    if (!ok) {
      return std::nullopt;
    }
  }
  return config;
}

/// The original's standard output: the histogram CSV (io::write_histogram_csv) as a string.
std::string histogram_csv(dyng::array_view<const std::uint64_t> counts) {
  std::ostringstream out;
  dyng::io::write_histogram_csv(out, counts);
  return out.str();
}

double ms_since(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
}

/// The generated batch as the exporter's `batch_text()`: "- u v" per deletion, then "+ u v".
template <typename batch_t>
void write_batch_text(const std::string& path, const batch_t& batch) {
  std::ofstream out(path);
  for (std::size_t i = 0; i < batch.num_deletions(); ++i) {
    out << "- " << batch.delete_src()[i] << ' ' << batch.delete_dst()[i] << '\n';
  }
  for (std::size_t i = 0; i < batch.num_insertions(); ++i) {
    out << "+ " << batch.insert_src()[i] << ' ' << batch.insert_dst()[i] << '\n';
  }
  out.flush();
  if (!out) {
    throw dyng::io_error("dyng-compat-cycle-enum: cannot write " + path);
  }
}

/// What dynG does not port: other algorithms, modes and the CUDA backend (exit status 1).
void expect_ported(const cli_config& config) {
  if (config.algorithm != "johnson") {
    throw dyng::not_supported_error("dyng-compat-cycle-enum: --algorithm " + config.algorithm +
                                    " is not ported (dynG counts with Johnson's search)");
  }
  if (config.mode != "simple") {
    throw dyng::not_supported_error("dyng-compat-cycle-enum: --mode " + config.mode +
                                    " arrives with the time-window and temporal modes (0.4)");
  }
}

/// The device time of a stage in the calls completed after the first `from` samples (the timed
/// call only: the profiler of a shared handle also records the untimed calls before it).
double device_ms(const dyng::profiler& prof, std::size_t from, std::string_view name) {
  double total = 0.0;
  const auto& samples = prof.samples();
  for (std::size_t i = from; i < samples.size(); ++i) {
    if (samples[i].name == name) {
      total += samples[i].device_ms;
    }
  }
  return total;
}

template <typename graph_t>
int run(const cli_config& config, const std::chrono::steady_clock::time_point start) {
  const bool cuda = config.backend == "cuda";
  const dyng::resources res = cuda ? dyng::resources::cuda(config.cuda_device_id)
                              : config.backend == "openmp"
                                  ? dyng::resources::openmp(config.openmp_threads)
                                  : dyng::resources::sequential();
  if (cuda) {
    // The context and every kernel are loaded before any timed region (the original's
    // initialize_device() and occupancy queries; CUDA_MODULE_LOADING=EAGER).
    res.warm_up();
  }
  dyng::profiler_options popt;
  popt.cuda_events = cuda && config.report_timing;
  dyng::profiler prof(popt);
  dyng::resources timed = res;  // a copy shares the handle: the profiler records every call
  if (!config.timing.empty() || popt.cuda_events) {
    timed.attach_profiler(&prof);
  }
  dyng::cycle_count::options opt;
  // The library sizes everything by min(k, max(n, 2)), so k is passed through. A k beyond the int
  // range is exact as INT_MAX: the vertex ids are int32_t, so no simple cycle is longer.
  opt.max_length = config.max_cycle_length
                       ? static_cast<int>(std::min<std::size_t>(
                             *config.max_cycle_length,
                             static_cast<std::size_t>(std::numeric_limits<int>::max())))
                       : -1;
  opt.scheduler = config.scheduler;
  opt.work_items = config.work_items;
  const bool resident = cuda && config.scope == "resident";

  auto t = clock_type::now();
  const auto edges = dyng::io::read_edge_list<std::int32_t, dyng::unweighted>(config.input_path);
  const double read_ms = ms_since(t);
  t = clock_type::now();
  graph_t g =
      graph_t::from_edges(res, edges.view(), dyng::graph_properties::cycle_enum_compatible());
  const double build_ms = ms_since(t);

  double prior_ms = 0.0;
  double generate_ms = 0.0;
  double compute_ms = 0.0;
  double update_ms = 0.0;
  std::string histogram;
  double kernel_ms = 0.0;
  double memcpy_ms = 0.0;
  double total_ms = 0.0;
  double update_device_ms = 0.0;
  std::vector<double> chain_ms;
  std::string chain_match;
  if (config.task == task_kind::count) {
    if (resident) {
      // Upload the graph (untimed) with the lightest count there is: 2-cycles, one root per
      // queue claim (a millisecond on the gate graphs). A full count right before the timed one
      // would leave the GPU in the state of a long busy period, which the original's process
      // never has before its kernel (measured on GitHub k = 4 at locked clocks: 58 ms for a first
      // count, 61-62 ms for a count right after it, 58 ms again after a pause).
      dyng::cycle_count::options upload = opt;
      upload.max_length = 2;
      upload.work_items = dyng::cycle_count::cuda_work_items::roots;
      (void)dyng::cycle_count::compute(res, g, upload);
      prof.reset();  // the profiler of the shared handle recorded that call: keep the timed one
    }
    const std::size_t from = prof.samples().size();
    t = clock_type::now();
    const dyng::cycle_count::result r = dyng::cycle_count::compute(timed, g, opt);
    compute_ms = ms_since(t);
    histogram = histogram_csv(r.counts());
    if (popt.cuda_events) {
      kernel_ms = device_ms(prof, from, "cycle_count.count");
      memcpy_ms =
          device_ms(prof, from, "graph.upload") + device_ms(prof, from, "cycle_count.finalize");
      total_ms = device_ms(prof, from, "cycle_count.compute");
      std::cerr << "vertices: " << g.num_vertices() << '\n'
                << "edges: " << g.num_edges() << '\n'
                << "kernel_ms: " << kernel_ms << '\n'
                << "memcpy_ms: " << memcpy_ms << '\n'
                << "total_ms: " << total_ms << '\n';
    }
  } else {
    // run_update(): the prior (not timed), the batch, the timed update.
    t = clock_type::now();
    dyng::cycle_count::result r = dyng::cycle_count::compute(res, g, opt);
    prior_ms = ms_since(t);
    t = clock_type::now();
    const auto base = g.to_csr(res);
    const auto batch = dyng::generators::legacy::cycle_enum_batch(base.view(), config.batch);
    generate_ms = ms_since(t);
    // --chain: the later batches, generated from the same graph before any update.
    std::vector<decltype(dyng::generators::legacy::cycle_enum_batch(base.view(), config.batch))>
        chained;
    for (std::size_t i = 1; i < config.chain; ++i) {
      auto params = config.batch;
      params.seed = config.batch.seed + i;
      chained.push_back(dyng::generators::legacy::cycle_enum_batch(base.view(), params));
    }
    if (!config.write_batch.empty()) {
      write_batch_text(config.write_batch, batch);
    }
    if (cuda && !resident) {
      // The original uploads G_t in every update: time the update of a graph whose device copy is
      // not resident (a clone keeps the state, so the result still matches it).
      g = g.clone(res);
    }
    const std::size_t from = prof.samples().size();
    t = clock_type::now();
    const dyng::cycle_count::stats st = dyng::cycle_count::update(timed, g, batch.view(), r);
    update_ms = ms_since(t);
    if (popt.cuda_events) {
      update_device_ms = device_ms(prof, from, "cycle_count.update");
    }
    std::cerr << "deletions=" << batch.num_deletions() << " insertions=" << batch.num_insertions()
              << '\n';
    std::cerr << "update_seconds=" << update_ms / 1000.0 << '\n';
    (void)st;
    if (config.compare_recompute) {
      t = clock_type::now();
      const dyng::cycle_count::result recomputed = dyng::cycle_count::compute(res, g, opt);
      const double recompute_ms = ms_since(t);
      const auto a = r.counts();
      const auto b = recomputed.counts();
      const bool match = std::vector<std::uint64_t>(a.begin(), a.end()) ==
                         std::vector<std::uint64_t>(b.begin(), b.end());
      std::cerr << "recompute_seconds=" << recompute_ms / 1000.0 << '\n';
      std::cerr << "match=" << (match ? "yes" : "no") << '\n';
    }
    histogram = histogram_csv(r.counts());
    if (config.chain > 1) {
      chain_ms.push_back(update_ms);
      for (const auto& next : chained) {
        t = clock_type::now();
        (void)dyng::cycle_count::update(timed, g, next.view(), r);
        chain_ms.push_back(ms_since(t));
      }
      const dyng::cycle_count::result recomputed = dyng::cycle_count::compute(res, g, opt);
      const auto a = r.counts();
      const auto b = recomputed.counts();
      chain_match = std::vector<std::uint64_t>(a.begin(), a.end()) ==
                            std::vector<std::uint64_t>(b.begin(), b.end())
                        ? "yes"
                        : "no";
    }
  }
  std::cout << histogram;
  std::cout.flush();
  if (!config.timing.empty()) {
    std::ofstream csv(config.timing);
    prof.write_csv(csv);
  }
  std::cerr << "RESULT task=" << (config.task == task_kind::count ? "count" : "update")
            << " read_ms=" << read_ms << " build_ms=" << build_ms << " prior_ms=" << prior_ms
            << " generate_ms=" << generate_ms << " compute_ms=" << compute_ms
            << " update_ms=" << update_ms << " end_to_end_ms=" << ms_since(start)
            << " threads=" << res.num_threads() << " backend=" << config.backend
            << " scope=" << (cuda ? config.scope : std::string("host"))
            << " kernel_ms=" << kernel_ms << " memcpy_ms=" << memcpy_ms << " total_ms=" << total_ms
            << " update_device_ms=" << update_device_ms;
  if (!chain_ms.empty()) {
    std::cerr << " chain_ms=";
    for (std::size_t i = 0; i < chain_ms.size(); ++i) {
      std::cerr << (i == 0 ? "" : ",") << chain_ms[i];
    }
    std::cerr << " chain_match=" << chain_match;
  }
  std::cerr << '\n';
  return 0;
}

int run(const cli_config& config) {
  const auto start = clock_type::now();
  expect_ported(config);
  const std::string edge =
      config.edge_type.empty() ? (config.backend == "cuda" ? "int32" : "int64") : config.edge_type;
  if (edge == "int32") {
    return run<dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>>(config, start);
  }
  return run<dyng::graph<std::int32_t, std::int64_t, dyng::unweighted>>(config, start);
}

}  // namespace

int main(int argc, char** argv) {
  const std::optional<cli_config> config = parse_args(argc, argv, std::cerr);
  if (!config) {
    print_usage(std::cerr);
    return 1;
  }
  if (config->show_help) {
    print_usage(std::cout);
    return 0;
  }
  try {
    return run(*config);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
