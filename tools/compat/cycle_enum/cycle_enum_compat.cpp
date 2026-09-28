// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/cli/cycle_enum_main.cpp (the `cycle-enum` driver:
// options, their validation, run_update, the output)
/**
 * @file cycle_enum_compat.cpp
 * @brief dyng-compat-cycle-enum: a drop-in clone of CycleEnumeration-GPU's `cycle-enum` for the
 *        static simple-cycle count and the incremental update on the host backends, built on
 *        dynG, for the parity harness (PLAN Section 5.6, tools/compat).
 *
 * Usage (the original's flags and their aliases):
 *   dyng-compat-cycle-enum --input <path> [--backend sequential|openmp] [--openmp-threads t]
 *       [--max-cycle-length k] [--task count|update] [--deletes d --inserts i --batch-seed s
 *       [--batch-locality w]] [--compare-recompute] [--timing <csv>]
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
 * and --backend cuda (M2b) exit with status 1 and a message; the CUDA tuning flags
 * (--cuda-device, --cuda-scheduler, --cuda-work-items, --report-timing) are parsed and ignored.
 *
 * dynG additions: --timing <csv> writes the profiler stages (kind,name,value), and a summary line
 * on standard error:
 *   RESULT task=<count|update> read_ms=<> build_ms=<> prior_ms=<> generate_ms=<> compute_ms=<>
 *          update_ms=<> end_to_end_ms=<> threads=<>
 * where compute_ms is the static count (task count) and update_ms the timed update (the region of
 * the original's update_seconds; parity/timed_regions/cycle_count.toml).
 */
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/edge_list_io.hpp>
#include <dyng/io/result_io.hpp>

#include <charconv>
#include <chrono>
#include <cstdint>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <string_view>
#include <system_error>
#include <vector>

namespace {

using graph_t = dyng::graph<std::int32_t, std::int64_t, dyng::unweighted>;
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
  std::string timing;
};

void print_usage(std::ostream& out) {
  out << "Usage: dyng-compat-cycle-enum --input <path> [options]\n\n"
      << "Options:\n"
      << "  --algorithm <johnson>\n"
      << "  --backend <sequential|openmp>\n"
      << "  --mode <simple>\n"
      << "  --max-cycle-length <integer >= 2>\n"
      << "  --openmp-threads <positive integer>\n"
      << "  --task <count|update>\n"
      << "  --deletes <count>  --inserts <count>  (update task)\n"
      << "  --batch-seed <integer>  --batch-locality <window>  (update task)\n"
      << "  --compare-recompute  (update task: verify against a full recompute\n"
      << "                        with the same backend and time it)\n"
      << "  --timing <csv>  (dynG: write the profiler stages)\n"
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
      if (*value != "naive" && *value != "work-queue" && *value != "work_queue" &&
          *value != "workqueue" && *value != "queue") {
        err << "unknown cuda scheduler: " << *value << '\n';
        return std::nullopt;
      }
    } else if (option == "--cuda-work-items") {
      const auto value = take();
      if (!value) {
        return std::nullopt;
      }
      if (*value != "auto" && *value != "roots" && *value != "root" && *value != "edges" &&
          *value != "edge" && *value != "two-hop" && *value != "two_hop" && *value != "twohop") {
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
      // CUDA timing in the original; nothing to report on the host backends.
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

double ms_since(clock_type::time_point start) {
  return std::chrono::duration<double, std::milli>(clock_type::now() - start).count();
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
  if (config.backend == "cuda") {
    throw dyng::not_supported_error(
        "dyng-compat-cycle-enum: the cuda backend of cycle_count arrives with M2b");
  }
}

int run(const cli_config& config) {
  const auto start = clock_type::now();
  expect_ported(config);
  const dyng::resources res = config.backend == "openmp"
                                  ? dyng::resources::openmp(config.openmp_threads)
                                  : dyng::resources::sequential();
  dyng::profiler prof;
  dyng::resources timed = res;  // a copy shares the handle: the profiler records every call
  if (!config.timing.empty()) {
    timed.attach_profiler(&prof);
  }
  dyng::cycle_count::options opt;
  opt.max_length = config.max_cycle_length ? static_cast<int>(*config.max_cycle_length) : -1;
  if (config.max_cycle_length && *config.max_cycle_length > 1000000) {
    opt.max_length = 1000000;  // longer than any simple cycle of a graph this driver can read
  }

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
  if (config.task == task_kind::count) {
    t = clock_type::now();
    const dyng::cycle_count::result r = dyng::cycle_count::compute(timed, g, opt);
    compute_ms = ms_since(t);
    histogram = dyng::io::format_histogram_csv(r.counts());
  } else {
    // run_update(): the prior (not timed), the batch, the timed update.
    t = clock_type::now();
    dyng::cycle_count::result r = dyng::cycle_count::compute(res, g, opt);
    prior_ms = ms_since(t);
    t = clock_type::now();
    const auto batch =
        dyng::generators::legacy::cycle_enum_batch(g.to_csr(res).view(), config.batch);
    generate_ms = ms_since(t);
    t = clock_type::now();
    const dyng::cycle_count::stats st = dyng::cycle_count::update(timed, g, batch.view(), r);
    update_ms = ms_since(t);
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
    histogram = dyng::io::format_histogram_csv(r.counts());
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
            << " threads=" << res.num_threads() << '\n';
  return 0;
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
