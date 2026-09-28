// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file profiler.hpp
 * @brief An instance-based profiler for library stages and counters.
 * @ingroup core
 *
 * A profiler is attached to dyng::resources (resources::attach_profiler); there is no global
 * registry. Stage and counter names follow the fixed scheme `<algo>.<hook>[.<sub>]` in dotted
 * lower case, e.g. `sssp.identify_affected` or `graph.apply` (PLAN Section 4.7.5), so every
 * algorithm's profile has the same shape. Output: CSV `kind,name,value` (compatible with the
 * MOSP `--timing` CSV) and JSON.
 */
#pragma once

#include <chrono>
#include <cstdint>
#include <iosfwd>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

namespace dyng {

class resources;

namespace detail {
class cuda_event_timer;
}  // namespace detail

/**
 * @brief Profiler settings.
 * @ingroup core
 */
struct profiler_options {
  /// Synchronize the resources' stream at stage boundaries, so stage times are paper-comparable
  /// breakdowns (adds synchronization; host backends are unaffected).
  bool sync_stages = false;
  /// Emit an NVTX range per stage (push at its start, pop at its end), for Nsight Systems and other
  /// NVTX tools. Needs a build with DYNG_WITH_NVTX (ON in CUDA builds whose toolkit ships NVTX 3);
  /// ignored otherwise.
  bool nvtx = false;
  /// Time stages with CUDA events on the device (CUDA backend only; ignored otherwise).
  bool cuda_events = false;
};

/**
 * @brief Aggregate of all calls of one stage.
 * @ingroup core
 */
struct stage_record {
  std::string name;        ///< `<algo>.<hook>[.<sub>]`
  int depth = 0;           ///< nesting depth at the first call (0 = outermost)
  std::int64_t calls = 0;  ///< number of completed calls
  double host_ms = 0.0;    ///< total host wall time in milliseconds
  double device_ms = 0.0;  ///< total device time in milliseconds (0 without CUDA events)
};

/**
 * @brief One completed call of a stage.
 * @ingroup core
 */
struct stage_sample {
  std::string name;        ///< `<algo>.<hook>[.<sub>]`
  int depth = 0;           ///< nesting depth of this call
  double host_ms = 0.0;    ///< host wall time in milliseconds
  double device_ms = 0.0;  ///< device time in milliseconds (0 without CUDA events)
};

/**
 * @brief A named counter (summed over all additions).
 * @ingroup core
 */
struct counter_record {
  std::string name;        ///< `<algo>.<name>[.<sub>]`
  std::int64_t value = 0;  ///< the sum of all added values
};

/**
 * @brief Records stage times and counters of library calls.
 *
 * Not thread-safe: record from the thread that calls the library (library code records stages
 * only outside its parallel regions).
 * @ingroup core
 */
class profiler {
 public:
  /**
   * @brief Create a profiler.
   * @param[in] options Settings.
   */
  explicit profiler(profiler_options options = {});

  /**
   * @brief The settings.
   * @return The options given at construction.
   */
  [[nodiscard]] const profiler_options& options() const noexcept {
    return options_;
  }

  /**
   * @brief Start a stage (prefer scoped_stage).
   * @param[in] name Stage name `<algo>.<hook>[.<sub>]` in dotted lower case.
   * @throws invalid_argument_error if `name` does not follow the naming scheme.
   */
  void begin_stage(std::string_view name);

  /**
   * @brief End the innermost open stage.
   * @param[in] device_ms Device time of the stage in milliseconds, if measured (else 0).
   * @throws invalid_argument_error if no stage is open.
   */
  void end_stage(double device_ms = 0.0);

  /**
   * @brief Add to a counter (created with value 0 on first use).
   * @param[in] name  Counter name `<algo>.<name>[.<sub>]` in dotted lower case.
   * @param[in] value The amount to add.
   * @throws invalid_argument_error if `name` does not follow the naming scheme.
   */
  void add_counter(std::string_view name, std::int64_t value);

  /**
   * @brief Aggregated stages in order of first call.
   * @return One record per stage name.
   */
  [[nodiscard]] const std::vector<stage_record>& stages() const noexcept {
    return stages_;
  }

  /**
   * @brief Every completed stage call in completion order.
   * @return The samples.
   */
  [[nodiscard]] const std::vector<stage_sample>& samples() const noexcept {
    return samples_;
  }

  /**
   * @brief Counters in order of first use.
   * @return One record per counter name.
   */
  [[nodiscard]] const std::vector<counter_record>& counters() const noexcept {
    return counters_;
  }

  /**
   * @brief Total host time of a stage.
   * @param[in] name Stage name.
   * @return The summed host milliseconds, or 0 if the stage was never recorded.
   */
  [[nodiscard]] double total_host_ms(std::string_view name) const noexcept;

  /**
   * @brief Value of a counter.
   * @param[in] name Counter name.
   * @return The counter value, or 0 if the counter was never used.
   */
  [[nodiscard]] std::int64_t counter(std::string_view name) const noexcept;

  /**
   * @brief Drop all records, samples and counters.
   * @throws invalid_argument_error if a stage is still open.
   */
  void reset();

  /**
   * @brief Write CSV `kind,name,value`: one `stage` row per sample (milliseconds, %.6f) followed
   *        by one `counter` row per counter, as the MOSP `--timing` CSV.
   * @param[out] out The stream.
   */
  void write_csv(std::ostream& out) const;

  /**
   * @brief Write JSON `{"stages": [...], "counters": [...]}` with the aggregated records.
   * @param[out] out The stream.
   */
  void write_json(std::ostream& out) const;

  /**
   * @brief Whether a name follows `<algo>.<hook>[.<sub>]`: two or more dot-separated parts, each
   *        starting with a lower-case letter followed by lower-case letters, digits or '_'.
   * @param[in] name The name.
   * @return True if the name is valid.
   */
  [[nodiscard]] static bool is_valid_name(std::string_view name) noexcept;

 private:
  struct open_stage {
    std::size_t record;
    std::chrono::steady_clock::time_point start;
  };

  profiler_options options_;
  std::vector<stage_record> stages_;
  std::vector<stage_sample> samples_;
  std::vector<counter_record> counters_;
  std::vector<open_stage> open_;
};

/**
 * @brief RAII stage: begins a stage on construction and ends it on destruction.
 *
 * Library code writes `scoped_stage stage(res, "sssp.loop");`. It does nothing when no profiler
 * is attached. With profiler_options::sync_stages the resources are synchronized at both ends.
 * With profiler_options::cuda_events and resources of the CUDA backend, the stage also records a
 * CUDA event on the stream of the resources at both ends; the end waits for the second event and
 * adds the elapsed device time to the stage's `device_ms`.
 * @ingroup core
 */
class scoped_stage {
 public:
  /**
   * @brief Begin a stage on the profiler attached to `res` (if any).
   * @param[in] res  Resources; must outlive the scoped_stage.
   * @param[in] name Stage name `<algo>.<hook>[.<sub>]`.
   * @throws invalid_argument_error if a profiler is attached and `name` is invalid.
   * @throws cuda_error            if the CUDA events of profiler_options::cuda_events cannot be
   *         created or recorded.
   * @sync With profiler_options::sync_stages the resources are synchronized first; otherwise
   *       only a host timestamp is taken (and, with profiler_options::cuda_events on the CUDA
   *       backend, an event is recorded on the stream).
   */
  scoped_stage(const resources& res, std::string_view name);

  /**
   * @brief Begin a stage on `p` (if not null), without synchronization.
   * @param[in] p    The profiler, or nullptr.
   * @param[in] name Stage name `<algo>.<hook>[.<sub>]`.
   * @throws invalid_argument_error if `p` is not null and `name` is invalid.
   */
  scoped_stage(profiler* p, std::string_view name);

  scoped_stage(const scoped_stage&) = delete;
  scoped_stage& operator=(const scoped_stage&) = delete;
  scoped_stage(scoped_stage&&) = delete;
  scoped_stage& operator=(scoped_stage&&) = delete;

  /**
   * @brief End the stage (if still open).
   */
  ~scoped_stage();

  /**
   * @brief End the stage early.
   */
  void stop() noexcept;

 private:
  const resources* res_ = nullptr;
  profiler* profiler_ = nullptr;
  std::unique_ptr<detail::cuda_event_timer> events_;  ///< device timing (cuda_events), or null
};

}  // namespace dyng
