// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay.hpp
 * @brief The triad_count replay of ESCHER-GPU@abcf9b2's (A's) golden set triad_count_escher_gpu
 *        (M9a step 2; PLAN 6.4.6, 8.3; ADR 0050): a case's hypergraph and batch files replayed
 *        through dyng::hypergraph on one backend, a counter evaluated after the build and after
 *        every batch. parity/compare.py triad_count drives it (dyng_triad_replay) and compares
 *        what it reports with the goldens.
 *
 * The container is built from the case's `.hg` (A's loaded rows, hyperedge i = A's key i + 1,
 * read with io::hypergraph_read_options::escher_gpu() and A's vertex count) under
 * {id_reuse::erase_first, capacity_policy::escher_gpu(<the case's H2V payload capacity>),
 * line_graph_policy::sorted_merge}, the properties of A's run. Every batch file (`.dgb`, flags
 * deleted_members and expected_ids) is applied in order, in one or two containers:
 *
 * - **by_key** deletes the exported ids (A's keys - 1) and requires
 *   hyper_apply_summary::inserted_ids == the exported expected ids, batch by batch;
 * - **by_content** resolves each exported deleted member set to a live hyperedge with that member
 *   set (escher_dump::by_content_resolver, on the host: the smallest live id with the set not taken
 *   earlier in the batch) and inserts A's rows in A's order, so its counts depend on no id.
 *
 * check_integrity() runs after every batch. The counter is
 *
 * - **oracle**: dyng::testing::triad_reference on to_hyperedge_list() (the live hyperedges copied
 *   to the host), every requested pattern;
 * - **algorithm**: triad_count (M9a steps 3-4; until then not available, the tool exits 2).
 *
 * The report carries per mode the full count and, per batch, the counts after it and the net
 * delta per pattern, the deletions and insertions, the unresolved and ambiguous deletions
 * (by_content), whether the inserted ids were the expected ones (by_key), the integrity verdict,
 * and the launch counters a counter records (empty for the oracle; the hook for steps 3-4).
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/testing/triad_reference.hpp>

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace dyng::test::triad_replay {

/// What a replay does.
struct options {
  std::string mode = "both";       ///< "by_content", "by_key" or "both"
  std::string counter = "oracle";  ///< "oracle" or "algorithm"
  /// The patterns to count (default: all four).
  std::vector<testing::triad_pattern> patterns = {
      testing::triad_pattern::h_motif, testing::triad_pattern::stathyper_type1,
      testing::triad_pattern::stathyper_type2, testing::triad_pattern::stathyper_type3};
  int oracle_threads = 0;             ///< the oracle's threads (0: OpenMP's default)
  bool integrity = true;              ///< check_integrity() after every batch
  std::int64_t payload_capacity = 0;  ///< the H2V payload capacity of escher_gpu(capacity)
  std::int32_t num_vertices = -1;     ///< A's vertex count (-1: the file's)
};

/// Counts of every requested pattern, by pattern name.
using pattern_counts = std::map<std::string, std::vector<std::int64_t>>;

/// One batch of one mode.
struct batch_report {
  std::int64_t index = 0;               ///< 1-based, in file order
  std::int64_t deletions = 0;           ///< deletions of the batch file
  std::int64_t insertions = 0;          ///< insertions of the batch file
  std::int64_t deleted = 0;             ///< hyper_apply_summary::deleted_hyperedges
  std::int64_t ignored = 0;             ///< hyper_apply_summary::ignored_deletions
  std::int64_t unresolved = 0;          ///< by_content: deleted sets without a live hyperedge
  std::int64_t ambiguous = 0;           ///< by_content: sets held by more than one live id
  bool ids_as_expected = true;          ///< by_key: inserted_ids == the expected ids
  std::int64_t first_id_mismatch = -1;  ///< by_key: the first insertion whose id differs
  bool integrity_ok = true;             ///< check_integrity() passed (or was not run)
  std::string integrity_error;          ///< its message when it failed
  std::int64_t num_hyperedges_after = 0;
  std::int64_t id_bound_after = 0;
  std::int64_t device_bytes_after = 0;  ///< store_stats().device_bytes
  pattern_counts counts_after;          ///< the counter after the batch
  pattern_counts delta;                 ///< counts_after - the counts before the batch
};

/// One mode's replay.
struct mode_report {
  std::string mode;                     ///< "by_content" or "by_key"
  pattern_counts full;                  ///< the counter after the build
  std::int64_t device_bytes_build = 0;  ///< store_stats().device_bytes after the build
  std::vector<batch_report> batches;
  std::int64_t oracle_evaluations = 0;  ///< oracle evaluations (others reused an equal list)
};

/// The result of one case.
struct report {
  std::string backend, counter;
  int threads = 0;
  std::string hg;
  std::int64_t num_vertices = 0, num_hyperedges = 0;
  std::vector<mode_report> modes;
  /// Launch counters per (mode, shape) the counter recorded (the algorithm's; empty for the
  /// oracle).
  std::map<std::string, std::int64_t> launches;
  /// The kernel paths of the (first) container at the end: fill, reuse, unfill x thread, warp,
  /// block (store_stats()).
  std::int64_t paths[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double seconds = 0;
  [[nodiscard]] std::string to_json() const;
};

/// The counter is not available (exit status 2 of the tool).
struct not_available : std::runtime_error {
  using std::runtime_error::runtime_error;
};

/**
 * @brief Replays one case.
 * @param[in] res     The backend's resources.
 * @param[in] hg      The case's hypergraph file.
 * @param[in] batches Its batch files, in order.
 * @param[in] opt     What to do.
 * @return The report.
 * @throws not_available for `--counter algorithm` before triad_count exists.
 */
report replay(const resources& res, const std::string& hg, const std::vector<std::string>& batches,
              const options& opt);

/**
 * @brief The pattern of a name ("h_motif", "stathyper_type1", ... or the short "type1"...).
 * @param[in] name The name.
 * @return The pattern.
 * @throws std::invalid_argument for an unknown name.
 */
testing::triad_pattern pattern_of(const std::string& name);

}  // namespace dyng::test::triad_replay
