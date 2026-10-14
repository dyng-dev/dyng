// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_ids.hpp
 * @brief Container-level id parity (M8b, criterion 4; ADR 0041, section 11): the batch records of
 *        the CBST traces of ESCHER-GPU@abcf9b2 (A) and MOSP_ESCHER@4b86159 (B) replayed through
 *        dyng::hypergraph on one backend, every inserted id, row and skipped count compared with
 *        the original's, in lockstep with dyng::testing::cbst_host_model and check_integrity()
 *        after every batch.
 *
 * A: `escher_gpu.build` gives the loaded rows (vertex v of A = dynG vertex v - 1; row key k = id
 * k - 1), `escher_gpu.update` the deleted keys and rows, the inserted rows and the keys the CBST
 * insert gave; replayed under {erase_first, escher_gpu() with the trace's payload capacity},
 * deleting by id (key - 1) and, in a second container, by content (each deleted vertex set
 * resolved to the smallest live id with that set not yet taken in the batch).
 *
 * B: `mosp_escher.bulk_load` gives the rows of a session (B's vertices are 0-based, B id i = dynG
 * id i - 1; the virtual source and target hyperedges are ordinary rows, their weight 0 loaded as
 * 1), `begin_batch` the operations and the final ids; replayed under {insert_first,
 * mosp_escher(extra of the session's trees)}, ops naming a virtual id filtered out and counted;
 * a session is replayed up to its first batch with incidence edits (vertex-kind, M11).
 */
#pragma once

#include "parity/escher/cbst_trace.hpp"

#include <dyng/core/resources.hpp>

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hypergraph_ids {

/// What a replay does.
struct options {
  std::string mode = "both";        ///< "by_id", "by_content" (A only) or "both"
  bool model = true;                ///< the lockstep with the host model
  bool integrity = true;            ///< check_integrity() after every batch
  std::size_t max_mismatches = 20;  ///< mismatches recorded in full
};

/// The result of one trace.
struct report {
  std::string trace, repo, backend, mode;
  int threads = 0;
  std::int64_t sessions = 0;          ///< bulk loads (B) or builds (A)
  std::int64_t batches = 0;           ///< batches replayed
  std::int64_t vertex_batches = 0;    ///< B batches with incidence edits (not replayed, M11)
  std::int64_t ids_compared = 0;      ///< inserted ids compared with the original's
  std::int64_t rows_compared = 0;     ///< h2v rows compared
  std::int64_t filtered_virtual = 0;  ///< B ops on a virtual id, filtered by the harness
  std::int64_t skipped_compared = 0;  ///< B batches whose skipped ops were compared
  std::int64_t skipped_ops = 0;       ///< the sum of B's skippedOps over those batches
  std::int64_t dropped_members = 0;   ///< repeated members dropped (dynG) = B's deduplicated ones
  std::int64_t ambiguous = 0;         ///< by-content resolutions not of the id A deleted
  std::int64_t content_diverged_at = -1;  ///< first batch whose by-content ids differ (-1: none)
  std::int64_t original_exceptions = 0;   ///< batches the original rejected (compared as errors)
  std::int64_t mismatch_count = 0;
  std::vector<std::string> mismatches;
  /// (dynG store_stats().device_bytes, the original's recorded device bytes) after the build and
  /// every batch (A: DynamicHypergraph::deviceBytes(), which also counts H2H; B: none).
  std::vector<std::pair<std::int64_t, std::int64_t>> device_bytes;
  /// A's accounting of the same scope as the container's (after the build and every batch): its
  /// H2V and V2H trees' deviceBytes() (the trees' op records) plus its H2V row view
  /// (RowLocator::deviceBytes(), (numHyperedgeKeys + 1) x 3 ints); -1 where the trace has no
  /// per-tree bytes. A's DynamicHypergraph::deviceBytes() adds its H2H tree and view (M8b step 2).
  std::vector<std::int64_t> original_store_bytes;
  std::int64_t h2v_capacity_trace = 0, h2v_capacity_dyng = 0;
  /// The kernel paths of the (by-id) container at the end: fill, reuse, unfill x (thread, warp,
  /// block), from store_stats() (criterion 12).
  std::int64_t paths[9] = {0, 0, 0, 0, 0, 0, 0, 0, 0};
  double seconds = 0;
  [[nodiscard]] bool ok() const {
    return mismatch_count == 0;
  }
  [[nodiscard]] std::string to_json() const;
};

/**
 * @brief Replays one trace through hypergraph containers on `res`.
 * @param[in,out] in  The trace (its side file set).
 * @param[in] res     The backend's resources.
 * @param[in] opt     What to do.
 * @param[in] name    The trace's name for the report.
 * @return The report.
 */
report replay(cbst_trace::reader& in, const resources& res, const options& opt,
              const std::string& name);

}  // namespace dyng::test::hypergraph_ids
