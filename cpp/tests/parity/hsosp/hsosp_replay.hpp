// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_replay.hpp
 * @brief The container side of hyper_sssp's parity (M11a criteria 8 and 10; ADRs 0085, 0086):
 *        the sessions of MOSP_ESCHER@4b86159's H-SOSP dumps (the golden set hyper_sssp_mosp_escher,
 *        ADR 0081) replayed WHOLE through dyng::hypergraph with slack_csr storage, by id.
 *
 * Every session is loaded as the by_id mode loads it (ADR 0080): B's rows with its virtual
 * source and target rows at their positions (weight 1; dynG id = B id - 1), slack_csr storage
 * and line graph, id_reuse::insert_first, capacity_policy::mosp_escher(extra) with B's headroom
 * factor, and B's CSR entry headroom and node capacity (the internal knobs; the device_build
 * record, every apply_batch record's headroom). Every batch replays B's HgBatch: the operations
 * naming a virtual row are filtered out by the harness and counted. After every batch, by id:
 *  - inserted_ids == final_ids - 1;
 *  - (skipped_ops - dropped_members) + filtered == B's skippedOps (ignored deletions + skipped
 *    incidence edits + filtered), dropped_members == B's deduplicated members;
 *  - the dead ids == dead_he - 1, the inserted and edited counts;
 *  - the rows B recorded (every row, or the touched ones) and their weights;
 *  - the deleted and inserted pairs of the line graph == B's (pairs.*), its pair count ==
 *    h2hPairCount (and B's candidates, recorded: equal on the CUDA store, M11a step 3);
 *  - check_integrity() (the slack CSR and the incidence lists, the allocator) and, with the
 *    oracle, h2h() == testing::rebuild_line_graph(h2v) and the pairs ==
 *    testing::line_graph_delta(pre, post).
 * A batch or load MOSP_ESCHER rejected must be rejected by dynG before anything changes.
 */
#pragma once

#include "parity/hsosp/hsosp_dump.hpp"

#include <dyng/core/resources.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::test::hsosp_replay {

/// What a replay checks.
struct options {
  bool oracle = true;     ///< rebuild_line_graph and line_graph_delta after every batch
  bool integrity = true;  ///< check_integrity() after the load and every batch
  std::size_t max_mismatches = 20;
};

/// The result of one dump.
struct report {
  std::string file, backend;
  int threads = 0;
  std::int64_t sessions = 0;          ///< bulk loads replayed
  std::int64_t stopped_sessions = 0;  ///< sessions not replayed to their end (must be 0)
  std::int64_t batches = 0;           ///< batches replayed
  std::int64_t vertex_batches = 0;    ///< batches with incidence edits
  std::int64_t ids_compared = 0;      ///< inserted ids compared with final_ids - 1
  std::int64_t rows_compared = 0;     ///< rows compared with B's
  std::int64_t pair_batches = 0;      ///< batches whose pairs were compared
  std::int64_t deleted_pairs = 0, inserted_pairs = 0;  ///< pairs compared
  std::int64_t filtered_virtual = 0;                   ///< B ops on a virtual row, filtered
  std::int64_t skipped_ops = 0;                        ///< the sum of B's skippedOps
  std::int64_t skipped_edits = 0;                      ///< incidence edits dynG skipped
  std::int64_t removals = 0, additions = 0;            ///< incidence edits dynG applied
  std::int64_t original_exceptions = 0;  ///< loads, builds and batches B rejected (compared)
  /// B's loads refused for a weight in (2^28, 2^31 - 1], which the container takes (its frozen
  /// [1, 2^31 - 1]) and hyper_sssp refuses (HD11; M11a step 4): recorded, not a difference.
  std::int64_t hyper_sssp_weight_limit = 0;
  std::int64_t oracle_states = 0;                     ///< states checked against the oracle
  std::int64_t candidates_equal = 0;                  ///< batches whose candidate count equals B's
  std::int64_t candidates_differ = 0;                 ///< (informational on the host stores)
  std::int64_t csr_rebuilds = 0, b_csr_rebuilds = 0;  ///< dynG's and B's tail rebuilds
  std::int64_t mirror_reuploads = 0, b_mirror_reuploads = 0;  ///< the CUDA store's and B's
  std::int64_t device_layouts_compared =
      0;                            ///< device builds and batches whose layout was compared
  std::int64_t digest_batches = 0;  ///< batches whose pairs were compared by digest
  std::int64_t node_growths = 0;    ///< dynG's node-capacity growths
  std::int64_t mismatch_count = 0;
  std::vector<std::string> mismatches;
  double seconds = 0;
  /// @brief No mismatch and every session to its end. @return The verdict.
  [[nodiscard]] bool ok() const {
    return mismatch_count == 0 && stopped_sessions == 0;
  }
  /// @brief The report as JSON. @return The text.
  [[nodiscard]] std::string to_json() const;
};

/**
 * @brief Replays the sessions of one dump (see the file comment).
 * @param[in] records The dump (a paper case with its loaded rows spliced back in).
 * @param[in] res     The backend's resources (sequential, openmp or cuda).
 * @param[in] opt     What to check.
 * @param[in] name    The dump's name for the report.
 * @return The report.
 */
report replay(const std::vector<hsosp_dump::record>& records, const resources& res,
              const options& opt, const std::string& name);

}  // namespace dyng::test::hsosp_replay
