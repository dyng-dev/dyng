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
  std::string mode = "both";  ///< "by_id", "by_content" (A only) or "both"
  bool model = true;          ///< the lockstep with the host model
  bool integrity = true;      ///< check_integrity() after every batch
  bool compare = true;        ///< compare the rows with the original's (off: apply only)
  /// A: the by-id container keeps the line graph (line_graph_policy::sorted_merge) and its H2H
  /// is compared with A's after the build and every batch (M8b step 2, criterion 7): the layout
  /// (row offsets and payload) against the hashes of A's H2H construct and rebuild inputs, and,
  /// where the trace has state digests, every key's row hash and the rows the digests record.
  /// The device bytes are then in the whole scope of A's DynamicHypergraph::deviceBytes() and
  /// must be within 1.05x of A's at every point (criterion 8).
  bool line_graph = false;
  /// A (M8b criterion 16): before every batch, generators::legacy::escher_batch_generator draws
  /// the batch from the by-id container's live state with A's options (its command line) and
  /// one std::mt19937_64(--seed), and the batch must equal A's (deleted ids and rows, inserted
  /// rows); a synthetic hypergraph is drawn with generators::legacy::escher_hypergraph and must
  /// equal A's loaded rows.
  bool legacy_generator = false;
  /// M8b criterion 13: the `.hg` file the trace's run loaded (A: --input, B: --hg), read with the
  /// original's option set (io::hypergraph_read_options::escher_gpu(); hsosp(--seed, --maxcard)),
  /// must give the original's loaded rows (B: and vertex count and weights). For A the by-id
  /// container is then built from the file's list (the public reader's path; M8b step 5).
  std::string hg_file;
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
  std::int64_t content_diverged_at = -1;     ///< first batch whose by-content ids differ (-1: none)
  std::int64_t generated_graphs = 0;         ///< synthetic hypergraphs drawn and compared (legacy)
  std::int64_t generated_batches = 0;        ///< batches drawn and compared (legacy)
  std::int64_t generated_batches_equal = 0;  ///< ... equal to A's
  std::int64_t hg_hyperedges = 0;   ///< hyperedges of the .hg file compared with the original's
  std::int64_t content_states = 0;  ///< by-content states compared with A's (live sets)
  std::int64_t content_states_equal = 0;  ///< ... equal (the same live member sets)
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
  /// The line graph (options::line_graph): states compared (the build and every batch), their
  /// layouts equal to A's construct / rebuild inputs (offsets and payload hashes), the payload
  /// entries compared that way, the keys whose row hash was compared with a state digest and the
  /// rows compared in full (digest rows).
  std::int64_t h2h_states = 0, h2h_layouts_equal = 0, h2h_payload_entries = 0;
  std::int64_t h2h_unrecorded = 0;  ///< states whose A input the trace does not keep (extracts)
  std::int64_t h2h_key_hashes = 0, h2h_full_rows = 0;
  /// The largest ratio of the container's device bytes to A's DynamicHypergraph::deviceBytes()
  /// (options::line_graph; 0 when the trace records none).
  double memory_ratio_max = 0;
  /// The profiler stages of the (by-id) container per batch (0: the build), when the resources
  /// carry a profiler: (stage, batch, host milliseconds), in completion order.
  struct stage_time {
    std::string stage;
    std::int64_t batch = 0;
    double ms = 0;
  };
  std::vector<stage_time> stage_times;
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
