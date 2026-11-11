// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file lp_replay.hpp
 * @brief The replays of label_propagation's oracles and legacy generator against
 *        LabelPropagation-CUDA@a276a3a's recorded traces (M10a step 2; ADRs 0067, 0068): shared by
 *        the tool dyng_lp_replay (parity/label_propagation_goldens.py drives it over the golden
 *        corpus) and the CPU test suite (the committed 5k5 / 50k digests).
 *
 * Every comparison is in the original's layout: its arrays (int32 ids, float bits, the CSR with
 * 32-bit offsets, BatchData's fields) and its trace fields, with the statuses mapped by name
 * (ADR 0068): inactive = VERTEX_INACTIVE (3), deleted = VERTEX_DELETED (1), active with a seed
 * label = VERTEX_GROUND_TRUTH (2), active without one = VERTEX_ACTIVE (0).
 */
#pragma once

#include "generators/dynlp_stream.hpp"

#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/testing/lp_host_model.hpp>

#include <cstddef>
#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace dyng::test::lp {

/// O's VertexStatus codes (headers/graphTypes.h).
enum original_status : std::int32_t {
  original_active = 0,
  original_deleted = 1,
  original_ground_truth = 2,
  original_inactive = 3
};

/// The model's state of every vertex in O's codes (by name, ADR 0068).
[[nodiscard]] std::vector<std::int32_t> original_statuses(const testing::lp_host_model& model);

/// O's BatchData as a model batch: the labels from groundTruthNodeIds by O's < 0.5f rule.
/// Throws std::runtime_error if a ground-truth id does not arrive in the batch.
[[nodiscard]] testing::lp_model_batch model_batch(const detail::dynlp_legacy_batch& batch,
                                                  std::int32_t num_vertices);

/// The model's row capacities from O's full-graph offsets.
[[nodiscard]] std::vector<std::int64_t> row_capacities(const std::vector<std::int32_t>& row_ptr);

/// One named array of the original's layout: its element count and its bytes (little-endian),
/// owned, or viewed in the vector it came from (the generator's arrays, up to 50M: no copies).
struct named_array {
  std::uint64_t count = 0;
  std::vector<std::uint8_t> bytes;     ///< owned bytes (when `view` is null)
  const std::uint8_t* view = nullptr;  ///< viewed bytes
  std::size_t view_size = 0;           ///< their number
  /// The bytes.
  [[nodiscard]] const std::uint8_t* data() const noexcept {
    return view != nullptr ? view : bytes.data();
  }
  /// The number of bytes.
  [[nodiscard]] std::size_t size() const noexcept {
    return view != nullptr ? view_size : bytes.size();
  }
};

/// The arrays of the generator in O's layout (views into `out`, which must outlive them), keyed
/// as the recorder's fields:
/// "row_ptr", "col_ind", "values", "true_labels", then per batch b "b<b>.inserted", "b<b>.gt_ids",
/// "b<b>.gt_labels", "b<b>.deleted", "b<b>.edge_src", "b<b>.edge_dst", "b<b>.edge_weight".
[[nodiscard]] std::map<std::string, named_array> generator_arrays(
    const detail::dynlp_legacy_output& out);

/// The model's prediction of O's trace fields for one batch, keyed "<record>.<field>" as the
/// recorder names them (apply_vertex_changes.status / .gt_class / .inserted / .num_inserted,
/// append_edges.row_end / .slot_col_ind / .slot_values / .error_flags (/ .row_ptr in b0),
/// sparsify.tau / .num_sub_edges / .sub_edges_sorted, connected_components.num_components /
/// .component_of_inserted, label_init.num_components / .w0 / .w1 / .initial_labels_of_inserted,
/// propagation_begin.frontier_size / .initial_frontier). The three Ins_eff stages are present only
/// when Ins_eff is not empty (O skips them then).
[[nodiscard]] std::map<std::string, named_array> model_fields(
    const testing::lp_host_model& model, const testing::lp_model_report& report, float tau,
    std::int32_t batch);

/// The hex SHA-256 of an array's bytes.
[[nodiscard]] std::string digest(const named_array& a);

}  // namespace dyng::test::lp
