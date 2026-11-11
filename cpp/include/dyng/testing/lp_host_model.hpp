// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/checker.cu (DynLPChecker: the host model)
/**
 * @file lp_host_model.hpp
 * @brief dyng::testing::lp_host_model: an exact host model of label_propagation's container and
 *        deterministic stages (vertex-arrival batches on fixed slot rows), the oracle of the port
 *        of LabelPropagation-CUDA\@a276a3a (O) (PLAN 6.4.7, 8.3; ADR 0068).
 * @ingroup testing
 *
 * It extends O's --check host model (DynLPChecker) to the slot layout O's device graph keeps, so
 * that every deterministic field of O's per-batch stage trace can be predicted exactly:
 *
 * - **Vertex sets** (single-use ids): a batch inserts Ins and deletes Del. The net effect is
 *   Ins_eff = Ins \ Del (in Ins order) and Del_eff = Del ∩ V_t. Every id named in Del is retired
 *   (it ends deleted: an id in Ins ∩ Del never enters the graph, and deleting an id that was never
 *   inserted retires it); an id inserted while it is not inactive (inserted or retired before, or
 *   listed twice) raises O's sticky "reinserted" flag (1 + the largest such id). V_{t+1} =
 *   (V_t \ Del_eff) ∪ Ins_eff ("present"); previously_present() is V_t.
 * - **Seeds:** an arrival may carry a seed label (O's ground truth, class 0 or 1); a seeded vertex
 *   is "active with a seed label" (O's VERTEX_GROUND_TRUTH, ADR 0068's mapping). A seed label on a
 *   vertex that is also deleted in the batch is refused.
 * - **Rows:** vertex v owns the fixed slots [row_ptr(v), row_ptr(v + 1)), capacity = its degree in
 *   the full graph (the row capacities given at construction); its entries are the slots up to
 *   row_end(v). A batch's new edges are appended in half-edge-id order (edge e gives the entry v in
 *   u's row, then u in v's row; per row this is the batch's edge order, O's stable radix sort)
 *   when both endpoints are present after the batch; an entry past the row's capacity is dropped
 *   and raises the sticky "row overflow" flag. Rows are never compacted (entries to deleted
 *   vertices stay); unused slots hold column -1 and weight 0.
 * - **Affected vertices** (Algorithm 2): V_aff = N_t(Del_eff) ∪ Ins_eff ∪ N_{t+1}(Ins_eff) on the
 *   slot rows (before and after the batch's appends), restricted to present unlabelled vertices:
 *   O's initial propagation frontier.
 * - **G'** (sparsification): the batch's edges with weight > tau (strict) whose two endpoints are
 *   both in Ins_eff, as sorted (min, max) pairs.
 * - **Components** of G' on Ins_eff (singletons included): a component's root is its smallest id
 *   (union_find), and its id is the rank of its root among the roots in Ins_eff order (O's
 *   numbering).
 * - **Supernode weights** W0 / W1 per component: the sums over its vertices' row entries to seeded
 *   vertices (of any status) of class 0 / 1 of round-to-nearest-even(double(w) * 2^32) as
 *   unsigned 64-bit integers (O's __double2ull_rn, emulated with std::nearbyint).
 * - **Initial labels** of Ins_eff: a seeded vertex gets exactly 0 or 1; any other
 *   float(W1' / (W0' + W1')) with Wk' = double(Wk) / 2^32 when W0' + W1' > 1e-10, else 0.5.
 * - **tau:** average_weight(), O's computeAvgEdgeWeight over a full graph.
 *
 * Propagation is not modelled: labels are judged by harmonic_solution(). Standard library only;
 * it shares no code with the container or label_propagation (PLAN 5.8), which an include-graph
 * and link test checks.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief One batch for lp_host_model::apply(): arrivals with seed labels, new edges, departures.
 * @ingroup testing
 */
struct lp_model_batch {
  std::vector<std::int32_t> insert_vertices;    ///< Ins, in order
  std::vector<std::int8_t> insert_labels;       ///< per arrival: -1 or its seed label 0 / 1
  std::vector<std::int32_t> delete_vertices;    ///< Del, in order
  std::vector<std::int32_t> edge_sources;       ///< new edges (u, v), in order
  std::vector<std::int32_t> edge_destinations;  ///< see edge_sources
  std::vector<float> edge_weights;              ///< the weight of each new edge
};

/**
 * @brief What one lp_host_model::apply() predicts for the batch.
 * @ingroup testing
 */
struct lp_model_report {
  std::vector<std::int32_t> inserted;               ///< Ins_eff, in Ins order
  std::vector<std::int32_t> deleted;                ///< Del_eff, in Del order
  std::vector<std::int32_t> affected;               ///< V_aff (present, unlabelled), sorted
  std::vector<std::int32_t> sparsified;             ///< G' as (min, max) pairs, sorted, flattened
  std::int32_t num_components = 0;                  ///< components of G' on Ins_eff
  std::vector<std::int32_t> component_of_inserted;  ///< component id per Ins_eff vertex
  std::vector<std::uint64_t> w0;                    ///< supernode weight to class 0, per component
  std::vector<std::uint64_t> w1;                    ///< supernode weight to class 1, per component
  std::vector<float> initial_labels;                ///< per Ins_eff vertex
  std::int64_t appended = 0;                        ///< entries appended (both directions)
  std::int64_t dropped_absent = 0;    ///< edges not appended: an endpoint is not present
  std::int64_t dropped_overflow = 0;  ///< entries dropped past their row's capacity
};

/**
 * @brief The host model; see the file documentation.
 * @ingroup testing
 */
class lp_host_model {
 public:
  /// A vertex's state (O's statuses: active = VERTEX_ACTIVE or VERTEX_GROUND_TRUTH by its seed).
  enum class vertex_state : std::uint8_t {
    inactive = 0,  ///< never inserted (O's VERTEX_INACTIVE)
    active = 1,    ///< in the current graph
    deleted = 2,   ///< retired (O's VERTEX_DELETED)
  };

  /**
   * @brief An empty graph whose rows have the given slot capacities.
   * @param[in] row_capacity Slots of each vertex's row (>= 0): its full-graph degree.
   * @throws std::invalid_argument if a capacity is negative.
   */
  explicit lp_host_model(const std::vector<std::int64_t>& row_capacity);

  /**
   * @brief Apply one batch and predict its stages.
   * @param[in] batch The batch.
   * @param[in] tau   The sparsification threshold.
   * @return The prediction.
   * @throws std::invalid_argument before any change if an id or an edge endpoint is out of range,
   *         the label array does not match the arrivals, a label is not -1, 0 or 1, or a seeded
   *         arrival is also deleted.
   */
  lp_model_report apply(const lp_model_batch& batch, float tau);

  /**
   * @brief Number of vertices.
   * @return The size of the capacity array given at construction.
   */
  [[nodiscard]] std::int32_t num_vertices() const noexcept {
    return static_cast<std::int32_t>(state_.size());
  }
  /**
   * @brief The state of every vertex.
   * @return One state per vertex.
   */
  [[nodiscard]] const std::vector<vertex_state>& states() const noexcept {
    return state_;
  }
  /**
   * @brief The vertices of V_{t+1}: the active ones after the last batch.
   * @return 1 for a present vertex, 0 otherwise.
   */
  [[nodiscard]] const std::vector<std::uint8_t>& present() const noexcept {
    return present_;
  }
  /**
   * @brief The vertices of V_t: those present before the last batch.
   * @return 1 for a vertex that was present, 0 otherwise.
   */
  [[nodiscard]] const std::vector<std::uint8_t>& previously_present() const noexcept {
    return previous_;
  }
  /**
   * @brief The seed labels (O's gtClass).
   * @return Per vertex -1 or its seed label 0 / 1.
   */
  [[nodiscard]] const std::vector<std::int8_t>& seed_labels() const noexcept {
    return seed_;
  }
  /**
   * @brief The first slot of every row.
   * @return num_vertices + 1 offsets (the last one: the number of slots).
   */
  [[nodiscard]] const std::vector<std::int64_t>& row_ptr() const noexcept {
    return row_ptr_;
  }
  /**
   * @brief The end of every row's entries.
   * @return One slot index per vertex.
   */
  [[nodiscard]] const std::vector<std::int64_t>& row_end() const noexcept {
    return row_end_;
  }
  /**
   * @brief The column of every slot.
   * @return The columns (-1 for an unused slot).
   */
  [[nodiscard]] const std::vector<std::int32_t>& slot_columns() const noexcept {
    return col_;
  }
  /**
   * @brief The weight of every slot.
   * @return The weights (0 for an unused slot).
   */
  [[nodiscard]] const std::vector<float>& slot_weights() const noexcept {
    return weight_;
  }
  /**
   * @brief O's sticky ERROR_ROW_OVERFLOW flag.
   * @return 1 once an entry was dropped past its row's capacity, else 0.
   */
  [[nodiscard]] std::int32_t row_overflow_flag() const noexcept {
    return overflow_;
  }
  /**
   * @brief O's sticky ERROR_REINSERTED flag.
   * @return 1 + the largest id inserted while it was not inactive, else 0.
   */
  [[nodiscard]] std::int32_t reinserted_flag() const noexcept {
    return reinserted_;
  }

  /**
   * @brief The published threshold of a full graph: O's computeAvgEdgeWeight, the mean of the
   *        weights of the entries with a column >= 0, summed in double in entry order, rounded to
   *        float (0.5 for a graph without entries).
   * @param[in] col_ind The columns of the full CSR.
   * @param[in] weights Its weights.
   * @return tau.
   * @throws std::invalid_argument if the sizes differ.
   */
  [[nodiscard]] static float average_weight(const std::vector<std::int32_t>& col_ind,
                                            const std::vector<float>& weights);

 private:
  std::vector<vertex_state> state_;
  std::vector<std::uint8_t> present_;
  std::vector<std::uint8_t> previous_;
  std::vector<std::int8_t> seed_;
  std::vector<std::int64_t> row_ptr_;
  std::vector<std::int64_t> row_end_;
  std::vector<std::int32_t> col_;
  std::vector<float> weight_;
  std::int32_t overflow_ = 0;
  std::int32_t reinserted_ = 0;
};

}  // namespace dyng::testing
