// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slack_rows.hpp
 * @brief The slack-row accessor of slack_csr storage: the line graph's node rows and weights as
 *        the algorithms read them (hyper_sssp's engines, M11a step 4; label_propagation reuses it
 *        in M10b), the store's counters and knobs (ADR 0086). Host-only: no CUDA header.
 *
 * Reached with detail::hypergraph_access::slack(h). The pointers are in the store's memory space
 * (host for the sequential and openmp backends, device for cuda, M11a step 3) and valid until the
 * next apply(). Row r is col_ind[row_start[r] .. row_start[r] + deg[r]) (neighbour ids, a set in
 * no particular order), for r below num_nodes; node_weight[r] is the weight of id r (a dead id
 * keeps its last weight, as MOSP_ESCHER's device weights do).
 */
#pragma once

#include "hypergraph/slack_csr/slack_layout.hpp"

#include <dyng/core/memory.hpp>

#include <cstdint>
#include <vector>

namespace dyng::detail::hyper {

/// The node rows and weights of a slack_csr line graph.
struct slack_row_view {
  memory_space space = memory_space::host;    ///< where the arrays are
  int device = -1;                            ///< the CUDA device (space == device)
  std::int64_t num_nodes = 0;                 ///< rows (the largest id bound so far)
  std::int64_t node_capacity = 0;             ///< the size of the node-indexed arrays
  std::int64_t num_entries = 0;               ///< entries (each pair in both rows)
  std::int64_t used_entries = 0;              ///< slots used (the tail cursor)
  std::int64_t cap_entries = 0;               ///< the entry capacity
  std::int64_t pair_count = 0;                ///< the live pairs, maintained per batch
  const std::int64_t* row_start = nullptr;    ///< per node: the offset of its slot
  const std::int32_t* deg = nullptr;          ///< per node: its entries
  const std::int32_t* cap = nullptr;          ///< per node: its slot
  const std::int32_t* col_ind = nullptr;      ///< the neighbour ids
  const std::int64_t* node_weight = nullptr;  ///< per node: its weight
  /// The incidence mirror's value arrays (the CUDA store; 0 on the host stores, which keep no
  /// mirror): their capacities and tail cursors, hyperedge rows and vertex rows.
  std::int64_t mirror_he_capacity = 0, mirror_he_tail = 0;
  std::int64_t mirror_v_capacity = 0, mirror_v_tail = 0;  ///< the vertex rows
};

/// The last batch's pair delta in the store's memory space (the CUDA store, ADR 0089; MOSP_ESCHER's
/// DeviceDelta, which its hsospUpdate reads): `pairs` holds num_deleted deleted pairs, then
/// num_inserted inserted ones, each (a, b) with a < b as two int32 ids, increasing; `ids` holds
/// the num_new inserted ids (recycled ones included), then the num_dead deleted ids that stay
/// dead; new_weights the weights of the inserted ones. Valid until the next apply(). The host
/// stores give the pairs in the applied record instead (all pointers null).
struct slack_delta_view {
  memory_space space = memory_space::host;    ///< where the arrays are
  int device = -1;                            ///< the CUDA device (space == device)
  const std::int32_t* pairs = nullptr;        ///< 2 x (num_deleted + num_inserted) ids
  const std::int32_t* ids = nullptr;          ///< num_new + num_dead ids
  const std::int64_t* new_weights = nullptr;  ///< num_new weights
  std::int64_t num_deleted = 0;               ///< deleted pairs
  std::int64_t num_inserted = 0;              ///< inserted pairs
  std::int64_t num_new = 0;                   ///< inserted ids
  std::int64_t num_dead = 0;                  ///< dead ids
};

/// The counters of a slack_csr store (the tests force each path and read it here).
struct slack_counters {
  std::int64_t csr_rebuilds = 0;       ///< full rebuilds after a tail overflow (csrRebuilt)
  std::int64_t relocations = 0;        ///< rows moved to the tail (the host store)
  std::int64_t tail_entries = 0;       ///< slots taken at the tail by moved rows (every store)
  std::int64_t node_growths = 0;       ///< growths of the node-indexed arrays
  std::int64_t mirror_reuploads = 0;   ///< incidence-mirror re-uploads (the CUDA store)
  std::int64_t candidates = 0;         ///< the last batch's candidate entries
  std::int64_t line_graph_builds = 0;  ///< builds from the incidence lists
};

/// What a slack_csr store gives the container and the algorithms.
class slack_access {
 public:
  slack_access() = default;
  slack_access(const slack_access&) = delete;
  slack_access& operator=(const slack_access&) = delete;
  slack_access(slack_access&&) = delete;
  slack_access& operator=(slack_access&&) = delete;
  virtual ~slack_access() = default;

  /// @brief The node rows (the line graph must be maintained). @return The view.
  [[nodiscard]] virtual slack_row_view rows() const = 0;
  /// @brief The counters. @return They.
  [[nodiscard]] virtual slack_counters counters() const noexcept = 0;
  /// @brief The knobs. @return They.
  [[nodiscard]] virtual slack::knobs knobs() const noexcept = 0;
  /// @brief The entry headroom of the next rebuilds (H-SOSP's applyBatch argument).
  /// @param[in] headroom At least 1.
  virtual void set_entry_headroom(double headroom) = 0;
  /// @brief The number of hyperedges of each vertex (the applied record's post lengths).
  /// @param[in] vertices The vertices. @param[out] out Their lengths.
  virtual void vertex_lengths(const std::vector<std::int32_t>& vertices,
                              std::vector<std::int32_t>& out) const = 0;
  /// @brief The last batch's pair delta on the device (the CUDA store). @return The view (null
  /// pointers on the host stores, whose pairs are in the applied record).
  [[nodiscard]] virtual slack_delta_view last_delta() const noexcept {
    return {};
  }
  /// @brief The bytes of the line graph in MOSP_ESCHER's graph_mb scope (its
  /// DeviceH2H::deviceBytes(): the node arrays, the entries, the tail cursor and flag, the
  /// incidence mirror; hsospDevice.cu:599-604), the memory gate's store accounting (HD6).
  /// @return The bytes (0 on the host stores).
  [[nodiscard]] virtual std::int64_t graph_scope_bytes() const noexcept {
    return 0;
  }
};

}  // namespace dyng::detail::hyper
