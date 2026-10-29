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
};

/// The counters of a slack_csr store (the tests force each path and read it here).
struct slack_counters {
  std::int64_t csr_rebuilds = 0;       ///< full rebuilds after a tail overflow (csrRebuilt)
  std::int64_t relocations = 0;        ///< rows moved to the tail
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
};

}  // namespace dyng::detail::hyper
