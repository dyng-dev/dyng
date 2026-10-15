// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file batch_checks.hpp
 * @brief The checks the batch writers (`.dgt`, `.dgb`) make before writing a batch, so that
 *        every written file reads back: one rule for both formats.
 */
#pragma once

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

/// Whether an array view may be read on the host (empty arrays may be anywhere).
template <typename view_t>
bool host_readable(const view_t& view) noexcept {
  return view.empty() || is_host_accessible(view.space());
}

/**
 * @brief Check a hyperedge batch for a batch file: host memory; num_insertions() + 1 offsets
 *        from 0, non-decreasing, ending at the member count; no empty insertion; members and
 *        deletion ids >= 0; one weight >= 1 per insertion (none for unweighted).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @param[in] batch The batch.
 * @param[in] who   The writer's name, for the message.
 * @param[in] index The batch's index, for the message.
 * @throws invalid_argument_error if a check fails.
 */
template <typename vertex_t, typename weight_t>
void check_hyperedge_batch_arrays(const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                  const char* who, std::size_t index) {
  DYNG_EXPECTS(host_readable(batch.delete_hyperedges) && host_readable(batch.insert_offsets) &&
                   host_readable(batch.insert_members) && host_readable(batch.insert_weights),
               who, ": batch ", index, " must be in host-accessible memory");
  const std::size_t n = batch.num_insertions();
  const std::size_t num_members = batch.insert_members.size();
  DYNG_EXPECTS(n > 0 ? batch.insert_offsets[0] == 0 &&
                           batch.insert_offsets[n] == static_cast<std::int64_t>(num_members)
                     : num_members == 0,
               who, ": the insertion offsets of batch ", index,
               " must start at 0 and end at the member count");
  for (std::size_t i = 0; i < n; ++i) {
    DYNG_EXPECTS(batch.insert_offsets[i] < batch.insert_offsets[i + 1], who, ": insertion ", i,
                 " of batch ", index,
                 batch.insert_offsets[i] == batch.insert_offsets[i + 1]
                     ? " is empty (a hyperedge has at least one member)"
                     : " has decreasing offsets");
  }
  for (std::size_t j = 0; j < num_members; ++j) {
    DYNG_EXPECTS(batch.insert_members[j] >= 0, who, ": batch ", index, " has the negative member ",
                 static_cast<std::int64_t>(batch.insert_members[j]));
  }
  for (std::size_t j = 0; j < batch.delete_hyperedges.size(); ++j) {
    DYNG_EXPECTS(batch.delete_hyperedges[j] >= 0, who, ": batch ", index,
                 " deletes the negative id ", batch.delete_hyperedges[j],
                 " (a batch file holds ids >= 0)");
  }
  if constexpr (is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(batch.insert_weights.empty(), who, ": batch ", index,
                 " has weights for an unweighted type");
  } else {
    DYNG_EXPECTS(batch.insert_weights.size() == n, who, ": batch ", index, " has ",
                 batch.insert_weights.size(), " weights for ", n, " insertions");
    for (std::size_t i = 0; i < n; ++i) {
      DYNG_EXPECTS(batch.insert_weights[i] >= 1, who, ": the weight of insertion ", i, " of batch ",
                   index, " is below 1");
    }
  }
}

/**
 * @brief Check an edge batch for a batch file: host memory, no vertex operations (0.4),
 *        consistent sizes, ids >= 0.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @param[in] batch The batch.
 * @param[in] who   The writer's name, for the message.
 * @param[in] index The batch's index, for the message.
 * @throws invalid_argument_error if a check fails.
 */
template <typename vertex_t, typename weight_t>
void check_edge_batch_arrays(const edge_batch_view<vertex_t, weight_t>& batch, const char* who,
                             std::size_t index) {
  const std::size_t n = batch.insert_src.size();
  DYNG_EXPECTS(batch.insert_vertices.empty() && batch.delete_vertices.empty(), who, ": batch ",
               index, " has vertex operations, which the batch files of this release do not hold");
  DYNG_EXPECTS(batch.num_weights >= 0 && batch.insert_dst.size() == n &&
                   batch.insert_weights.size() == n * static_cast<std::size_t>(batch.num_weights) &&
                   batch.delete_dst.size() == batch.delete_src.size(),
               who, ": the arrays of batch ", index, " disagree in size");
  DYNG_EXPECTS(host_readable(batch.insert_src) && host_readable(batch.insert_dst) &&
                   host_readable(batch.insert_weights) && host_readable(batch.delete_src) &&
                   host_readable(batch.delete_dst),
               who, ": batch ", index, " must be in host-accessible memory");
  for (std::size_t i = 0; i < n; ++i) {
    DYNG_EXPECTS(batch.insert_src[i] >= 0 && batch.insert_dst[i] >= 0, who, ": batch ", index,
                 " has a negative vertex id");
  }
  for (std::size_t i = 0; i < batch.delete_src.size(); ++i) {
    DYNG_EXPECTS(batch.delete_src[i] >= 0 && batch.delete_dst[i] >= 0, who, ": batch ", index,
                 " has a negative vertex id");
  }
}

}  // namespace dyng::detail
