// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file apply_common.hpp
 * @brief Checks shared by the host batch applications (apply_host.cpp, apply_set_host.cpp).
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>

/// Throws dyng::invalid_argument_error with a message and file:line (no condition text).
#define DYNG_THROW_INVALID_ARGUMENT(...)                                                           \
  ::dyng::detail::throw_with_location<::dyng::invalid_argument_error>(__FILE__, __LINE__, nullptr, \
                                                                      __VA_ARGS__)

namespace dyng::detail {

/**
 * @brief Throws unless a non-empty view is readable on the host.
 * @tparam value_t Element type.
 * @param[in] view The view.
 * @param[in] what Its name, for the message.
 * @throws invalid_argument_error if the view is in device-only memory.
 */
template <typename value_t>
void expect_host(const array_view<value_t>& view, const char* what) {
  DYNG_EXPECTS(view.empty() || is_host_accessible(view.space()), what,
               " must be in host-accessible memory for a host graph");
}

/**
 * @brief Converts an edge count to edge_t, or throws capacity_error.
 * @tparam edge_t Edge offset type.
 * @param[in] count The count.
 * @return The count as edge_t.
 * @throws capacity_error if it does not fit.
 */
template <typename edge_t>
edge_t checked_edge_count(std::int64_t count) {
  if (count > static_cast<std::int64_t>(std::numeric_limits<edge_t>::max())) {
    throw capacity_error("dyng: " + std::to_string(count) +
                         " edges do not fit the edge offset type; use a graph with 64-bit "
                         "edge_t (int64)");
  }
  return static_cast<edge_t>(count);
}

/**
 * @brief A self-loop in a batch or an input: whether it is to be skipped.
 * @param[in] policy The graph's self-loop policy.
 * @param[in] what   "insertion", "deletion" or "edge", for the message.
 * @param[in] index  Position of the operation.
 * @param[in] u      The vertex.
 * @return true for self_loop::drop, false for self_loop::keep.
 * @throws invalid_argument_error for self_loop::error.
 */
inline bool skip_self_loop(batch_semantics::self_loop policy, const char* what, std::int64_t index,
                           std::int64_t u) {
  switch (policy) {
    case batch_semantics::self_loop::keep:
      return false;
    case batch_semantics::self_loop::drop:
      return true;
    case batch_semantics::self_loop::error:
      break;
  }
  DYNG_THROW_INVALID_ARGUMENT(what, " ", index, " is a self-loop (", u, ", ", u,
                              ") and the graph's batch_semantics::on_self_loop is error");
  return true;
}

/**
 * @brief The checks every host apply makes before Step 0: no vertex operations, consistent array
 *        sizes and weights, host-accessible arrays.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @param[in] batch          The batch.
 * @param[in] num_objectives The graph's number of weight columns.
 * @throws not_supported_error    for vertex insertions or deletions.
 * @throws invalid_argument_error for inconsistent sizes or device-only arrays.
 */
template <typename vertex_t, typename weight_t>
void validate_batch_shape(const edge_batch_view<vertex_t, weight_t>& batch, int num_objectives) {
  const std::size_t num_inserts = batch.insert_src.size();
  const std::size_t num_deletes = batch.delete_src.size();
  if (!batch.insert_vertices.empty() || !batch.delete_vertices.empty() ||
      !batch.insert_vertex_labels.empty()) {
    throw not_supported_error(
        "dyng: vertex insertions and deletions are not supported yet (planned for 0.3)");
  }
  DYNG_EXPECTS(batch.insert_dst.size() == num_inserts, "the batch has ", num_inserts,
               " insertion sources but ", batch.insert_dst.size(), " destinations");
  DYNG_EXPECTS(batch.delete_dst.size() == num_deletes, "the batch has ", num_deletes,
               " deletion sources but ", batch.delete_dst.size(), " destinations");
  if (num_inserts > 0) {
    DYNG_EXPECTS(batch.num_weights == num_objectives, "the batch has ", batch.num_weights,
                 " weights per insertion but the graph has ", num_objectives, " weight columns");
    DYNG_EXPECTS(
        batch.insert_weights.size() == num_inserts * static_cast<std::size_t>(num_objectives),
        "the batch has ", batch.insert_weights.size(), " insertion weights, expected ", num_inserts,
        " x ", num_objectives);
  }
  expect_host(batch.insert_src, "edge_batch_view::insert_src");
  expect_host(batch.insert_dst, "edge_batch_view::insert_dst");
  expect_host(batch.insert_weights, "edge_batch_view::insert_weights");
  expect_host(batch.delete_src, "edge_batch_view::delete_src");
  expect_host(batch.delete_dst, "edge_batch_view::delete_dst");
}

}  // namespace dyng::detail
