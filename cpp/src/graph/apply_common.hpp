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

/// Throws dyng::invalid_argument_error with a message and file:line (no condition text).
#define DYNG_THROW_INVALID_ARGUMENT(...)                                                           \
  ::dyng::detail::throw_with_location<::dyng::invalid_argument_error>(__FILE__, __LINE__, nullptr, \
                                                                      __VA_ARGS__)

namespace dyng::detail {

/**
 * @brief Throws unless the batch semantics of `props` can be applied: batch_semantics::as_sets
 *        needs deletions_first, an on_existing_insert other than upsert (a weight upsert is not a
 *        structural change), row_order::sorted and multi_edges::forbid. Called when a graph is
 *        constructed (so an unusable combination fails there, not at the first apply) and again
 *        by the set apply.
 * @param[in] props The graph properties.
 * @throws not_supported_error for an unsupported combination (the message names it and the fix).
 */
inline void expect_supported_semantics(const graph_properties& props) {
  const batch_semantics& semantics = props.semantics;
  if (!semantics.as_sets) {
    return;
  }
  if (!semantics.deletions_first) {
    throw not_supported_error(
        "dyng: batch_semantics::as_sets needs deletions_first (a set batch deletes, then inserts)");
  }
  if (semantics.on_existing_insert == batch_semantics::existing_insert::upsert) {
    throw not_supported_error(
        "dyng: batch_semantics::as_sets supports on_existing_insert ignore or error; an upsert of "
        "the weights of an existing edge is not a structural change (delete and re-insert the "
        "edge to change its weights)");
  }
  if (props.order != row_order::sorted || props.parallel_edges != multi_edges::forbid) {
    throw not_supported_error(
        "dyng: batch_semantics::as_sets needs row_order::sorted and multi_edges::forbid (the "
        "changes are merged into sorted rows of a simple graph)");
  }
}

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
        "dyng: vertex insertions and deletions are not supported yet (planned for 0.4)");
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
