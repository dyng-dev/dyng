// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_impl.hpp
 * @brief The private state of hypergraph<V,W>, its internal applied record and the internal
 *        access the algorithms use (ADR 0041).
 *
 * Algorithms (triad_count in M9, hyper_sssp in M11) reach the storage through
 * detail::hypergraph_access: the store (its row views), the applied record of the last batch
 * (inserted ids on the host and the device, the deleted ids with their members, the touched
 * hyperedges and vertices, the incidence-delta hook) and the state identity.
 */
#pragma once

#include "hypergraph/store.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>
#include <memory>
#include <vector>

namespace dyng::detail {

/**
 * @brief A process-wide unique identifier for a container state (never 0); shared with graph
 *        (graph/graph.cpp).
 * @return A value never returned before in this process.
 */
std::uint64_t next_graph_state_id() noexcept;

/**
 * @brief The incidence changes of one batch, (vertex, hyperedge id, sign) per incidence: the
 *        hook of M11's line-graph delta (`incidence_delta`, filled only when asked for).
 */
struct hyper_incidence_delta {
  std::vector<std::int32_t> vertices;  ///< the vertex of every change
  std::vector<std::int64_t> ids;       ///< the hyperedge of every change
  std::vector<std::int8_t> signs;      ///< -1: removed, +1: added
};

/**
 * @brief What the last applied batch did, in the terms the algorithms read (the internal applied
 *        record; ADR 0041, section 10).
 */
struct hyper_applied {
  std::uint64_t version_before = 0;              ///< the hypergraph version before the batch
  std::uint64_t state_before = 0;                ///< the state identity before the batch
  std::uint64_t version_after = 0;               ///< the version after the batch
  std::uint64_t state_after = 0;                 ///< the state identity after the batch
  std::vector<std::int64_t> inserted_ids;        ///< the id of every inserted hyperedge
  buffer<std::int64_t> inserted_ids_device;      ///< the same in device memory (CUDA)
  std::vector<std::int64_t> deleted_ids;         ///< the live ids deleted, in batch order
  hyper::member_rows deleted_members;            ///< their members (row i: deleted_ids[i])
  std::vector<std::int64_t> touched_hyperedges;  ///< deleted and inserted ids, sorted, unique
  std::vector<std::int32_t> touched_vertices;    ///< their members, sorted, unique
  std::int64_t ignored_deletions = 0;            ///< skipped deletions
  std::int64_t dropped_members = 0;              ///< dropped repeated members
  bool has_incidence_delta = false;              ///< whether incidence_delta was filled
  hyper_incidence_delta incidence_delta;         ///< the M11 hook
};

/**
 * @brief The state behind hypergraph<V,W>: properties, version, state identity, placement, the
 *        store, the id-indexed weights and the applied record.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
class hypergraph_impl {
 public:
  hypergraph_properties props;                     ///< the properties
  std::uint64_t version = 0;                       ///< +1 per applied batch
  std::uint64_t state_id = next_graph_state_id();  ///< unique per state; copied by clone()
  backend home = backend::sequential;              ///< backend of the resources that built it
  int home_device = -1;                            ///< CUDA device of a cuda hypergraph
  bool poisoned = false;                           ///< a store operation failed
  std::unique_ptr<hyper::incidence_store> store;   ///< the storage
  std::vector<weight_t> weights;                   ///< by id (weighted: 0 for a dead id)
  hyper_applied applied;                           ///< the last applied batch
};

/**
 * @brief Internal access to a hypergraph's state (for the algorithms and the tests).
 */
struct hypergraph_access {
  /**
   * @brief The state of a hypergraph.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in] h The hypergraph.
   * @return Its implementation object.
   */
  template <typename vertex_t, typename weight_t>
  static const hypergraph_impl<vertex_t, weight_t>& impl(const hypergraph<vertex_t, weight_t>& h) {
    return h.impl();
  }

  /**
   * @brief Apply a batch (hypergraph::apply), optionally recording the incidence delta.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in]     res   Execution resources.
   * @param[in,out] h     The hypergraph.
   * @param[in]     batch The batch.
   * @param[in]     record_incidence_delta Fill hyper_applied::incidence_delta.
   * @return What the batch did.
   */
  template <typename vertex_t, typename weight_t>
  static hyper_apply_summary apply(const resources& res, hypergraph<vertex_t, weight_t>& h,
                                   const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                   bool record_incidence_delta = false);

  /**
   * @brief The applied record of the last batch.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in] h The hypergraph.
   * @return The record (empty before the first batch).
   */
  template <typename vertex_t, typename weight_t>
  static const hyper_applied& applied(const hypergraph<vertex_t, weight_t>& h) {
    return h.impl().applied;
  }

  /**
   * @brief The state identity.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in] h The hypergraph.
   * @return The identity of the current state.
   */
  template <typename vertex_t, typename weight_t>
  static std::uint64_t state_id(const hypergraph<vertex_t, weight_t>& h) {
    return h.impl().state_id;
  }

  /**
   * @brief The store (its views and counters).
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in] h The hypergraph.
   * @return The store.
   */
  template <typename vertex_t, typename weight_t>
  static const hyper::incidence_store& store(const hypergraph<vertex_t, weight_t>& h) {
    return *h.impl().store;
  }

  /**
   * @brief Check that `res` may be used with `h` (PLAN 4.6 rule 5): a hypergraph built with CUDA
   *        resources needs CUDA resources of the same device, a host hypergraph host resources.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in] res  Execution resources.
   * @param[in] h    The hypergraph.
   * @param[in] what The calling function, for the message.
   * @throws invalid_argument_error on a mismatch (the message names h.to_backend(res)).
   */
  template <typename vertex_t, typename weight_t>
  static void expect_placement(const resources& res, const hypergraph<vertex_t, weight_t>& h,
                               const char* what);

  /**
   * @brief Poison a hypergraph (the tests' forced store failure).
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Weight type.
   * @param[in,out] h The hypergraph.
   */
  template <typename vertex_t, typename weight_t>
  static void poison(hypergraph<vertex_t, weight_t>& h) {
    h.impl().poisoned = true;
  }
};

}  // namespace dyng::detail
