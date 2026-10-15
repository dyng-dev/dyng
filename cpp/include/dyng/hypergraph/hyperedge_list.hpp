// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyperedge_list.hpp
 * @brief hyperedge_list<V,W> (an owning CSR of member lists) and hyperedge_list_view<V,W>.
 * @ingroup hypergraph
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <new>
#include <stdexcept>
#include <vector>

namespace dyng {

/**
 * @brief A non-owning list of hyperedges: the members of hyperedge i are
 *        `members[offsets[i] .. offsets[i + 1])`.
 *
 * The input of hypergraph::from_hyperedges(); the arrays may live in any memory space. Hyperedge
 * i becomes the hyperedge with id i.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (unweighted: no weights).
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t = unweighted>
struct hyperedge_list_view {
  vertex_t num_vertices = 0;               ///< vertex ids lie in [0, num_vertices)
  array_view<const std::int64_t> offsets;  ///< num_hyperedges() + 1 offsets from 0 (or empty)
  array_view<const vertex_t> members;      ///< the members of every hyperedge, back to back
  array_view<const weight_t> weights;      ///< one weight per hyperedge; empty for unweighted

  /**
   * @brief Number of hyperedges.
   * @return offsets.size() - 1, or 0 for empty offsets.
   */
  [[nodiscard]] std::int64_t num_hyperedges() const noexcept {
    return offsets.empty() ? 0 : static_cast<std::int64_t>(offsets.size()) - 1;
  }
};

/**
 * @brief An owning host list of hyperedges (a CSR of member lists), for building hypergraphs and
 *        for hypergraph::to_hyperedge_list().
 *
 * A plain aggregate: the fields may be filled directly. hypergraph::from_hyperedges() validates
 * it.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (unweighted: `weights` stays empty).
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t = unweighted>
struct hyperedge_list {
  vertex_t num_vertices = 0;             ///< vertex ids lie in [0, num_vertices)
  std::vector<std::int64_t> offsets{0};  ///< num_hyperedges() + 1 offsets, offsets[0] == 0
  std::vector<vertex_t> members;         ///< the members of every hyperedge, back to back
  std::vector<weight_t> weights;         ///< one weight per hyperedge; empty for unweighted

  /**
   * @brief Number of hyperedges.
   * @return offsets.size() - 1, or 0 for empty offsets.
   */
  [[nodiscard]] std::int64_t num_hyperedges() const noexcept {
    return offsets.empty() ? 0 : static_cast<std::int64_t>(offsets.size()) - 1;
  }

  /**
   * @brief Append one hyperedge.
   * @param[in] hyperedge_members Its members (host memory).
   * @param[in] w                 Its weight (ignored for unweighted).
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the list is unchanged.
   */
  void add_hyperedge(array_view<const vertex_t> hyperedge_members, weight_t w = {}) {
    const std::size_t n_offsets = offsets.size();
    const std::size_t n_members = members.size();
    const std::size_t n_weights = weights.size();
    const auto undo = [&]() noexcept {
      offsets.erase(offsets.begin() + static_cast<std::ptrdiff_t>(n_offsets), offsets.end());
      members.erase(members.begin() + static_cast<std::ptrdiff_t>(n_members), members.end());
      weights.erase(weights.begin() + static_cast<std::ptrdiff_t>(n_weights), weights.end());
    };
    try {
      if (offsets.empty()) offsets.push_back(0);
      members.insert(members.end(), hyperedge_members.begin(), hyperedge_members.end());
      offsets.push_back(static_cast<std::int64_t>(members.size()));
      if constexpr (!is_unweighted_v<weight_t>) {
        weights.push_back(w);
      } else {
        (void)w;
      }
    } catch (const std::bad_alloc& e) {
      undo();
      detail::throw_host_allocation_failure("hyperedge_list::add_hyperedge", e.what());
    } catch (const std::length_error& e) {
      undo();
      detail::throw_host_allocation_failure("hyperedge_list::add_hyperedge", e.what());
    }
  }

  /**
   * @brief Append one hyperedge.
   * @param[in] hyperedge_members Its members.
   * @param[in] w                 Its weight (ignored for unweighted).
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the list is unchanged.
   */
  void add_hyperedge(std::initializer_list<vertex_t> hyperedge_members, weight_t w = {}) {
    add_hyperedge(array_view<const vertex_t>(hyperedge_members.begin(), hyperedge_members.size()),
                  w);
  }

  /**
   * @brief A read-only view of the arrays.
   * @return A hyperedge_list_view in host memory; valid while this object is alive and
   *         unmodified.
   */
  [[nodiscard]] hyperedge_list_view<vertex_t, weight_t> view() const noexcept {
    return hyperedge_list_view<vertex_t, weight_t>{num_vertices, host_view(offsets),
                                                   host_view(members), host_view(weights)};
  }
};

}  // namespace dyng
