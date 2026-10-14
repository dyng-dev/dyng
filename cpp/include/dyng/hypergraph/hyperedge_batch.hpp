// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyperedge_batch.hpp
 * @brief hyperedge_batch_view<V,W> (non-owning) and hyperedge_batch<V,W> (owning builder): the
 *        changes applied to a hypergraph.
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
 * @brief A non-owning batch of hypergraph changes. An empty array means "none of this kind".
 *
 * Four kinds of operations: deletions of whole hyperedges (by id), insertions of whole
 * hyperedges (a CSR of member lists, one weight each for a weighted hypergraph), and the
 * incidence edits (a vertex removed from or added to a live hyperedge; planned for 0.4:
 * hypergraph::apply() raises not_supported_error when they are not empty). A "replace" is a
 * deletion plus an insertion. The order in which deletions and insertions reach the store is a
 * property of the hypergraph (hypergraph_properties::reuse).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (unweighted: no weights).
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
struct hyperedge_batch_view {
  array_view<const std::int64_t> delete_hyperedges;  ///< ids of hyperedges to delete
  array_view<const std::int64_t> insert_offsets;     ///< num_insertions() + 1 offsets from 0
  array_view<const vertex_t> insert_members;         ///< members of inserted hyperedge i are
                                                     ///< [insert_offsets[i], insert_offsets[i+1])
  array_view<const weight_t> insert_weights;         ///< one weight per inserted hyperedge
  array_view<const std::int64_t> delete_incidence_hyperedge;  ///< (h, v) removals: h (0.4)
  array_view<const vertex_t> delete_incidence_vertex;         ///< (h, v) removals: v (0.4)
  array_view<const std::int64_t> insert_incidence_hyperedge;  ///< (h, v) additions: h (0.4)
  array_view<const vertex_t> insert_incidence_vertex;         ///< (h, v) additions: v (0.4)

  /**
   * @brief Number of hyperedge insertions.
   * @return insert_offsets.size() - 1, or 0 for empty offsets.
   */
  [[nodiscard]] std::size_t num_insertions() const noexcept {
    return insert_offsets.empty() ? 0 : insert_offsets.size() - 1;
  }

  /**
   * @brief Number of hyperedge deletions.
   * @return delete_hyperedges.size().
   */
  [[nodiscard]] std::size_t num_deletions() const noexcept {
    return delete_hyperedges.size();
  }

  /**
   * @brief Whether the batch has no operation at all.
   * @return True if it has no deletion, insertion or incidence edit.
   */
  [[nodiscard]] bool empty() const noexcept {
    return num_insertions() == 0 && delete_hyperedges.empty() &&
           delete_incidence_hyperedge.empty() && insert_incidence_hyperedge.empty();
  }
};

/**
 * @brief An owning host batch of hypergraph changes with a builder interface.
 *
 * Operations of one kind keep the order in which they were added (the inserted ids are reported
 * in insertion order).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type (unweighted: no weights).
 * @ingroup hypergraph
 */
template <typename vertex_t, typename weight_t>
class hyperedge_batch {
 public:
  /**
   * @brief An empty batch.
   */
  hyperedge_batch() noexcept = default;

  /**
   * @brief Add the deletion of a hyperedge.
   * @param[in] id The hyperedge id (a dead or out-of-range id is skipped and counted by apply()).
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void delete_hyperedge(std::int64_t id) {
    try {
      delete_hyperedges_.push_back(id);
    } catch (const std::bad_alloc& e) {
      detail::throw_host_allocation_failure("hyperedge_batch::delete_hyperedge", e.what());
    } catch (const std::length_error& e) {
      detail::throw_host_allocation_failure("hyperedge_batch::delete_hyperedge", e.what());
    }
  }

  /**
   * @brief Add the insertion of a hyperedge.
   * @param[in] members Its members (host memory; repeated members are dropped by apply()).
   * @param[in] w       Its weight (ignored for unweighted).
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_hyperedge(array_view<const vertex_t> members, weight_t w = {}) {
    const std::size_t n_offsets = insert_offsets_.size();
    const std::size_t n_members = insert_members_.size();
    const std::size_t n_weights = insert_weights_.size();
    try {
      if (insert_offsets_.empty()) insert_offsets_.push_back(0);
      insert_members_.insert(insert_members_.end(), members.begin(), members.end());
      insert_offsets_.push_back(static_cast<std::int64_t>(insert_members_.size()));
      if constexpr (!is_unweighted_v<weight_t>) {
        insert_weights_.push_back(w);
      } else {
        (void)w;
      }
    } catch (const std::bad_alloc& e) {
      shrink_insertions(n_offsets, n_members, n_weights);
      detail::throw_host_allocation_failure("hyperedge_batch::insert_hyperedge", e.what());
    } catch (const std::length_error& e) {
      shrink_insertions(n_offsets, n_members, n_weights);
      detail::throw_host_allocation_failure("hyperedge_batch::insert_hyperedge", e.what());
    }
  }

  /**
   * @brief Add the insertion of a hyperedge.
   * @param[in] members Its members (repeated members are dropped by apply()).
   * @param[in] w       Its weight (ignored for unweighted).
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_hyperedge(std::initializer_list<vertex_t> members, weight_t w = {}) {
    insert_hyperedge(array_view<const vertex_t>(members.begin(), members.size()), w);
  }

  /**
   * @brief Add the removal of vertex `v` from the live hyperedge `h` (planned for 0.4: apply()
   *        raises not_supported_error for a batch that has one).
   * @param[in] h The hyperedge id.
   * @param[in] v The vertex.
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void delete_incidence(std::int64_t h, vertex_t v) {
    append_pair(delete_incidence_hyperedge_, delete_incidence_vertex_, h, v,
                "hyperedge_batch::delete_incidence");
  }

  /**
   * @brief Add vertex `v` to the live hyperedge `h` (planned for 0.4: apply() raises
   *        not_supported_error for a batch that has one).
   * @param[in] h The hyperedge id.
   * @param[in] v The vertex.
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_incidence(std::int64_t h, vertex_t v) {
    append_pair(insert_incidence_hyperedge_, insert_incidence_vertex_, h, v,
                "hyperedge_batch::insert_incidence");
  }

  /**
   * @brief Reserve capacity.
   * @param[in] insertions Expected number of inserted hyperedges.
   * @param[in] members    Expected number of their members in total.
   * @param[in] deletions  Expected number of deleted hyperedges.
   * @throws out_of_memory_error if the capacity cannot be allocated.
   * @guarantee Strong for the contents (the operations are unchanged; some arrays may keep a
   *            larger capacity).
   */
  void reserve(std::size_t insertions, std::size_t members, std::size_t deletions) {
    try {
      insert_offsets_.reserve(insertions + 1);
      insert_members_.reserve(members);
      if constexpr (!is_unweighted_v<weight_t>) insert_weights_.reserve(insertions);
      delete_hyperedges_.reserve(deletions);
    } catch (const std::bad_alloc& e) {
      detail::throw_host_allocation_failure("hyperedge_batch::reserve", e.what());
    } catch (const std::length_error& e) {
      detail::throw_host_allocation_failure("hyperedge_batch::reserve", e.what());
    }
  }

  /**
   * @brief Remove every operation (keeps the capacity).
   * @guarantee No-throw.
   */
  void clear() noexcept {
    delete_hyperedges_.clear();
    insert_offsets_.clear();
    insert_members_.clear();
    insert_weights_.clear();
    delete_incidence_hyperedge_.clear();
    delete_incidence_vertex_.clear();
    insert_incidence_hyperedge_.clear();
    insert_incidence_vertex_.clear();
  }

  /**
   * @brief Number of hyperedge insertions.
   * @return The insertion count.
   */
  [[nodiscard]] std::size_t num_insertions() const noexcept {
    return insert_offsets_.empty() ? 0 : insert_offsets_.size() - 1;
  }

  /**
   * @brief Number of hyperedge deletions.
   * @return The deletion count.
   */
  [[nodiscard]] std::size_t num_deletions() const noexcept {
    return delete_hyperedges_.size();
  }

  /**
   * @brief Whether the batch is empty.
   * @return True if it has no operation.
   */
  [[nodiscard]] bool empty() const noexcept {
    return num_insertions() == 0 && delete_hyperedges_.empty() &&
           delete_incidence_hyperedge_.empty() && insert_incidence_hyperedge_.empty();
  }

  /**
   * @brief The ids of the deletions.
   * @return The array, in the order they were added.
   */
  [[nodiscard]] const std::vector<std::int64_t>& delete_hyperedges() const noexcept {
    return delete_hyperedges_;
  }

  /**
   * @brief The offsets of the inserted hyperedges' members.
   * @return num_insertions() + 1 offsets (empty without insertions).
   */
  [[nodiscard]] const std::vector<std::int64_t>& insert_offsets() const noexcept {
    return insert_offsets_;
  }

  /**
   * @brief The members of the inserted hyperedges, back to back.
   * @return The array.
   */
  [[nodiscard]] const std::vector<vertex_t>& insert_members() const noexcept {
    return insert_members_;
  }

  /**
   * @brief The weights of the inserted hyperedges.
   * @return One per insertion (empty for unweighted).
   */
  [[nodiscard]] const std::vector<weight_t>& insert_weights() const noexcept {
    return insert_weights_;
  }

  /**
   * @brief The hyperedges of the incidence removals (planned for 0.4).
   * @return One per removal, in the order they were added.
   */
  [[nodiscard]] const std::vector<std::int64_t>& delete_incidence_hyperedge() const noexcept {
    return delete_incidence_hyperedge_;
  }

  /**
   * @brief The vertices of the incidence removals (planned for 0.4).
   * @return One per removal, in the order they were added.
   */
  [[nodiscard]] const std::vector<vertex_t>& delete_incidence_vertex() const noexcept {
    return delete_incidence_vertex_;
  }

  /**
   * @brief The hyperedges of the incidence additions (planned for 0.4).
   * @return One per addition, in the order they were added.
   */
  [[nodiscard]] const std::vector<std::int64_t>& insert_incidence_hyperedge() const noexcept {
    return insert_incidence_hyperedge_;
  }

  /**
   * @brief The vertices of the incidence additions (planned for 0.4).
   * @return One per addition, in the order they were added.
   */
  [[nodiscard]] const std::vector<vertex_t>& insert_incidence_vertex() const noexcept {
    return insert_incidence_vertex_;
  }

  /**
   * @brief A read-only view of the batch.
   * @return A host hyperedge_batch_view; valid while this batch is alive and unmodified.
   */
  [[nodiscard]] hyperedge_batch_view<vertex_t, weight_t> view() const noexcept {
    hyperedge_batch_view<vertex_t, weight_t> out;
    out.delete_hyperedges = host_view(delete_hyperedges_);
    out.insert_offsets = host_view(insert_offsets_);
    out.insert_members = host_view(insert_members_);
    out.insert_weights = host_view(insert_weights_);
    out.delete_incidence_hyperedge = host_view(delete_incidence_hyperedge_);
    out.delete_incidence_vertex = host_view(delete_incidence_vertex_);
    out.insert_incidence_hyperedge = host_view(insert_incidence_hyperedge_);
    out.insert_incidence_vertex = host_view(insert_incidence_vertex_);
    return out;
  }

 private:
  /// Undo a partial insertion (no allocation).
  void shrink_insertions(std::size_t n_offsets, std::size_t n_members,
                         std::size_t n_weights) noexcept {
    insert_offsets_.erase(insert_offsets_.begin() + static_cast<std::ptrdiff_t>(n_offsets),
                          insert_offsets_.end());
    insert_members_.erase(insert_members_.begin() + static_cast<std::ptrdiff_t>(n_members),
                          insert_members_.end());
    insert_weights_.erase(insert_weights_.begin() + static_cast<std::ptrdiff_t>(n_weights),
                          insert_weights_.end());
  }

  /// Append one (h, v) pair to two parallel arrays with the strong guarantee.
  static void append_pair(std::vector<std::int64_t>& hs, std::vector<vertex_t>& vs, std::int64_t h,
                          vertex_t v, const char* what) {
    const std::size_t n = hs.size();
    try {
      hs.push_back(h);
      vs.push_back(v);
    } catch (const std::bad_alloc& e) {
      hs.erase(hs.begin() + static_cast<std::ptrdiff_t>(n), hs.end());
      vs.erase(vs.begin() + static_cast<std::ptrdiff_t>(n), vs.end());
      detail::throw_host_allocation_failure(what, e.what());
    } catch (const std::length_error& e) {
      hs.erase(hs.begin() + static_cast<std::ptrdiff_t>(n), hs.end());
      vs.erase(vs.begin() + static_cast<std::ptrdiff_t>(n), vs.end());
      detail::throw_host_allocation_failure(what, e.what());
    }
  }

  std::vector<std::int64_t> delete_hyperedges_;
  std::vector<std::int64_t> insert_offsets_;
  std::vector<vertex_t> insert_members_;
  std::vector<weight_t> insert_weights_;
  std::vector<std::int64_t> delete_incidence_hyperedge_;
  std::vector<vertex_t> delete_incidence_vertex_;
  std::vector<std::int64_t> insert_incidence_hyperedge_;
  std::vector<vertex_t> insert_incidence_vertex_;
};

}  // namespace dyng
