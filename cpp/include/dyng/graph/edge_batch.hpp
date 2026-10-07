// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file edge_batch.hpp
 * @brief edge_batch_view<V,W> (non-owning) and edge_batch<V,W> (owning builder).
 * @ingroup batch
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

/**
 * @defgroup batch Batches
 * @brief Edge batches: the changes applied to a graph by graph::apply() and update().
 */

namespace dyng {

/**
 * @brief A non-owning batch of changes. An empty array means "none of this kind".
 *
 * How the batch is interpreted (upsert, deletions first, self-loops, ...) is a property of the
 * graph it is applied to (graph_properties::semantics), not of the batch.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @ingroup batch
 */
template <typename vertex_t, typename weight_t>
struct edge_batch_view {
  array_view<const vertex_t> insert_src;       ///< source of each insertion
  array_view<const vertex_t> insert_dst;       ///< destination of each insertion
  array_view<const weight_t> insert_weights;   ///< num_weights per insertion, insertion-major
  array_view<const vertex_t> delete_src;       ///< source of each deletion
  array_view<const vertex_t> delete_dst;       ///< destination of each deletion
  array_view<const vertex_t> insert_vertices;  ///< new vertices (0.4, DynLP)
  array_view<const std::int8_t> insert_vertex_labels;   ///< optional labels of new vertices
  array_view<const vertex_t> delete_vertices;           ///< removed vertices (0.4, DynLP)
  int num_weights = is_unweighted_v<weight_t> ? 0 : 1;  ///< weights per insertion (0: unweighted)

  /**
   * @brief Number of edge insertions.
   * @return insert_src.size().
   */
  [[nodiscard]] std::size_t num_insertions() const noexcept {
    return insert_src.size();
  }

  /**
   * @brief Number of edge deletions.
   * @return delete_src.size().
   */
  [[nodiscard]] std::size_t num_deletions() const noexcept {
    return delete_src.size();
  }

  /**
   * @brief Whether the batch has no operation at all.
   * @return True if every array is empty.
   */
  [[nodiscard]] bool empty() const noexcept {
    return insert_src.empty() && delete_src.empty() && insert_vertices.empty() &&
           delete_vertices.empty();
  }
};

/**
 * @brief An owning host batch with a builder interface.
 *
 * Operations keep the order in which they were added; the order matters (the last insertion of
 * a pair wins under upsert semantics).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 * @ingroup batch
 */
template <typename vertex_t, typename weight_t>
class edge_batch {
 public:
  /// The default number of weights per insertion: 0 for weight_t = unweighted, else 1.
  static constexpr int default_num_weights = is_unweighted_v<weight_t> ? 0 : 1;

  /**
   * @brief An empty batch.
   * @param[in] num_weights Weights per insertion (the number of objectives of the graph; 0 for
   *                        weight_t = unweighted).
   * @throws invalid_argument_error if `num_weights` is negative, or not 0 for an unweighted batch.
   */
  explicit edge_batch(int num_weights = default_num_weights) : num_weights_(num_weights) {
    DYNG_EXPECTS(num_weights >= 0, "edge_batch: num_weights must be >= 0, got ", num_weights);
    DYNG_EXPECTS(!is_unweighted_v<weight_t> || num_weights == 0,
                 "edge_batch: an unweighted batch has no weights, got num_weights = ", num_weights);
  }

  /**
   * @brief Add an edge insertion (or an upsert of an existing edge).
   * @param[in] u       Source.
   * @param[in] v       Destination.
   * @param[in] weights Its num_weights() weights.
   * @throws invalid_argument_error if `weights` does not hold num_weights() values.
   * @throws out_of_memory_error    if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_edge(vertex_t u, vertex_t v, std::initializer_list<weight_t> weights) {
    insert_edge(u, v, array_view<const weight_t>(weights.begin(), weights.size()));
  }

  /**
   * @brief Add an edge insertion without weights (a batch with num_weights() == 0, e.g. of an
   *        unweighted graph).
   * @param[in] u Source.
   * @param[in] v Destination.
   * @throws invalid_argument_error if num_weights() is not 0.
   * @throws out_of_memory_error    if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_edge(vertex_t u, vertex_t v) {
    insert_edge(u, v, array_view<const weight_t>());
  }

  /**
   * @brief Add an edge insertion (or an upsert of an existing edge).
   * @param[in] u       Source.
   * @param[in] v       Destination.
   * @param[in] weights Its num_weights() weights (host memory).
   * @throws invalid_argument_error if `weights` does not hold num_weights() values.
   * @throws out_of_memory_error    if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void insert_edge(vertex_t u, vertex_t v, array_view<const weight_t> weights) {
    DYNG_EXPECTS(weights.size() == static_cast<std::size_t>(num_weights_),
                 "edge_batch::insert_edge got ", weights.size(), " weights, expected ",
                 num_weights_);
    const std::size_t n = insert_src_.size();
    const std::size_t w = insert_weights_.size();
    try {
      insert_src_.push_back(u);
      insert_dst_.push_back(v);
      insert_weights_.insert(insert_weights_.end(), weights.begin(), weights.end());
    } catch (const std::bad_alloc& e) {
      shrink_insertions(n, w);
      detail::throw_host_allocation_failure("edge_batch::insert_edge", e.what());
    } catch (const std::length_error& e) {
      shrink_insertions(n, w);
      detail::throw_host_allocation_failure("edge_batch::insert_edge", e.what());
    }
  }

  /**
   * @brief Add an edge deletion.
   * @param[in] u Source.
   * @param[in] v Destination.
   * @throws out_of_memory_error if the arrays cannot grow.
   * @guarantee Strong: after an exception the batch is unchanged.
   */
  void delete_edge(vertex_t u, vertex_t v) {
    const std::size_t n = delete_src_.size();
    try {
      delete_src_.push_back(u);
      delete_dst_.push_back(v);
    } catch (const std::bad_alloc& e) {
      shrink_deletions(n);
      detail::throw_host_allocation_failure("edge_batch::delete_edge", e.what());
    } catch (const std::length_error& e) {
      shrink_deletions(n);
      detail::throw_host_allocation_failure("edge_batch::delete_edge", e.what());
    }
  }

  /**
   * @brief Reserve capacity.
   * @param[in] insertions Expected number of insertions.
   * @param[in] deletions  Expected number of deletions.
   * @throws out_of_memory_error if the capacity cannot be allocated.
   * @guarantee Strong for the contents (the operations are unchanged; some arrays may keep a
   *            larger capacity).
   */
  void reserve(std::size_t insertions, std::size_t deletions) {
    try {
      insert_src_.reserve(insertions);
      insert_dst_.reserve(insertions);
      insert_weights_.reserve(insertions * static_cast<std::size_t>(num_weights_));
      delete_src_.reserve(deletions);
      delete_dst_.reserve(deletions);
    } catch (const std::bad_alloc& e) {
      detail::throw_host_allocation_failure("edge_batch::reserve", e.what());
    } catch (const std::length_error& e) {
      detail::throw_host_allocation_failure("edge_batch::reserve", e.what());
    }
  }

  /**
   * @brief Remove every operation (keeps num_weights() and the capacity).
   * @guarantee No-throw.
   */
  void clear() noexcept {
    insert_src_.clear();
    insert_dst_.clear();
    insert_weights_.clear();
    delete_src_.clear();
    delete_dst_.clear();
  }

  /**
   * @brief Weights per insertion.
   * @return The value given at construction.
   */
  [[nodiscard]] int num_weights() const noexcept {
    return num_weights_;
  }

  /**
   * @brief Number of edge insertions.
   * @return The insertion count.
   */
  [[nodiscard]] std::size_t num_insertions() const noexcept {
    return insert_src_.size();
  }

  /**
   * @brief Number of edge deletions.
   * @return The deletion count.
   */
  [[nodiscard]] std::size_t num_deletions() const noexcept {
    return delete_src_.size();
  }

  /**
   * @brief Whether the batch is empty.
   * @return True if it has no operation.
   */
  [[nodiscard]] bool empty() const noexcept {
    return insert_src_.empty() && delete_src_.empty();
  }

  /**
   * @brief Sources of the insertions.
   * @return The array, in insertion order.
   */
  [[nodiscard]] const std::vector<vertex_t>& insert_src() const noexcept {
    return insert_src_;
  }

  /**
   * @brief Destinations of the insertions.
   * @return The array, in insertion order.
   */
  [[nodiscard]] const std::vector<vertex_t>& insert_dst() const noexcept {
    return insert_dst_;
  }

  /**
   * @brief Weights of the insertions, num_weights() per insertion.
   * @return The insertion-major array.
   */
  [[nodiscard]] const std::vector<weight_t>& insert_weights() const noexcept {
    return insert_weights_;
  }

  /**
   * @brief Sources of the deletions.
   * @return The array, in deletion order.
   */
  [[nodiscard]] const std::vector<vertex_t>& delete_src() const noexcept {
    return delete_src_;
  }

  /**
   * @brief Destinations of the deletions.
   * @return The array, in deletion order.
   */
  [[nodiscard]] const std::vector<vertex_t>& delete_dst() const noexcept {
    return delete_dst_;
  }

  /**
   * @brief A read-only view of the batch.
   * @return A host edge_batch_view; valid while this batch is alive and unmodified.
   */
  [[nodiscard]] edge_batch_view<vertex_t, weight_t> view() const noexcept {
    edge_batch_view<vertex_t, weight_t> out;
    out.insert_src = host_view(insert_src_);
    out.insert_dst = host_view(insert_dst_);
    out.insert_weights = host_view(insert_weights_);
    out.delete_src = host_view(delete_src_);
    out.delete_dst = host_view(delete_dst_);
    out.num_weights = num_weights_;
    return out;
  }

 private:
  /// Undo a partial insertion: the arrays back to `n` insertions and `w` weights (no allocation).
  void shrink_insertions(std::size_t n, std::size_t w) noexcept {
    insert_src_.erase(insert_src_.begin() + static_cast<std::ptrdiff_t>(n), insert_src_.end());
    insert_dst_.erase(insert_dst_.begin() + static_cast<std::ptrdiff_t>(n), insert_dst_.end());
    insert_weights_.erase(insert_weights_.begin() + static_cast<std::ptrdiff_t>(w),
                          insert_weights_.end());
  }

  /// Undo a partial deletion: the arrays back to `n` deletions (no allocation).
  void shrink_deletions(std::size_t n) noexcept {
    delete_src_.erase(delete_src_.begin() + static_cast<std::ptrdiff_t>(n), delete_src_.end());
    delete_dst_.erase(delete_dst_.begin() + static_cast<std::ptrdiff_t>(n), delete_dst_.end());
  }

  int num_weights_;
  std::vector<vertex_t> insert_src_;
  std::vector<vertex_t> insert_dst_;
  std::vector<weight_t> insert_weights_;
  std::vector<vertex_t> delete_src_;
  std::vector<vertex_t> delete_dst_;
};

}  // namespace dyng
