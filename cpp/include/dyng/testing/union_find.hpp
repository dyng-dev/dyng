// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/checker.cu (UnionFind)
/**
 * @file union_find.hpp
 * @brief dyng::testing::union_find: a deterministic disjoint-set forest whose representative is
 *        the smallest member of its set (the oracle of connected-component labellings).
 * @ingroup testing
 *
 * The union of two sets makes the smaller root the parent of the larger one, so the root of every
 * set is its minimum element whatever the order of the unions: the convention of
 * Shiloach-Vishkin's "hook the larger root to the smaller" (LabelPropagation-CUDA's components)
 * and of LabelPropagation-CUDA's own --check oracle. find() halves paths (every visited node is
 * pointed at its grandparent), a form of path compression; the answers do not depend on it.
 *
 * Header-only, standard library only: it shares no code with the algorithms it checks (PLAN 5.8;
 * ADR 0068).
 */
#pragma once

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <vector>

namespace dyng::testing {

/**
 * @brief Disjoint sets over the integers [0, size), the root of each set its smallest member.
 * @ingroup testing
 */
class union_find {
 public:
  /**
   * @brief `size` singleton sets {0}, {1}, ..., {size - 1}.
   * @param[in] size Number of elements (>= 0).
   * @throws std::invalid_argument if `size` is negative.
   */
  explicit union_find(std::int64_t size) {
    if (size < 0) {
      throw std::invalid_argument("union_find: negative size");
    }
    parent_.resize(static_cast<std::size_t>(size));
    std::iota(parent_.begin(), parent_.end(), std::int64_t{0});
  }

  /**
   * @brief Number of elements.
   * @return The size given at construction.
   */
  [[nodiscard]] std::int64_t size() const noexcept {
    return static_cast<std::int64_t>(parent_.size());
  }

  /**
   * @brief The root (smallest member) of the set of `x`; halves the path to it.
   * @param[in] x An element in [0, size()).
   * @return The root.
   * @throws std::out_of_range if `x` is out of range.
   */
  std::int64_t find(std::int64_t x) {
    check(x);
    auto i = static_cast<std::size_t>(x);
    while (parent_[i] != static_cast<std::int64_t>(i)) {
      parent_[i] = parent_[static_cast<std::size_t>(parent_[i])];
      i = static_cast<std::size_t>(parent_[i]);
    }
    return static_cast<std::int64_t>(i);
  }

  /**
   * @brief Merge the sets of `a` and `b`: the smaller root becomes the root of the union.
   * @param[in] a An element.
   * @param[in] b An element.
   * @return True if they were in different sets.
   * @throws std::out_of_range if an element is out of range.
   */
  bool unite(std::int64_t a, std::int64_t b) {
    a = find(a);
    b = find(b);
    if (a == b) {
      return false;
    }
    if (a < b) {
      parent_[static_cast<std::size_t>(b)] = a;
    } else {
      parent_[static_cast<std::size_t>(a)] = b;
    }
    return true;
  }

  /**
   * @brief Whether `a` and `b` are in the same set.
   * @param[in] a An element.
   * @param[in] b An element.
   * @return True if find(a) == find(b).
   */
  bool same(std::int64_t a, std::int64_t b) {
    return find(a) == find(b);
  }

 private:
  void check(std::int64_t x) const {
    if (x < 0 || x >= size()) {
      throw std::out_of_range("union_find: element out of range");
    }
  }
  std::vector<std::int64_t> parent_;
};

}  // namespace dyng::testing
