// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file types.hpp
 * @brief Scalar conventions shared by every module: sentinels, weight tags, engine and determinism.
 * @ingroup core
 *
 * PLAN Section 4.4.2: public ids are 0-based; "none" is invalid_id<T>() == -1 for signed ids;
 * shortest-path distances use infinite_distance<T>() == max / 4 (the MOSP DISTANCE_INF, kept for
 * byte parity with the original tools).
 */
#pragma once

#include <cstdint>
#include <limits>
#include <type_traits>

namespace dyng {

/**
 * @brief Weight tag for graphs without edge weights (e.g. cycle_count).
 *
 * A graph<V, E, unweighted> has no weight columns (num_weights() == 0), and its batches carry no
 * insertion weights. All values of the tag are equal, so containers of it compare as usual.
 * @ingroup core
 */
struct unweighted {
  /**
   * @brief Equality: every value equals every other.
   * @return true.
   */
  friend constexpr bool operator==(unweighted /*a*/, unweighted /*b*/) noexcept {
    return true;
  }
  /**
   * @brief Inequality: no value differs from another.
   * @return false.
   */
  friend constexpr bool operator!=(unweighted /*a*/, unweighted /*b*/) noexcept {
    return false;
  }
  /**
   * @brief Order: no value is smaller than another.
   * @return false.
   */
  friend constexpr bool operator<(unweighted /*a*/, unweighted /*b*/) noexcept {
    return false;
  }
  /**
   * @brief Order: no value is larger than another.
   * @return false.
   */
  friend constexpr bool operator>(unweighted /*a*/, unweighted /*b*/) noexcept {
    return false;
  }
};

/**
 * @brief Whether a weight type is the unweighted tag.
 * @tparam weight_t A weight type.
 * @ingroup core
 */
template <typename weight_t>
inline constexpr bool is_unweighted_v = std::is_same_v<weight_t, unweighted>;

/**
 * @brief The "no id" sentinel of a signed id type (vertex, edge, hyperedge or parent).
 * @tparam id_t A signed integer type.
 * @return -1 converted to `id_t`.
 * @ingroup core
 */
template <typename id_t>
constexpr id_t invalid_id() noexcept {
  static_assert(std::is_integral_v<id_t> && std::is_signed_v<id_t>,
                "invalid_id<T>() is defined for signed integer id types");
  return static_cast<id_t>(-1);
}

/**
 * @brief The "unreachable" distance of an integer distance type.
 *
 * Equals `std::numeric_limits<distance_t>::max() / 4`, so that adding any valid edge weight to it
 * cannot overflow. Writers print values >= infinite_distance<distance_t>() / 2 as `INF`.
 *
 * @tparam distance_t A signed integer distance type (std::int64_t in the public API).
 * @return max() / 4 of `distance_t`.
 * @ingroup core
 */
template <typename distance_t>
constexpr distance_t infinite_distance() noexcept {
  static_assert(std::is_integral_v<distance_t> && std::is_signed_v<distance_t>,
                "infinite_distance<T>() is defined for signed integer distance types");
  return std::numeric_limits<distance_t>::max() / 4;
}

/**
 * @brief Which CUDA update engine an algorithm uses (PLAN Section 4.5.4).
 * @ingroup core
 */
enum class engine : std::uint8_t {
  automatic,  ///< the library chooses (fused when the device supports it)
  fused,      ///< hand-fused paper kernels (e.g. a persistent cooperative kernel)
  operators,  ///< the multi-kernel framework operators
};

/**
 * @brief What two runs of an algorithm are guaranteed to agree on (PLAN Section 5.1).
 * @ingroup core
 */
enum class determinism : std::uint8_t {
  bitwise,      ///< identical bits across runs and backends (canonical tie-breaking)
  exact_value,  ///< identical values (e.g. counts); internal order may differ
  tolerance,    ///< equal within a documented tolerance
};

}  // namespace dyng
