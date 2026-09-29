// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file policies.hpp
 * @brief The policies the enactors apply (PLAN Section 4.5.3): the convergence rule of a
 *        fixed-point Step 2 with its iteration cap and on_limit policy (invariant I3), the sign
 *        of an aggregate-delta count and the ownership rules of exactly-once counting (I2).
 *
 * Only the policies that sssp and cycle_count use, or that the enactors themselves apply, live
 * here (rule of two). The plan's schedule (push / pull), sync_rule and tie_break policies arrive
 * with their first two users (the operators engine of sssp and label_propagation); the engine
 * choice is public (dyng::engine in <dyng/core/types.hpp>).
 */
#pragma once

#include <type_traits>

namespace dyng::detail::framework {

/**
 * @brief What an enactor does when a fixed-point Step 2 reaches its iteration cap.
 */
enum class on_limit {
  error,               ///< throw convergence_error naming the algorithm and the cap
  report,              ///< stop, finalize the partial result and report stats.converged = false
  fallback_recompute,  ///< discard Step 2 and recompute from scratch (the problem's recompute hook)
};

/**
 * @brief The convergence rule of a fixed-point Step 2 (invariant I3: a problem whose invalidating
 *        changes are not routed through an invalidation step sets max_iterations and at_limit).
 *
 * The rule itself is evaluated by the problem's is_converged hook (the default: the frontier is
 * empty); the enactor applies the cap. The default, no cap, is what sssp uses: its distances only
 * decrease, so its loop terminates without one.
 */
struct convergence {
  /// How convergence is decided (read by the problem's is_converged hook).
  enum class rule {
    empty_frontier,  ///< no element changed in the last round
    tolerance,       ///< the largest change of the last round is at most `tolerance`
  };
  rule kind = rule::empty_frontier;     ///< the rule
  double tolerance = 0.0;               ///< the bound of rule::tolerance
  int max_iterations = -1;              ///< calls of the loop hook allowed; -1: no cap
  on_limit at_limit = on_limit::error;  ///< what happens at the cap
};

/**
 * @brief The sign of one aggregate-delta count: P_{t+1} = P_t - count(G_t, deletions) +
 *        count(G_{t+1}, insertions).
 */
enum class sign : int {
  minus = -1,  ///< the structures through the deleted elements, counted on G_t (before the commit)
  plus = +1,   ///< the structures through the inserted elements, counted on G_{t+1}
};

/**
 * @brief Ownership rules for exactly-once counting (invariant I2). An aggregate-delta problem
 *        names its rule as `ownership_type`; the enactor passes it to every count, so there is no
 *        default rule to forget.
 */
namespace ownership {

/**
 * @brief A structure that contains several changed elements of one phase is counted only by the
 *        changed element with the smallest id (the id of a change is its position in the
 *        normalized change list). cycle_count's rule: a cycle is attributed to its smallest-id
 *        changed edge (changed_edge_index, cpp/src/algorithms/cycle_count/problem.hpp).
 */
struct min_member {};

}  // namespace ownership

/**
 * @brief Whether a type is one of the ownership rules above.
 * @tparam policy_t The type.
 */
template <typename policy_t>
struct is_ownership_policy : std::false_type {};

/// @brief ownership::min_member is an ownership rule.
template <>
struct is_ownership_policy<ownership::min_member> : std::true_type {};

/**
 * @brief is_ownership_policy<policy_t>::value.
 * @tparam policy_t The type.
 */
template <typename policy_t>
inline constexpr bool is_ownership_policy_v = is_ownership_policy<policy_t>::value;

}  // namespace dyng::detail::framework
