// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file frontier.hpp
 * @brief The frontier types of the framework that are in use (PLAN Section 4.5.3).
 *
 * An enactor owns two frontiers of the problem's `frontier_type` for the length of one run (the
 * input and the output of a loop round) and passes them to the hooks. A frontier type must be
 * default-constructible without allocating (invariant I9): the storage of a frontier belongs in
 * the problem's pooled workspace (ADR 0015), which a frontier object refers to.
 *
 * In 0.1 both algorithms keep their frontiers inside their engines' workspaces and iterate there:
 * sssp's affected lists, near-far piles and invalidation lists (sssp_workspace), cycle_count's
 * change lists with their ownership index (cycle_count_workspace). Their frontier type is
 * internal_frontier. The plan's sparse, dense, bucketed and binned frontiers and the work items
 * enter this header with their second user (rule of two); a problem may meanwhile define its own
 * frontier type (any type with `bool empty() const`).
 */
#pragma once

#include <type_traits>
#include <utility>

namespace dyng::detail::framework {

/**
 * @brief The frontier of a problem that keeps its frontiers in its workspace and runs Step 2 to
 *        its fixed point inside one call of the loop hook.
 *
 * With this frontier type the default is_converged hook reports convergence once the loop hook
 * has run (one call), and the enactor's iteration cap counts calls, not rounds.
 */
struct internal_frontier {};

/**
 * @brief Whether a frontier type answers `empty()` (the default convergence test).
 * @tparam frontier_t The frontier type.
 */
template <typename frontier_t, typename = void>
struct has_empty : std::false_type {};

/// @brief A frontier type with `bool empty() const`.
template <typename frontier_t>
struct has_empty<frontier_t, std::void_t<decltype(std::declval<const frontier_t&>().empty())>>
    : std::true_type {};

}  // namespace dyng::detail::framework
