// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file conformance.hpp
 * @brief Compile-time checks of a problem (PLAN Section 8.2): the declarations every problem
 *        needs and the hooks its family requires, as static_asserts with plain-English messages
 *        that point to docs/developer/framework.md.
 *
 * The enactors instantiate these checks, so a problem that runs through the framework cannot
 * miss a required hook (a fixed-point problem without a loop, an aggregate-delta problem without
 * an ownership rule: invariant I2). The run-time conformance kit (C1-C12, test_traits) lives in
 * cpp/tests/support.
 */
#pragma once

#include "framework/context.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"

#include <dyng/core/stats.hpp>

#include <string_view>
#include <type_traits>
#include <utility>

namespace dyng::detail::framework {

/// Detection of the declarations and hooks of a problem (the traits behind the checks).
namespace hook_detail {

template <typename problem_t, typename = void>
struct has_family : std::false_type {};
template <typename problem_t>
struct has_family<problem_t, std::void_t<decltype(problem_t::problem_family)>> : std::true_type {};

template <typename problem_t, typename = void>
struct has_name : std::false_type {};
template <typename problem_t>
struct has_name<
    problem_t, std::enable_if_t<std::is_convertible_v<decltype(problem_t::name), std::string_view>>>
    : std::true_type {};

template <typename problem_t, typename = void>
struct has_container : std::false_type {};
template <typename problem_t>
struct has_container<problem_t, std::void_t<typename problem_t::container_type>> : std::true_type {
};

template <typename problem_t, typename = void>
struct has_stats : std::false_type {};
template <typename problem_t>
struct has_stats<problem_t, std::void_t<typename problem_t::stats_type>> : std::true_type {};

template <typename problem_t, typename = void>
struct has_ownership : std::false_type {};
template <typename problem_t>
struct has_ownership<problem_t, std::void_t<typename problem_t::ownership_type>> : std::true_type {
};

template <typename problem_t, typename = void>
struct has_loop : std::false_type {};
template <typename problem_t>
struct has_loop<problem_t, std::void_t<decltype(std::declval<problem_t&>().loop(
                               std::declval<context&>(),
                               std::declval<new_view<typename problem_t::container_type>>(),
                               std::declval<typename problem_t::frontier_type&>(),
                               std::declval<typename problem_t::frontier_type&>()))>>
    : std::true_type {};

template <typename problem_t, typename view_t, typename = void>
struct has_count : std::false_type {};
template <typename problem_t, typename view_t>
struct has_count<problem_t, view_t,
                 std::void_t<decltype(std::declval<problem_t&>().count(
                     std::declval<context&>(), std::declval<view_t>(),
                     std::declval<typename problem_t::frontier_type&>(), sign::plus,
                     std::declval<typename problem_t::ownership_type>()))>> : std::true_type {};

template <typename problem_t, typename applied_t, typename = void>
struct has_enact_fused : std::false_type {};
template <typename problem_t, typename applied_t>
struct has_enact_fused<
    problem_t, applied_t,
    std::void_t<decltype(std::declval<problem_t&>().enact_fused(
        std::declval<context&>(), std::declval<new_view<typename problem_t::container_type>>(),
        std::declval<const applied_t&>(), std::declval<typename problem_t::stats_type&>()))>>
    : std::true_type {};

template <typename problem_t, typename = void>
struct has_compute_fused : std::false_type {};
template <typename problem_t>
struct has_compute_fused<
    problem_t,
    std::void_t<decltype(std::declval<problem_t&>().compute_fused(
        std::declval<context&>(), std::declval<new_view<typename problem_t::container_type>>(),
        std::declval<typename problem_t::stats_type&>()))>> : std::true_type {};

template <typename problem_t, typename = void>
struct has_recompute : std::false_type {};
template <typename problem_t>
struct has_recompute<problem_t, std::void_t<decltype(std::declval<problem_t&>().recompute(
                                    std::declval<context&>(),
                                    std::declval<new_view<typename problem_t::container_type>>(),
                                    std::declval<typename problem_t::stats_type&>()))>>
    : std::true_type {};

template <typename problem_t, typename = void>
struct has_target : std::false_type {};
template <typename problem_t>
struct has_target<problem_t, std::enable_if_t<std::is_convertible_v<
                                 decltype(std::declval<const problem_t&>().target()), const void*>>>
    : std::true_type {};

}  // namespace hook_detail

/**
 * @brief The declarations every problem needs (checked when an enactor is instantiated).
 * @tparam problem_t The problem.
 * @return true (the checks are static_asserts).
 */
template <typename problem_t>
constexpr bool check_problem() {
  static_assert(hook_detail::has_family<problem_t>::value,
                "dyng framework: a problem must derive from "
                "framework::problem_base<the_problem, family::fixed_point or "
                "family::aggregate_delta> (docs/developer/framework.md, 'Writing a problem')");
  if constexpr (hook_detail::has_family<problem_t>::value) {
    static_assert(
        std::is_base_of_v<problem_base<problem_t, problem_t::problem_family>, problem_t>,
        "dyng framework: a problem must derive from problem_base<itself, family> (CRTP: the first "
        "template argument is the problem's own type)");
  }
  static_assert(hook_detail::has_name<problem_t>::value,
                "dyng framework: a problem must declare `static constexpr std::string_view name "
                "= \"<algo>\";` (the prefix of its profiler stages '<algo>.<hook>' and messages)");
  static_assert(hook_detail::has_container<problem_t>::value,
                "dyng framework: a problem must declare `using container_type = ...;` (the graph "
                "or hypergraph type its old_view / new_view show)");
  static_assert(hook_detail::has_stats<problem_t>::value,
                "dyng framework: a problem must declare `using stats_type = <algo>::stats;` (the "
                "stats its update() returns)");
  if constexpr (hook_detail::has_stats<problem_t>::value) {
    static_assert(std::is_base_of_v<update_stats, typename problem_t::stats_type>,
                  "dyng framework: a problem's stats_type must derive from dyng::update_stats "
                  "(PLAN Section 5.1: every algorithm's stats share the common counters)");
  }
  if constexpr (hook_detail::has_family<problem_t>::value) {
    static_assert(std::is_nothrow_default_constructible_v<typename problem_t::frontier_type>,
                  "dyng framework: a frontier type must be default-constructible without "
                  "allocating or throwing (its storage belongs in the pooled workspace; "
                  "invariant I9). Use internal_frontier if the frontiers live in the problem's "
                  "workspace");
    if constexpr (problem_t::problem_family == family::aggregate_delta) {
      static_assert(hook_detail::has_ownership<problem_t>::value,
                    "dyng framework: an aggregate_delta problem must declare `using "
                    "ownership_type = ownership::min_member;` (or another ownership rule): "
                    "exactly-once counting needs an explicit rule and there is no default "
                    "(invariant I2)");
      if constexpr (hook_detail::has_ownership<problem_t>::value) {
        static_assert(is_ownership_policy_v<typename problem_t::ownership_type>,
                      "dyng framework: ownership_type must be one of the rules in "
                      "framework/policies.hpp (namespace ownership)");
        static_assert(
            hook_detail::has_count<problem_t, old_view<typename problem_t::container_type>>::value,
            "dyng framework: an aggregate_delta problem must provide `count(context&, "
            "old_view<container_type>, frontier_type&, sign, ownership_type)`: the structures "
            "through the deleted elements are subtracted on G_t, before the commit "
            "(invariant I1)");
      }
    }
  }
  return true;
}

/**
 * @brief The hooks the family of a problem requires for update() (checked by the update enactor,
 *        which knows the type of the applied batch).
 * @tparam problem_t The problem.
 * @tparam applied_t What the commit reports (framework::applied_batch for graphs).
 * @return true (the checks are static_asserts).
 */
template <typename problem_t, typename applied_t>
constexpr bool check_update_hooks() {
  constexpr bool fused = hook_detail::has_enact_fused<problem_t, applied_t>::value;
  if constexpr (problem_t::problem_family == family::fixed_point) {
    static_assert(hook_detail::has_loop<problem_t>::value || fused,
                  "dyng framework: a fixed_point problem must provide `loop(context&, "
                  "new_view<container_type>, frontier_type& in, frontier_type& out)` (Tier A) "
                  "or `enact_fused(context&, new_view<container_type>, const applied&, "
                  "stats_type&)` (Tier B): Step 2 iterates to a fixed point");
  } else {
    static_assert(
        hook_detail::has_count<problem_t, new_view<typename problem_t::container_type>>::value ||
            fused,
        "dyng framework: an aggregate_delta problem must provide `count(context&, "
        "new_view<container_type>, frontier_type&, sign, ownership_type)` (Tier A) or "
        "`enact_fused(...)` (Tier B): the structures through the inserted elements are added "
        "on G_{t+1}, after the commit");
  }
  return true;
}

/**
 * @brief The hooks the family of a problem requires for compute() (checked by the static
 *        enactor).
 * @tparam problem_t The problem.
 * @return true (the checks are static_asserts).
 */
template <typename problem_t>
constexpr bool check_static_hooks() {
  constexpr bool fused = hook_detail::has_compute_fused<problem_t>::value;
  if constexpr (problem_t::problem_family == family::fixed_point) {
    static_assert(hook_detail::has_loop<problem_t>::value || fused,
                  "dyng framework: compute() of a fixed_point problem needs `loop(context&, "
                  "new_view<container_type>, frontier_type& in, frontier_type& out)` or "
                  "`compute_fused(context&, new_view<container_type>, stats_type&)`");
  } else {
    static_assert(
        hook_detail::has_count<problem_t, new_view<typename problem_t::container_type>>::value ||
            fused,
        "dyng framework: compute() of an aggregate_delta problem needs `count(context&, "
        "new_view<container_type>, frontier_type&, sign, ownership_type)` (the full count) or "
        "`compute_fused(context&, new_view<container_type>, stats_type&)`");
  }
  return true;
}

}  // namespace dyng::detail::framework
