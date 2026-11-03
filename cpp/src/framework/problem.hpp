// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file problem.hpp
 * @brief problem_base: the thesis update template as a CRTP base with no-op default hooks
 *        (PLAN Sections 4.5.1 and 4.5.2). Internal-stable; guide: docs/developer/framework.md.
 *
 * Template card:
 *
 *     normalize -> translate -> prepare -> [before_apply -> (AG: count -)] -> commit ->
 *     identify_affected -> seed -> { FP: loop until is_converged | AG: count + } -> finalize
 *
 * A problem is a class that derives from problem_base<itself, family::...> and declares
 *
 *     static constexpr std::string_view name = "<algo>";   // profiler stages "<algo>.<hook>"
 *     using container_type = graph<V, E, W>;                // what the views show
 *     using stats_type     = <algo>::stats;                 // derived from update_stats
 *     using ownership_type = ownership::min_member;         // aggregate_delta only (I2)
 *     using frontier_type  = ...;                           // optional (internal_frontier)
 *
 * and overrides the hooks it needs. Hooks are resolved at compile time (no virtual calls): a
 * default hook of this base returns not_provided, and the enactors neither call such a hook nor
 * open its profiler stage, so a problem's profile lists exactly the hooks it implements. The
 * required hooks have no default (framework/conformance.hpp checks them with plain-English
 * messages):
 *
 *   - family::fixed_point: loop(ctx, new_view, frontier& in, frontier& out) or, Tier B,
 *     enact_fused(ctx, new_view, applied, stats&);
 *   - family::aggregate_delta: count(ctx, old_view, frontier&, sign, ownership_type) for the
 *     subtraction on G_t and count(ctx, new_view, frontier&, sign, ownership_type) (or
 *     enact_fused) for the addition on G_{t+1}.
 *
 * Hook signatures (every hook receives the run's context first; old_view / new_view are
 * framework/views.hpp; `batch` is the requested batch, `applied` what the commit did,
 * framework/composition.hpp for graphs):
 *
 * | Hook (profiler stage)                     | When                        | Reads   |
 * |-------------------------------------------|-----------------------------|---------|
 * | normalize(ctx, old, batch)                | Step 0, the problem's own   | G_t     |
 * | prepare(ctx, old, batch)                  | Step 0                      | G_t     |
 * | before_apply(ctx, old, batch, frontier&)  | Step 1a                     | G_t     |
 * | count(ctx, old, frontier&, sign::minus, o)| Step 1a (AG, `count_minus`) | G_t     |
 * | identify_affected(ctx, new, applied, f&)  | Step 1b                     | G_{t+1} |
 * | seed(ctx, new, frontier&)                 | Step 1b                     | G_{t+1} |
 * | loop(ctx, new, in&, out&)                 | Step 2 (FP) until converged | G_{t+1} |
 * | count(ctx, new, frontier&, sign::plus, o) | Step 2 (AG, `count_plus`)   | G_{t+1} |
 * | finalize(ctx, stats&)                     | finish                      | -       |
 * | enact_fused(ctx, new, applied, stats&)    | Tier B: replaces 1b..finish | G_{t+1} |
 *
 * compute() (static_enactor): reset(ctx) -> seed_static(ctx, new, frontier&) (stage `seed`) ->
 * loop until converged (FP) or count(ctx, new, frontier&, sign::plus, o) (AG, stage `count`) ->
 * finalize(ctx, stats&); Tier B: compute_fused(ctx, new, stats&) (stage `enact_fused`).
 *
 * Policy hooks (no stage): is_converged(ctx, frontier, iteration) (default below),
 * convergence_policy(ctx) -> convergence (default: no cap), select_engine(ctx) -> engine
 * (default: engine::automatic), fused_available(ctx) -> bool (whether the fused engine can run in
 * this call, e.g. on the CUDA backend with cooperative launch; default: not provided, read as
 * true), algorithm_budget(ctx) -> budget (default: unchecked), recompute(ctx, new, stats&)
 * (needed only for on_limit::fallback_recompute). engine::automatic resolves to fused when the
 * problem has a fused engine and it can run (fused_available), else to operators; a problem with
 * both a Tier A engine and a fused one must provide fused_available or select_engine
 * (framework/conformance.hpp), because a fused engine is written for one backend (PLAN 4.5.4).
 * The enactor chooses the engine before the commit and records it (context::chosen_engine()).
 *
 * Lifecycle members, used by the participant adapter of framework/composition.hpp (no stage; they
 * are the "argument validation, version check, result bookkeeping" of the algorithm's .cpp file,
 * PLAN Section 4.8): target() (required there), begin_update(ctx, old, batch) (validate and bind
 * before Step 0; must change nothing visible; the update enactor calls it first, before it
 * chooses the engine), resume(ctx, new, applied) (re-bind to G_{t+1} after
 * the commit: grow the result, lease and size the workspace, build per-run inputs; its own
 * sub-stages if any), end_update(ctx, new, stats) (record the graph state the result now matches),
 * poison() (a failed algorithm phase left the result unusable), reads_prepared_graph() (whether
 * the commit must prepare the in-edges or the device copy for this problem; default true).
 */
#pragma once

#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/views.hpp"

#include <type_traits>

namespace dyng::detail::framework {

/**
 * @brief The two property families of PLAN Section 4.5.1.
 */
enum class family {
  /// A per-element value; Step 2 iterates to a fixed point (sssp).
  fixed_point,
  /// A global aggregate; P_{t+1} = P_t - count(G_t, deletions) + count(G_{t+1}, insertions), one
  /// signed count per side under an ownership rule, no iteration (cycle_count).
  aggregate_delta,
};

/**
 * @brief The return type of a default hook: the problem does not implement it.
 *
 * The enactors test the return type of a hook call at compile time and skip hooks that return
 * not_provided (no call, no profiler stage).
 */
struct not_provided {};

/**
 * @brief Whether a hook's return type means "implemented".
 * @tparam result_t The return type of the hook call.
 */
template <typename result_t>
inline constexpr bool is_provided_v = !std::is_same_v<std::decay_t<result_t>, not_provided>;

/**
 * @brief CRTP base of every problem, with the no-op default hooks.
 * @tparam derived_t The problem (derives from problem_base<derived_t, kind>).
 * @tparam kind      Its family.
 */
template <typename derived_t, family kind>
struct problem_base {
  /// The problem's family.
  static constexpr family problem_family = kind;

  /// The default frontier: the problem keeps its frontiers in its workspace.
  using frontier_type = internal_frontier;

  // ---- lifecycle (participant adapter; no profiler stage) ----------------------------------

  /**
   * @brief Validate the call and bind the problem before Step 0 (default: nothing).
   * @return not_provided.
   */
  template <typename view_t, typename batch_t>
  not_provided begin_update(context& /*ctx*/, old_view<view_t> /*g*/, const batch_t& /*batch*/) {
    return {};
  }

  /**
   * @brief Re-bind the problem to G_{t+1} after the commit (default: nothing).
   * @return not_provided.
   */
  template <typename view_t, typename applied_t>
  not_provided resume(context& /*ctx*/, new_view<view_t> /*g*/, const applied_t& /*applied*/) {
    return {};
  }

  /**
   * @brief Record the graph state the result matches after the update (default: nothing).
   * @return not_provided.
   */
  template <typename view_t, typename stats_t>
  not_provided end_update(context& /*ctx*/, new_view<view_t> /*g*/, const stats_t& /*stats*/) {
    return {};
  }

  /**
   * @brief Mark the result unusable after its algorithm phase failed (default: nothing).
   * @return not_provided.
   */
  not_provided poison() noexcept {
    return {};
  }

  /**
   * @brief Whether the commit prepares the in-edges / device copy for this problem (default:
   *        not provided, which the adapter reads as true).
   * @return not_provided.
   */
  [[nodiscard]] not_provided reads_prepared_graph() const noexcept {
    return {};
  }

  // ---- policies (no profiler stage) --------------------------------------------------------

  /**
   * @brief The engine of this run (default: engine::automatic, see the file comment).
   * @return not_provided.
   */
  not_provided select_engine(context& /*ctx*/) {
    return {};
  }

  /**
   * @brief Whether the fused engine (enact_fused / compute_fused) can run in this call, for
   *        example only on the CUDA backend and only with cooperative launch (default: not
   *        provided, read as true).
   * @return not_provided.
   */
  not_provided fused_available(context& /*ctx*/) {
    return {};
  }

  /**
   * @brief The convergence policy of Step 2 (default: convergence{}, no iteration cap).
   * @return not_provided.
   */
  not_provided convergence_policy(context& /*ctx*/) {
    return {};
  }

  /**
   * @brief The budget of the update's algorithm work, both halves around the commit (default:
   *        budget::unchecked(); framework/budgets.hpp).
   * @return not_provided.
   */
  not_provided algorithm_budget(context& /*ctx*/) {
    return {};
  }

  /**
   * @brief Whether Step 2 has converged before loop call number `iteration` (0-based).
   *
   * The default: a frontier with `empty()` has converged when it is empty; a problem with an
   * internal_frontier (or any frontier without `empty()`) runs its loop hook exactly once, and
   * that call iterates to the fixed point inside the problem's engine.
   * @tparam frontier_t The frontier type.
   * @param[in] f         The current input frontier.
   * @param[in] iteration Loop calls so far.
   * @return true if the enactor stops Step 2.
   */
  template <typename frontier_t>
  [[nodiscard]] bool is_converged(context& /*ctx*/, const frontier_t& f, int iteration) const {
    if constexpr (has_empty<frontier_t>::value) {
      (void)iteration;
      return f.empty();
    } else {
      (void)f;
      return iteration > 0;
    }
  }

  // ---- Step 0 and Step 1a: on G_t, before the commit ---------------------------------------

  /**
   * @brief Step 0, the problem's own normalization (G1; the framework's set normalization of
   *        batch_semantics::as_sets has already run and is part of `batch`). Stage `normalize`.
   * @return not_provided.
   */
  template <typename view_t, typename batch_t>
  not_provided normalize(context& /*ctx*/, old_view<view_t> /*g*/, const batch_t& /*batch*/) {
    return {};
  }

  /**
   * @brief Step 0: prepare the batch (classify, summarize, check). Stage `prepare`.
   * @return not_provided.
   */
  template <typename view_t, typename batch_t>
  not_provided prepare(context& /*ctx*/, old_view<view_t> /*g*/, const batch_t& /*batch*/) {
    return {};
  }

  /**
   * @brief Step 1a: read G_t before it changes (G4). Stage `before_apply`.
   * @return not_provided.
   */
  template <typename view_t, typename batch_t, typename frontier_t>
  not_provided before_apply(context& /*ctx*/, old_view<view_t> /*g*/, const batch_t& /*batch*/,
                            frontier_t& /*frontier*/) {
    return {};
  }

  // ---- Step 1b: on G_{t+1}, after the commit ----------------------------------------------

  /**
   * @brief Step 1b: roots -> invalidate -> the affected frontier (G3). Stage
   *        `identify_affected`.
   * @return not_provided.
   */
  template <typename view_t, typename applied_t, typename frontier_t>
  not_provided identify_affected(context& /*ctx*/, new_view<view_t> /*g*/,
                                 const applied_t& /*applied*/, frontier_t& /*frontier*/) {
    return {};
  }

  /**
   * @brief Step 1b: initial values of new or invalidated elements (G9). Stage `seed`.
   * @return not_provided.
   */
  template <typename view_t, typename frontier_t>
  not_provided seed(context& /*ctx*/, new_view<view_t> /*g*/, frontier_t& /*frontier*/) {
    return {};
  }

  // ---- finish ------------------------------------------------------------------------------

  /**
   * @brief Combine, apply deltas, unpack; fill the stats (G8). Stage `finalize`.
   * @return not_provided.
   */
  template <typename stats_t>
  not_provided finalize(context& /*ctx*/, stats_t& /*stats*/) {
    return {};
  }

  // ---- compute() (static_enactor) ------------------------------------------------------------

  /**
   * @brief compute(): reset the result before the static run. Stage `reset`.
   * @return not_provided.
   */
  not_provided reset(context& /*ctx*/) {
    return {};
  }

  /**
   * @brief compute(): the initial frontier (e.g. the source vertex). Stage `seed`.
   * @return not_provided.
   */
  template <typename view_t, typename frontier_t>
  not_provided seed_static(context& /*ctx*/, new_view<view_t> /*g*/, frontier_t& /*frontier*/) {
    return {};
  }
};

}  // namespace dyng::detail::framework
