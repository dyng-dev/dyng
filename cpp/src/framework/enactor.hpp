// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file enactor.hpp
 * @brief static_enactor (compute) and update_enactor (update): the fixed hook order of the
 *        thesis template, one profiler stage per implemented hook, the convergence policy, the
 *        device error check and the budget of the algorithm phase (PLAN Sections 4.5.1-4.5.5).
 *
 * update_enactor<problem_t> runs one update of one result in two halves, because several results
 * share one commit (framework/composition.hpp, run_update()):
 *
 *     before_commit(ctx, old_view, batch):  normalize -> prepare -> before_apply
 *                                           [-> AG: count(-) on G_t, stage count_minus]
 *                                           -> device error check
 *     ... the commit (G_t -> G_{t+1}, once for every result) ...
 *     after_commit(ctx, new_view, applied, stats):
 *         resume -> Tier B: enact_fused
 *                   Tier A: identify_affected -> seed
 *                           -> { FP: loop until is_converged (stage loop, cap + on_limit)
 *                              | AG: count(+) on G_{t+1}, stage count_plus }
 *                           -> finalize
 *         -> device error check -> budget check (DYNG_DEBUG_BUDGETS)
 *
 * static_enactor<problem_t>::run(ctx, view) runs compute(): reset -> seed_static (stage seed) ->
 * { FP: loop until is_converged | AG: count (stage count) } -> finalize, or compute_fused (stage
 * enact_fused), then the device error check.
 *
 * Stages are named "<algo>.<hook>" (problem_t::name); a hook the problem does not implement is
 * neither called nor staged. The enactor sets the common update_stats it decides: engine_used
 * (fused when enact_fused ran, operators otherwise; finalize may refine it), converged (false only
 * under on_limit::report) and fallback_used (on_limit::fallback_recompute). The problem fills the
 * rest (affected, iterations, frontier_visits and its own counters) in finalize or enact_fused.
 */
#pragma once

#include "framework/budgets.hpp"
#include "framework/conformance.hpp"
#include "framework/context.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/views.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/types.hpp>

#include <string>
#include <string_view>
#include <utility>

namespace dyng::detail::framework {

/**
 * @brief The profiler stage names of one algorithm, "<algo>.<hook>" (built once per problem
 *        type, so running a hook allocates nothing).
 */
struct hook_stages {
  std::string update;             ///< "<algo>.update" (the whole update() call)
  std::string compute;            ///< "<algo>.compute" (the whole compute() call)
  std::string commit;             ///< "<algo>.commit" (the commit of a single-result update)
  std::string normalize;          ///< "<algo>.normalize"
  std::string prepare;            ///< "<algo>.prepare"
  std::string before_apply;       ///< "<algo>.before_apply"
  std::string count_minus;        ///< "<algo>.count_minus" (AG, on G_t)
  std::string identify_affected;  ///< "<algo>.identify_affected"
  std::string seed;               ///< "<algo>.seed" (also seed_static)
  std::string loop;               ///< "<algo>.loop" (the whole Step 2 of a fixed point)
  std::string count_plus;         ///< "<algo>.count_plus" (AG, on G_{t+1})
  std::string count;              ///< "<algo>.count" (AG, compute())
  std::string finalize;           ///< "<algo>.finalize"
  std::string enact_fused;        ///< "<algo>.enact_fused" (Tier B, update and compute)
  std::string recompute;          ///< "<algo>.recompute" (on_limit::fallback_recompute)
  std::string reset;              ///< "<algo>.reset" (compute())
  std::string update_function;    ///< "<algo>::update" (messages)
  std::string compute_function;   ///< "<algo>::compute" (messages)

  /**
   * @brief The names of an algorithm.
   * @param[in] algorithm The algorithm's name (e.g. "sssp").
   * @return Every stage name.
   */
  static hook_stages of(std::string_view algorithm) {
    const std::string a(algorithm);
    hook_stages s;
    s.update = a + ".update";
    s.compute = a + ".compute";
    s.commit = a + ".commit";
    s.normalize = a + ".normalize";
    s.prepare = a + ".prepare";
    s.before_apply = a + ".before_apply";
    s.count_minus = a + ".count_minus";
    s.identify_affected = a + ".identify_affected";
    s.seed = a + ".seed";
    s.loop = a + ".loop";
    s.count_plus = a + ".count_plus";
    s.count = a + ".count";
    s.finalize = a + ".finalize";
    s.enact_fused = a + ".enact_fused";
    s.recompute = a + ".recompute";
    s.reset = a + ".reset";
    s.update_function = a + "::update";
    s.compute_function = a + "::compute";
    return s;
  }
};

/**
 * @brief The stage names of a problem type (built on first use; thread-safe).
 * @tparam problem_t The problem.
 * @return The names.
 */
template <typename problem_t>
const hook_stages& stages_of() {
  static const hook_stages stages = hook_stages::of(problem_t::name);
  return stages;
}

/// Pieces shared by the two enactors.
namespace enactor_detail {

/// Throw the recorded device errors of a phase (and forget them).
inline void throw_if_device_errors(context& ctx, const std::string& function) {
  const std::uint32_t bits = ctx.device_errors();
  if (bits != 0) {
    const std::string detail = ctx.device_error_detail();
    ctx.clear_device_errors();
    throw_device_errors(bits, function, detail);
  }
}

/// The engine of a run: the problem's choice, or fused if it has a fused engine.
template <typename problem_t>
engine choose_engine(problem_t& p, context& ctx, bool has_fused) {
  using chosen_t = decltype(p.select_engine(ctx));
  engine e = engine::automatic;
  if constexpr (is_provided_v<chosen_t>) {
    e = p.select_engine(ctx);
  }
  if (e == engine::automatic) {
    e = has_fused ? engine::fused : engine::operators;
  }
  return e;
}

/// The convergence policy of a problem (default: no cap).
template <typename problem_t>
convergence convergence_of(problem_t& p, context& ctx) {
  if constexpr (is_provided_v<decltype(p.convergence_policy(ctx))>) {
    return p.convergence_policy(ctx);
  } else {
    return convergence{};
  }
}

/// How a fixed-point Step 2 ended.
enum class loop_end {
  converged,  ///< is_converged returned true
  capped,     ///< the cap was reached under on_limit::report
  recompute,  ///< the cap was reached under on_limit::fallback_recompute
};

/// Step 2 of a fixed point: call loop until is_converged, under the convergence policy's cap.
template <typename problem_t, typename view_t, typename frontier_t>
loop_end run_loop(problem_t& p, context& ctx, view_t g, frontier_t& in, frontier_t& out,
                  const convergence& policy, const std::string& function) {
  frontier_t* current = &in;
  frontier_t* next = &out;
  for (int iteration = 0; !p.is_converged(ctx, *current, iteration); ++iteration) {
    if (policy.max_iterations >= 0 && iteration >= policy.max_iterations) {
      switch (policy.at_limit) {
        case on_limit::report:
          return loop_end::capped;
        case on_limit::fallback_recompute:
          return loop_end::recompute;
        case on_limit::error:
          break;
      }
      throw convergence_error(concat_message(
          "dyng: ", function, ": no fixed point after ", policy.max_iterations,
          " iteration(s) (the convergence cap max_iterations); raise the cap, or choose "
          "on_limit::report or on_limit::fallback_recompute"));
    }
    p.loop(ctx, g, *current, *next);
    std::swap(current, next);
  }
  return loop_end::converged;
}

}  // namespace enactor_detail

/**
 * @brief Runs one update() of one problem in the fixed hook order (see the file comment).
 *
 * The enactor owns the problem's two frontiers for the length of the update; it is made for one
 * update and discarded (the participant adapter holds one per result).
 * @tparam problem_t The problem (framework/problem.hpp).
 */
template <typename problem_t>
class update_enactor {
  static_assert(check_problem<problem_t>());  // before the member types that need a problem

 public:
  using container_type = typename problem_t::container_type;  ///< the container
  using frontier_type = typename problem_t::frontier_type;    ///< the frontiers it owns
  using stats_type = typename problem_t::stats_type;          ///< the stats it fills

  /**
   * @brief An enactor for one update of `problem`.
   * @param[in,out] problem The problem (must outlive the enactor).
   */
  explicit update_enactor(problem_t& problem) noexcept : p_(problem) {}

  /**
   * @brief Steps 0 and 1a on G_t: normalize, prepare, before_apply and (aggregate_delta) the
   *        subtraction count(-). Nothing visible may change here (a later participant can still
   *        reject the batch).
   * @tparam batch_t The requested batch (framework::requested_batch for graphs).
   * @param[in,out] ctx   The run's context.
   * @param[in]     g     G_t.
   * @param[in]     batch The requested batch.
   * @throws what the hooks throw; the recorded device errors (throw_device_errors).
   */
  template <typename batch_t>
  void before_commit(context& ctx, old_view<container_type> g, const batch_t& batch) {
    const hook_stages& names = stages_of<problem_t>();
    const resources& res = ctx.res();
    if constexpr (is_provided_v<decltype(p_.normalize(ctx, g, batch))>) {
      scoped_stage stage(res, names.normalize);
      p_.normalize(ctx, g, batch);
    }
    if constexpr (is_provided_v<decltype(p_.prepare(ctx, g, batch))>) {
      scoped_stage stage(res, names.prepare);
      p_.prepare(ctx, g, batch);
    }
    if constexpr (is_provided_v<decltype(p_.before_apply(ctx, g, batch, in_))>) {
      scoped_stage stage(res, names.before_apply);
      p_.before_apply(ctx, g, batch, in_);
    }
    if constexpr (problem_t::problem_family == family::aggregate_delta) {
      scoped_stage stage(res, names.count_minus);
      p_.count(ctx, g, in_, sign::minus, typename problem_t::ownership_type{});
    }
    enactor_detail::throw_if_device_errors(ctx, names.update_function);
  }

  /**
   * @brief The algorithm phase on G_{t+1}: resume, then Tier B (enact_fused) or Tier A
   *        (identify_affected, seed, loop or count(+), finalize); then the device error and budget
   *        checks.
   * @tparam applied_t What the commit reports (framework::applied_batch for graphs).
   * @param[in,out] ctx     The run's context.
   * @param[in]     g       G_{t+1}.
   * @param[in]     applied What the commit did.
   * @param[in,out] stats   The update's stats (common fields set here, the rest by the hooks).
   * @throws convergence_error   if the cap is reached under on_limit::error.
   * @throws not_supported_error if the chosen engine does not exist for the problem.
   * @throws internal_error      if the phase exceeds its budget (DYNG_DEBUG_BUDGETS builds).
   * @throws what the hooks throw; the recorded device errors (throw_device_errors).
   */
  template <typename applied_t>
  void after_commit(context& ctx, new_view<container_type> g, const applied_t& applied,
                    stats_type& stats) {
    static_assert(check_update_hooks<problem_t, applied_t>());
    constexpr bool has_fused = hook_detail::has_enact_fused<problem_t, applied_t>::value;
    constexpr bool has_operators =
        problem_t::problem_family == family::fixed_point
            ? hook_detail::has_loop<problem_t>::value
            : hook_detail::has_count<problem_t, new_view<container_type>>::value;
    const hook_stages& names = stages_of<problem_t>();
    const resources& res = ctx.res();
    const budget_scope phase;
    stats.fallback_used = false;
    stats.converged = true;
    if constexpr (is_provided_v<decltype(p_.resume(ctx, g, applied))>) {
      p_.resume(ctx, g, applied);
    }
    const engine chosen = enactor_detail::choose_engine(p_, ctx, has_fused);
    if (chosen == engine::fused) {
      if constexpr (has_fused) {
        stats.engine_used = engine::fused;
        scoped_stage stage(res, names.enact_fused);
        p_.enact_fused(ctx, g, applied, stats);
      } else {
        throw not_supported_error(
            concat_message("dyng: ", names.update_function, ": this backend has no fused engine"));
      }
    } else {
      if constexpr (has_operators) {
        stats.engine_used = engine::operators;
        run_operators(ctx, g, applied, stats);
      } else {
        throw not_supported_error(concat_message(
            "dyng: ", names.update_function,
            ": there is no operators engine for this backend in this release; use engine::fused "
            "or engine::automatic"));
      }
    }
    enactor_detail::throw_if_device_errors(ctx, names.update_function);
    check_budget(problem_t::name, budget_of(ctx), phase.used());
  }

 private:
  /// Tier A: identify_affected -> seed -> loop | count(+) -> finalize.
  template <typename applied_t>
  void run_operators(context& ctx, new_view<container_type> g, const applied_t& applied,
                     stats_type& stats) {
    const hook_stages& names = stages_of<problem_t>();
    const resources& res = ctx.res();
    if constexpr (is_provided_v<decltype(p_.identify_affected(ctx, g, applied, in_))>) {
      scoped_stage stage(res, names.identify_affected);
      p_.identify_affected(ctx, g, applied, in_);
    }
    if constexpr (is_provided_v<decltype(p_.seed(ctx, g, in_))>) {
      scoped_stage stage(res, names.seed);
      p_.seed(ctx, g, in_);
    }
    if constexpr (problem_t::problem_family == family::fixed_point) {
      enactor_detail::loop_end end = enactor_detail::loop_end::converged;
      {
        scoped_stage stage(res, names.loop);
        end = enactor_detail::run_loop(
            p_, ctx, g, in_, out_, enactor_detail::convergence_of(p_, ctx), names.update_function);
      }
      if (end == enactor_detail::loop_end::capped) {
        stats.converged = false;
      } else if (end == enactor_detail::loop_end::recompute) {
        if constexpr (hook_detail::has_recompute<problem_t>::value) {
          scoped_stage stage(res, names.recompute);
          p_.recompute(ctx, g, stats);
          stats.fallback_used = true;
          return;  // the recompute replaces the rest of Step 2 and finalize
        } else {
          throw not_supported_error(
              concat_message("dyng: ", names.update_function,
                             ": on_limit::fallback_recompute needs the problem's recompute hook"));
        }
      }
    } else {
      scoped_stage stage(res, names.count_plus);
      p_.count(ctx, g, in_, sign::plus, typename problem_t::ownership_type{});
    }
    if constexpr (is_provided_v<decltype(p_.finalize(ctx, stats))>) {
      scoped_stage stage(res, names.finalize);
      p_.finalize(ctx, stats);
    }
  }

  /// The problem's budget of the algorithm phase (default: unchecked).
  budget budget_of(context& ctx) {
    if constexpr (is_provided_v<decltype(p_.algorithm_budget(ctx))>) {
      return p_.algorithm_budget(ctx);
    } else {
      return budget::unchecked();
    }
  }

  problem_t& p_;
  frontier_type in_{};
  frontier_type out_{};
};

/**
 * @brief Runs compute() of one problem: reset -> seed_static -> loop | count -> finalize, or
 *        compute_fused (see the file comment).
 * @tparam problem_t The problem (framework/problem.hpp).
 */
template <typename problem_t>
class static_enactor {
  static_assert(check_problem<problem_t>());  // before the member types that need a problem

 public:
  using container_type = typename problem_t::container_type;  ///< the container
  using frontier_type = typename problem_t::frontier_type;    ///< the frontiers it owns
  using stats_type = typename problem_t::stats_type;          ///< what finalize fills

  /**
   * @brief An enactor for one compute() of `problem`.
   * @param[in,out] problem The problem (must outlive the enactor).
   */
  explicit static_enactor(problem_t& problem) noexcept : p_(problem) {
    static_assert(check_static_hooks<problem_t>());
  }

  /**
   * @brief Run compute() on `g`.
   * @param[in,out] ctx The run's context.
   * @param[in]     g   The container.
   * @return The stats the hooks filled (compute() itself returns only the result).
   * @throws convergence_error   if the cap is reached (on_limit::error, or
   *                             fallback_recompute, which compute() cannot fall back from).
   * @throws not_supported_error if the chosen engine does not exist for the problem.
   * @throws what the hooks throw; the recorded device errors (throw_device_errors).
   */
  stats_type run(context& ctx, new_view<container_type> g) {
    constexpr bool has_fused = hook_detail::has_compute_fused<problem_t>::value;
    constexpr bool has_operators =
        problem_t::problem_family == family::fixed_point
            ? hook_detail::has_loop<problem_t>::value
            : hook_detail::has_count<problem_t, new_view<container_type>>::value;
    const hook_stages& names = stages_of<problem_t>();
    const resources& res = ctx.res();
    stats_type stats;
    const engine chosen = enactor_detail::choose_engine(p_, ctx, has_fused);
    if (chosen == engine::fused) {
      if constexpr (has_fused) {
        stats.engine_used = engine::fused;
        scoped_stage stage(res, names.enact_fused);
        p_.compute_fused(ctx, g, stats);
      } else {
        throw not_supported_error(
            concat_message("dyng: ", names.compute_function, ": this backend has no fused engine"));
      }
    } else {
      if constexpr (has_operators) {
        stats.engine_used = engine::operators;
        run_operators(ctx, g, stats);
      } else {
        throw not_supported_error(concat_message(
            "dyng: ", names.compute_function,
            ": there is no operators engine for this backend in this release; use engine::fused "
            "or engine::automatic"));
      }
    }
    enactor_detail::throw_if_device_errors(ctx, names.compute_function);
    return stats;
  }

 private:
  void run_operators(context& ctx, new_view<container_type> g, stats_type& stats) {
    const hook_stages& names = stages_of<problem_t>();
    const resources& res = ctx.res();
    if constexpr (is_provided_v<decltype(p_.reset(ctx))>) {
      scoped_stage stage(res, names.reset);
      p_.reset(ctx);
    }
    if constexpr (is_provided_v<decltype(p_.seed_static(ctx, g, in_))>) {
      scoped_stage stage(res, names.seed);
      p_.seed_static(ctx, g, in_);
    }
    if constexpr (problem_t::problem_family == family::fixed_point) {
      convergence policy = enactor_detail::convergence_of(p_, ctx);
      if (policy.at_limit == on_limit::fallback_recompute) {
        policy.at_limit = on_limit::error;  // compute() is the recomputation
      }
      scoped_stage stage(res, names.loop);
      if (enactor_detail::run_loop(p_, ctx, g, in_, out_, policy, names.compute_function) ==
          enactor_detail::loop_end::capped) {
        stats.converged = false;
      }
    } else {
      scoped_stage stage(res, names.count);
      p_.count(ctx, g, in_, sign::plus, typename problem_t::ownership_type{});
    }
    if constexpr (is_provided_v<decltype(p_.finalize(ctx, stats))>) {
      scoped_stage stage(res, names.finalize);
      p_.finalize(ctx, stats);
    }
  }

  problem_t& p_;
  frontier_type in_{};
  frontier_type out_{};
};

}  // namespace dyng::detail::framework
