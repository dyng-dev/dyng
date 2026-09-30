// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file composition.hpp
 * @brief How problems take part in run_update() (PLAN Section 4.5.2, "Composition"): the batch
 *        types the hooks see, the participant adapter of a problem, the single-result update and
 *        the result bookkeeping shared by every algorithm.
 *
 * run_update(res, g, batch, participants, n, commit_stage) (cpp/include/dyng/update.hpp,
 * composition.cpp) stages the batch to the host, normalizes it once under batch_semantics::as_sets
 * (stage "<algo>.normalize", or "update.normalize" for dyng::update), runs every participant's
 * before_apply on G_t, commits once (stage "<algo>.commit" or "update.commit"), then runs every
 * participant's after_apply on G_{t+1}; a participant that fails after the commit is poisoned and
 * the others still run. problem_participant<problem_t> is the participant of one problem: its
 * before_apply is update_enactor::before_commit (begin_update first), its after_apply is
 * update_enactor::after_commit + end_update. So
 *
 *   - `<algo>::update(res, g, batch, r)` is `update_one<problem_t>(res, g, batch, r, ...)`, and
 *   - `dyng::update(res, g, batch, r1, r2, ...)` builds one participant per result with
 *     make_participant<problem_t>(...) (each algorithm's update_traits) and runs them together:
 *     every Step 0 and 1a on G_t, one commit, every Step 1b and 2 on G_{t+1}.
 */
#pragma once

#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/views.hpp"
#include "graph/graph_impl.hpp"
#include "graph/normalized_batch.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/update.hpp>

#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <type_traits>
#include <utility>

namespace dyng::detail::framework {

/**
 * @brief The batch as the hooks before the commit see it (normalize, prepare, before_apply,
 *        count(-)).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
struct requested_batch {
  const edge_batch_view<vertex_t, weight_t>& edges;  ///< the requested batch (host memory)
  /// Step 0 of batch_semantics::as_sets, computed once by run_update() for every participant and
  /// the commit (ADR 0020); nullptr for other semantics.
  const normalized_batch<vertex_t>* normalized;
};

/**
 * @brief What the commit did, as the hooks after it see it (resume, identify_affected,
 *        enact_fused).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct applied_batch {
  const apply_summary& summary;        ///< the counters of the commit
  const apply_delta<vertex_t>& delta;  ///< its effective changes, classified per objective
  /// The normalized batch of batch_semantics::as_sets, or nullptr (as in requested_batch).
  const normalized_batch<vertex_t>* normalized;
};

/// Detection of the optional parts of a problem for the adapter.
namespace composition_detail {

template <typename stats_t, typename = void>
struct has_batch_summary : std::false_type {};
template <typename stats_t>
struct has_batch_summary<stats_t, std::void_t<decltype(std::declval<stats_t&>().batch)>>
    : std::is_same<std::decay_t<decltype(std::declval<stats_t&>().batch)>, apply_summary> {};

template <typename container_t>
struct graph_types;
template <typename vertex_t, typename edge_t, typename weight_t>
struct graph_types<graph<vertex_t, edge_t, weight_t>> {
  using participant = update_participant<vertex_t, edge_t, weight_t>;
  using batch = edge_batch_view<vertex_t, weight_t>;
  using requested = requested_batch<vertex_t, weight_t>;
  using applied = applied_batch<vertex_t>;
  using normalized = normalized_batch<vertex_t>;
};

}  // namespace composition_detail

/**
 * @brief The participant of one problem in run_update(): the problem's lifecycle members around
 *        the two halves of its update_enactor.
 *
 * The adapter owns the problem (constructed from the arguments after `out`, typically the
 * result), its context and its enactor for the length of one update. Its stats are written to
 * `out` only when the update succeeds (the out-parameter of a failed update keeps its value).
 * @tparam problem_t The problem; its container_type is graph<V,E,W>, and it provides
 *                   `const void* target() const` (the result it updates, to reject duplicates).
 */
template <typename problem_t>
class problem_participant final
    : public composition_detail::graph_types<typename problem_t::container_type>::participant {
  using types = composition_detail::graph_types<typename problem_t::container_type>;

 public:
  using graph_type = typename problem_t::container_type;  ///< the container
  using batch_type = typename types::batch;               ///< the batch view
  using stats_type = typename problem_t::stats_type;      ///< the problem's stats

  /**
   * @brief The participant of a problem built from `args`.
   * @tparam args_t The problem's constructor arguments.
   * @param[out] out  Receives the update's stats when it succeeds (must outlive the participant).
   * @param[in]  args Forwarded to the problem's constructor.
   */
  template <typename... args_t>
  explicit problem_participant(stats_type& out, args_t&&... args)
      : out_(out), problem_(std::forward<args_t>(args)...), enactor_(problem_) {
    static_assert(hook_detail::has_target<problem_t>::value,
                  "dyng framework: a problem that takes part in run_update() must provide `const "
                  "void* target() const` (the address of the result it updates, so that "
                  "dyng::update() can reject a result passed twice)");
  }

  /**
   * @brief The result this participant updates.
   * @return problem.target().
   */
  [[nodiscard]] const void* target() const noexcept override {
    return problem_.target();
  }

  /**
   * @brief Whether the commit prepares the in-edges / device copy for this problem.
   * @return problem.reads_prepared_graph(), true if the problem does not say.
   */
  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    if constexpr (is_provided_v<decltype(problem_.reads_prepared_graph())>) {
      return problem_.reads_prepared_graph();
    } else {
      return true;
    }
  }

  /**
   * @brief Receive the framework's Step 0 (batch_semantics::as_sets) before before_apply().
   * @param[in] normalized The normalized batch, or nullptr.
   */
  void use_normalized(const typename types::normalized* normalized) noexcept override {
    normalized_ = normalized;
  }

  /**
   * @brief begin_update, the choice of the engine, then Steps 0 and 1a on G_t
   *        (update_enactor::before_commit).
   * @param[in] res   Execution resources.
   * @param[in] g     G_t.
   * @param[in] batch The batch (host memory).
   */
  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    ctx_.emplace(res, problem_t::name);
    const old_view<graph_type> view(g);
    const typename types::requested requested{batch, normalized_};
    enactor_.template before_commit<typename types::applied>(*ctx_, view, requested);
  }

  /**
   * @brief The algorithm phase on G_{t+1} (update_enactor::after_commit), then end_update.
   * @param[in] res     Execution resources (the same as in before_apply()).
   * @param[in] g       G_{t+1}.
   * @param[in] summary What the commit did.
   * @param[in] delta   The effective changes of the commit.
   */
  void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                   const apply_delta<typename graph_type::vertex_type>& delta) override {
    (void)res;
    const new_view<graph_type> view(g);
    const typename types::applied applied{summary, delta, normalized_};
    stats_type stats;
    if constexpr (composition_detail::has_batch_summary<stats_type>::value) {
      stats.batch = summary;
    }
    enactor_.after_commit(*ctx_, view, applied, stats);
    if constexpr (is_provided_v<decltype(problem_.end_update(*ctx_, view, stats))>) {
      problem_.end_update(*ctx_, view, stats);
    }
    out_ = stats;
  }

  /**
   * @brief Mark the result unusable (its algorithm phase failed).
   */
  void poison() noexcept override {
    if constexpr (is_provided_v<decltype(problem_.poison())>) {
      problem_.poison();
    }
  }

  /**
   * @brief The problem (for tests).
   * @return The problem the participant owns.
   */
  [[nodiscard]] problem_t& problem() noexcept {
    return problem_;
  }

 private:
  stats_type& out_;
  problem_t problem_;
  update_enactor<problem_t> enactor_;
  std::optional<context> ctx_;
  const typename types::normalized* normalized_ = nullptr;
};

/**
 * @brief A participant for dyng::update() (an algorithm's update_traits::make_participant).
 * @tparam problem_t The problem.
 * @tparam args_t    Its constructor arguments.
 * @param[out] out  Receives the stats when the update succeeds.
 * @param[in]  args Forwarded to the problem's constructor.
 * @return The participant.
 */
template <typename problem_t, typename... args_t>
std::unique_ptr<
    typename composition_detail::graph_types<typename problem_t::container_type>::participant>
make_participant(typename problem_t::stats_type& out, args_t&&... args) {
  return std::make_unique<problem_participant<problem_t>>(out, std::forward<args_t>(args)...);
}

/**
 * @brief `<algo>::update(res, g, batch, r)`: one problem through run_update(), inside the stage
 *        "<algo>.update", with the commit stage "<algo>.commit".
 * @tparam problem_t The problem.
 * @tparam args_t    Its constructor arguments.
 * @param[in]     res   Execution resources.
 * @param[in,out] g     The graph (version + 1).
 * @param[in]     batch The batch (any memory space; run_update() stages it to the host).
 * @param[in]     args  Forwarded to the problem's constructor.
 * @return The update's stats.
 * @throws what run_update() and the hooks throw.
 */
template <typename problem_t, typename... args_t>
typename problem_t::stats_type update_one(
    const resources& res, typename problem_t::container_type& g,
    const typename composition_detail::graph_types<typename problem_t::container_type>::batch&
        batch,
    args_t&&... args) {
  const hook_stages& names = stages_of<problem_t>();
  scoped_stage stage(res, names.update);
  typename problem_t::stats_type out;
  problem_participant<problem_t> participant(out, std::forward<args_t>(args)...);
  typename composition_detail::graph_types<typename problem_t::container_type>::participant*
      participants[] = {&participant};
  run_update(res, g, batch, participants, 1, names.commit);
  return out;
}

/**
 * @brief The stale-result checks every update() makes before Step 0 (ADR 0006, "Graph
 *        identity"): the result must match the graph's version and its state identifier.
 * @tparam graph_t The graph type.
 * @param[in] function    The call, for the message (e.g. "sssp::update").
 * @param[in] version     The graph version the result matches.
 * @param[in] graph_state The graph state identifier the result matches.
 * @param[in] g           The graph.
 * @throws stale_result_error if either differs.
 */
template <typename graph_t>
void expect_current_result(std::string_view function, std::uint64_t version,
                           std::uint64_t graph_state, const graph_t& g) {
  if (version != g.version()) {
    throw stale_result_error(concat_message(
        "dyng: ", function, ": the result matches graph version ", version,
        " but the graph is at version ", g.version(),
        " (the graph was changed without updating this result; recompute it, or update all "
        "results together with dyng::update(res, g, batch, results...))"));
  }
  if (graph_state != graph_access::impl(g).state_id) {
    throw stale_result_error(concat_message(
        "dyng: ", function,
        ": the result was computed on another graph (or on an earlier state of a graph variable "
        "that was reassigned since), although both are at version ",
        g.version(), "; recompute it on this graph"));
  }
}

/**
 * @brief Record the graph state a result matches after compute() or a successful update().
 * @tparam graph_t The graph type.
 * @param[out] version     Receives g.version().
 * @param[out] graph_state Receives the graph's state identifier.
 * @param[in]  g           The graph.
 */
template <typename graph_t>
void stamp_result(std::uint64_t& version, std::uint64_t& graph_state, const graph_t& g) {
  version = g.version();
  graph_state = graph_access::impl(g).state_id;
}

}  // namespace dyng::detail::framework
