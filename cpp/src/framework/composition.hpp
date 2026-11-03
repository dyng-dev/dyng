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
 *
 * The same adapter serves both containers (ADR 0043): composition_detail::graph_types<container>
 * names, per container, the participant interface, the batch view, what the hooks see before
 * (requested) and after (applied) the commit, Step 0's normalized batch, the summary of the
 * commit (the type of stats::batch) and the record the participant receives with it. For
 * hypergraph<V,W> these are hyper_update_participant, hyperedge_batch_view, hyper_requested_batch,
 * hyper_applied_batch, hyper_staged_batch (the transaction's Step 0), hyper_apply_summary and
 * hyper_applied (the applied record), and run_update() is the hypergraph's
 * (cpp/src/hypergraph/hyper_update.cpp).
 */
#pragma once

#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/views.hpp"
#include "graph/graph_impl.hpp"
#include "graph/normalized_batch.hpp"
#include "hypergraph/hypergraph_impl.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
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
  const apply_summary& summary;  ///< the counters of the commit
  /// The changes the batch requested (NOT its net change, see apply_delta), classified per
  /// objective.
  const apply_delta<vertex_t>& delta;
  /// The normalized batch of batch_semantics::as_sets, or nullptr (as in requested_batch).
  const normalized_batch<vertex_t>* normalized;
};

/**
 * @brief A hypergraph batch as the hooks before the commit see it (normalize, prepare,
 *        before_apply, count(-)).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
struct hyper_requested_batch {
  const hyperedge_batch_view<vertex_t, weight_t>& hyperedges;  ///< the requested batch (host)
  /// Step 0, computed once by run_update() for every participant and the commit (the
  /// transaction's staged batch: the live deletions of G_t, the inserted rows normalized).
  const hyper_staged_batch<weight_t>* normalized;
};

/**
 * @brief What the commit of a hypergraph batch did, as the hooks after it see it (resume,
 *        identify_affected, count(+), enact_fused).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename weight_t>
struct hyper_applied_batch {
  const hyper_apply_summary& summary;  ///< the counters and the inserted ids of the commit
  /// The applied record: inserted ids on the host and (CUDA) the device, deleted ids with their
  /// members, touched hyperedges and vertices, the skipped counts.
  const hyper_applied& record;
  const hyper_staged_batch<weight_t>* normalized;  ///< Step 0 (as in hyper_requested_batch)
};

/// Detection of the optional parts of a problem for the adapter.
namespace composition_detail {

template <typename stats_t, typename summary_t, typename = void>
struct has_batch_summary : std::false_type {};
template <typename stats_t, typename summary_t>
struct has_batch_summary<stats_t, summary_t, std::void_t<decltype(std::declval<stats_t&>().batch)>>
    : std::is_same<std::decay_t<decltype(std::declval<stats_t&>().batch)>, summary_t> {};

template <typename container_t>
struct graph_types;
template <typename vertex_t, typename edge_t, typename weight_t>
struct graph_types<graph<vertex_t, edge_t, weight_t>> {
  using participant = update_participant<vertex_t, edge_t, weight_t>;
  using batch = edge_batch_view<vertex_t, weight_t>;
  using requested = requested_batch<vertex_t, weight_t>;
  using applied = applied_batch<vertex_t>;
  using normalized = normalized_batch<vertex_t>;
  using summary = apply_summary;         ///< stats::batch
  using record = apply_delta<vertex_t>;  ///< what after_apply() receives with the summary
  /// reads_prepared_graph() of a problem that does not say: the in-edges / device copy.
  static constexpr bool reads_prepared_default = true;
};
template <typename vertex_t, typename weight_t>
struct graph_types<hypergraph<vertex_t, weight_t>> {
  using participant = hyper_update_participant<vertex_t, weight_t>;
  using batch = hyperedge_batch_view<vertex_t, weight_t>;
  using requested = hyper_requested_batch<vertex_t, weight_t>;
  using applied = hyper_applied_batch<vertex_t, weight_t>;
  using normalized = hyper_staged_batch<weight_t>;
  using summary = hyper_apply_summary;  ///< stats::batch
  using record = hyper_applied;         ///< what after_apply() receives with the summary
  /// reads_prepared_graph() of a problem that does not say: the line graph is not required.
  static constexpr bool reads_prepared_default = false;
};

}  // namespace composition_detail

/**
 * @brief The participant of one problem in run_update(): the problem's lifecycle members around
 *        the two halves of its update_enactor.
 *
 * The adapter owns the problem (constructed from the arguments after `out`, typically the
 * result), its context and its enactor for the length of one update. Its stats are written to
 * `out` only when the update succeeds (the out-parameter of a failed update keeps its value).
 * @tparam problem_t The problem; its container_type is graph<V,E,W> or hypergraph<V,W>, and it
 *                   provides `const void* target() const` (the result it updates, to reject
 *                   duplicates).
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
   * @brief Whether the commit prepares the in-edges / device copy (graphs), or the update keeps
   *        the line graph (hypergraphs), for this problem.
   * @return problem.reads_prepared_graph(); if the problem does not say, true for graphs and
   *         false for hypergraphs.
   */
  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    if constexpr (is_provided_v<decltype(problem_.reads_prepared_graph())>) {
      return problem_.reads_prepared_graph();
    } else {
      return types::reads_prepared_default;
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
   * @param[in] record  Graphs: the changes the commit applied as requested (apply_delta);
   *                    hypergraphs: the applied record (hyper_applied).
   */
  void after_apply(const resources& res, const graph_type& g,
                   const typename types::summary& summary,
                   const typename types::record& record) override {
    (void)res;
    const new_view<graph_type> view(g);
    const typename types::applied applied{summary, record, normalized_};
    stats_type stats;
    if constexpr (composition_detail::has_batch_summary<stats_type,
                                                        typename types::summary>::value) {
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
 * @brief The state identity of a container (ADR 0006, "Graph identity").
 * @param[in] g The graph.
 * @return Its state identifier.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
std::uint64_t state_identity(const graph<vertex_t, edge_t, weight_t>& g) {
  return graph_access::impl(g).state_id;
}

/**
 * @brief The state identity of a hypergraph.
 * @param[in] h The hypergraph.
 * @return Its state identifier.
 */
template <typename vertex_t, typename weight_t>
std::uint64_t state_identity(const hypergraph<vertex_t, weight_t>& h) {
  return hypergraph_access::state_id(h);
}

/// The container's name in the stale-result messages.
template <typename vertex_t, typename edge_t, typename weight_t>
constexpr const char* container_noun(const graph<vertex_t, edge_t, weight_t>& /*g*/) {
  return "graph";
}
/// @copydoc container_noun
template <typename vertex_t, typename weight_t>
constexpr const char* container_noun(const hypergraph<vertex_t, weight_t>& /*h*/) {
  return "hypergraph";
}

/**
 * @brief The stale-result checks every update() makes before Step 0 (ADR 0006, "Graph
 *        identity"): the result must match the container's version and its state identifier.
 * @tparam graph_t The container type (graph or hypergraph).
 * @param[in] function    The call, for the message (e.g. "sssp::update").
 * @param[in] version     The graph version the result matches.
 * @param[in] graph_state The graph state identifier the result matches.
 * @param[in] g           The graph.
 * @throws stale_result_error if either differs.
 */
template <typename graph_t>
void expect_current_result(std::string_view function, std::uint64_t version,
                           std::uint64_t graph_state, const graph_t& g) {
  const char* noun = container_noun(g);
  if (version != g.version()) {
    throw stale_result_error(concat_message(
        "dyng: ", function, ": the result matches ", noun, " version ", version, " but the ", noun,
        " is at version ", g.version(), " (the ", noun,
        " was changed without updating this result; recompute it, or update all results "
        "together with dyng::update(res, g, batch, results...))"));
  }
  if (graph_state != state_identity(g)) {
    throw stale_result_error(
        concat_message("dyng: ", function, ": the result was computed on another ", noun,
                       " (or on an earlier state of a ", noun,
                       " variable that was reassigned since), although both are at version ",
                       g.version(), "; recompute it on this ", noun));
  }
}

/**
 * @brief Record the container state a result matches after compute() or a successful update().
 * @tparam graph_t The container type (graph or hypergraph).
 * @param[out] version     Receives g.version().
 * @param[out] graph_state Receives the graph's state identifier.
 * @param[in]  g           The graph.
 */
template <typename graph_t>
void stamp_result(std::uint64_t& version, std::uint64_t& graph_state, const graph_t& g) {
  version = g.version();
  graph_state = state_identity(g);
}

}  // namespace dyng::detail::framework
