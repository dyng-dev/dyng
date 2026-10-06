// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file update.hpp
 * @brief dyng::update(res, g, batch, results...): apply one batch and update several results.
 * @ingroup core
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>

#include <cstddef>
#include <memory>
#include <new>
#include <stdexcept>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng {

namespace detail {

template <typename vertex_t>
struct apply_delta;
template <typename vertex_t>
struct normalized_batch;

/**
 * @brief One result taking part in an update (type-erased; each algorithm implements it).
 *
 * run_update() calls before_apply() of every participant on G_t, commits the batch once, then
 * calls after_apply() of every participant on G_{t+1} (invariant I1: subtract on G_t,
 * add on G_{t+1}). The
 * participant of an algorithm is the framework's adapter of its problem (problem_participant in
 * the library's cpp/src/framework/composition.hpp): before_apply() runs the problem's Steps 0 and
 * 1a through its update_enactor, after_apply() its Steps 1b and 2.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
class update_participant {
 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;               ///< the container
  using batch_type = edge_batch_view<vertex_t, weight_t>;             ///< the batch view
  update_participant() = default;                                     ///< default
  update_participant(const update_participant&) = delete;             ///< not copyable
  update_participant& operator=(const update_participant&) = delete;  ///< not copyable
  update_participant(update_participant&&) = delete;                  ///< not movable
  update_participant& operator=(update_participant&&) = delete;       ///< not movable
  virtual ~update_participant() = default;                            ///< destructor

  /**
   * @brief The result object this participant updates (to reject duplicates).
   * @return Its address.
   */
  [[nodiscard]] virtual const void* target() const noexcept = 0;

  /**
   * @brief Steps 0 and 1a on G_t: validate (version, batch) and prepare; mutates nothing visible.
   * @param[in] res   Execution resources.
   * @param[in] g     The graph before the batch.
   * @param[in] batch The batch.
   */
  virtual void before_apply(const resources& res, const graph_type& g, const batch_type& batch) = 0;

  /**
   * @brief Steps 1b and 2 on G_{t+1}: bring the result up to date.
   * @param[in] res     Execution resources.
   * @param[in] g       The graph after the batch.
   * @param[in] summary What the commit did.
   * @param[in] delta   The effective changes of the commit.
   */
  virtual void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                           const apply_delta<vertex_t>& delta) = 0;

  /**
   * @brief Mark the result unusable after after_apply() threw (later updates throw).
   */
  virtual void poison() noexcept = 0;

  /**
   * @brief Receive Step 0 of batch_semantics::as_sets (the normalized batch of G_t), which
   *        run_update() computes once for every participant and the commit; called before
   *        before_apply(), with nullptr for graphs without set semantics. The object lives until
   *        run_update() returns.
   * @param[in] normalized The normalized batch, or nullptr.
   */
  virtual void use_normalized(const normalized_batch<vertex_t>* normalized) noexcept {
    (void)normalized;
  }

  /**
   * @brief Whether after_apply() reads what the commit prepares for the engines of G_{t+1}: the
   *        in-edges on the host backends, the device copy on cuda (graph_access::prepare()).
   *
   * run_update() prepares the graph inside the commit only if some participant reads it, so an
   * update of results that read only the out-edges (cycle_count on the host backends) does not
   * pay a transposition per batch.
   * @return true (the default).
   */
  [[nodiscard]] virtual bool reads_prepared_graph() const noexcept {
    return true;
  }
};

/**
 * @brief Run one update over several participants: every before_apply() on G_t, one commit,
 *        then every after_apply() on G_{t+1}.
 *
 * If a before_apply() or the commit throws, nothing was changed. If an after_apply() throws, that
 * result is poisoned, the remaining participants still run, and the first exception is rethrown.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 * @param[in]     res          Execution resources.
 * @param[in,out] g            The graph (version + 1).
 * @param[in]     batch        The batch.
 * @param[in]     participants The participants (distinct targets).
 * @param[in]     count        Number of participants.
 * @param[in]     commit_stage Profiler stage name of the commit (e.g. "sssp.commit").
 * @return What the commit did.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary run_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         update_participant<vertex_t, edge_t, weight_t>* const* participants,
                         std::size_t count, std::string_view commit_stage);

/**
 * @brief Maps a result type to its stats type and its participant; each algorithm header
 *        specializes it.
 * @tparam result_t The result type.
 */
template <typename result_t>
struct update_traits;

/**
 * @brief How dyng::update() drives a container type: its participant interface, its batch view
 *        type and its run_update(). The customization point of the unified update: a container
 *        header specializes it (graph<V,E,W> below; the hypergraph in 0.3), so update() and
 *        update_each() need no change for a new container. The primary template marks a type that
 *        is not a container of dyng::update().
 * @tparam container_t The container type.
 */
template <typename container_t>
struct participant_of {
  /// Whether dyng::update() accepts this container type.
  static constexpr bool supported = false;
};

/**
 * @brief The participant interface, batch view and run_update() of graph<V,E,W>.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct participant_of<graph<vertex_t, edge_t, weight_t>> {
  static constexpr bool supported = true;                       ///< a container of update()
  using type = update_participant<vertex_t, edge_t, weight_t>;  ///< the interface
  using batch_type = edge_batch_view<vertex_t, weight_t>;       ///< the batch view it takes

  /**
   * @brief Run one update over the participants (run_update() of graphs).
   * @param[in]     res          Execution resources.
   * @param[in,out] g            The graph.
   * @param[in]     batch        The batch.
   * @param[in]     participants The participants.
   * @param[in]     count        Number of participants.
   * @param[in]     commit_stage Profiler stage name of the commit.
   * @return What the commit did.
   */
  static apply_summary run(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                           const batch_type& batch, type* const* participants, std::size_t count,
                           std::string_view commit_stage) {
    return run_update(res, g, batch, participants, count, commit_stage);
  }
};

/**
 * @brief The plain-English checks of dyng::update()'s container and batch arguments.
 * @tparam container_t  The container type.
 * @tparam batch_view_t The batch argument's type.
 * @return true (the checks are static_asserts).
 */
template <typename container_t, typename batch_view_t>
constexpr bool check_update_arguments() {
  static_assert(participant_of<container_t>::supported,
                "dyng::update: the container is not one dyng::update() drives (in 0.1: "
                "dyng::graph<V, E, W>)");
  if constexpr (participant_of<container_t>::supported) {
    static_assert(std::is_same_v<batch_view_t, typename participant_of<container_t>::batch_type>,
                  "dyng::update: the batch must be the container's batch view, "
                  "edge_batch_view<V, W> of the graph's vertex and weight types: pass "
                  "`batch.view()` of an edge_batch");
  }
  return true;
}

/**
 * @brief The stats type of a result type.
 * @tparam result_t The result type.
 */
template <typename result_t>
struct stats_of {
  using type = typename update_traits<result_t>::stats_type;  ///< the stats type
};

}  // namespace detail

/**
 * @brief Apply `batch` to `g` once and bring every result up to date.
 *
 * All before-apply work of every result runs on G_t first, then the batch is applied once, then
 * every result is updated on G_{t+1}. Every result must match `g.version()`, and no result may be
 * passed twice. The postcondition of each algorithm's own update() holds for each result (for
 * sssp: equal to sssp::compute() on the new graph).
 *
 * Thread safety: the call changes `g` and every result, so it must not overlap any other call on
 * `g` or on one of the results (graph, "Thread safety"); updates of distinct graphs may run
 * concurrently.
 *
 * @tparam container_t  The container type (graph<V,E,W>).
 * @tparam batch_view_t The batch view type (edge_batch_view<V,W>).
 * @tparam results_t    The result types (e.g. sssp::result<V>).
 * @param[in]     res     Execution resources.
 * @param[in,out] g       The container; its version increases by one.
 * @param[in]     batch   The batch (any memory space; read on the host in this release, so arrays
 *                        in device memory are copied once under res.get_copy_policy()).
 * @param[in,out] results The results to update.
 * @return One stats object per result, in order.
 * @throws stale_result_error     if a result does not match `g` (its version, or the graph state
 *         it was computed on).
 * @throws invalid_argument_error if a result is passed twice, the batch is invalid, or a batch
 *         array must be copied to the host and the copy policy is copy_policy::error.
 * @throws not_supported_error    if the backend of `res` cannot apply the batch or update a
 *         result (vertex operations before 0.4), or a batch array is in device memory and CUDA
 *         is not built.
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws error                  any other exception of an algorithm's own update() (for example
 *         cycle_count's internal_error or capacity_error, a cuda_error); the first one is rethrown.
 * @sync
 * @guarantee Strong for every error found before the batch is applied (the before-apply work of
 *            every result runs on G_t before the commit): the graph and every result are
 *            unchanged. Basic for an error after the commit: the graph holds the new version, the
 *            result whose update threw is poisoned (every later use throws stale_result_error),
 *            the other results are still brought up to date, and the first exception is rethrown.
 * @ingroup core
 */
template <typename container_t, typename batch_view_t, typename... results_t>
auto update(const resources& res, container_t& g, const batch_view_t& batch, results_t&... results)
    -> std::tuple<typename detail::stats_of<results_t>::type...> {
  static_assert(sizeof...(results_t) > 0, "dyng::update() needs at least one result");
  static_assert(detail::check_update_arguments<container_t, batch_view_t>());
  using participant_t = typename detail::participant_of<container_t>::type;
  std::tuple<typename detail::stats_of<results_t>::type...> out;
  try {
    std::apply(
        [&](auto&... stats) {
          std::unique_ptr<participant_t> owned[] = {
              detail::update_traits<results_t>::template make_participant<container_t>(results,
                                                                                       stats)...};
          participant_t* raw[sizeof...(results_t)] = {};
          for (std::size_t i = 0; i < sizeof...(results_t); ++i) {
            raw[i] = owned[i].get();
          }
          detail::participant_of<container_t>::run(res, g, batch, raw, sizeof...(results_t),
                                                   "update.commit");
        },
        out);
  } catch (const std::bad_alloc& e) {
    detail::throw_host_allocation_failure("dyng::update (the participants)", e.what());
  } catch (const std::length_error& e) {
    detail::throw_host_allocation_failure("dyng::update (the participants)", e.what());
  }
  return out;
}

/**
 * @brief Apply `batch` to `g` once and bring a run-time list of results of one type up to date
 *        (for example the K objectives of a multi-objective graph).
 *
 * The same contract as update(res, g, batch, results...), for a list whose length is known only at
 * run time. The list must not be empty (the variadic update() rejects zero results at compile
 * time; here it is invalid_argument_error, checked before anything changes).
 *
 * @tparam container_t  The container type (graph<V,E,W>).
 * @tparam batch_view_t The batch view type (edge_batch_view<V,W>).
 * @tparam result_t     The result type (e.g. sssp::result<V>).
 * @param[in]     res     Execution resources.
 * @param[in,out] g       The container; its version increases by one.
 * @param[in]     batch   The batch (any memory space; read on the host in this release, so arrays
 *                        in device memory are copied once under res.get_copy_policy()).
 * @param[in,out] results Pointers to the results to update (host memory, at least one, none
 *                        null).
 * @return One stats object per result, in order.
 * @throws stale_result_error     if a result does not match `g` (its version, or the graph state
 *         it was computed on).
 * @throws invalid_argument_error if the list is empty or not in host memory, a pointer is null or
 *         a result is listed twice, the batch is invalid, or a batch array must be copied to the
 *         host and the copy policy is copy_policy::error.
 * @throws not_supported_error    if the backend of `res` cannot apply the batch or update a
 *         result (vertex operations before 0.4), or a batch array is in device memory and CUDA
 *         is not built.
 * @throws out_of_memory_error    if host or device memory cannot be allocated.
 * @throws error                  any other exception of an algorithm's own update() (for example
 *         cycle_count's internal_error or capacity_error, a cuda_error); the first one is rethrown.
 * @sync
 * @guarantee Strong for every error found before the batch is applied (the before-apply work of
 *            every result runs on G_t before the commit): the graph and every result are
 *            unchanged. Basic for an error after the commit: the graph holds the new version, the
 *            result whose update threw is poisoned (every later use throws stale_result_error),
 *            the other results are still brought up to date, and the first exception is rethrown.
 * @ingroup core
 */
template <typename container_t, typename batch_view_t, typename result_t>
auto update_each(const resources& res, container_t& g, const batch_view_t& batch,
                 array_view<result_t* const> results)
    -> std::vector<typename detail::stats_of<result_t>::type> {
  static_assert(detail::check_update_arguments<container_t, batch_view_t>());
  using participant_t = typename detail::participant_of<container_t>::type;
  DYNG_EXPECTS(is_host_accessible(results.space()),
               "dyng::update_each: the list of results must be in host memory (it is in ",
               to_string(results.space()), ")");
  DYNG_EXPECTS(!results.empty(),
               "dyng::update_each: the list of results is empty (applying a batch without "
               "updating any result would leave every result of the graph stale; use g.apply())");
  for (std::size_t i = 0; i < results.size(); ++i) {
    DYNG_EXPECTS(results[i] != nullptr, "dyng::update_each: result ", i, " is null");
  }
  try {
    std::vector<typename detail::stats_of<result_t>::type> out(results.size());
    std::vector<std::unique_ptr<participant_t>> owned;
    std::vector<participant_t*> raw;
    owned.reserve(results.size());
    raw.reserve(results.size());
    for (std::size_t i = 0; i < results.size(); ++i) {
      owned.push_back(detail::update_traits<result_t>::template make_participant<container_t>(
          *results[i], out[i]));
      raw.push_back(owned.back().get());
    }
    detail::participant_of<container_t>::run(res, g, batch, raw.data(), raw.size(),
                                             "update.commit");
    return out;
  } catch (const std::bad_alloc& e) {
    detail::throw_host_allocation_failure(
        detail::concat_message("dyng::update_each (", results.size(), " results)"), e.what());
  } catch (const std::length_error& e) {
    detail::throw_host_allocation_failure(
        detail::concat_message("dyng::update_each (", results.size(), " results)"), e.what());
  }
}

/**
 * @brief update_each() for a view of mutable pointers, as `host_view(std::vector<result_t*>&)`
 *        gives (the same call; the element type does not deduce through the conversion).
 *
 * @tparam container_t  The container type (graph<V,E,W>).
 * @tparam batch_view_t The batch view type (edge_batch_view<V,W>).
 * @tparam result_t     The result type (e.g. sssp::result<V>).
 * @param[in]     res     Execution resources.
 * @param[in,out] g       The container; its version increases by one.
 * @param[in]     batch   The batch (any memory space, as for update_each()).
 * @param[in,out] results Pointers to the results to update (host memory, none null).
 * @return One stats object per result, in order.
 * @throws stale_result_error     as update_each().
 * @throws invalid_argument_error as update_each().
 * @throws not_supported_error    as update_each().
 * @throws out_of_memory_error    as update_each().
 * @throws error                  as update_each().
 * @sync
 * @guarantee As update_each().
 * @ingroup core
 */
template <typename container_t, typename batch_view_t, typename result_t>
auto update_each(const resources& res, container_t& g, const batch_view_t& batch,
                 array_view<result_t*> results)
    -> std::vector<typename detail::stats_of<result_t>::type> {
  return update_each<container_t, batch_view_t, result_t>(res, g, batch,
                                                          array_view<result_t* const>(results));
}

}  // namespace dyng
