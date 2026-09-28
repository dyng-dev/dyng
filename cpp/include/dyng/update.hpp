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
#include <dyng/core/resources.hpp>
#include <dyng/graph/apply_summary.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>

#include <cstddef>
#include <memory>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace dyng {

namespace detail {

template <typename vertex_t>
struct apply_delta;

/**
 * @brief One result taking part in an update (type-erased; each algorithm implements it).
 *
 * run_update() calls before_apply() of every participant on G_t, commits the batch once, then
 * calls after_apply() of every participant on G_{t+1} (PLAN Section 4.5.1, invariant I1).
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
 * @brief The participant interface of a container type.
 * @tparam container_t The container type.
 */
template <typename container_t>
struct participant_of;

/**
 * @brief The participant interface of graph<V,E,W>.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
struct participant_of<graph<vertex_t, edge_t, weight_t>> {
  using type = update_participant<vertex_t, edge_t, weight_t>;  ///< the interface
};

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
 *         result (vertex operations before 0.3), or a batch array is in device memory and CUDA
 *         is not built.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup core
 */
template <typename container_t, typename batch_view_t, typename... results_t>
auto update(const resources& res, container_t& g, const batch_view_t& batch, results_t&... results)
    -> std::tuple<typename detail::stats_of<results_t>::type...> {
  static_assert(sizeof...(results_t) > 0, "dyng::update() needs at least one result");
  using participant_t = typename detail::participant_of<container_t>::type;
  std::tuple<typename detail::stats_of<results_t>::type...> out;
  std::apply(
      [&](auto&... stats) {
        std::unique_ptr<participant_t> owned[] = {
            detail::update_traits<results_t>::template make_participant<container_t>(results,
                                                                                     stats)...};
        participant_t* raw[sizeof...(results_t)] = {};
        for (std::size_t i = 0; i < sizeof...(results_t); ++i) {
          raw[i] = owned[i].get();
        }
        detail::run_update(res, g, batch, raw, sizeof...(results_t), "update.commit");
      },
      out);
  return out;
}

/**
 * @brief Apply `batch` to `g` once and bring a run-time list of results of one type up to date
 *        (for example the K objectives of a multi-objective graph).
 *
 * The same contract as update(res, g, batch, results...), for a list whose length is known only at
 * run time.
 *
 * @tparam container_t  The container type (graph<V,E,W>).
 * @tparam batch_view_t The batch view type (edge_batch_view<V,W>).
 * @tparam result_t     The result type (e.g. sssp::result<V>).
 * @param[in]     res     Execution resources.
 * @param[in,out] g       The container; its version increases by one.
 * @param[in]     batch   The batch (any memory space; read on the host in this release, so arrays
 *                        in device memory are copied once under res.get_copy_policy()).
 * @param[in,out] results Pointers to the results to update (host memory, none null).
 * @return One stats object per result, in order.
 * @throws stale_result_error     if a result does not match `g` (its version, or the graph state
 *         it was computed on).
 * @throws invalid_argument_error if a pointer is null or a result is listed twice, the batch is
 *         invalid, or a batch array must be copied to the host and the copy policy is
 *         copy_policy::error.
 * @throws not_supported_error    if the backend of `res` cannot apply the batch or update a
 *         result (vertex operations before 0.3), or a batch array is in device memory and CUDA
 *         is not built.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup core
 */
template <typename container_t, typename batch_view_t, typename result_t>
auto update_each(const resources& res, container_t& g, const batch_view_t& batch,
                 array_view<result_t* const> results)
    -> std::vector<typename detail::stats_of<result_t>::type> {
  using participant_t = typename detail::participant_of<container_t>::type;
  std::vector<typename detail::stats_of<result_t>::type> out(results.size());
  std::vector<std::unique_ptr<participant_t>> owned;
  std::vector<participant_t*> raw;
  owned.reserve(results.size());
  raw.reserve(results.size());
  for (std::size_t i = 0; i < results.size(); ++i) {
    DYNG_EXPECTS(results[i] != nullptr, "dyng::update_each: result ", i, " is null");
    owned.push_back(detail::update_traits<result_t>::template make_participant<container_t>(
        *results[i], out[i]));
    raw.push_back(owned.back().get());
  }
  detail::run_update(res, g, batch, raw.data(), raw.size(), "update.commit");
  return out;
}

}  // namespace dyng
