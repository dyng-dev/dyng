// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file composition.cpp
 * @brief run_update(): the fixed order of one update over several results (PLAN Section 4.5.2,
 *        "Composition"): every before_apply() on G_t, one commit, every after_apply() on G_{t+1}.
 *
 * This is the first piece of the framework layer; the problem hooks and enactors are extracted in
 * M3 once a second algorithm (cycle_count) exists (rule of two).
 */
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/update.hpp>

#include <cstddef>
#include <exception>

namespace dyng::detail {

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary run_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         update_participant<vertex_t, edge_t, weight_t>* const* participants,
                         std::size_t count, std::string_view commit_stage) {
  DYNG_EXPECTS(count == 0 || participants != nullptr, "run_update: no participant array");
  for (std::size_t i = 0; i < count; ++i) {
    DYNG_EXPECTS(participants[i] != nullptr, "run_update: participant ", i, " is null");
    for (std::size_t j = 0; j < i; ++j) {
      DYNG_EXPECTS(participants[i]->target() != participants[j]->target(), "dyng::update: result ",
                   i, " is the same object as result ", j, "; pass each result once");
    }
  }
  // Steps 0 and 1a on G_t. Nothing is mutated until every participant accepted the batch.
  for (std::size_t i = 0; i < count; ++i) {
    participants[i]->before_apply(res, g, batch);
  }
  // Commit: G_t -> G_{t+1}, exactly once.
  apply_delta<vertex_t> delta;
  apply_summary summary;
  {
    scoped_stage stage(res, commit_stage);
    summary = graph_access::apply(res, g, batch, &delta);
  }
  // Steps 1b and 2 on G_{t+1}. A failing participant is poisoned; the others still run.
  std::exception_ptr first_error;
  for (std::size_t i = 0; i < count; ++i) {
    try {
      participants[i]->after_apply(res, g, summary, delta);
    } catch (...) {
      participants[i]->poison();
      if (!first_error) {
        first_error = std::current_exception();
      }
    }
  }
  if (first_error) {
    std::rethrow_exception(first_error);
  }
  return summary;
}

#define DYNG_INSTANTIATE_RUN_UPDATE(V, E, W)                           \
  template apply_summary run_update<V, E, W>(                          \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, \
      update_participant<V, E, W>* const*, std::size_t, std::string_view);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_RUN_UPDATE)
#undef DYNG_INSTANTIATE_RUN_UPDATE

}  // namespace dyng::detail
