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
#include "core/resources_access.hpp"
#include "core/staging.hpp"
#include "framework/workspace.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "graph/normalized_batch.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/update.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>

namespace dyng::detail {

namespace {

/// The normalized batch of one update, leased from the workspace pool (its vectors and device
/// buffers keep their capacity across updates).
template <typename vertex_t>
struct normalized_workspace final : pooled_workspace {
  normalized_batch<vertex_t> batch;  ///< Step 0 of the current update

  [[nodiscard]] std::size_t bytes() const noexcept override {
    return batch.bytes();
  }
};

/// "<algo>.normalize" for the commit stage "<algo>.commit" (update.normalize for dyng::update).
std::string normalize_stage(std::string_view commit_stage) {
  const std::size_t dot = commit_stage.rfind('.');
  return std::string(commit_stage.substr(0, dot == std::string_view::npos ? 0 : dot)) +
         (dot == std::string_view::npos ? "update" : "") + ".normalize";
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary run_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         update_participant<vertex_t, edge_t, weight_t>* const* participants,
                         std::size_t count, std::string_view commit_stage) try {
  DYNG_EXPECTS(count == 0 || participants != nullptr, "run_update: no participant array");
  for (std::size_t i = 0; i < count; ++i) {
    DYNG_EXPECTS(participants[i] != nullptr, "run_update: participant ", i, " is null");
    for (std::size_t j = 0; j < i; ++j) {
      DYNG_EXPECTS(participants[i]->target() != participants[j]->target(), "dyng::update: result ",
                   i, " is the same object as result ", j, "; pass each result once");
    }
  }
  // The batch is read on the host (the participants' checks and the host apply of this release):
  // arrays in device memory are copied once, under the copy policy of `res` (PLAN 4.7.1).
  const host_batch<vertex_t, weight_t> staged(
      res, batch, commit_stage == "sssp.commit" ? "sssp::update" : "dyng::update");
  const edge_batch_view<vertex_t, weight_t>& host = staged.view();
  // Step 0 of set semantics, once for every participant and the commit (ADR 0020): the
  // normalized batch of G_t (graph/normalized_batch.hpp).
  std::optional<workspace_pool::lease<normalized_workspace<vertex_t>>> normalized_lease;
  const normalized_batch<vertex_t>* normalized = nullptr;
  if (g.properties().semantics.as_sets) {
    scoped_stage stage(res, normalize_stage(commit_stage));
    normalized_lease.emplace(
        resources_access::workspaces(res).acquire<normalized_workspace<vertex_t>>(res));
    normalized_batch<vertex_t>& nb = (*normalized_lease)->batch;
    const auto& state = graph_access::impl(g);
    normalize_set_batch(state.host_edges(), host, g.properties(), nb);
    nb.state_id = state.state_id;
    normalized = &nb;
  }
  for (std::size_t i = 0; i < count; ++i) {
    participants[i]->use_normalized(normalized);
  }
  // Steps 0 and 1a on G_t. Nothing is mutated until every participant accepted the batch.
  for (std::size_t i = 0; i < count; ++i) {
    participants[i]->before_apply(res, g, host);
  }
  // Commit: G_t -> G_{t+1}, exactly once.
  apply_delta<vertex_t> delta;
  apply_summary summary;
  {
    scoped_stage stage(res, commit_stage);
    summary = graph_access::apply(res, g, host, &delta, normalized);
    // What the engines read of G_{t+1} is built once here, inside the commit, for every
    // participant: the host in-edges (the graph builds them lazily; MOSP-OpenMP builds its reverse
    // graph in "prepare", before the objectives) or, on the CUDA backend, the device copy (MOSP-CUDA
    // uploads the updated graph and builds its reverse graph on the device in "upload"). Skipped
    // when no participant reads them (cycle_count on the host backends reads the out-edges only).
    bool prepare = false;
    for (std::size_t i = 0; i < count; ++i) {
      prepare = prepare || participants[i]->reads_prepared_graph();
    }
    if (prepare) {
      graph_access::prepare(res, g);
    }
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
DYNG_TRANSLATE_ALLOCATION_FAILURE("dyng::update (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges; batch of ", batch.num_insertions(), " insertions, ",
                                  batch.num_deletions(), " deletions; ", count, " result(s))")

#define DYNG_INSTANTIATE_RUN_UPDATE(V, E, W)                           \
  template apply_summary run_update<V, E, W>(                          \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, \
      update_participant<V, E, W>* const*, std::size_t, std::string_view);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_RUN_UPDATE)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_RUN_UPDATE)
#undef DYNG_INSTANTIATE_RUN_UPDATE

}  // namespace dyng::detail
