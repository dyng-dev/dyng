// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_update.cpp
 * @brief run_update() of hypergraphs (ADR 0043): the fixed order of one dyng::update() over
 *        several results of one hypergraph, through the hypergraph's transaction
 *        (transaction.hpp): begin (Step 0 once), every before_apply() on G_t, one commit, every
 *        after_apply() on G_{t+1}.
 */
#include "framework/budgets.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/instantiate.hpp"
#include "hypergraph/transaction.hpp"
#include "util/allocation.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstddef>
#include <exception>
#include <optional>
#include <string>
#include <string_view>

namespace dyng::detail {

namespace {

/// "<algo>.normalize" for the commit stage "<algo>.commit" (update.normalize for dyng::update),
/// as for graphs (framework/composition.cpp).
std::string hyper_normalize_stage(std::string_view commit_stage) {
  const std::size_t dot = commit_stage.rfind('.');
  return std::string(commit_stage.substr(0, dot == std::string_view::npos ? 0 : dot)) +
         (dot == std::string_view::npos ? "update" : "") + ".normalize";
}

/// The function a user called, for messages: "<algo>::update" for "<algo>.commit",
/// "dyng::update" for "update.commit" and a name without a dot.
std::string hyper_update_function(std::string_view commit_stage) {
  const std::size_t dot = commit_stage.rfind('.');
  const std::string_view algo = commit_stage.substr(0, dot == std::string_view::npos ? 0 : dot);
  return algo.empty() || algo == "update" ? std::string("dyng::update")
                                          : std::string(algo) + "::update";
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyper_apply_summary run_update(const resources& res, hypergraph<vertex_t, weight_t>& h,
                               const hyperedge_batch_view<vertex_t, weight_t>& batch,
                               hyper_update_participant<vertex_t, weight_t>* const* participants,
                               std::size_t count, std::string_view commit_stage) try {
  DYNG_EXPECTS(count == 0 || participants != nullptr, "run_update: no participant array");
  for (std::size_t i = 0; i < count; ++i) {
    DYNG_EXPECTS(participants[i] != nullptr, "run_update: participant ", i, " is null");
    for (std::size_t j = 0; j < i; ++j) {
      DYNG_EXPECTS(participants[i]->target() != participants[j]->target(), "dyng::update: result ",
                   i, " is the same object as result ", j, "; pass each result once");
    }
  }
  const std::string function = hyper_update_function(commit_stage);
  // begin: the placement and poisoning checks, the batch staged to the host once under the copy
  // policy of `res`, and Step 0 once for every participant and the commit (the transaction).
  std::optional<hyper_transaction<vertex_t, weight_t>> txn;
  {
    const scoped_stage stage(res, hyper_normalize_stage(commit_stage));
    txn.emplace(res, h, batch, function.c_str());
  }
  for (std::size_t i = 0; i < count; ++i) {
    participants[i]->use_normalized(&txn->staged());
  }
  // The line graph of G_t for the participants that read it (line_graph_policy::automatic builds
  // it here, once, and the commit maintains it from then on). The row views of every state are
  // rebuilt by its commit; the public views stay lazy (never built on the apply path).
  bool line_graph = false;
  for (std::size_t i = 0; i < count; ++i) {
    line_graph = line_graph || participants[i]->reads_prepared_graph();
  }
  if (line_graph) {
    hypergraph_access::require_line_graph(h, function.c_str());
  }
  // Steps 0 and 1a on G_t. Nothing is mutated until every participant accepted the batch.
  for (std::size_t i = 0; i < count; ++i) {
    participants[i]->before_apply(res, h, txn->requested());
  }
  // Commit: G_t -> G_{t+1}, exactly once.
  hyper_apply_summary summary;
  {
    scoped_stage stage(res, commit_stage);
    const framework::budget_scope commit;  // container growth: reported, never failed (I9)
    {
      const scoped_stage apply_stage(res, "hypergraph.apply");
      summary = txn->commit();
    }
    framework::record_commit_counts(commit.used());
  }
  // Steps 1b and 2 on G_{t+1}. A failing participant is poisoned; the others still run.
  const hyper_applied& applied = txn->applied();
  std::exception_ptr first_error;
  for (std::size_t i = 0; i < count; ++i) {
    try {
      participants[i]->after_apply(res, h, summary, applied);
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
DYNG_TRANSLATE_ALLOCATION_FAILURE("dyng::update (", h.hyperedge_id_bound(),
                                  " hyperedge ids; batch of ", batch.num_insertions(),
                                  " insertions, ", batch.num_deletions(), " deletions; ", count,
                                  " result(s))")

#define DYNG_INSTANTIATE_HYPER_RUN_UPDATE(V, W)                               \
  template hyper_apply_summary run_update<V, W>(                              \
      const resources&, hypergraph<V, W>&, const hyperedge_batch_view<V, W>&, \
      hyper_update_participant<V, W>* const*, std::size_t, std::string_view);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPER_RUN_UPDATE)
#undef DYNG_INSTANTIATE_HYPER_RUN_UPDATE

}  // namespace dyng::detail
