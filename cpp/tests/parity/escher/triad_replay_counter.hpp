// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay_counter.hpp
 * @brief The counter interface of the triad replay (triad_replay.cpp): what is evaluated after the
 *        build and after every batch, and the counter of experimental::count_local_patterns
 *        (triad_replay_local_patterns.cu, M9b, ADR 0058), which needs nvcc.
 */
#pragma once

#include "parity/escher/triad_replay.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>

namespace dyng::test::triad_replay {

/// The container of the replays.
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
/// Its batch view.
using batch_view = dyng::hyperedge_batch_view<std::int32_t, dyng::unweighted>;

/**
 * The counter evaluated after the build and after every batch. apply() applies the batch (a
 * counter that maintains counts applies it through dyng::update) and returns the summary; counts()
 * gives the counts of the current state.
 */
class counter {
 public:
  counter() = default;
  counter(const counter&) = delete;
  counter& operator=(const counter&) = delete;
  counter(counter&&) = delete;
  counter& operator=(counter&&) = delete;
  virtual ~counter() = default;
  virtual void build(const resources& res, hgraph& h) = 0;
  virtual hyper_apply_summary apply(const resources& res, hgraph& h, const batch_view& b) = 0;
  virtual pattern_counts counts(const resources& res, const hgraph& h) = 0;
  /// The launch counters per (mode, shape) recorded so far (the hook of M9a steps 3-4).
  virtual void launches(std::map<std::string, std::int64_t>& out) const {
    (void)out;
  }
  std::int64_t evaluations = 0;
};

/**
 * @brief The counter `local_patterns` (M9b criterion 3): the four built-in Ops through
 *        experimental::count_local_patterns beside triad_count's results, all updated by one
 *        commit per batch; after every batch the counts, the per-bin deltas, removed / added and
 *        the frontiers must be equal bit for bit, and the counts equal a fresh compute(); the
 *        counts reported are count_local_patterns'.
 * @param[in] opt The replay's options (the patterns, recount).
 * @return The counter.
 */
std::unique_ptr<counter> make_local_patterns_counter(const options& opt);

}  // namespace dyng::test::triad_replay
