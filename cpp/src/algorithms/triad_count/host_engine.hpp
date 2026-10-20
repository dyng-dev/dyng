// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (countWedgeShape, forEachWedge, TriangleBody,
// PairBody, keyKernel: the wedges of a full count and of a frontier phase as loops)
/**
 * @file host_engine.hpp
 * @brief The incremental host engine of triad_count (sequential.cpp, openmp.cpp): ESCHER-GPU's
 *        wedge engine transliterated to loops over the internal row views in host memory, with the
 *        CUDA engine's operations (engine/count_ops.hpp, motif_ops.hpp, stathyper_ops.hpp), its
 *        exactly-once rule (engine/ownership.hpp) and its shared pieces (engine/phases.hpp), run by
 *        operators::count_owned (count_delta.hpp) on an executor (M9a step 4, ADR 0054; the
 *        phases as unsigned magnitudes, ADR 0057).
 *
 * One count is a pattern's shape over a set of source keys, the work of one source being what
 * the device engine's threads do for that source's wedges (engine/wedge.cuh):
 *   - full mode (compute()): the sources are the keys 1..full_count_keys(); a source a takes its
 *     out-wedges (a, b), b > a (`out[a]`); a triangle (a, b, k) is closed by the k > b common to
 *     the rows of a after b and of b after `out[b]`; nothing is filtered;
 *   - frontier mode (a phase of update()): the sources are the frontier's keys; a source f takes
 *     every wedge (f, b) of its row (from `loc[f]`), skips the wedge when b is in the frontier and
 *     b < f, and skips the closing k when k is in the frontier and k < f
 *     (ownership::min_member, invariant I2), so a triad through several frontier members is
 *     counted once, at its smallest;
 *   - the shapes: triangles (h_motif: motif_op's 30 bins; stathyper_type3: type3_op), pairs
 *     (stathyper_type2: every wedge is one pair), keys (stathyper_type1: C(|e|, 3) of each source);
 *   - the early outs of ESCHER-GPU's countWedgeShape: no sources (an empty frontier), and sources
 *     without a wedge (the wedges are counted first); a count that runs is one pass.
 * The rows are the store's encoding (h2v members + 1 by key = id + 1, h2h neighbour keys); the
 * operations see 0-based ids. A loop over h2v keys never goes past views.h2v.max_key (the full
 * count's type-1 sources are 1..views.h2v.max_key; the frontiers hold live ids); the flags cover
 * the line graph's keys 0..views.h2h.max_key.
 *
 * count_owned gives every thread private bins and adds them up in thread order after the pass,
 * so 1, 8 or 56 OpenMP threads give the same counts (a static partition of the sources). A phase
 * is counted as unsigned magnitudes with its sign beside them (triad_phase_bins; ADR 0057: a
 * phase of 2^63 or more triads in a bin is exact), as on cuda. A pattern of no class (motif_op's
 * bin -1) and a count past 2^64 - 1 (an operation's product or sum, a thread's bin, a total) are
 * reported by count_owned and raised here: internal_error and capacity_error.
 */
#pragma once

#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/ownership.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/engine/sets.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "framework/policies.hpp"
#include "hypergraph/row_views.hpp"
#include "operators/count_delta.hpp"

#include <dyng/core/error.hpp>
#include <dyng/triad_count.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace dyng::detail::triad_host {

/// The rows handle of a state's h2v view.
inline triad::hyperedge_rows rows_of(const hyper::row_views& views) {
  return triad::hyperedge_rows{views.h2v.flat, views.h2v.loc, views.h2v.len};
}

/// The key of source s: the frontier's keys[s], or s + 1 in a full count (sourceKey).
inline int source_key(const int* keys, std::int64_t s) noexcept {
  return keys != nullptr ? keys[s] : static_cast<int>(s) + 1;
}

/// The first wedge of source a: every neighbour in a frontier phase, the out-neighbours (b > a)
/// in a full count (wedgeBegin).
inline int wedge_begin(const escher::row_view& adj, int a, bool all_neighbours) noexcept {
  return all_neighbours ? adj.loc[a] : adj.out[a];
}

/**
 * @brief The wedges of n sources (count_wedges_kernel and the scan's total): zero is the second
 *        early out.
 * @param[in] adj  The line graph's rows.
 * @param[in] keys The frontier's keys, or nullptr for the keys 1..n (out-wedges only).
 * @param[in] n    The sources.
 * @return Their wedges.
 */
inline std::int64_t count_wedges(const escher::row_view& adj, const int* keys, int n) noexcept {
  const bool all_neighbours = keys != nullptr;
  std::int64_t wedges = 0;
  for (int s = 0; s < n; ++s) {
    const int a = source_key(keys, s);
    wedges += adj.loc[a] + adj.len[a] - wedge_begin(adj, a, all_neighbours);
  }
  return wedges;
}

/**
 * @brief The work of one source for the triangle patterns (forEachWedge with TriangleBody): every
 *        wedge (a, b) of the source that passes the frontier filter, closed into its triangles
 *        (a, b, k), k > b, that pass the closing filter.
 * @tparam op_t The operation (motif_op, type3_op).
 */
template <class op_t>
struct triangle_item {
  escher::row_view adj;              ///< the line graph's rows
  triad::hyperedge_rows rows;        ///< the h2v rows
  const int* keys;                   ///< the frontier's keys (nullptr: the full count)
  const unsigned char* in_frontier;  ///< the frontier's flags (nullptr: the full count)

  /// Source s (operators::count_owned's element), its counts added to `bins`.
  template <class bins_t>
  void operator()(std::int64_t s, framework::ownership::min_member /*rule*/, bins_t& bins) const {
    const bool all_neighbours = keys != nullptr;
    const int a = source_key(keys, s);
    const int end_a = adj.loc[a] + adj.len[a];
    op_t op{};
    for (int p = wedge_begin(adj, a, all_neighbours); p < end_a; ++p) {
      const int b = adj.flat[p];
      if (triad::min_member_skips_wedge(in_frontier, b, a)) {
        continue;
      }
      op.begin_wedge(rows, a - 1, b - 1);
      int x = p + 1;
      int y = adj.out[b];
      const int end_b = adj.loc[b] + adj.len[b];
      while (x < end_a && y < end_b) {
        const int ka = adj.flat[x];
        const int kb = adj.flat[y];
        if (ka == kb) {
          if (!triad::min_member_skips_closing(in_frontier, ka, a)) {
            op.triangle(rows, ka - 1, bins);
          }
          ++x;
          ++y;
        } else if (ka < kb) {
          ++x;
        } else {
          ++y;
        }
      }
    }
    op.flush(bins);
  }
};

/**
 * @brief The work of one source for the pair pattern (forEachWedge with PairBody): every wedge
 *        (a, b) of the source that passes the frontier filter is one pair.
 * @tparam op_t The operation (type2_op).
 */
template <class op_t>
struct pair_item {
  escher::row_view adj;              ///< the line graph's rows
  triad::hyperedge_rows rows;        ///< the h2v rows
  const int* keys;                   ///< the frontier's keys (nullptr: the full count)
  const unsigned char* in_frontier;  ///< the frontier's flags (nullptr: the full count)

  /// Source s, its counts added to `bins`.
  template <class bins_t>
  void operator()(std::int64_t s, framework::ownership::min_member /*rule*/, bins_t& bins) const {
    const bool all_neighbours = keys != nullptr;
    const int a = source_key(keys, s);
    const int end_a = adj.loc[a] + adj.len[a];
    op_t op{};
    for (int p = wedge_begin(adj, a, all_neighbours); p < end_a; ++p) {
      const int b = adj.flat[p];
      if (triad::min_member_skips_wedge(in_frontier, b, a)) {
        continue;
      }
      op.pair(rows, a - 1, b - 1, bins);
    }
    op.flush(bins);
  }
};

/**
 * @brief The work of one key for the key pattern (keyKernel): the source's own count (no wedges,
 *        no filter: a key is counted once, by itself).
 * @tparam op_t The operation (choose3_op).
 */
template <class op_t>
struct key_item {
  triad::hyperedge_rows rows;  ///< the h2v rows
  const int* keys;             ///< the frontier's keys (nullptr: the full count)

  /// Source s, its count added to `bins`.
  template <class bins_t>
  void operator()(std::int64_t s, framework::ownership::min_member /*rule*/, bins_t& bins) const {
    op_t op{};
    op.key(rows, source_key(keys, s) - 1, bins);
    op.flush(bins);
  }
};

/// Raise count_owned's error bits (after the pass: the result is poisoned by the caller).
inline void raise_status(std::uint32_t status, const char* what) {
  if ((status & operators::count_invalid_bin) != 0) {
    DYNG_FAIL(what, ": a triangle of the line graph has a Venn pattern of no h-motif class");
  }
  if ((status & operators::count_overflow) != 0) {
    throw capacity_error(concat_message("dyng: ", what, ": a count exceeds the 64-bit range"));
  }
}

/**
 * @brief One count of a pattern over n sources (countWedgeShape on the host): the early outs,
 *        then `run(item)` with the source work of the pattern's shape.
 * @param[in] kind  The pattern.
 * @param[in] views The row views (host memory).
 * @param[in] keys  The frontier's keys, or nullptr for the keys 1..n.
 * @param[in] n     The sources.
 * @param[in] flags The frontier's flags (nullptr: the full count).
 * @param[in] run   Called once with the source work of the pattern's shape (a triangle_item,
 *                  pair_item or key_item): the pass (count_owned with the op's
 *                  bins, bins_of).
 * @return The outcome.
 */
template <typename run_t>
triad::wedge::launch dispatch(triad_count::pattern kind, const hyper::row_views& views,
                              const int* keys, int n, const unsigned char* flags, run_t&& run) {
  if (n <= 0) {
    return triad::wedge::launch::no_sources;
  }
  const triad::hyperedge_rows rows = rows_of(views);
  if (kind == triad_count::pattern::stathyper_type1) {
    run(key_item<triad::choose3_op>{rows, keys});
    return triad::wedge::launch::kernel;
  }
  if (count_wedges(views.h2h, keys, n) == 0) {
    return triad::wedge::launch::no_wedges;
  }
  switch (kind) {
    case triad_count::pattern::h_motif:
      run(triangle_item<triad::motif_op>{views.h2h, rows, keys, flags});
      break;
    case triad_count::pattern::stathyper_type2:
      run(pair_item<triad::type2_op>{views.h2h, rows, keys, flags});
      break;
    case triad_count::pattern::stathyper_type3:
      run(triangle_item<triad::type3_op>{views.h2h, rows, keys, flags});
      break;
    case triad_count::pattern::stathyper_type1:
      break;
  }
  return triad::wedge::launch::kernel;
}

/// The bins of an item's operation.
template <typename item_t>
struct bins_of;
/// The bins of a triangle item.
template <class op_t>
struct bins_of<triangle_item<op_t>> {
  static constexpr int value = op_t::num_bins;  ///< the op's bins
};
/// The bins of a pair item.
template <class op_t>
struct bins_of<pair_item<op_t>> {
  static constexpr int value = op_t::num_bins;  ///< the op's bins
};
/// The bins of a key item.
template <class op_t>
struct bins_of<key_item<op_t>> {
  static constexpr int value = op_t::num_bins;  ///< the op's bins
};

/**
 * @brief The words of scratch the host engine's counts need on an executor (the 30 bins of
 *        h_motif per thread).
 * @tparam exec_t The executor.
 * @param[in] exec The executor.
 * @return The words.
 */
template <typename exec_t>
std::size_t scratch_words(const exec_t& exec) noexcept {
  return operators::count_scratch_words<triad_count::max_bins>(exec);
}

/**
 * @brief The full count of compute(): every out-wedge of the keys 1..full_count_keys() (no
 *        frontier, nothing filtered), the totals into `counts` (zeroed first).
 * @tparam exec_t The executor.
 * @param[in]  exec    The executor.
 * @param[in]  kind    The pattern.
 * @param[in]  views   The row views (host memory).
 * @param[in]  rule    The exactly-once rule (a full count has no frontier: nothing is skipped).
 * @param[out] scratch scratch_words(exec) words.
 * @param[out] counts  The counts.
 * @return The outcome.
 * @throws internal_error for a triad of no class; capacity_error past 2^64 - 1.
 */
template <typename exec_t>
triad::wedge::launch count_full(const exec_t& exec, triad_count::pattern kind,
                                const hyper::row_views& views,
                                framework::ownership::min_member rule, std::uint64_t* scratch,
                                triad_bins& counts) {
  counts.fill(0);
  std::uint32_t status = 0;
  const triad::wedge::launch outcome = dispatch(
      kind, views, nullptr, triad::full_count_keys(kind, views), nullptr, [&](const auto& item) {
        constexpr int bins = bins_of<std::decay_t<decltype(item)>>::value;
        status = operators::count_owned<bins>(exec, rule, triad::full_count_keys(kind, views), item,
                                              scratch, counts.data());
      });
  raise_status(status, "triad_count::compute");
  return outcome;
}

/**
 * @brief One frontier phase of update(): the triads (pairs, keys) through the frontier's keys,
 *        each counted once at its smallest frontier member (`rule`), as unsigned magnitudes with
 *        the phase's sign beside them (ADR 0057; count_owned, the sign applied by apply_delta()).
 * @tparam exec_t The executor.
 * @param[in]     exec    The executor.
 * @param[in]     kind    The pattern.
 * @param[in]     views   G_t's row views (the subtract phase) or G_{t+1}'s (the add phase).
 * @param[in]     keys    The frontier's keys (ids + 1).
 * @param[in]     n       Their number.
 * @param[in]     flags   A flag per line-graph key 0..views.h2h.max_key (nullptr for type 1).
 * @param[in]     s       The phase's sign (minus: G_t and the deletions; plus: G_{t+1} and the
 *                        ASSIGNED inserted ids).
 * @param[in]     rule    The exactly-once rule (invariant I2), passed explicitly.
 * @param[out]    scratch scratch_words(exec) words.
 * @param[out]    out     The phase: `s` and its triads per bin.
 * @return The outcome.
 * @throws internal_error for a triad of no class; capacity_error if a count exceeds 2^64 - 1.
 */
template <typename exec_t>
triad::wedge::launch count_phase(const exec_t& exec, triad_count::pattern kind,
                                 const hyper::row_views& views, const int* keys, int n,
                                 const unsigned char* flags, framework::sign s,
                                 framework::ownership::min_member rule, std::uint64_t* scratch,
                                 triad_phase_bins& out) {
  std::uint32_t status = 0;
  out.sign = s;
  out.counts.fill(0);
  const triad::wedge::launch outcome = dispatch(kind, views, keys, n, flags, [&](const auto& item) {
    constexpr int bins = bins_of<std::decay_t<decltype(item)>>::value;
    status = operators::count_owned<bins>(exec, rule, n, item, scratch, out.counts.data());
  });
  raise_status(status, "triad_count::update");
  return outcome;
}

/**
 * @brief Without an ownership rule: a compile error (I2: exactly-once counting has no default
 *        rule; cpp/tests/compile_fail/count_delta_ownership.cpp matches the message).
 * @tparam exec_t The executor.
 * @return Nothing (never compiles).
 */
template <typename exec_t>
triad::wedge::launch count_phase(const exec_t&, triad_count::pattern, const hyper::row_views&,
                                 const int*, int, const unsigned char*, framework::sign,
                                 std::uint64_t*, triad_phase_bins&) {
  static_assert(operators::count_detail::always_false<exec_t>,
                "triad_host::count_phase needs an ownership rule: pass "
                "framework::ownership::min_member{} after the sign (invariant I2: exactly-once "
                "counting has no default rule)");
  return triad::wedge::launch::no_sources;
}

}  // namespace dyng::detail::triad_host
