// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (countWedgeShape, forEachWedge, TriangleBody,
// PairBody, keyKernel: the wedges of a full count and of a frontier phase as loops)
/**
 * @file host_engine.hpp
 * @brief The incremental host engine of every local pattern (sequential.cpp, openmp.cpp):
 *        ESCHER-GPU's wedge engine transliterated to loops over the internal row views in host
 *        memory, the work of one source being the pattern's (local_patterns::host_source, the
 *        CUDA engine's operations, its exactly-once rule), run by operators::count_owned
 *        (count_delta.hpp) on an executor (M9a step 4, ADR 0054; the phases as unsigned
 *        magnitudes, ADR 0057; one engine for the built-in patterns and the Ops of users,
 *        ADR 0058).
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
 * Since M9b (ADR 0058) the work of one source is the pattern's table's `source` function (the
 * template local_patterns::host_source of <dyng/experimental/detail/local_patterns_engine.hpp>,
 * instantiated for the four built-in operations in builtin_ops.cpp and for an Op of a user in the
 * user's translation unit), called once per source by count_owned through a type-erased item; the
 * loops above moved there unchanged. count_owned runs with max_pattern_bins bins per thread, and
 * the source's own bins handle bounds its bins to the pattern's num_bins.
 *
 * count_owned gives every thread private bins and adds them up in thread order after the pass,
 * so 1, 8 or 56 OpenMP threads give the same counts (a static partition of the sources). A phase
 * is counted as unsigned magnitudes with its sign beside them (triad_phase_bins; ADR 0057: a
 * phase of 2^63 or more triads in a bin is exact), as on cuda. A pattern of no class (motif_op's
 * bin -1) and a count past 2^64 - 1 (an operation's product or sum, a thread's bin, a total) are
 * reported by count_owned and raised here: internal_error and capacity_error.
 */
#pragma once

#include "algorithms/triad_count/engine/ownership.hpp"
#include "algorithms/triad_count/engine/phases.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "framework/policies.hpp"
#include "hypergraph/row_views.hpp"
#include "operators/count_delta.hpp"

#include <dyng/core/error.hpp>
#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/local_patterns.hpp>

#include <cstddef>
#include <cstdint>

namespace dyng::detail::triad_host {

static_assert(local_patterns::host_invalid_bin == operators::count_invalid_bin &&
                  local_patterns::host_overflow == operators::count_overflow,
              "the Op's host bins handle reports with count_owned's status bits");

/// The line graph's rows of the host work (the internal row view's layout).
inline local_patterns::line_rows line_rows_of(const escher::row_view& v) noexcept {
  return local_patterns::line_rows{v.flat, v.loc, v.len, v.out, v.max_key};
}

/// What the work of a source reads: the state's rows, the frontier's keys and flags.
inline local_patterns::host_pass pass_of(const hyper::row_views& views, const int* keys,
                                         const unsigned char* flags) noexcept {
  local_patterns::host_pass p;
  p.adj = line_rows_of(views.h2h);
  p.rows = local_patterns::rows_factory::make(views.h2v.flat, views.h2v.loc, views.h2v.len);
  p.keys = keys;
  p.in_frontier = flags;
  return p;
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
    const int a = local_patterns::host_source_key(keys, s);
    wedges += adj.loc[a] + adj.len[a] - (all_neighbours ? adj.loc[a] : adj.out[a]);
  }
  return wedges;
}

/**
 * @brief The work of one source (operators::count_owned's element): the pattern's `source`
 *        function with the worker's bins.
 */
struct erased_source {
  const local_patterns::kernels* k;       ///< the pattern's table
  const void* op;                         ///< the Op's value
  const local_patterns::host_pass* pass;  ///< what the sources read

  /// Source s, its counts added to `bins` (a host_count_bins of max_pattern_bins bins).
  template <class bins_t>
  void operator()(std::int64_t s, framework::ownership::min_member /*rule*/, bins_t& bins) const {
    k->source(op, *pass, s, bins.bins, bins.status);
  }
};

/// Raise count_owned's error bits (after the pass: the result is poisoned by the caller).
inline void raise_status(const local_patterns::kernels& k, std::uint32_t status, const char* what) {
  if ((status & operators::count_invalid_bin) != 0) {
    if (k.builtin) {
      DYNG_FAIL(what, ": a triangle of the line graph has a Venn pattern of no h-motif class");
    }
    throw invalid_argument_error(concat_message("dyng: ", what, ": the Op named a bin outside [0, ",
                                                k.num_bins, ") (its num_bins)"));
  }
  if ((status & operators::count_overflow) != 0) {
    throw capacity_error(concat_message("dyng: ", what, ": a count exceeds the 64-bit range"));
  }
}

/**
 * @brief One count of a pattern over n sources (countWedgeShape on the host): the early outs,
 *        then `run(n)` (the pass).
 * @param[in] k     The pattern's table.
 * @param[in] views The row views (host memory).
 * @param[in] keys  The frontier's keys, or nullptr for the keys 1..n.
 * @param[in] n     The sources.
 * @param[in] run   Called once when the pass runs.
 * @return The outcome.
 */
template <typename run_t>
triad::wedge::launch dispatch(const local_patterns::kernels& k, const hyper::row_views& views,
                              const int* keys, int n, run_t&& run) {
  if (n <= 0) {
    return triad::wedge::launch::no_sources;
  }
  if (k.pattern_shape == experimental::shape::keys) {
    run();
    return triad::wedge::launch::kernel;
  }
  if (count_wedges(views.h2h, keys, n) == 0) {
    return triad::wedge::launch::no_wedges;
  }
  run();
  return triad::wedge::launch::kernel;
}

/**
 * @brief The words of scratch the host engine's counts need on an executor (max_pattern_bins
 *        bins and a status word per thread).
 * @tparam exec_t The executor.
 * @param[in] exec The executor.
 * @return The words.
 */
template <typename exec_t>
std::size_t scratch_words(const exec_t& exec) noexcept {
  return operators::count_scratch_words<experimental::max_pattern_bins>(exec);
}

/**
 * @brief The full count of compute(): every out-wedge of the keys 1..full_count_keys() (no
 *        frontier, nothing filtered), the totals into `counts` (max_pattern_bins words).
 * @tparam exec_t The executor.
 * @param[in]  exec    The executor.
 * @param[in]  k       The pattern's table.
 * @param[in]  op      The Op's value.
 * @param[in]  views   The row views (host memory).
 * @param[in]  rule    The exactly-once rule (a full count has no frontier: nothing is skipped).
 * @param[out] scratch scratch_words(exec) words.
 * @param[out] counts  The counts.
 * @return The outcome.
 * @throws internal_error for a triad of no class (a built-in pattern); invalid_argument_error for
 *         a bin an Op named outside its range; capacity_error past 2^64 - 1.
 */
template <typename exec_t>
triad::wedge::launch count_full(const exec_t& exec, const local_patterns::kernels& k,
                                const void* op, const hyper::row_views& views,
                                framework::ownership::min_member rule, std::uint64_t* scratch,
                                std::uint64_t* counts) {
  for (int b = 0; b < experimental::max_pattern_bins; ++b) {
    counts[b] = 0;
  }
  std::uint32_t status = 0;
  const int n = triad::full_count_keys(k.pattern_shape, views);
  const local_patterns::host_pass pass = pass_of(views, nullptr, nullptr);
  const triad::wedge::launch outcome = dispatch(k, views, nullptr, n, [&]() {
    status = operators::count_owned<experimental::max_pattern_bins>(
        exec, rule, n, erased_source{&k, op, &pass}, scratch, counts);
  });
  raise_status(k, status, k.builtin ? "triad_count::compute" : "count_local_patterns::compute");
  return outcome;
}

/**
 * @brief One frontier phase of update(): the triads (pairs, keys) through the frontier's keys,
 *        each counted once at its smallest frontier member (`rule`), as unsigned magnitudes with
 *        the phase's sign beside them (ADR 0057; count_owned, the sign applied by apply_delta()).
 * @tparam exec_t The executor.
 * @param[in]     exec    The executor.
 * @param[in]     k       The pattern's table.
 * @param[in]     op      The Op's value.
 * @param[in]     views   G_t's row views (the subtract phase) or G_{t+1}'s (the add phase).
 * @param[in]     keys    The frontier's keys (ids + 1).
 * @param[in]     n       Their number.
 * @param[in]     flags   A flag per line-graph key 0..views.h2h.max_key (nullptr for keys).
 * @param[in]     s       The phase's sign (minus: G_t and the deletions; plus: G_{t+1} and the
 *                        ASSIGNED inserted ids).
 * @param[in]     rule    The exactly-once rule (invariant I2), passed explicitly.
 * @param[out]    scratch scratch_words(exec) words.
 * @param[out]    out     The phase: `s` and its counts per bin.
 * @return The outcome.
 * @throws as count_full().
 */
template <typename exec_t>
triad::wedge::launch count_phase(const exec_t& exec, const local_patterns::kernels& k,
                                 const void* op, const hyper::row_views& views, const int* keys,
                                 int n, const unsigned char* flags, framework::sign s,
                                 framework::ownership::min_member rule, std::uint64_t* scratch,
                                 triad_phase_bins& out) {
  std::uint32_t status = 0;
  out.sign = s;
  out.counts.fill(0);
  const local_patterns::host_pass pass =
      pass_of(views, keys, k.pattern_shape == experimental::shape::keys ? nullptr : flags);
  const triad::wedge::launch outcome = dispatch(k, views, keys, n, [&]() {
    status = operators::count_owned<experimental::max_pattern_bins>(
        exec, rule, n, erased_source{&k, op, &pass}, scratch, out.counts.data());
  });
  raise_status(k, status, k.builtin ? "triad_count::update" : "count_local_patterns::update");
  return outcome;
}

/**
 * @brief Without an ownership rule: a compile error (I2: exactly-once counting has no default
 *        rule; cpp/tests/compile_fail/count_delta_ownership.cpp matches the message).
 * @tparam exec_t The executor.
 * @return Nothing (never compiles).
 */
template <typename exec_t>
triad::wedge::launch count_phase(const exec_t&, const local_patterns::kernels&, const void*,
                                 const hyper::row_views&, const int*, int, const unsigned char*,
                                 framework::sign, std::uint64_t*, triad_phase_bins&) {
  static_assert(operators::count_detail::always_false<exec_t>,
                "triad_host::count_phase needs an ownership rule: pass "
                "framework::ownership::min_member{} after the sign (invariant I2: exactly-once "
                "counting has no default rule)");
  return triad::wedge::launch::no_sources;
}

}  // namespace dyng::detail::triad_host
