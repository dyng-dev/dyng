// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file problem.hpp
 * @brief The triad_count problem on the framework (cpp/src/framework), over the hypergraph: its
 *        hooks, its state, and the interfaces of the host and CUDA engines; since M9b the one
 *        driver of every local pattern (triad_count's four and the Ops of
 *        experimental::count_local_patterns, ADR 0058).
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md):
 *
 *     normalize -> prepare -> before_apply -> count(-) on G_t -> commit ->
 *     identify_affected -> count(+) on G_{t+1} -> finalize
 *
 * triad_count is an aggregate-delta problem (family::aggregate_delta, ownership_type =
 * ownership::min_member: a triad is counted by the changed hyperedge with the smallest id in it,
 * invariant I2). Its hooks, one profiler stage each:
 *   - begin_update (no stage): the backend, poisoned and stale-result checks, the placement, the
 *     CUDA engine; Step 0 is the transaction's (stage triad_count.normalize, run_update()): the
 *     live distinct deletions of G_t and the normalized insertions (hyper_staged_batch);
 *   - count(-) on the old view (triad_count.count_minus, on G_t): the frontier of Step 0's
 *     deletions (on cuda one upload of the host ids) and the subtract phase on G_t's internal row
 *     views, with everything both phases need reserved first (invariant I9);
 *   - commit (triad_count.commit, run_update()): hypergraph.apply;
 *   - cuda, Tier B (select_engine: engine::fused; triad_count.enact_fused): identify_affected
 *     (triad_count.identify_affected: the frontier of the ASSIGNED inserted ids, from the applied
 *     record's device copy), count_plus (triad_count.count_plus: the add phase on G_{t+1}'s
 *     internal row views) and finalize (triad_count.finalize: one read of both phases' bins, the
 *     two phases applied, a negative bin an internal_error);
 *   - host backends, Tier A (the incremental host engine, host_engine.hpp; ADR 0054): count(-)
 *     on G_t (triad_count.count_minus: the frontier of Step 0's deletions, the subtract phase),
 *     identify_affected (triad_count.identify_affected: the frontier of the ASSIGNED inserted
 *     ids), count(+) (triad_count.count_plus: the add phase on G_{t+1}) and finalize
 *     (triad_count.finalize: the two phases applied as on cuda, a negative bin an
 *     internal_error);
 *   - end_update (no stage): record the hypergraph state the result matches, return the scratch.
 * compute() is the static enactor: reset (triad_count.reset) -> count (triad_count.count, the
 * full count of the host engine) -> finalize on the host backends; compute_fused
 * (triad_count.enact_fused, with the sub-stages reset, count and finalize) on cuda.
 *
 * The engines read the hypergraph through its internal row views
 * (detail::hypergraph_access::views(), ADR 0042): rows by CBST key (id + 1), h2v members + 1,
 * h2h neighbour keys with `out`. The h2v view covers the keys 1..hyperedge_id_bound(), the h2h
 * view 1..the largest bound so far (its rows above the bound are empty), so every loop over h2v
 * keys is bounded by views.h2v.max_key, and the frontier flags cover views.h2h.max_key + 1 keys.
 * The public views (h2v(), h2h()) are never built on the counting path.
 *
 * One driver, one engine (M9b, ADR 0058): the problem is local_pattern_problem<V, W, flavor_t>,
 * the same hooks for every pattern; what depends on the pattern is its table
 * (local_patterns::kernels: num_bins, shape, the CUDA kernel's launch and the host work of one
 * source) and the Op's value, both kept in the result's state (local_pattern_core). The flavor
 * gives the name of the stages ("triad_count.*" for triad_count::result,
 * "count_local_patterns.*" for the experimental results), the stats type and the result's
 * handle; triad_count_problem<V, W> is the triad flavor.
 */
#pragma once

#include "algorithms/triad_count/engine/phases.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string_view>
#include <vector>

namespace dyng::detail {

namespace framework {
struct hook_stages;
}  // namespace framework

/// The counts of one phase of a pattern, up to max_pattern_bins bins (num_bins of them are used).
using pattern_bins = std::array<std::uint64_t, experimental::max_pattern_bins>;

/**
 * @brief One phase of an update: the sign it counts with and its counts per bin as unsigned
 *        magnitudes (ADR 0057: a phase of 2^63 or more triads in a bin is exact; signed 64-bit
 *        phase bins refused it). The subtract and the add phase are kept apart; apply_delta()
 *        checks the signs and applies both once.
 */
struct triad_phase_bins {
  framework::sign sign = framework::sign::plus;  ///< the sign the phase counted with
  pattern_bins counts{};                         ///< the phase's counts per bin
};

/**
 * @brief The launch counters of the engines (M9a criterion 9), per mode (0: the full count, 1: a
 *        frontier phase) and shape (0: triangles, 1: pairs, 2: keys): the counting passes (kernels
 *        on cuda, count_owned passes on the host backends) and the calls that ended at one of the
 *        two early outs (no sources: an empty frontier; no wedges: sources without a wedge). Kept
 *        by the result across its calls.
 */
struct triad_launch_counters {
  std::int64_t kernels[2][3] = {};     ///< counting passes run (kernels launched)
  std::int64_t no_sources[2][3] = {};  ///< calls without sources
  std::int64_t no_wedges[2][3] = {};   ///< calls whose sources have no wedge
};

/**
 * @brief The state of every local-pattern result (triad_count::result and the experimental
 *        results): the pattern's table and Op value, the counts and the matched hypergraph.
 */
struct local_pattern_core {
  local_patterns::kernels kernels;    ///< the pattern's table (num_bins, shape, the passes)
  std::vector<unsigned char> op;      ///< the Op's value (kernels.op_size bytes)
  std::vector<std::uint64_t> counts;  ///< num_bins counts
  int num_bins = 0;                   ///< the pattern's bins
  std::uint64_t version = 0;          ///< hypergraph version matched
  std::uint64_t graph_state = 0;      ///< hypergraph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;              ///< a failed update left the result unusable
  triad_launch_counters launches;     ///< the engine's passes (all calls of the result)
};

/// The state behind a triad_count::result.
struct triad_count_state : local_pattern_core {
  triad_count::options opt;  ///< the options (kind fixed at compute())
};

/// The state behind an experimental::local_patterns_result.
struct local_patterns_state : local_pattern_core {};

/// Access to the state of a result.
struct triad_count_access {
  /// The state of `r`, or nullptr for a moved-from result.
  static const triad_count_state* find(const triad_count::result& r) noexcept {
    return r.impl_.get();
  }
  /// The state of `r`, or nullptr for a moved-from result.
  static triad_count_state* find(triad_count::result& r) noexcept {
    return r.impl_.get();
  }
  /// The state of `r`.
  static triad_count_state& state(triad_count::result& r);
  /// The state of `r`.
  static const triad_count_state& state(const triad_count::result& r);
  /// A result owning `st`.
  static triad_count::result make(std::unique_ptr<triad_count_state> st);
};

/**
 * @brief The table of one of triad_count's built-in patterns (the four Ops of the engine, each
 *        with its CUDA kernel in counters.cu and its host work in builtin_ops.cpp).
 * @param[in] kind The pattern.
 * @return The table (static storage).
 */
[[nodiscard]] const local_patterns::kernels& triad_kernels(triad_count::pattern kind) noexcept;

/**
 * @brief The value-initialized Op of a built-in pattern, as bytes (motif_op{}, choose3_op{},
 *        type2_op{}, type3_op{}; builtin_ops.cpp).
 * @param[in]  kind The pattern.
 * @param[out] out  The Op's bytes.
 */
void triad_op_value(triad_count::pattern kind, std::vector<unsigned char>& out);

/**
 * @brief The CUDA launch of a built-in pattern's kernel (counters.cu: local_patterns::launch of
 *        its Op), or nullptr in a build without CUDA.
 * @param[in] kind The pattern.
 * @return The function.
 */
[[nodiscard]] local_patterns::launch_fn triad_device_launch(triad_count::pattern kind) noexcept;

/**
 * @brief Whether a pattern reads the line graph (h2h) and the vertex-to-hyperedge rows.
 * @param[in] k The pattern's table.
 * @return false for the keys shape (type 1: it reads the hyperedges' rows only), true otherwise.
 */
[[nodiscard]] inline bool reads_line_graph(const local_patterns::kernels& k) noexcept {
  return k.pattern_shape != experimental::shape::keys;
}

/**
 * @brief The shape index of the launch counters (0 triangles, 1 pairs, 2 keys).
 * @param[in] k The pattern's table.
 * @return The index.
 */
[[nodiscard]] inline int shape_index(const local_patterns::kernels& k) noexcept {
  switch (k.pattern_shape) {
    case experimental::shape::keys:
      return 2;
    case experimental::shape::pairs:
      return 1;
    case experimental::shape::triangles:
      break;
  }
  return 0;
}

// ------------------------------------------------------------------------------------------------
// The host backends (sequential.cpp, openmp.cpp): the incremental host engine (host_engine.hpp)
// ------------------------------------------------------------------------------------------------

/**
 * @brief The host memory of the host engine, leased from the workspace pool of the resources
 *        handle for one compute() or update() (ADR 0015) and sized before the commit for both
 *        phases (invariant I9).
 */
struct triad_host_workspace final : pooled_workspace {
  scratch_buffer<int> keys;             ///< the frontier's keys (id + 1)
  scratch_buffer<unsigned char> flags;  ///< a flag per line-graph key (the frontier)
  scratch_buffer<std::uint64_t> bins;   ///< count_owned's bins, one set per thread

  /// The memory held.
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return keys.bytes() + flags.bytes() + bins.bytes();
  }
};

/// The host engine of one call: the leased workspace, the sizes it was reserved for, the bins of
/// both phases (kept apart) and the passes. The problem holds it (no heap allocation per call).
struct triad_host_engine {
  const local_patterns::kernels* k = nullptr;                     ///< the pattern's table
  const void* op = nullptr;                                       ///< the Op's value
  std::optional<workspace_pool::lease<triad_host_workspace>> ws;  ///< the host memory
  int num_keys = 0;                ///< the keys of the current phase's frontier
  int max_keys = 0;                ///< the frontier keys reserved
  std::int64_t max_flags = 0;      ///< the flags reserved
  triad_phase_bins minus;          ///< the subtract phase (sign minus)
  triad_phase_bins plus;           ///< the add phase (sign plus)
  triad_launch_counters launches;  ///< this call's passes
};

/**
 * @brief The full count of the host engine on the sequential backend (host_engine.hpp,
 *        count_full on operators::sequential_exec).
 * @param[in]  res     Resources of the sequential backend.
 * @param[in]  k       The pattern's table.
 * @param[in]  op      The Op's value.
 * @param[in]  views   The internal row views (host memory).
 * @param[in]  rule    The exactly-once rule (nothing is skipped in a full count).
 * @param[out] scratch triad_host_scratch_words(res) words.
 * @param[out] counts  max_pattern_bins counts (zeroed first; k.num_bins of them used).
 * @return The outcome (an early out or a pass).
 * @throws internal_error for a triad of no class (a built-in pattern); invalid_argument_error for
 *         a bin an Op of a user named outside its range; capacity_error if a count exceeds
 *         2^64 - 1.
 */
triad::wedge::launch triad_sequential_full(const resources& res, const local_patterns::kernels& k,
                                           const void* op, const hyper::row_views& views,
                                           framework::ownership::min_member rule,
                                           std::uint64_t* scratch, std::uint64_t* counts);

/**
 * @brief One frontier phase of the host engine on the sequential backend (host_engine.hpp,
 *        count_phase on operators::sequential_exec).
 * @param[in]     res     Resources of the sequential backend.
 * @param[in]     k       The pattern's table.
 * @param[in]     op      The Op's value.
 * @param[in]     views   The row views of the phase's hypergraph state (host memory).
 * @param[in]     keys    The frontier's keys (ids + 1).
 * @param[in]     n       Their number.
 * @param[in]     flags   A flag per line-graph key (nullptr for the keys shape).
 * @param[in]     s       The phase's sign.
 * @param[in]     rule    The exactly-once rule (invariant I2).
 * @param[out]    scratch triad_host_scratch_words(res) words.
 * @param[out]    out     The phase: its sign and its counts per bin.
 * @return The outcome.
 * @throws as triad_sequential_full().
 */
triad::wedge::launch triad_sequential_phase(const resources& res, const local_patterns::kernels& k,
                                            const void* op, const hyper::row_views& views,
                                            const int* keys, int n, const unsigned char* flags,
                                            framework::sign s,
                                            framework::ownership::min_member rule,
                                            std::uint64_t* scratch, triad_phase_bins& out);

/**
 * @brief The full count on the OpenMP backend (as triad_sequential_full(), on
 *        operators::openmp_exec: a static partition of the sources, private bins per thread added
 *        in thread order).
 * @param[in]  res     Resources of the OpenMP backend (their thread count).
 * @param[in]  k       The pattern's table.
 * @param[in]  op      The Op's value.
 * @param[in]  views   The internal row views (host memory).
 * @param[in]  rule    The exactly-once rule.
 * @param[out] scratch triad_host_scratch_words(res) words.
 * @param[out] counts  max_pattern_bins counts.
 * @return The outcome.
 * @throws as triad_sequential_full().
 */
triad::wedge::launch triad_openmp_full(const resources& res, const local_patterns::kernels& k,
                                       const void* op, const hyper::row_views& views,
                                       framework::ownership::min_member rule,
                                       std::uint64_t* scratch, std::uint64_t* counts);

/**
 * @brief One frontier phase on the OpenMP backend (as triad_sequential_phase()).
 * @param[in]     res     Resources of the OpenMP backend.
 * @param[in]     k       The pattern's table.
 * @param[in]     op      The Op's value.
 * @param[in]     views   The row views of the phase's hypergraph state.
 * @param[in]     keys    The frontier's keys.
 * @param[in]     n       Their number.
 * @param[in]     flags   A flag per line-graph key (nullptr for the keys shape).
 * @param[in]     s       The phase's sign.
 * @param[in]     rule    The exactly-once rule.
 * @param[out]    scratch triad_host_scratch_words(res) words.
 * @param[out]    out     The phase: its sign and its counts per bin.
 * @return The outcome.
 * @throws as triad_sequential_full().
 */
triad::wedge::launch triad_openmp_phase(const resources& res, const local_patterns::kernels& k,
                                        const void* op, const hyper::row_views& views,
                                        const int* keys, int n, const unsigned char* flags,
                                        framework::sign s, framework::ownership::min_member rule,
                                        std::uint64_t* scratch, triad_phase_bins& out);

/**
 * @brief The scratch words of the host engine's counts on the backend of `res` (sequential.cpp,
 *        openmp.cpp).
 * @param[in] res Resources of a host backend.
 * @return The words (max_pattern_bins bins and a status word per thread).
 */
[[nodiscard]] std::size_t triad_host_scratch_words(const resources& res) noexcept;

/**
 * @brief The ids of the add phase's frontier (shared by both engines): the ids the commit
 *        ASSIGNED to the inserted hyperedges (hyper_apply_summary::inserted_ids), never the
 *        requested ones (invariant I2's frontier; M9a criterion 6).
 */
struct triad_inserted_frontier {
  const std::int64_t* host = nullptr;    ///< the ids (host memory)
  const std::int64_t* device = nullptr;  ///< the same in device memory (cuda), or nullptr
  int size = 0;                          ///< their number
};

/**
 * @brief The add phase's frontier of an applied batch (triad_inserted_frontier).
 * @param[in]  record          The applied record.
 * @param[in]  id_bound_before hyperedge_id_bound() of G_t (Step 0's).
 * @param[out] scratch         Host ids when they are not the record's (the recorded mutation
 *                             triad_inserted_requested_ids only).
 * @return The frontier's ids.
 */
[[nodiscard]] triad_inserted_frontier triad_inserted_ids(const hyper_applied& record,
                                                         std::int64_t id_bound_before,
                                                         std::vector<std::int64_t>& scratch);

// ------------------------------------------------------------------------------------------------
// The CUDA backend (cuda.cu; ESCHER-GPU's wedge engine, engine/)
// ------------------------------------------------------------------------------------------------

/**
 * @brief The device memory of the CUDA engine, leased from the workspace pool of the resources
 *        handle for one compute() or update() (ADR 0015) and sized before the commit for both
 *        phases (invariant I9).
 */
struct triad_cuda_workspace final : pooled_workspace {
  scratch_buffer<long long> offsets;        ///< the wedge offsets of a phase's sources
  scratch_buffer<unsigned char> scan_temp;  ///< CUB's storage of their scan
  /// compute(): the offsets of every key (20 MB on coauth-DBLP) and their scan's storage,
  /// released when compute() returns (the updates' scratch stays reserved for the next update).
  scratch_buffer<long long> full_offsets;
  scratch_buffer<unsigned char> full_scan_temp;  ///< compute(): CUB's storage
  scratch_buffer<std::int64_t> ids;              ///< the deleted ids (uploaded)
  scratch_buffer<int> keys;                      ///< the frontier's keys (id + 1)
  scratch_buffer<unsigned char> flags;           ///< a flag per line-graph key (the frontier)
  scratch_buffer<unsigned long long> bins;       ///< subtract bins, add bins, the error word
  /// Pinned: the read-back of the bins and, after them, the wedge count of a phase.
  buffer<unsigned long long> host;

  /// The memory held.
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return offsets.bytes() + scan_temp.bytes() + full_offsets.bytes() + full_scan_temp.bytes() +
           ids.bytes() + keys.bytes() + flags.bytes() + bins.bytes() +
           host.size() * sizeof(unsigned long long);
  }
};

/// The CUDA engine of one call: the leased workspace and the sizes it was reserved for. The
/// problem holds it (no heap allocation per call, conformance check C8).
struct triad_cuda_engine {
  const local_patterns::kernels* k = nullptr;                     ///< the pattern's table
  const void* op = nullptr;                                       ///< the Op's value
  int stride = 30;                                                ///< the phases' bin stride
  std::optional<workspace_pool::lease<triad_cuda_workspace>> ws;  ///< the device memory
  std::size_t scan_bytes = 0;                                     ///< the reserved CUB storage
  int num_keys = 0;            ///< the keys of the current phase's frontier
  int max_keys = 0;            ///< the frontier keys (and wedge sources) reserved
  std::int64_t max_flags = 0;  ///< the flags reserved
  framework::sign minus_sign = framework::sign::minus;  ///< the subtract phase's sign
  framework::sign plus_sign = framework::sign::plus;    ///< the add phase's sign
  triad_launch_counters launches;                       ///< this call's launches
};

/**
 * @brief Start the CUDA engine of one call for a pattern: lease the workspace and size its bins
 *        and pinned buffer.
 * @param[in]  res Resources of the CUDA backend.
 * @param[in]  k   The pattern's table.
 * @param[in]  op  The Op's value.
 * @param[out] e   The engine.
 */
void triad_cuda_open(const resources& res, const local_patterns::kernels& k, const void* op,
                     triad_cuda_engine& e);

/**
 * @brief compute() on cuda: the full count of a state's row views (ESCHER-GPU's count()).
 * @param[in]     res    Resources of the CUDA backend.
 * @param[in,out] e      The engine.
 * @param[in]     views  The internal row views (device memory).
 * @param[out]    counts num_bins counts.
 * @param[in]     names  The stages of the driver's flavor (count, finalize, compute_function).
 */
void triad_cuda_compute(const resources& res, triad_cuda_engine& e, const hyper::row_views& views,
                        std::uint64_t* counts, const framework::hook_stages& names);

/**
 * @brief update() on cuda, count(-): the subtract phase on G_t (ESCHER-GPU's subtract()), with the
 *        frontier of the deleted ids; sizes the engine for the add phase.
 * @param[in]     res        Resources of the CUDA backend.
 * @param[in,out] e          The engine.
 * @param[in]     views      G_t's internal row views.
 * @param[in]     deletions  Step 0's live distinct deletions (ids, host memory).
 * @param[in]     insertions The number of hyperedges the batch inserts.
 * @param[in]     s          The phase's sign (triad::phase_sign(sign::minus)).
 * @param[in]     rule       The exactly-once rule of the phase (invariant I2).
 */
void triad_cuda_count_minus(const resources& res, triad_cuda_engine& e,
                            const hyper::row_views& views,
                            const std::vector<std::int64_t>& deletions, std::int64_t insertions,
                            framework::sign s, framework::ownership::min_member rule);

/**
 * @brief update() on cuda, identify_affected: the frontier of the add phase, the ids the commit
 *        ASSIGNED to the inserted hyperedges (triad_inserted_ids(): their device copy, uploaded
 *        only when the ids have none).
 * @param[in]     res   Resources of the CUDA backend.
 * @param[in,out] e     The engine.
 * @param[in]     views G_{t+1}'s internal row views.
 * @param[in]     ids   The add phase's frontier.
 */
void triad_cuda_identify_affected(const resources& res, triad_cuda_engine& e,
                                  const hyper::row_views& views,
                                  const triad_inserted_frontier& ids);

/**
 * @brief update() on cuda, count(+): the add phase on G_{t+1} (ESCHER-GPU's add()).
 * @param[in]     res   Resources of the CUDA backend.
 * @param[in,out] e     The engine.
 * @param[in]     views G_{t+1}'s internal row views.
 * @param[in]     s     The phase's sign (triad::phase_sign(sign::plus)).
 * @param[in]     rule  The exactly-once rule of the phase (invariant I2).
 */
void triad_cuda_count_plus(const resources& res, triad_cuda_engine& e,
                           const hyper::row_views& views, framework::sign s,
                           framework::ownership::min_member rule);

/**
 * @brief update() on cuda, finalize: the bins of both phases on the host.
 * @param[in]     res   Resources of the CUDA backend.
 * @param[in,out] e     The engine.
 * @param[out]    minus The subtract phase (its sign and the counts removed per bin).
 * @param[out]    plus  The add phase (its sign and the counts added per bin).
 * @param[in]     what  The function named in an error ("<flavor>::update").
 * @throws internal_error if the device engine raised its error word for an unclassifiable triad
 *         (a built-in pattern); invalid_argument_error for a bin an Op of a user named outside its
 *         range; capacity_error if it raised it for a count past 2^64 - 1.
 */
void triad_cuda_finalize(const resources& res, triad_cuda_engine& e, triad_phase_bins& minus,
                         triad_phase_bins& plus, const char* what);

/**
 * @brief Add the engine's launch counters of this call to `into`.
 * @param[in]     e    The engine.
 * @param[in,out] into The result's counters.
 */
void triad_cuda_take_launches(const triad_cuda_engine& e, triad_launch_counters& into) noexcept;

/**
 * @brief The host synchronizations of one update on cuda (the algorithm budget, I9 / C8).
 * @param[in] k The pattern's table.
 * @return At most one per phase that launches wedges (its wedge count) and one for the bins.
 */
[[nodiscard]] std::int64_t triad_cuda_host_syncs(const local_patterns::kernels& k) noexcept;

// ------------------------------------------------------------------------------------------------
// The flavors of the driver
// ------------------------------------------------------------------------------------------------

/// triad_count::result: the stages "triad_count.*" and triad_count::stats.
struct triad_flavor {
  static constexpr std::string_view name = "triad_count";  ///< the stages' prefix
  /// The stats' per-bin delta (messages).
  static constexpr std::string_view delta_name = "triad_count::stats::delta";
  using stats_type = triad_count::stats;  ///< the stats of update()
  /// The result an update changes.
  struct handle {
    triad_count::result* r = nullptr;  ///< the result
  };
  /// The state of the result, or nullptr for a moved-from result.
  static local_pattern_core* find(handle h) noexcept {
    return triad_count_access::find(*h.r);
  }
  /// The state of the result (invalid_argument_error for a moved-from result).
  static local_pattern_core& state(handle h) {
    return triad_count_access::state(*h.r);
  }
  /// The result's address (dyng::update()'s duplicate check).
  static const void* target(handle h) noexcept {
    return h.r;
  }
  /// The checks of the options before anything changes (the CUDA engine on cuda).
  static void check_update(const local_pattern_core& st, bool on_cuda, const char* what);
  /// The counters of the bins in the stats.
  static void write(stats_type& stats, std::uint64_t removed, std::uint64_t added,
                    const std::int64_t* delta, int n) noexcept {
    stats.triads_removed = removed;
    stats.triads_added = added;
    for (int k = 0; k < n && k < triad_count::max_bins; ++k) {
      stats.delta[static_cast<std::size_t>(k)] = delta[k];
    }
  }
};

/// experimental::local_patterns_result<op_t>: the stages "count_local_patterns.*" and
/// experimental::local_patterns_stats.
struct local_patterns_flavor {
  static constexpr std::string_view name = "count_local_patterns";  ///< the stages' prefix
  /// The stats' per-bin delta (messages).
  static constexpr std::string_view delta_name = "experimental::local_patterns_stats::delta";
  using stats_type = experimental::local_patterns_stats;  ///< the stats of update()
  /// The result an update changes: its state's owner and its address.
  struct handle {
    local_patterns_state_ptr* owner = nullptr;  ///< the result's state
    const void* target = nullptr;               ///< the result
  };
  /// The state of the result, or nullptr for a moved-from result.
  static local_pattern_core* find(handle h) noexcept;
  /// The state of the result (invalid_argument_error for a moved-from result).
  static local_pattern_core& state(handle h);
  /// The result's address.
  static const void* target(handle h) noexcept {
    return h.target;
  }
  /// No options to check.
  static void check_update(const local_pattern_core& /*st*/, bool /*on_cuda*/,
                           const char* /*what*/) noexcept {}
  /// The counters of the bins in the stats (delta: one value per bin; allocates).
  static void write(stats_type& stats, std::uint64_t removed, std::uint64_t added,
                    const std::int64_t* delta, int n) {
    stats.removed = removed;
    stats.added = added;
    stats.delta.assign(delta, delta + n);
  }
};

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

/**
 * @brief The local-pattern problem (family::aggregate_delta) that framework::update_enactor and
 *        framework::static_enactor run: the hooks of the file comment, for every pattern (its
 *        table in the result's state). Its members are defined in triad_count.cpp, the only
 *        translation unit that runs it.
 *
 * One problem serves one call: an update of one result (constructed from the result's handle;
 * the participant adapter of framework/composition.hpp owns it) or a compute() (constructed from
 * the state being built). The CUDA engine of an update lives from count(-) (before the commit) to
 * end_update(); everything it needs after the commit is sized before it (invariant I9).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (ignored).
 * @tparam flavor_t triad_flavor or local_patterns_flavor (the stages, the stats, the result).
 */
template <typename vertex_t, typename weight_t, typename flavor_t>
class local_pattern_problem final
    : public framework::problem_base<local_pattern_problem<vertex_t, weight_t, flavor_t>,
                                     framework::family::aggregate_delta> {
 public:
  static constexpr std::string_view name = flavor_t::name;  ///< stages "<name>.<hook>"
  using container_type = hypergraph<vertex_t, weight_t>;    ///< the container
  using stats_type = typename flavor_t::stats_type;         ///< the stats of update()
  using frontier_type = framework::internal_frontier;       ///< frontiers are internal
  using ownership_type = framework::ownership::min_member;  ///< the counting rule (I2)
  using old_graph = framework::old_view<container_type>;    ///< G_t
  using new_graph = framework::new_view<container_type>;    ///< G_{t+1}
  /// The batch before the commit (batch.normalized: Step 0, the live deletions of G_t).
  using requested = framework::hyper_requested_batch<vertex_t, weight_t>;
  /// What the commit did (applied.record: the applied record with the assigned ids).
  using applied = framework::hyper_applied_batch<vertex_t, weight_t>;

  /**
   * @brief The problem of one update of a result.
   * @param[in] h The result's handle (the result must outlive the problem).
   */
  explicit local_pattern_problem(typename flavor_t::handle h);

  /**
   * @brief The problem of one compute() that fills `st`.
   * @param[in,out] st The state of the result being built (must outlive the problem).
   */
  explicit local_pattern_problem(local_pattern_core& st) noexcept : state_(&st), computing_(true) {}

  // ---- lifecycle (no stage) ------------------------------------------------------------------

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return computing_ ? nullptr : flavor_t::target(handle_);
  }
  /// Whether the update keeps the line graph: true, except for the keys shape (type 1, which
  /// reads the hyperedges' rows only).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return reads_line_graph_;
  }
  /// The checks before anything changes (backend, poisoned and stale result, placement, the CUDA
  /// engine); keeps Step 0's batch.
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Bind G_{t+1}.
  void resume(framework::context& ctx, new_graph g, const applied& applied);
  /// Record the hypergraph state the result matches; return the engine.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (its algorithm phase failed).
  void poison() noexcept;

  // ---- policies (no stage) -------------------------------------------------------------------

  /// engine::fused (the wedge engine) on cuda, engine::operators (the hooks and the host engine)
  /// on the host backends.
  [[nodiscard]] engine select_engine(framework::context& ctx) const noexcept;
  /// The budget of the update's algorithm work (I9, C8): no allocation once reserved; on cuda at
  /// most triad_cuda_host_syncs() host synchronizations, none on the host backends.
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept;

  // ---- Step 1a, on G_t -------------------------------------------------------------------------

  /// <name>.count_minus: the subtract phase on G_t (cuda: the wedge engine; host: the host
  /// engine), after reserving what both phases need.
  void count(framework::context& ctx, old_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);

  // ---- Tier A (host backends), on G_{t+1} ------------------------------------------------------

  /// <name>.identify_affected: the add phase's frontier, the ASSIGNED inserted ids.
  void identify_affected(framework::context& ctx, new_graph g, const applied& applied,
                         frontier_type& f);
  /// update(): <name>.count_plus, the add phase on G_{t+1}; compute(): <name>.count, the full
  /// count.
  void count(framework::context& ctx, new_graph g, frontier_type& f, framework::sign s,
             ownership_type rule);
  /// update(): <name>.finalize, the delta and the stats; compute(): the hypergraph state.
  void finalize(framework::context& ctx, stats_type& stats);

  // ---- Tier B (cuda), on G_{t+1} -----------------------------------------------------------

  /// <name>.enact_fused: identify_affected, count_plus and finalize (sub-stages).
  void enact_fused(framework::context& ctx, new_graph g, const applied& applied, stats_type& stats);

  // ---- compute() -----------------------------------------------------------------------------

  /// <name>.reset: the zero counts.
  void reset(framework::context& ctx);
  /// <name>.enact_fused (cuda): the full count, with the sub-stages reset, count, finalize.
  void compute_fused(framework::context& ctx, new_graph g, stats_type& stats);

 private:
  /// Both phases applied to the counts (their signs checked), and the stats.
  void apply_delta(const triad_phase_bins& minus, const triad_phase_bins& plus, stats_type& stats);
  /// The counters of the update in the stats (all but the bins).
  void fill_counts(stats_type& stats) const;
  /// Host backends: lease the host engine's workspace for the call and size its scratch.
  void open_host(framework::context& ctx);
  /// Host backends: the frontier of a phase in the workspace (keys and, for the shapes that read
  /// the line graph, flags over views.h2h.max_key + 1 keys).
  void host_frontier(const hyper::row_views& views, const std::int64_t* ids, int n);
  /// The Op's value of the state.
  [[nodiscard]] const void* op_value() const noexcept {
    return state_->op.data();
  }

  local_pattern_core* state_ = nullptr;
  typename flavor_t::handle handle_{};
  bool computing_ = false;
  bool reads_line_graph_ = true;
  const hyper_staged_batch<weight_t>* normalized_ = nullptr;  ///< Step 0 of the update
  const container_type* graph_ = nullptr;                     ///< G_{t+1} (compute(): G)
  std::int64_t deleted_ = 0;                                  ///< the subtract frontier's size
  std::int64_t inserted_ = 0;                                 ///< the add frontier's size
  std::optional<triad_cuda_engine> cuda_;                     ///< cuda: the engine of the call
  std::optional<triad_host_engine> host_;  ///< host backends: the engine of the call
  std::vector<std::int64_t> requested_;    ///< triad_inserted_ids()'s scratch (mutation only)
};

/// The triad_count problem: the driver's triad flavor.
template <typename vertex_t, typename weight_t>
using triad_count_problem = local_pattern_problem<vertex_t, weight_t, triad_flavor>;

}  // namespace dyng::detail
