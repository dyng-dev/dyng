// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/include/hsosp.cuh (UpdateConfig, UpdateStats)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (the host emulation's state)
/**
 * @file problem.hpp
 * @brief The hyper_sssp problem on the framework (cpp/src/framework), over the hypergraph's
 *        slack_csr line graph: its state, its hooks and the two engines' entry points (M11a step
 *        4; ADRs 0090-0092).
 *
 * Template card (PLAN Section 4.5.2; docs/developer/framework.md):
 *
 *     normalize -> prepare -> before_apply -> commit (hypergraph.apply: Step 0, the ids, the
 *     incidence edits, the order-free line-graph delta, the slack CSR apply) ->
 *     identify_affected -> seed -> { loop until is_converged } [-> recompute] -> finalize
 *
 * A fixed-point problem with two engines:
 *   - sequential, Tier A (sequential.cpp): MOSP_ESCHER@4b86159's host emulation
 *     (hypergraph/src/HostHypergraph.cpp:365-507) ported as is: identify_affected = the roots and
 *     the children-list invalidation of emulateSospUpdate; seed = the pulls of the invalidated
 *     hyperedges and the inserted pairs' relaxations (its first offers); loop = its priority queue
 *     drained in one call, the work budget asked after it (framework over_budget, ADR 0092);
 *     recompute and compute() = emulateSospRecompute's pull rounds;
 *   - cuda, Tier B (cuda.cu, kernels.cuh, state.cuh): MOSP_ESCHER's hsospUpdate and
 *     hsospRecompute (hsosp/src/hsospDevice.cu) transplanted as is behind enact_fused /
 *     compute_fused, which open the stages identify_affected, seed, loop, recompute (on every
 *     update, 0 ms without a fallback, so the region map's sosp_update row is defined) and
 *     finalize themselves (ADR 0091).
 *
 * Both engines read the container's slack line graph (detail::hypergraph_access::slack(h):
 * rowStart / deg / colInd / the node weights, host or device memory) and the applied record's pair
 * delta (host vectors, or slack_access::last_delta() on the CUDA store), and write nothing of the
 * container (the CUDA store zeroes the dead degrees in its own apply, ADR 0089).
 *
 * The two modes (HD2; ADR 0091): the public mode (a source vertex s: the hyperedges holding s
 * start at their own weight with parent -1, kept under incidence edits and hyperedge changes) and
 * the internal virtual-endpoint mode of the parity harness (virtual_endpoints.hpp: a source
 * hyperedge at distance 0 and up to two ids whose weight hyper_sssp reads as 0, MOSP_ESCHER's
 * virtual source and target rows; no public name).
 */
#pragma once

#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/frontier.hpp"
#include "framework/policies.hpp"
#include "framework/problem.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/views.hpp"
#include "framework/workspace.hpp"
#include "hypergraph/slack_csr/slack_rows.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::detail {

/// The largest hyperedge weight hyper_sssp accepts (MOSP_ESCHER's limit; PLAN 4.4.2, HD11).
inline constexpr std::int64_t hyper_sssp_max_weight = std::int64_t{1} << 28;

/// The distance of a dead or unreachable hyperedge (MOSP_ESCHER's INF, HostHypergraph::INF).
inline constexpr std::int64_t hyper_sssp_infinity = std::numeric_limits<std::int64_t>::max() / 4;

/// The rounds of pointer jumping the CUDA invalidation may run (MOSP_ESCHER's cap).
inline constexpr int hyper_sssp_jump_cap = 64;

/**
 * @brief The endpoints of a result: the public source vertex, or the internal virtual-endpoint
 *        mode (HD2; virtual_endpoints.hpp).
 */
struct hyper_sssp_endpoints {
  std::int32_t source_vertex = -1;     ///< the public mode: the source vertex (-1: virtual)
  std::int64_t source_hyperedge = -1;  ///< the virtual mode: the source row (distance 0)
  std::int64_t zero_weight_a = -1;     ///< the virtual mode: an id whose weight reads as 0
  std::int64_t zero_weight_b = -1;     ///< the virtual mode: another one
  /// @brief Whether this is the virtual-endpoint mode. @return source_hyperedge >= 0.
  [[nodiscard]] bool is_virtual() const noexcept {
    return source_hyperedge >= 0;
  }
};

/**
 * @brief The engine's own counters of the last call (MOSP_ESCHER's UpdateStats; the harness logs
 *        them, they are not parity targets: HD7).
 */
struct hyper_sssp_engine_counters {
  std::int64_t iterations = 0;           ///< update: push rounds or expansions; recompute: rounds
  bool fallback = false;                 ///< the update fell back to a recompute
  std::int64_t fallback_iterations = 0;  ///< rounds of the fallback recompute
  std::int64_t max_frontier = 0;         ///< the largest frontier
  std::int64_t seed_count = 0;           ///< update: the invalidated hyperedges
  std::int64_t jump_rounds = 0;          ///< pointer-jumping rounds (cuda)
  std::int64_t work = 0;                 ///< relaxations of the update
  hyper_sssp::fallback_reason reason = hyper_sssp::fallback_reason::none;  ///< why it fell back
  engine engine_used = engine::fused;  ///< the CUDA engine that ran (operators on the host)
  std::int64_t epochs = 0;             ///< near-far threshold raises (the shared engine)
  /// The host synchronizations of the shared engine's run (its fallback included; -1: the as-is
  /// engine, whose count algorithm_budget derives from the counters above).
  std::int64_t host_syncs = -1;
};

/**
 * @brief The internal knobs of the shared engine (engine::operators on cuda; ADR 0095). Not
 *        public: the parity harness sets them (hyper_sssp_virtual::set_operators_knobs) to read
 *        the engine's variants; the defaults are the variant the O12 reading chose.
 */
struct hyper_sssp_operators_knobs {
  bool warp_rows = true;  ///< one warp per row in the pull and the push (else one thread)
  /// The near-far width; 0: unbounded (every improved node joins the next round, MOSP_ESCHER's
  /// push rounds).
  std::int64_t delta = 0;
  /// compute(), the update's fallback and the static recompute on the shared engine (true) or on
  /// the ported recompute (hsospRecompute, false).
  bool shared_compute = false;
};

namespace hsosp {
/// The device state of a CUDA result (MOSP_ESCHER's HsospState; state.cuh).
struct device_state;
}  // namespace hsosp
/// Deletes a device state (cuda.cu; a no-op stub without CUDA).
struct hyper_sssp_device_state_deleter {
  /// @brief Delete. @param[in] st The state.
  void operator()(hsosp::device_state* st) const noexcept;
};
/// The owner of a device state.
using hyper_sssp_device_ptr = std::unique_ptr<hsosp::device_state, hyper_sssp_device_state_deleter>;

namespace hyper_ops {
/// The shared engine's own device scratch of a CUDA result (operators.cu: its control block and
/// the batch's change lists; allocated on the engine's first use).
struct device_scratch;
}  // namespace hyper_ops
/// Deletes the shared engine's scratch (operators.cu; a no-op stub without CUDA).
struct hyper_sssp_operators_scratch_deleter {
  /// @brief Delete. @param[in] s The scratch.
  void operator()(hyper_ops::device_scratch* s) const noexcept;
};
/// The owner of the shared engine's scratch.
using hyper_sssp_operators_ptr =
    std::unique_ptr<hyper_ops::device_scratch, hyper_sssp_operators_scratch_deleter>;

/// The state behind a hyper_sssp::result.
struct hyper_sssp_state {
  hyper_sssp::options opt;             ///< the options
  hyper_sssp_endpoints endpoints;      ///< the source (public or virtual mode)
  int block_size = 256;                ///< the recompute kernels' threads per block (internal knob:
                                       ///< MOSP_ESCHER's UpdateConfig::blockSize, the harness's)
  backend home = backend::sequential;  ///< the backend of the arrays
  int device = -1;                     ///< the CUDA device of a cuda result
  std::int64_t n = 0;                  ///< the ids the arrays cover (the id bound matched)
  // ---- the host engine (sequential) ----
  std::vector<std::int64_t> dist;    ///< by id (host)
  std::vector<std::int32_t> parent;  ///< by id (host), -1 = none
  /// The public mode: 1 for the ids that hold the source (the host copy on every backend; the
  /// CUDA engine keeps a device copy in its state).
  std::vector<std::uint8_t> holds_source;
  // ---- the CUDA engines ----
  hyper_sssp_device_ptr dev;             ///< the device arrays (cuda; both engines)
  hyper_sssp_operators_ptr ops;          ///< the shared engine's scratch (cuda, engine::operators)
  hyper_sssp_operators_knobs ops_knobs;  ///< the shared engine's variant
  // ---- bookkeeping ----
  std::uint64_t version = 0;      ///< hypergraph version matched
  std::uint64_t graph_state = 0;  ///< hypergraph state matched (ADR 0006, "Graph identity")
  bool poisoned = false;          ///< a failed update left the result unusable
  std::int64_t max_weight = 0;    ///< the largest weight seen by compute() and the updates (the
                                  ///< running maximum of the check; HD11)
  /// The hypergraph the result matches (distance_to(), vertex_distances()): its lifetime token
  /// and its implementation (hypergraph_impl<int32_t, int32_t>).
  std::weak_ptr<const int> graph_alive;
  const void* graph_impl = nullptr;
  hyper_sssp_engine_counters last;  ///< the engine's counters of the last compute() / update()
  // ---- caches of the accessors (cleared by every compute() and update()) ----
  mutable std::vector<std::int64_t> host_dist;    ///< cuda: the distances on the host
  mutable bool host_dist_valid = false;           ///< host_dist is current
  mutable std::vector<std::int64_t> vertex_dist;  ///< vertex_distances()
  mutable bool vertex_dist_valid = false;         ///< vertex_dist is current

  hyper_sssp_state() = default;
  hyper_sssp_state(const hyper_sssp_state&) = delete;
  hyper_sssp_state& operator=(const hyper_sssp_state&) = delete;
  ~hyper_sssp_state() = default;

  /// @brief Forget the caches (the arrays changed).
  void invalidate_caches() noexcept {
    host_dist_valid = false;
    vertex_dist_valid = false;
  }
};

/// Access to the state of a result.
struct hyper_sssp_access {
  /// The state of `r` (invalid_argument_error for a moved-from result).
  static hyper_sssp_state& state(hyper_sssp::result<std::int32_t>& r);
  /// The state of `r`.
  static const hyper_sssp_state& state(const hyper_sssp::result<std::int32_t>& r);
  /// A result owning `st`.
  static hyper_sssp::result<std::int32_t> make(std::unique_ptr<hyper_sssp_state> st);
};

// ------------------------------------------------------------------------------------------------
// The line graph and the batch as the engines read them
// ------------------------------------------------------------------------------------------------

/// What an engine reads of the hypergraph after the commit.
struct hyper_sssp_graph {
  hyper::slack_row_view rows;  ///< the slack line graph and the node weights
  /// The host rows by id (liveness: an empty row is dead), the host model of slack_csr storage.
  const std::vector<std::vector<std::int32_t>>* host_rows = nullptr;
  /// The hyperedges of every vertex (host; the public mode's base hyperedges are those of s).
  const std::vector<std::vector<std::int64_t>>* host_v2h = nullptr;
  /// The public mode: 1 for the ids that hold the source (host engine; nullptr otherwise).
  const std::vector<std::uint8_t>* holds = nullptr;
  hyper_sssp_endpoints endpoints;  ///< the source

  /// @brief The weight hyper_sssp reads for id v (host arrays). @param[in] v The id.
  /// @return 0 for the virtual mode's zero-weight ids, the node weight otherwise.
  [[nodiscard]] long long weight(std::int64_t v) const noexcept {
    return v == endpoints.zero_weight_a || v == endpoints.zero_weight_b ? 0 : rows.node_weight[v];
  }
  /// @brief Whether id v is live (host model). @param[in] v The id. @return Live.
  [[nodiscard]] bool live(std::int64_t v) const noexcept {
    return v >= 0 && static_cast<std::size_t>(v) < host_rows->size() &&
           !(*host_rows)[static_cast<std::size_t>(v)].empty();
  }
};

/// The scratch of the host engine and the batch's host lists, leased from the workspace pool of
/// the resources handle (ADR 0015): sized by the first updates of a shape, then reused, so a
/// steady-state update allocates nothing (invariant I9, check C8). MOSP_ESCHER's
/// emulateSospUpdate allocates its locals per call; the port keeps them here (ADR 0091).
struct hyper_sssp_workspace final : pooled_workspace {
  using key = std::pair<long long, int>;  ///< (distance, parent id): ties go to the lower id
  using entry = std::pair<key, int>;      ///< (key, node)
  // The batch's lists (resume()).
  std::vector<std::int64_t> root_ids;       ///< new, recreated and dead ids (host stores)
  std::vector<std::int64_t> lost_source;    ///< public mode: live ids that lost the source
  std::vector<std::int64_t> gained_source;  ///< public mode: live ids that gained it
  std::vector<std::int64_t> changed;        ///< public mode: ids whose membership changed
  std::vector<std::int64_t> holding;        ///< public mode: changed ids holding the source after
  std::vector<std::int64_t> not_holding;    ///< public mode: changed ids not holding it after
  // The CUDA driver's int32 lists (the engine's node ids).
  std::vector<int> base32;            ///< public mode: the live hyperedges of the source
  std::vector<int> lost32, gained32;  ///< public mode: lost_source, gained_source
  std::vector<int> initial32;         ///< the recompute's first list
  // The shared engine's device lists of an update (engine::operators on cuda; operators.cu): the
  // batch as sssp_changes, the insertion heads, the uploaded lost and gained nodes.
  scratch_buffer<int> ops_from;   ///< the changed edges' tails
  scratch_buffer<int> ops_to;     ///< the changed edges' heads
  scratch_buffer<int> ops_heads;  ///< the insertion heads
  scratch_buffer<int> ops_lists;  ///< the public mode's lost, then gained nodes
  // emulateSospUpdate's locals.
  std::vector<std::uint8_t> inv;  ///< the invalidated ids
  std::vector<int> child_off;     ///< the children of the tree before the batch (CSR:
  std::vector<int> child_ids;     ///< offsets, then the ids in id order)
  std::vector<int> stack;         ///< the subtree walk
  std::vector<entry> heap;        ///< the priority queue (std::push_heap / pop_heap)
  // emulateLoop's locals (compute(), the recompute).
  std::vector<int> candidates;        ///< the candidates of a round
  std::vector<int> next;              ///< the next round's (collected)
  std::vector<int> affected;          ///< the changed candidates of a round
  std::vector<std::uint8_t> in_cand;  ///< listed as a candidate
  std::vector<std::uint8_t> holds;    ///< public mode: the recompute's source flags
  // The update's counters.
  std::int64_t invalidated = 0;  ///< the invalidated hyperedges (the source excluded)
  std::int64_t work = 0;         ///< the relaxations (MOSP_ESCHER's work units)
  std::int64_t expansions = 0;   ///< the nodes popped and expanded
  bool over_budget = false;      ///< the loop stopped at the work budget

  /// @brief Start an update (the counters cleared, the lists emptied; capacities kept).
  void begin() noexcept;
  /// @brief Bytes of the vectors' capacities. @return They.
  [[nodiscard]] std::size_t bytes() const noexcept override;
};

/// Make `v` hold `n` elements without allocating in the steady state: a growth is a reservation
/// (invariant I9).
template <typename vector_t>
void reserve_scratch(vector_t& v, std::size_t n) {
  if (v.capacity() < n) {
    note_reservation();
    v.reserve(n + n / 2 + 16);
  }
}

/// What changed in the batch, as the engines read it.
struct hyper_sssp_batch {
  const std::vector<std::int64_t>* deleted_a = nullptr;   ///< deleted pairs (host stores)
  const std::vector<std::int64_t>* deleted_b = nullptr;   ///< (their second ids)
  const std::vector<std::int64_t>* inserted_a = nullptr;  ///< inserted pairs (host stores)
  const std::vector<std::int64_t>* inserted_b = nullptr;  ///< (their second ids)
  hyper::slack_delta_view device;  ///< the pairs and ids on the device (the CUDA store)
  std::int64_t num_deleted = 0;    ///< deleted pairs
  std::int64_t num_inserted = 0;   ///< inserted pairs
  /// The new, recreated and dead ids (MOSP_ESCHER's newHe and deadHe; host stores).
  const std::vector<std::int64_t>* root_ids = nullptr;
  /// The public mode: the live ids that lost the source vertex (a root if their parent is -1),
  /// and those that gained it (offered their own weight with parent -1).
  const std::vector<std::int64_t>* lost_source = nullptr;
  const std::vector<std::int64_t>* gained_source = nullptr;  ///< (see lost_source)
  /// The CUDA driver's scratch (the workspace's int32 lists; null on the host).
  hyper_sssp_workspace* scratch = nullptr;
};

/// The budget of one update (options::work_budget, max_iterations, on_budget; ADR 0092).
struct hyper_sssp_budget {
  double work_limit = 0;            ///< work_budget x max(1, live entries)
  std::int64_t max_iterations = 0;  ///< push rounds (cuda)
  int block_size = 256;             ///< the recompute kernels' block size
  hyper_sssp::budget_action on_budget = hyper_sssp::budget_action::recompute;  ///< the action
};

// ------------------------------------------------------------------------------------------------
// The host engine (sequential.cpp): MOSP_ESCHER's emulateSospRecompute / emulateSospUpdate
// ------------------------------------------------------------------------------------------------

/**
 * @brief emulateSospRecompute: the static solve on the host, rounds of pulls and collections from
 *        the source (MOSP_ESCHER's emulateLoop), into `dist` / `parent` sized to the node count.
 * @param[in]     g      The line graph.
 * @param[out]    dist   Distances by id.
 * @param[out]    parent Parents by id.
 * @param[in]     cap    The rounds allowed (max(max_iterations, n + 1): enough for every input).
 * @param[in,out] ws     The scratch.
 * @return The rounds run (-1 if the cap was reached first).
 */
std::int64_t hyper_sssp_host_recompute(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
                                       std::vector<std::int32_t>& parent, std::int64_t cap,
                                       hyper_sssp_workspace& ws);

/**
 * @brief emulateSospUpdate's first part (identify_affected): the roots and their subtrees in the
 *        tree before the batch are invalidated.
 */
void hyper_sssp_host_identify(const hyper_sssp_graph& g, const hyper_sssp_batch& b,
                              std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent,
                              hyper_sssp_workspace& ws);

/**
 * @brief emulateSospUpdate's second part (seed): the pulls of the invalidated live hyperedges, the
 *        relaxations of the inserted pairs (and, public mode, the base offers).
 */
void hyper_sssp_host_seed(const hyper_sssp_graph& g, const hyper_sssp_batch& b,
                          std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent,
                          hyper_sssp_workspace& ws);

/**
 * @brief emulateSospUpdate's loop: the priority queue drained, the work budget checked before each
 *        expansion (ws.over_budget when it is exceeded; then the update recomputes).
 */
void hyper_sssp_host_loop(const hyper_sssp_graph& g, std::vector<std::int64_t>& dist,
                          std::vector<std::int32_t>& parent, double work_limit,
                          hyper_sssp_workspace& ws);

// ------------------------------------------------------------------------------------------------
// The CUDA engine (cuda.cu; stubs without CUDA in hyper_sssp.cpp)
// ------------------------------------------------------------------------------------------------

/**
 * @brief Size the device state for `nodes` node slots (MOSP_ESCHER's HsospState::allocate; a
 *        growth keeps the arrays' contents and initializes the new slots).
 */
void hyper_sssp_cuda_reserve(const resources& res, hyper_sssp_state& st, std::int64_t nodes);

/**
 * @brief hsospRecompute into `st` (compute(), the update's fallback and the harness's static
 *        recompute), with the profiler stages of the caller. `ws` (an update's pooled workspace,
 *        or null) holds its host lists, so a fallback allocates nothing in the steady state.
 */
hyper_sssp_engine_counters hyper_sssp_cuda_recompute(const resources& res, hyper_sssp_state& st,
                                                     const hyper_sssp_graph& g,
                                                     const hyper_sssp_budget& budget,
                                                     hyper_sssp_workspace* ws = nullptr);

/**
 * @brief hsospUpdate on `st` for the batch `b` (the stages identify_affected, seed, loop,
 *        recompute and finalize), its budget triggers as MOSP_ESCHER's (ADR 0092).
 * @throws convergence_error under budget_action::error for a spent work or iteration budget.
 */
hyper_sssp_engine_counters hyper_sssp_cuda_update(const resources& res, hyper_sssp_state& st,
                                                  const hyper_sssp_graph& g,
                                                  const hyper_sssp_batch& b,
                                                  const hyper_sssp_budget& budget,
                                                  const framework::hook_stages& names);

/// The distances and parents of a device state on the host (clone(), the accessors' caches).
void hyper_sssp_cuda_download(const resources& res, const hyper_sssp_state& st,
                              std::vector<std::int64_t>& dist, std::vector<std::int32_t>& parent);

/// Upload host distances and parents into a device state (clone() of a host result).
void hyper_sssp_cuda_upload(const resources& res, hyper_sssp_state& st,
                            const std::vector<std::int64_t>& dist,
                            const std::vector<std::int32_t>& parent);

/// The device arrays of the distances and parents (the result's views).
const std::int64_t* hyper_sssp_cuda_distances(const hyper_sssp_state& st) noexcept;
/// @copydoc hyper_sssp_cuda_distances
const std::int32_t* hyper_sssp_cuda_parents(const hyper_sssp_state& st) noexcept;
/// Set the public mode's source flags of `ids` on the device to `value`.
void hyper_sssp_cuda_set_holds(const resources& res, hyper_sssp_state& st,
                               const std::vector<std::int64_t>& ids, std::uint8_t value);
/// The bytes of a device state (the memory gate's graph scope: MOSP_ESCHER's
/// HsospState::deviceBytes(), plus the shared engine's scratch once it ran).
std::int64_t hyper_sssp_cuda_bytes(const hyper_sssp_state& st) noexcept;
/// The bytes of the shared engine's scratch of a result (0 before its first run).
std::int64_t hyper_sssp_operators_bytes(const hyper_sssp_state& st) noexcept;

// ------------------------------------------------------------------------------------------------
// The shared engine (operators.cu, operators.cuh; engine::operators on cuda; M11b, O12, ADR 0095)
// ------------------------------------------------------------------------------------------------

/**
 * @brief hyper_sssp on the shared sssp CUDA operators engine: sssp's kernels over the line graph
 *        (slack rows, node weights, the symmetric view, the base offers, the work counter) and the
 *        host loop of sssp's operators engine with MOSP_ESCHER's budget triggers where they apply
 *        (before each push round: the push rounds against max_iterations, then the relaxations
 *        against work_budget x max(1, live entries); an input distance that does not fit, or
 *        distance-only packing: distance_width; an unfinished pointer jumping:
 *        invalidation_depth). One member per framework hook (Tier A); the problem asks spent()
 *        after the loop (framework over_budget) and then recomputes.
 *
 * Bound to one run of one result (its device state sized; `b` null for compute()). The device
 * arrays are the result's (the distances, parents and packed words) with the as-is engine's work
 * arrays as sssp's scratch (ADR 0095 section 3), plus the shared engine's own scratch (st.ops).
 */
class hyper_sssp_cuda_operators_engine {
 public:
  /**
   * @brief Bind a run.
   * @param[in]     res        Resources of the CUDA backend (must outlive the engine).
   * @param[in,out] st         The result's state (its device state sized for the graph).
   * @param[in]     g          The line graph after the batch.
   * @param[in]     b          The batch (null: compute()).
   * @param[in]     budget     The budget (an update's).
   * @param[in]     max_weight The largest weight before or after the batch (the packing bound).
   * @throws not_supported_error if the cuda backend is not built.
   */
  hyper_sssp_cuda_operators_engine(const resources& res, hyper_sssp_state& st,
                                   const hyper_sssp_graph& g, const hyper_sssp_batch* b,
                                   const hyper_sssp_budget& budget, std::int64_t max_weight);
  /// update(): pack the old tree, the batch as sssp_changes, the roots, pointer jumping,
  /// invalidation, the insertion heads (1 synchronization).
  void identify_affected();
  /// update(): the pull pass of the candidates.
  void seed();
  /// compute(): every word infinite, the seeds (the source row, or the base nodes) the frontier.
  void reset();
  /// Step 2: near-far with the push, to the fixed point (update: under the budget).
  void loop();
  /// Unpack the words into the result's distances and parents (1 synchronization).
  void finalize();
  /// @brief Whether the update stopped for a trigger (then the problem recomputes).
  /// @return The update must fall back.
  [[nodiscard]] bool spent() const noexcept {
    return reason_ != hyper_sssp::fallback_reason::none;
  }
  /// @brief Why it stopped. @return The trigger (none: it did not).
  [[nodiscard]] hyper_sssp::fallback_reason reason() const noexcept {
    return reason_;
  }
  /// @brief Whether the words hold the parents (else compute() takes the ported recompute and an
  ///        update falls back with distance_width). @return Packed parents.
  [[nodiscard]] bool packed() const noexcept {
    return parent_bits_ > 0;
  }
  /// @brief The run's counters. @return They (host_syncs counted).
  [[nodiscard]] const hyper_sssp_engine_counters& counters() const noexcept {
    return counters_;
  }

 private:
  [[nodiscard]] unsigned int grid(std::int64_t work) const noexcept;
  int next_generation();

  const resources& res_;
  hyper_sssp_state& st_;
  hyper_sssp_graph g_;
  const hyper_sssp_batch* b_ = nullptr;
  hyper_sssp_budget budget_;
  hyper_sssp_engine_counters counters_{};
  hyper_sssp::fallback_reason reason_ = hyper_sssp::fallback_reason::none;
  std::int64_t n_ = 0;
  int source_ = -1;                  ///< the virtual mode's source row (-1: the public mode)
  int parent_bits_ = 0;              ///< the packing (0: distance-only words)
  std::uint64_t no_parent_ = 0;      ///< the packing's "no parent" value
  std::uint64_t max_distance_ = 0;   ///< the bound n x max weight
  std::uint64_t delta_ = 0;          ///< the near-far width
  int max_rounds_ = 0;               ///< pointer-jumping rounds enqueued
  int grid_cap_ = 1;                 ///< blocks the device holds at once
  std::int64_t candidates_ = 0;      ///< candidates of the pull pass
  std::int64_t frontier_bound_ = 0;  ///< an upper bound of the frontier's length
  bool stopped_ = false;             ///< a trigger stopped the update: the later hooks do nothing
};

// ------------------------------------------------------------------------------------------------
// The problem
// ------------------------------------------------------------------------------------------------

/**
 * @brief The hyper_sssp problem (see the file comment).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 */
template <typename vertex_t, typename weight_t>
class hyper_sssp_problem final
    : public framework::problem_base<hyper_sssp_problem<vertex_t, weight_t>,
                                     framework::family::fixed_point> {
 public:
  static constexpr std::string_view name = "hyper_sssp";  ///< stages "<name>.<hook>"
  using container_type = hypergraph<vertex_t, weight_t>;  ///< the container
  using stats_type = hyper_sssp::stats;                   ///< the stats of update()
  using frontier_type = framework::internal_frontier;     ///< frontiers are internal
  using old_graph = framework::old_view<container_type>;  ///< G_t
  using new_graph = framework::new_view<container_type>;  ///< G_{t+1}
  /// The batch before the commit (batch.normalized: Step 0).
  using requested = framework::hyper_requested_batch<vertex_t, weight_t>;
  /// What the commit did (applied.record: the applied record with the pair delta).
  using applied = framework::hyper_applied_batch<vertex_t, weight_t>;

  /// The problem of one update of `r`.
  explicit hyper_sssp_problem(hyper_sssp::result<vertex_t>& r)
      : state_(&hyper_sssp_access::state(r)), target_(&r) {}
  /// The problem of one compute() filling `st`.
  explicit hyper_sssp_problem(hyper_sssp_state& st) : state_(&st) {}

  // ---- lifecycle (no stage) ----

  /// The result this problem updates (nullptr for compute()).
  [[nodiscard]] const void* target() const noexcept {
    return target_;
  }
  /// The checks before anything changes (backend, placement, the stale or poisoned result, the
  /// storage and line graph, the weights of the batch's insertions).
  void begin_update(framework::context& ctx, old_graph g, const requested& batch);
  /// Bind G_{t+1}: the line graph, the arrays sized for its ids, the batch's lists.
  void resume(framework::context& ctx, new_graph g, const applied& applied);
  /// Record the hypergraph state the result matches.
  void end_update(framework::context& ctx, new_graph g, const stats_type& stats);
  /// Mark the result unusable (a failed algorithm phase).
  void poison() noexcept {
    state_->poisoned = true;
  }
  /// hyper_sssp reads the line graph: the commit keeps it (line_graph_policy::automatic builds
  /// it before the commit, so its delta is inside the commit).
  [[nodiscard]] bool reads_prepared_graph() const noexcept {
    return true;
  }
  /// The budget of the algorithm phase (invariant I9): nothing allocated once reserved; the host
  /// engine does not synchronize, the CUDA engine once per pointer-jumping and push round (an
  /// unbounded count: MOSP_ESCHER reads its counters every round).
  [[nodiscard]] framework::budget algorithm_budget(framework::context& ctx) const noexcept;
  /// The engine: MOSP_ESCHER's kernels on cuda (Tier B), the host emulation elsewhere (Tier A).
  [[nodiscard]] engine select_engine(framework::context& ctx) const noexcept;

  // ---- Tier A: the host engine (sequential) ----

  /// hyper_sssp.identify_affected: the roots and the invalidated subtrees.
  void identify_affected(framework::context& ctx, new_graph g, const applied& applied,
                         frontier_type& f);
  /// hyper_sssp.seed: the first offers (compute(): the source).
  void seed(framework::context& ctx, new_graph g, frontier_type& f);
  /// hyper_sssp.loop: the priority queue (update) or the pull rounds (compute()), one call.
  void loop(framework::context& ctx, new_graph g, frontier_type& in, frontier_type& out);
  /// The work budget (framework over_budget): spent when the loop stopped at it; throws
  /// convergence_error under budget_action::error.
  bool over_budget(framework::context& ctx);
  /// hyper_sssp.recompute: the fallback (emulateSospRecompute).
  void recompute(framework::context& ctx, new_graph g, stats_type& stats);
  /// hyper_sssp.finalize: the counters.
  void finalize(framework::context& ctx, stats_type& stats);
  /// hyper_sssp.reset (compute()): the arrays sized and cleared.
  void reset(framework::context& ctx);
  /// hyper_sssp.seed (compute()): the line graph bound.
  void seed_static(framework::context& ctx, new_graph g, frontier_type& f);

  // ---- Tier B: the CUDA engine ----

  /// hyper_sssp.enact_fused: hsospUpdate (its own stages inside).
  void enact_fused(framework::context& ctx, new_graph g, const applied& applied, stats_type& stats);
  /// hyper_sssp.enact_fused (compute()): hsospRecompute.
  void compute_fused(framework::context& ctx, new_graph g, stats_type& stats);

  /// The engine's counters of this run.
  [[nodiscard]] const hyper_sssp_engine_counters& counters() const noexcept {
    return counters_;
  }

 private:
  /// compute(): bind `h`, size the arrays, the public mode's source flags.
  void bind_static(framework::context& ctx, const container_type& h);
  /// Whether `id` held the source before the batch (the public mode's flags).
  [[nodiscard]] bool before_holds(std::int64_t id) const;
  /// The line graph of `h` as the engines read it.
  hyper_sssp_graph graph_of(const container_type& h) const;
  /// The budget of this update.
  hyper_sssp_budget budget_of() const;
  /// Size the result's arrays for the node count of `g` (a reservation when they grow).
  void size_arrays(framework::context& ctx, const hyper_sssp_graph& g);
  /// The stats both engines fill.
  void fill_stats(const hyper_sssp_engine_counters& c, stats_type& stats) const;
  /// The largest weight the packing of the shared engine must hold (the running maximum after
  /// the batch; 2^31 - 1 without validate_inputs).
  [[nodiscard]] std::int64_t weight_bound() const noexcept;
  /// The shared engine's static solve into the result (compute(), the fallback, the recompute):
  /// the shared compute when the knobs ask for it and the words hold parents, else the ported
  /// recompute (hsospRecompute). @return The counters of the solve.
  hyper_sssp_engine_counters cuda_operators_solve(framework::context& ctx);

  hyper_sssp_state* state_ = nullptr;
  const hyper_sssp::result<vertex_t>* target_ = nullptr;
  bool computing_ = false;
  hyper_sssp_graph graph_{};
  hyper_sssp_batch batch_{};
  std::optional<workspace_pool::lease<hyper_sssp_workspace>> ws_;
  std::optional<hyper_sssp_cuda_operators_engine> ops_;  ///< the shared engine of this run (cuda)
  hyper_sssp_engine_counters counters_{};
  std::int64_t pending_max_ = 0;  ///< the running maximum after the batch's weights
};

}  // namespace dyng::detail
