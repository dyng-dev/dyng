// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu (hsospUpdate's budget triggers and
// their order, lines 1065-1195) and MOSP_ESCHER@4b86159:mosp/src/sospUpdateGpu.cu (nearFar: the
// host loop of sssp's operators engine)
/**
 * @file operators.cu
 * @brief hyper_sssp's engine::operators on cuda: the host loop of the shared sssp engine over the
 *        line graph (M11b, decision O12; ADR 0095). The kernels are sssp's (algorithms/sssp/
 *        operators.cuh) and hyper_sssp's own (operators.cuh).
 *
 * The loop is sssp's operators engine (algorithms/sssp/operators.cu: the same kernels, the same
 * control block, the same synchronizations), with what hyper_sssp needs on top:
 *   - the batch: the slack_csr store's pair delta and ids become sssp_changes on the device
 *     (changes_kernel, heads_kernel), the public mode's lost and gained nodes with them;
 *   - the seeds: the virtual mode's source row at 0, or the public mode's base nodes at their own
 *     weight (and the base offers of every pull);
 *   - the budget, MOSP_ESCHER's triggers where they apply (hsospUpdate, hsospDevice.cu:1065-1195):
 *     before each push round, iterations >= max_iterations, then the relaxations (row lengths of
 *     the pulled candidates and of the pushed reachable nodes) > work_budget x max(1, live
 *     entries); an input distance that does not fit the packing (pack_tree's overflow), or
 *     distance-only packing at all (the bound n x max weight does not leave the parent bits room):
 *     distance_width; a pointer jumping that did not end in ceil(log2 n) + 1 rounds:
 *     invalidation_depth. The problem recomputes after a trigger (or throws under
 *     budget_action::error for the first two).
 * The device arrays are the result's: the distances, parents and packed words of its device
 * state (state.cuh), and the as-is engine's work arrays as sssp's scratch (ADR 0095 section 3):
 * flag = inv_a, ancestor = jump_a, candidates = inv_b, frontier = jump_b, near lists = front /
 * next, far piles = cand_list / aff_list, in_far = is_candidate (all zero between runs of either
 * engine), the stamps and their generations the as-is engine's (both only compare a stamp with a
 * newer generation). The shared engine's own scratch (st.ops) holds the control block, its pinned
 * mirror and the batch's change lists.
 */
#include "algorithms/hyper_sssp/operators.cuh"
#include "algorithms/hyper_sssp/problem.hpp"
#include "algorithms/hyper_sssp/state.cuh"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "framework/scratch_buffer.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace hyper_ops {

/// The shared engine's own scratch of a CUDA result (an update's lists are in the pooled
/// workspace, hyper_sssp_workspace::ops_*, so a steady-state update allocates nothing: I9).
struct device_scratch {
  scratch_buffer<unsigned char> control;  ///< run_control (device)
  buffer<unsigned char> host_control;     ///< its pinned mirror
  scratch_buffer<int> base;               ///< compute(): the public mode's base nodes

  /// @brief The device bytes held. @return They.
  [[nodiscard]] std::int64_t bytes() const noexcept {
    return static_cast<std::int64_t>(control.bytes() + base.bytes());
  }
};

}  // namespace hyper_ops

void hyper_sssp_operators_scratch_deleter::operator()(hyper_ops::device_scratch* s) const noexcept {
  delete s;
}

namespace {

namespace ops = sssp_operators;
using hyper_ops::run_control;
using sssp_kernels::block_size;
using sssp_kernels::u64;

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

/// The shared engine's scratch of `st` (allocated on first use).
hyper_ops::device_scratch& scratch_of(const resources& res, hyper_sssp_state& st) {
  if (st.ops == nullptr) {
    note_reservation();  // a result's first run on the shared engine (invariant I9)
    st.ops.reset(new hyper_ops::device_scratch);
    st.ops->control.reserve(res, sizeof(run_control));
    st.ops->host_control = buffer<unsigned char>(
        sizeof(run_control), res.stream(), resources_access::staging_memory(res), res.device());
  }
  return *st.ops;
}

run_control* device_control(hyper_ops::device_scratch& s) noexcept {
  return reinterpret_cast<run_control*>(s.control.data());
}

/// Copy the control block (and the work counter) to the pinned mirror and wait for it.
run_control read_control(const resources& res, hyper_ops::device_scratch& s,
                         hyper_sssp_engine_counters& c) {
  const cudaStream_t stream = native(res);
  DYNG_CUDA_TRY(cudaMemcpyAsync(s.host_control.data(), s.control.data(), sizeof(run_control),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // the budget of the phase (I9) counts it
  ++c.host_syncs;
  run_control out;
  std::memcpy(&out, s.host_control.data(), sizeof(out));
  return out;
}

/// Zero the control block and the work counter (the minimum slot idle).
void init_control(const resources& res, hyper_ops::device_scratch& s) {
  const cudaStream_t stream = native(res);
  run_control* rc = device_control(s);
  ops::init_control_kernel<int><<<1, 1, 0, stream>>>(&rc->c);
  DYNG_CHECK_KERNEL(stream);
  DYNG_CUDA_TRY(cudaMemsetAsync(&rc->work, 0, sizeof(u64), stream));
}

/// Upload a host list to `device` (ints).
void upload(const resources& res, int* device, const std::vector<int>& list) {
  if (list.empty()) return;
  DYNG_CUDA_TRY(cudaMemcpyAsync(device, list.data(), sizeof(int) * list.size(),
                                cudaMemcpyHostToDevice, native(res)));
}

/// The line graph as the shared engine's kernels read it.
hyper_ops::line_graph_view view_of(const hyper_sssp_graph& g, const hyper_sssp_state& st,
                                   run_control* rc) noexcept {
  hyper_ops::line_graph_view v;
  v.rows.row_start = reinterpret_cast<const long long*>(g.rows.row_start);
  v.rows.deg = g.rows.deg;
  v.rows.col_ind = g.rows.col_ind;
  v.node_w = reinterpret_cast<const long long*>(g.rows.node_weight);
  v.zero_a = g.endpoints.zero_weight_a < 0 ? -1 : static_cast<int>(g.endpoints.zero_weight_a);
  v.zero_b = g.endpoints.zero_weight_b < 0 ? -1 : static_cast<int>(g.endpoints.zero_weight_b);
  v.holds = st.endpoints.is_virtual() ? nullptr : st.dev->d_holds;
  v.work = &rc->work;
  return v;
}

int as_count(std::int64_t count, const char* what) {
  DYNG_EXPECTS(count <= std::numeric_limits<int>::max(), "hyper_sssp: ", count, " ", what,
               " exceed the engine's int32 lists");
  return static_cast<int>(count);
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Binding: the packing, the delta, the grid
// ------------------------------------------------------------------------------------------------

hyper_sssp_cuda_operators_engine::hyper_sssp_cuda_operators_engine(
    const resources& res, hyper_sssp_state& st, const hyper_sssp_graph& g,
    const hyper_sssp_batch* b, const hyper_sssp_budget& budget, std::int64_t max_weight)
    : res_(res), st_(st), g_(g), b_(b), budget_(budget), n_(g.rows.num_nodes) {
  counters_.engine_used = engine::operators;
  counters_.host_syncs = 0;
  DYNG_EXPECTS(g_.rows.space == memory_space::device,
               "hyper_sssp: the line graph is not on the device");
  DYNG_EXPECTS(n_ <= std::numeric_limits<int>::max(), "hyper_sssp: ", n_,
               " line-graph nodes exceed the engine's int32 ids");
  source_ = st_.endpoints.is_virtual() ? static_cast<int>(st_.endpoints.source_hyperedge) : -1;
  if (n_ == 0) return;
  // The bound of every shortest distance: a simple path holds at most n nodes of weight at most
  // the largest weight (the public mode counts the first node's weight too). The words keep the
  // parents next to it when it fits (sssp's packing), else hold distances only.
  const u64 weight = static_cast<u64>(std::max<std::int64_t>(max_weight, 1));
  const u64 nodes = static_cast<u64>(n_);
  DYNG_EXPECTS(weight <= sssp_kernels::output_max_distance / nodes, "hyper_sssp: distances up to ",
               weight, " * ", nodes, " do not fit in 62 bits");
  max_distance_ = weight * nodes;
  const sssp_kernels::packing packed = sssp_kernels::make_packing(n_, max_distance_);
  parent_bits_ = packed.parent_bits;
  no_parent_ = packed.no_parent;
  const std::int64_t delta = st_.ops_knobs.delta;
  delta_ = delta > 0 ? static_cast<u64>(delta) : (u64{1} << 62);  // unbounded: every round near
  int rounds = 0;
  while ((1LL << rounds) < n_) ++rounds;
  max_rounds_ = rounds + 1;
  const cuda_device_properties& props = resources_access::device_properties(res_);
  grid_cap_ = std::max(props.multiprocessor_count, 1) *
              std::max(props.max_threads_per_multiprocessor / block_size, 1);
  (void)scratch_of(res_, st_);
  st_.dev->stream = native(res_);
}

unsigned int hyper_sssp_cuda_operators_engine::grid(std::int64_t work) const noexcept {
  const std::int64_t blocks = (std::max<std::int64_t>(work, 1) + block_size - 1) / block_size;
  return static_cast<unsigned int>(std::min<std::int64_t>(blocks, grid_cap_));
}

int hyper_sssp_cuda_operators_engine::next_generation() {
  hsosp::device_state& dev = *st_.dev;
  if (dev.epoch >= std::numeric_limits<int>::max() - 1) {
    // The stamps start again (both engines compare a stamp only with a newer generation).
    DYNG_CUDA_TRY(cudaMemsetAsync(
        dev.d_stamp, 0, sizeof(int) * static_cast<std::size_t>(dev.max_nodes), native(res_)));
    dev.epoch = 0;
  }
  return ++dev.epoch;
}

// ------------------------------------------------------------------------------------------------
// update(): identify_affected, seed (MOSP_ESCHER's hsospUpdate steps 1-3 on sssp's kernels)
// ------------------------------------------------------------------------------------------------

void hyper_sssp_cuda_operators_engine::identify_affected() {
  if (n_ == 0) return;
  if (!packed()) {
    // Distance-only words cannot carry the canonical parents through the update: recompute
    // (MOSP_ESCHER's 32-bit packed overflow; HD8).
    reason_ = hyper_sssp::fallback_reason::distance_width;
    stopped_ = true;
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  hsosp::device_state& dev = *st_.dev;
  hyper_ops::device_scratch& s = *st_.ops;
  run_control* rc = device_control(s);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const int n = static_cast<int>(n_);
  const hyper_sssp_batch& b = *b_;
  const hyper::slack_delta_view& dd = b.device;
  const int num_del = as_count(dd.num_deleted, "deleted pairs");
  const int num_ins = as_count(dd.num_inserted, "inserted pairs");
  const int num_ids = as_count(dd.num_new + dd.num_dead, "new and dead ids");
  hyper_sssp_workspace& ws = *b.scratch;
  // The public mode's lists (the workspace's int32 copies, as the as-is engine uploads them).
  ws.lost32.clear();
  ws.gained32.clear();
  if (!st_.endpoints.is_virtual()) {
    reserve_scratch(ws.lost32, b.lost_source->size());
    reserve_scratch(ws.gained32, b.gained_source->size());
    for (const std::int64_t id : *b.lost_source) ws.lost32.push_back(static_cast<int>(id));
    for (const std::int64_t id : *b.gained_source) ws.gained32.push_back(static_cast<int>(id));
  }
  const int num_lost = as_count(static_cast<std::int64_t>(ws.lost32.size()), "lost nodes");
  const int num_gained = as_count(static_cast<std::int64_t>(ws.gained32.size()), "gained nodes");
  const std::int64_t num_changed = 2 * static_cast<std::int64_t>(num_del) + num_ids + num_lost;
  const std::int64_t num_heads = 2 * static_cast<std::int64_t>(num_ins) + num_gained;
  (void)as_count(num_changed, "changed edges");
  (void)as_count(num_heads, "insertion heads");
  const auto* pairs = reinterpret_cast<const int2*>(dd.pairs);
  int* from = ws.ops_from.reserve_with_headroom(res_, static_cast<std::size_t>(num_changed));
  int* to = ws.ops_to.reserve_with_headroom(res_, static_cast<std::size_t>(num_changed));
  int* heads = ws.ops_heads.reserve_with_headroom(res_, static_cast<std::size_t>(num_heads));
  int* lists =
      ws.ops_lists.reserve_with_headroom(res_, static_cast<std::size_t>(num_lost + num_gained));
  upload(res_, lists, ws.lost32);
  upload(res_, lists + num_lost, ws.gained32);

  init_control(res_, s);
  DYNG_CUDA_TRY(cudaMemsetAsync(dev.d_inv_a, 0, sizeof(int) * static_cast<std::size_t>(n), stream));
  ops::pack_tree_kernel<int><<<grid(n_), block_size, 0, stream>>>(
      n, dev.d_dist, dev.d_parent, dev.d_packed, packed, max_distance_, dev.d_jump_a, &rc->c);
  DYNG_CHECK_KERNEL(stream);
  // invalidate_subtree, strategy pointer_jumping: the batch's roots, then the rounds.
  if (num_changed > 0) {
    hyper_ops::changes_kernel<<<grid(num_changed), block_size, 0, stream>>>(
        pairs, num_del, dd.ids, num_ids, lists, num_lost, dev.d_parent, source_, from, to);
    DYNG_CHECK_KERNEL(stream);
    ops::mark_roots_kernel<int><<<grid(num_changed), block_size, 0, stream>>>(
        from, to, static_cast<int>(num_changed), dev.d_parent, dev.d_inv_a);
    DYNG_CHECK_KERNEL(stream);
    for (int round = 0; round < max_rounds_; ++round) {
      ops::pointer_jump_kernel<int>
          <<<grid(n_), block_size, 0, stream>>>(n, dev.d_jump_a, dev.d_inv_a, &rc->c, round);
      DYNG_CHECK_KERNEL(stream);
    }
  }
  const int generation = next_generation();
  ops::invalidate_kernel<int><<<grid(n_), block_size, 0, stream>>>(
      n, dev.d_inv_a, dev.d_packed, dev.d_stamp, generation, dev.d_inv_b, &rc->c);
  DYNG_CHECK_KERNEL(stream);
  if (num_heads > 0) {
    hyper_ops::heads_kernel<<<grid(num_heads), block_size, 0, stream>>>(
        pairs + num_del, num_ins, lists + num_lost, num_gained, heads);
    DYNG_CHECK_KERNEL(stream);
    ops::insert_heads_kernel<int><<<grid(num_heads), block_size, 0, stream>>>(
        heads, static_cast<int>(num_heads), source_, dev.d_stamp, generation, dev.d_inv_b, &rc->c);
    DYNG_CHECK_KERNEL(stream);
  }
  const run_control result = read_control(res_, s, counters_);
  int rounds_run = 0;
  while (rounds_run < max_rounds_ && result.c.active[rounds_run] != 0) ++rounds_run;
  counters_.jump_rounds = num_changed > 0 ? std::min(rounds_run + 1, max_rounds_) : 0;
  counters_.seed_count = static_cast<std::int64_t>(result.c.invalidated);
  if (result.c.overflow != 0) {
    reason_ = hyper_sssp::fallback_reason::distance_width;  // an input distance does not fit
    stopped_ = true;
    return;
  }
  if (num_changed > 0 && result.c.active[max_rounds_ - 1] != 0) {
    reason_ = hyper_sssp::fallback_reason::invalidation_depth;  // the jumping did not end
    stopped_ = true;
    return;
  }
  candidates_ = static_cast<std::int64_t>(result.c.candidates);
}

void hyper_sssp_cuda_operators_engine::seed() {
  frontier_bound_ = 0;
  if (n_ == 0 || stopped_ || candidates_ == 0) return;
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  hsosp::device_state& dev = *st_.dev;
  hyper_ops::device_scratch& s = *st_.ops;
  run_control* rc = device_control(s);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const hyper_ops::line_graph_view view = view_of(g_, st_, rc);
  const int generation = next_generation();
  const int count = static_cast<int>(candidates_);
  if (st_.ops_knobs.warp_rows) {
    hyper_ops::pull_warp_kernel<<<grid(32 * candidates_), block_size, 0, stream>>>(
        dev.d_inv_b, count, view, dev.d_packed, packed, max_distance_, dev.d_stamp, generation,
        dev.d_jump_b, &rc->c);
  } else {
    hyper_ops::pull_thread_kernel<<<grid(candidates_), block_size, 0, stream>>>(
        dev.d_inv_b, count, view, dev.d_packed, packed, max_distance_, dev.d_stamp, generation,
        dev.d_jump_b, &rc->c);
  }
  DYNG_CHECK_KERNEL(stream);
  frontier_bound_ = candidates_;  // the frontier is a subset of the candidates
}

// ------------------------------------------------------------------------------------------------
// compute(): reset (the seeds)
// ------------------------------------------------------------------------------------------------

void hyper_sssp_cuda_operators_engine::reset() {
  frontier_bound_ = 0;
  if (n_ == 0) return;
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  hsosp::device_state& dev = *st_.dev;
  hyper_ops::device_scratch& s = *st_.ops;
  run_control* rc = device_control(s);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const int n = static_cast<int>(n_);
  // The public mode's base: the live hyperedges of the source vertex (the flags' host copy).
  std::vector<int> base;
  if (!st_.endpoints.is_virtual()) {
    for (std::size_t i = 0; i < st_.holds_source.size() && i < static_cast<std::size_t>(n_); ++i) {
      if (st_.holds_source[i] != 0 && g_.live(static_cast<std::int64_t>(i))) {
        base.push_back(static_cast<int>(i));
      }
    }
  }
  int* lists = s.base.reserve_with_headroom(res_, base.size());
  upload(res_, lists, base);
  if (!st_.endpoints.is_virtual() && !st_.holds_source.empty()) {
    // The source flags on the device (the host copy is current: compute() rebuilt it); the pulls
    // of the later updates read them, of either engine.
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(dev.d_holds, st_.holds_source.data(),
                        std::min(st_.holds_source.size(), static_cast<std::size_t>(dev.max_nodes)),
                        cudaMemcpyHostToDevice, stream));
  }
  init_control(res_, s);
  DYNG_CUDA_TRY(
      cudaMemsetAsync(dev.d_is_candidate, 0, sizeof(int) * static_cast<std::size_t>(n), stream));
  hyper_ops::init_kernel<<<grid(n_), block_size, 0, stream>>>(n, dev.d_packed, source_, packed);
  DYNG_CHECK_KERNEL(stream);
  const int num_base = static_cast<int>(base.size());
  hyper_ops::seed_kernel<<<grid(std::max(num_base, 1)), block_size, 0, stream>>>(
      dev.d_packed, source_, lists, num_base, view_of(g_, st_, rc), packed, dev.d_jump_b, &rc->c);
  DYNG_CHECK_KERNEL(stream);
  frontier_bound_ = source_ >= 0 ? 1 : num_base;
  // Every distance is final once the loop ends: the stamps of compute() may be any older
  // generation.
}

// ------------------------------------------------------------------------------------------------
// Step 2: near-far with the push (sssp's operators loop; MOSP_ESCHER's budget before each round)
// ------------------------------------------------------------------------------------------------

void hyper_sssp_cuda_operators_engine::loop() {
  if (n_ == 0 || stopped_ || frontier_bound_ == 0) return;
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  hsosp::device_state& dev = *st_.dev;
  hyper_ops::device_scratch& s = *st_.ops;
  run_control* rc = device_control(s);
  ops::control<int>* c = &rc->c;
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const hyper_ops::line_graph_view view = view_of(g_, st_, rc);
  const bool budgeted = b_ != nullptr;  // compute() has no budget (MOSP_ESCHER's recompute)
  const bool warp = st_.ops_knobs.warp_rows;

  // The threshold and the first split over the frontier, whose length is still on the device.
  const ops::count_of<int> frontier{static_cast<int>(frontier_bound_), &c->frontier};
  ops::min_distance_kernel<int><<<grid(frontier_bound_), block_size, 0, stream>>>(
      dev.d_jump_b, frontier, dev.d_packed, packed, c);
  DYNG_CHECK_KERNEL(stream);
  ops::threshold_kernel<int><<<1, 1, 0, stream>>>(c, delta_, true);
  DYNG_CHECK_KERNEL(stream);
  int* current = dev.d_front;
  int* next = dev.d_next;
  int* far = dev.d_cand_list;
  int* far2 = dev.d_aff_list;
  ops::split_kernel<int><<<grid(frontier_bound_), block_size, 0, stream>>>(
      dev.d_jump_b, frontier, dev.d_packed, packed, dev.d_stamp, next_generation(), current,
      &c->near[0], dev.d_is_candidate, far, &c->far[0], false, nullptr, c);
  DYNG_CHECK_KERNEL(stream);
  run_control state = read_control(res_, s, counters_);
  int near_slot = 0;
  int far_slot = 0;
  int near_count = state.c.near[near_slot];
  int far_count = state.c.far[far_slot];
  std::int64_t iterations = 0;
  std::int64_t epochs = 0;
  std::int64_t largest = std::max<std::int64_t>(near_count, 0);
  while (true) {
    if (near_count > 0) {
      if (budgeted) {
        // MOSP_ESCHER's triggers before each push round, in its order (hsospDevice.cu).
        if (iterations >= budget_.max_iterations) {
          reason_ = hyper_sssp::fallback_reason::iterations;
          break;
        }
        if (static_cast<double>(state.work) > budget_.work_limit) {
          reason_ = hyper_sssp::fallback_reason::work_budget;
          break;
        }
      }
      ++iterations;
      const int gen = next_generation();
      if (warp) {
        hyper_ops::push_warp_kernel<<<grid(32LL * near_count), block_size, 0, stream>>>(
            current, near_count, view, dev.d_packed, packed, max_distance_, source_, dev.d_stamp,
            gen, next, &c->near[1 - near_slot], dev.d_is_candidate, far, &c->far[far_slot],
            &c->near[near_slot], c);
      } else {
        hyper_ops::push_thread_kernel<<<grid(near_count), block_size, 0, stream>>>(
            current, near_count, view, dev.d_packed, packed, max_distance_, source_, dev.d_stamp,
            gen, next, &c->near[1 - near_slot], dev.d_is_candidate, far, &c->far[far_slot],
            &c->near[near_slot], c);
      }
      DYNG_CHECK_KERNEL(stream);
      state = read_control(res_, s, counters_);
      near_slot = 1 - near_slot;
      near_count = state.c.near[near_slot];
      far_count = state.c.far[far_slot];
      largest = std::max<std::int64_t>(largest, near_count);
      std::swap(current, next);
      continue;
    }
    if (far_count == 0) break;
    // The near frontier is empty: raise the threshold past the far pile and re-split it.
    ++epochs;
    const ops::count_of<int> pile{far_count, nullptr};
    ops::min_distance_kernel<int>
        <<<grid(far_count), block_size, 0, stream>>>(far, pile, dev.d_packed, packed, c);
    DYNG_CHECK_KERNEL(stream);
    ops::threshold_kernel<int><<<1, 1, 0, stream>>>(c, delta_, false);
    DYNG_CHECK_KERNEL(stream);
    ops::split_kernel<int><<<grid(far_count), block_size, 0, stream>>>(
        far, pile, dev.d_packed, packed, dev.d_stamp, next_generation(), current,
        &c->near[near_slot], dev.d_is_candidate, far2, &c->far[1 - far_slot], true,
        &c->far[far_slot], c);
    DYNG_CHECK_KERNEL(stream);
    state = read_control(res_, s, counters_);
    far_slot = 1 - far_slot;
    near_count = state.c.near[near_slot];
    far_count = state.c.far[far_slot];
    largest = std::max<std::int64_t>(largest, near_count);
    std::swap(far, far2);
  }
  counters_.iterations = iterations;
  counters_.epochs = epochs;
  counters_.max_frontier = largest;
  counters_.work = static_cast<std::int64_t>(state.work);
  if (reason_ != hyper_sssp::fallback_reason::none) {
    // The update stops here; the near frontier and the far pile are dropped, their in_far marks
    // cleared for the next run (the recompute that follows writes every word).
    stopped_ = true;
    DYNG_CUDA_TRY(
        cudaMemsetAsync(dev.d_is_candidate, 0, sizeof(int) * static_cast<std::size_t>(n_), stream));
  }
}

// ------------------------------------------------------------------------------------------------
// finalize: unpack
// ------------------------------------------------------------------------------------------------

void hyper_sssp_cuda_operators_engine::finalize() {
  if (n_ == 0 || stopped_) return;
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  hsosp::device_state& dev = *st_.dev;
  hyper_ops::device_scratch& s = *st_.ops;
  run_control* rc = device_control(s);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const int n = static_cast<int>(n_);
  if (b_ != nullptr) {
    ops::unpack_counted_kernel<int><<<grid(n_), block_size, 0, stream>>>(
        n, dev.d_packed, packed, dev.d_dist, dev.d_parent, &rc->c);
  } else {
    ops::unpack_kernel<int>
        <<<grid(n_), block_size, 0, stream>>>(n, dev.d_packed, packed, dev.d_dist, dev.d_parent);
  }
  DYNG_CHECK_KERNEL(stream);
  (void)read_control(res_, s, counters_);  // a result complete when the call returns
}

// ------------------------------------------------------------------------------------------------
// Kernel registration (resources::warm_up())
// ------------------------------------------------------------------------------------------------

namespace {

/// Registers the shared engine's kernels of hyper_sssp at load time (sssp's int32 instantiations
/// are registered by sssp's operators.cu: one name per kernel address in this translation unit).
struct hyper_operators_kernel_registrar {
  hyper_operators_kernel_registrar() noexcept {
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::pull_thread_kernel),
                    "hyper_ops::pull_thread_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::pull_warp_kernel),
                    "hyper_ops::pull_warp_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::push_thread_kernel),
                    "hyper_ops::push_thread_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::push_warp_kernel),
                    "hyper_ops::push_warp_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::changes_kernel),
                    "hyper_ops::changes_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::heads_kernel),
                    "hyper_ops::heads_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::init_kernel),
                    "hyper_ops::init_kernel");
    register_kernel(reinterpret_cast<const void*>(&hyper_ops::seed_kernel),
                    "hyper_ops::seed_kernel");
  }
};

const hyper_operators_kernel_registrar hyper_operators_kernels;

}  // namespace

std::int64_t hyper_sssp_operators_bytes(const hyper_sssp_state& st) noexcept {
  return st.ops == nullptr ? 0 : st.ops->bytes();
}

}  // namespace dyng::detail
