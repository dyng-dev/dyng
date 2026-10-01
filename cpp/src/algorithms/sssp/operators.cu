// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:mosp/src/sospUpdateGpu.cu (nearFar, sospUpdateGpu,
// sospFromScratchGpu: the host loop) and MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu (choosePacking,
// the control-block checks of sospUpdateGpu)
/**
 * @file operators.cu
 * @brief The CUDA backend of sssp, operators engine (Tier A; decision O24): the host side of the
 *        multi-kernel engine, one member per framework hook. The kernels are in operators.cuh.
 *
 * The engine runs where the fused engine cannot (a device without cooperative launch:
 * engine::automatic falls back to it) and when engine::operators asks for it. It gives the fused
 * engine's bytes (operators.cuh, "Why the bytes are the same"; conformance check C4).
 *
 * Hooks and host synchronizations (each one reads the control block back):
 *   identify_affected  pack the old tree, roots, pointer-jumping rounds, invalidation, insertion
 *                      heads; 1 sync (the input checks and the number of candidates)
 *   seed               the pull pass; no sync (the frontier's length stays on the device)
 *   loop               threshold, first split, then per push iteration or threshold raise one
 *                      sync (the host decides whether to go on, as MOSP_ESCHER's nearFar())
 *   finalize           unpack (and `affected`); 1 sync
 * so an update synchronizes 3 + iterations + epochs times at most (the budget of
 * sssp_problem::algorithm_budget). compute() runs reset (the source only) and seed_static
 * instead of the first two hooks. Device errors (an input distance that does not fit, a parent
 * cycle) are recorded in the run as the fused engine records them, and the remaining hooks then do
 * nothing.
 */
#include "algorithms/sssp/operators.cuh"
#include "algorithms/sssp/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/instantiate.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>

namespace dyng::detail {

namespace {

namespace ops = sssp_operators;
using sssp_kernels::block_size;
using sssp_kernels::u64;

cudaStream_t native(const resources& res) noexcept {
  return static_cast<cudaStream_t>(res.stream().get());
}

template <typename vertex_t>
ops::control<vertex_t>* device_control(sssp_cuda_workspace<vertex_t>& ws) noexcept {
  return reinterpret_cast<ops::control<vertex_t>*>(ws.control.data());
}

/// Copy the control block to the pinned mirror and wait for it (one host synchronization).
template <typename vertex_t>
ops::control<vertex_t> read_control(const resources& res, sssp_cuda_workspace<vertex_t>& ws) {
  const cudaStream_t stream = native(res);
  DYNG_CUDA_TRY(cudaMemcpyAsync(ws.host_control.data(), ws.control.data(),
                                sizeof(ops::control<vertex_t>), cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();  // the budget of the phase (I9) counts it
  ops::control<vertex_t> out;
  std::memcpy(&out, ws.host_control.data(), sizeof(out));
  return out;
}

template <typename vertex_t>
vertex_t list_length(std::size_t count, const char* what) {
  DYNG_EXPECTS(count <= static_cast<std::size_t>(std::numeric_limits<vertex_t>::max()),
               "sssp: ", count, " ", what, " exceed the index range of the vertex type");
  return static_cast<vertex_t>(count);
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_kernels::device_csr<vertex_t, edge_t, weight_t> out_csr(
    const sssp_graph<vertex_t, edge_t, weight_t>& g) {
  return {g.num_vertices, g.out_row_ptr, g.out_col_ind, g.out_weights};
}

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_kernels::device_csr<vertex_t, edge_t, weight_t> in_csr(
    const sssp_graph<vertex_t, edge_t, weight_t>& g) {
  return {g.num_vertices, g.in_row_ptr, g.in_col_ind, g.in_weights};
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// Binding (MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu choosePacking; the checks of sospUpdateGpu)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::sssp_cuda_operators_engine(
    const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run)
    : res_(res), run_(run), n_(run.graph.num_vertices), update_(run.changes != nullptr) {
  run_.counters = sssp_counters{};
  run_.device_errors = 0;
  run_.device_error_detail = nullptr;
  if (n_ == 0) {
    return;
  }
  DYNG_EXPECTS(run_.delta > 0, "sssp: the near-far width delta must be > 0, got ", run_.delta);
  if (!update_) {
    DYNG_EXPECTS(run_.source >= 0 && run_.source < n_, "sssp: source ", run_.source,
                 " is out of range [0, ", n_, ")");
  }
  // The fused engine's packing (cuda.cu choose_packing): the bound (n - 1) * max weight, and
  // distance-only words when it does not fit next to the parent bits.
  const u64 weight = static_cast<u64>(std::max<std::int64_t>(run_.max_weight, 1));
  const u64 hops = static_cast<u64>(std::max<std::int64_t>(n_ - 1, 1));
  DYNG_EXPECTS(weight <= sssp_kernels::output_max_distance / hops, "sssp: distances up to ", weight,
               " * ", hops, " do not fit in 62 bits");
  max_distance_ = weight * hops;
  const sssp_kernels::packing packed = sssp_kernels::make_packing(n_, max_distance_);
  parent_bits_ = packed.parent_bits;
  no_parent_ = packed.no_parent;
  int rounds = 0;
  while ((1LL << rounds) < static_cast<long long>(n_)) {
    ++rounds;
  }
  max_rounds_ = rounds + 1;  // the fused kernel's max_rounds
  const cuda_device_properties& props = resources_access::device_properties(res_);
  grid_cap_ = std::max(props.multiprocessor_count, 1) *
              std::max(props.max_threads_per_multiprocessor / block_size, 1);
  run_.counters.packed_parents = parent_bits_ > 0;
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  ws.reserve(res_, n_);  // a no-op: the problem leased the workspace for n vertices
}

template <typename vertex_t, typename edge_t, typename weight_t>
unsigned int sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::grid(
    std::int64_t work) const noexcept {
  const std::int64_t blocks = (std::max<std::int64_t>(work, 1) + block_size - 1) / block_size;
  return static_cast<unsigned int>(std::min<std::int64_t>(blocks, grid_cap_));
}

// ------------------------------------------------------------------------------------------------
// compute(): reset, seed_static (MOSP_ESCHER's sospFromScratchGpu)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::reset() {
  if (n_ == 0) {
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  ops::control<vertex_t>* c = device_control(ws);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  ops::init_control_kernel<vertex_t><<<1, 1, 0, stream>>>(c);
  DYNG_CHECK_KERNEL(stream);
  ops::init_from_scratch_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
      n_, ws.packed.data(), run_.source, packed, ws.frontier.data(), c);
  DYNG_CHECK_KERNEL(stream);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::seed_static() {
  frontier_bound_ = n_ == 0 ? 0 : 1;  // the source (reset() listed it)
}

// ------------------------------------------------------------------------------------------------
// update(): identify_affected, seed (MOSP_ESCHER's sospUpdateGpu, Step 1)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::identify_affected() {
  if (n_ == 0) {
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  ops::control<vertex_t>* c = device_control(ws);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const sssp_changes<vertex_t>& changes = *run_.changes;
  const vertex_t num_changed = list_length<vertex_t>(changes.num_changed, "changed edges");
  const vertex_t num_heads = list_length<vertex_t>(changes.num_insert_heads, "insertion heads");
  auto* distances = reinterpret_cast<const long long*>(run_.distances);

  ops::init_control_kernel<vertex_t><<<1, 1, 0, stream>>>(c);
  DYNG_CHECK_KERNEL(stream);
  ops::pack_tree_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
      n_, distances, run_.parents, ws.packed.data(), packed, max_distance_, ws.ancestor.data(), c);
  DYNG_CHECK_KERNEL(stream);
  // invalidate_subtree, strategy pointer_jumping: roots, then the rounds (each round after a
  // converged one returns at once).
  if (num_changed > 0) {
    ops::mark_roots_kernel<vertex_t><<<grid(num_changed), block_size, 0, stream>>>(
        changes.changed_from, changes.changed_to, num_changed, run_.parents, ws.flag.data());
    DYNG_CHECK_KERNEL(stream);
    for (int round = 0; round < max_rounds_; ++round) {
      ops::pointer_jump_kernel<vertex_t>
          <<<grid(n_), block_size, 0, stream>>>(n_, ws.ancestor.data(), ws.flag.data(), c, round);
      DYNG_CHECK_KERNEL(stream);
    }
  }
  const int generation = ws.next_generation(res_);
  ops::invalidate_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
      n_, ws.flag.data(), ws.packed.data(), ws.stamp.data(), generation, ws.candidates.data(), c);
  DYNG_CHECK_KERNEL(stream);
  if (num_heads > 0) {
    ops::insert_heads_kernel<vertex_t><<<grid(num_heads), block_size, 0, stream>>>(
        changes.insert_heads, num_heads, run_.source, ws.stamp.data(), generation,
        ws.candidates.data(), c);
    DYNG_CHECK_KERNEL(stream);
  }
  const ops::control<vertex_t> result = read_control(res_, ws);
  if (result.overflow != 0) {
    run_.device_errors = bits_of(device_error::invalid_input);
    run_.device_error_detail =
        "an input distance is negative or larger than (n - 1) * max weight; the initial tree "
        "does not belong to this graph";
    stopped_ = true;
    return;
  }
  if (num_changed > 0 && result.active[max_rounds_ - 1] != 0) {
    run_.device_errors = bits_of(device_error::parent_cycle);
    run_.device_error_detail =
        "the input shortest-path tree has a parent cycle (pointer jumping did not reach a root "
        "in ceil(log2 n) + 1 rounds)";
    stopped_ = true;
    return;
  }
  run_.counters.invalidated = static_cast<std::int64_t>(result.invalidated);
  candidates_ = static_cast<std::int64_t>(result.candidates);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::seed() {
  frontier_bound_ = 0;
  if (n_ == 0 || stopped_ || candidates_ == 0) {
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  // neighbor_reduce over the in-edges with the packed argmin.
  const int generation = ws.next_generation(res_);
  ops::pull_kernel<vertex_t, edge_t, weight_t><<<grid(candidates_), block_size, 0, stream>>>(
      ws.candidates.data(), static_cast<vertex_t>(candidates_), in_csr(run_.graph),
      ws.packed.data(), packed, max_distance_, ws.stamp.data(), generation, ws.frontier.data(),
      device_control(ws));
  DYNG_CHECK_KERNEL(stream);
  frontier_bound_ = candidates_;  // the frontier is a subset of the candidates
}

// ------------------------------------------------------------------------------------------------
// Step 2 (MOSP_ESCHER's nearFar(); the decisions of MOSP-CUDA's persistent loop)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::loop() {
  if (n_ == 0 || stopped_ || frontier_bound_ == 0) {
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  ops::control<vertex_t>* c = device_control(ws);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  const auto delta = static_cast<u64>(run_.delta);
  const auto out = out_csr(run_.graph);

  // The threshold (smallest frontier distance + delta) and the first split, both over the
  // frontier, whose length is still on the device.
  const ops::count_of<vertex_t> frontier{static_cast<vertex_t>(frontier_bound_), &c->frontier};
  ops::min_distance_kernel<vertex_t><<<grid(frontier_bound_), block_size, 0, stream>>>(
      ws.frontier.data(), frontier, ws.packed.data(), packed, c);
  DYNG_CHECK_KERNEL(stream);
  ops::threshold_kernel<vertex_t><<<1, 1, 0, stream>>>(c, delta, true);
  DYNG_CHECK_KERNEL(stream);
  vertex_t* current = ws.list_a.data();
  vertex_t* next = ws.list_b.data();
  vertex_t* far = ws.far_a.data();
  vertex_t* far2 = ws.far_b.data();
  ops::split_kernel<vertex_t><<<grid(frontier_bound_), block_size, 0, stream>>>(
      ws.frontier.data(), frontier, ws.packed.data(), packed, ws.stamp.data(),
      ws.next_generation(res_), current, &c->near[0], ws.in_far.data(), far, &c->far[0], false,
      nullptr, c);
  DYNG_CHECK_KERNEL(stream);
  ops::control<vertex_t> state = read_control(res_, ws);
  int near_slot = 0;  // the counter of the current near frontier
  int far_slot = 0;   // the counter of the current far pile
  vertex_t near_count = state.near[near_slot];
  vertex_t far_count = state.far[far_slot];
  std::int64_t iterations = 0;
  std::int64_t epochs = 0;
  std::int64_t pushes = 0;
  while (true) {
    if (near_count > 0) {
      // One push iteration (advance with packed_min): the next frontier is appended to the other
      // near counter, which the previous iteration cleared; this one clears the current counter.
      ++iterations;
      pushes += near_count;
      ops::push_kernel<vertex_t, edge_t, weight_t><<<grid(near_count), block_size, 0, stream>>>(
          current, near_count, out, ws.packed.data(), packed, max_distance_, run_.source,
          ws.stamp.data(), ws.next_generation(res_), next, &c->near[1 - near_slot],
          ws.in_far.data(), far, &c->far[far_slot], &c->near[near_slot], c);
      DYNG_CHECK_KERNEL(stream);
      state = read_control(res_, ws);
      near_slot = 1 - near_slot;
      near_count = state.near[near_slot];
      far_count = state.far[far_slot];
      std::swap(current, next);
      continue;
    }
    if (far_count == 0) {
      break;
    }
    // The near frontier is empty: raise the threshold past the far pile and re-split it (the
    // vertices below join the current near frontier, the rest the other far pile; the split
    // clears the counter of the pile it reads).
    ++epochs;
    const ops::count_of<vertex_t> pile{far_count, nullptr};
    ops::min_distance_kernel<vertex_t>
        <<<grid(far_count), block_size, 0, stream>>>(far, pile, ws.packed.data(), packed, c);
    DYNG_CHECK_KERNEL(stream);
    ops::threshold_kernel<vertex_t><<<1, 1, 0, stream>>>(c, delta, false);
    DYNG_CHECK_KERNEL(stream);
    ops::split_kernel<vertex_t><<<grid(far_count), block_size, 0, stream>>>(
        far, pile, ws.packed.data(), packed, ws.stamp.data(), ws.next_generation(res_), current,
        &c->near[near_slot], ws.in_far.data(), far2, &c->far[1 - far_slot], true, &c->far[far_slot],
        c);
    DYNG_CHECK_KERNEL(stream);
    state = read_control(res_, ws);
    far_slot = 1 - far_slot;
    near_count = state.near[near_slot];
    far_count = state.far[far_slot];
    std::swap(far, far2);
  }
  run_.counters.iterations = iterations;
  run_.counters.epochs = epochs;
  run_.counters.pushes = pushes;
}

// ------------------------------------------------------------------------------------------------
// finalize: unpack (MOSP-CUDA's unpack pass, with the `affected` counter of an update)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_operators_engine<vertex_t, edge_t, weight_t>::finalize() {
  if (n_ == 0 || stopped_) {
    return;
  }
  const scoped_device guard(res_.device());
  const cudaStream_t stream = native(res_);
  sssp_cuda_workspace<vertex_t>& ws = *run_.cuda_ws;
  ops::control<vertex_t>* c = device_control(ws);
  const sssp_kernels::packing packed{parent_bits_, no_parent_};
  auto* distances = reinterpret_cast<long long*>(run_.distances);
  if (packed.has_parents()) {
    if (update_) {
      ops::unpack_counted_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
          n_, ws.packed.data(), packed, distances, run_.parents, c);
    } else {
      ops::unpack_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
          n_, ws.packed.data(), packed, distances, run_.parents);
    }
    DYNG_CHECK_KERNEL(stream);
  } else {
    // Distance-only words: recover the lowest-id parent over tight edges (the old parents and the
    // "distance changed" marks of an update kept in `ancestor` and `candidates`, free here).
    ops::unpack_distances_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
        n_, ws.packed.data(), run_.source, distances, run_.parents, update_, ws.ancestor.data(),
        ws.candidates.data());
    DYNG_CHECK_KERNEL(stream);
    ops::recover_parents_kernel<vertex_t, edge_t, weight_t><<<grid(n_), block_size, 0, stream>>>(
        out_csr(run_.graph), ws.packed.data(), run_.source, run_.parents);
    DYNG_CHECK_KERNEL(stream);
    if (update_) {
      ops::count_affected_kernel<vertex_t><<<grid(n_), block_size, 0, stream>>>(
          n_, ws.ancestor.data(), ws.candidates.data(), run_.parents, c);
      DYNG_CHECK_KERNEL(stream);
    }
  }
  // One synchronization: `affected`, and a result that is complete when the call returns (the
  // fused engine's contract, ADR 0017 item 7).
  const ops::control<vertex_t> result = read_control(res_, ws);
  if (update_) {
    run_.counters.affected = static_cast<std::int64_t>(result.affected);
  }
}

#define DYNG_INSTANTIATE_SSSP_OPERATORS(V, E, W) template class sssp_cuda_operators_engine<V, E, W>;
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_OPERATORS)
#undef DYNG_INSTANTIATE_SSSP_OPERATORS

// ------------------------------------------------------------------------------------------------
// Kernel registration (resources::warm_up())
// ------------------------------------------------------------------------------------------------

namespace {

template <typename vertex_t>
void register_vertex_kernels(const char* (&names)[14]) noexcept {
  const void* kernels[14] = {
      reinterpret_cast<const void*>(&ops::init_control_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::init_from_scratch_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::pack_tree_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::mark_roots_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::pointer_jump_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::invalidate_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::insert_heads_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::min_distance_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::threshold_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::split_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::unpack_counted_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::unpack_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::unpack_distances_kernel<vertex_t>),
      reinterpret_cast<const void*>(&ops::count_affected_kernel<vertex_t>),
  };
  for (int i = 0; i < 14; ++i) {
    register_kernel(kernels[i], names[i]);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void register_graph_kernels(const char* (&names)[3]) noexcept {
  register_kernel(reinterpret_cast<const void*>(&ops::pull_kernel<vertex_t, edge_t, weight_t>),
                  names[0]);
  register_kernel(reinterpret_cast<const void*>(&ops::push_kernel<vertex_t, edge_t, weight_t>),
                  names[1]);
  register_kernel(
      reinterpret_cast<const void*>(&ops::recover_parents_kernel<vertex_t, edge_t, weight_t>),
      names[2]);
}

#define DYNG_SSSP_OPERATORS_VERTEX_NAMES(V)           \
  {"sssp_operators::init_control_kernel<" V ">",      \
   "sssp_operators::init_from_scratch_kernel<" V ">", \
   "sssp_operators::pack_tree_kernel<" V ">",         \
   "sssp_operators::mark_roots_kernel<" V ">",        \
   "sssp_operators::pointer_jump_kernel<" V ">",      \
   "sssp_operators::invalidate_kernel<" V ">",        \
   "sssp_operators::insert_heads_kernel<" V ">",      \
   "sssp_operators::min_distance_kernel<" V ">",      \
   "sssp_operators::threshold_kernel<" V ">",         \
   "sssp_operators::split_kernel<" V ">",             \
   "sssp_operators::unpack_counted_kernel<" V ">",    \
   "sssp_operators::unpack_kernel<" V ">",            \
   "sssp_operators::unpack_distances_kernel<" V ">",  \
   "sssp_operators::count_affected_kernel<" V ">"}
#define DYNG_SSSP_OPERATORS_GRAPH_NAMES(T)                                     \
  {"sssp_operators::pull_kernel<" T ">", "sssp_operators::push_kernel<" T ">", \
   "sssp_operators::recover_parents_kernel<" T ">"}

/// Registers every kernel instantiation of the operators engine at load time.
struct operators_kernel_registrar {
  operators_kernel_registrar() noexcept {
    const char* v32[14] = DYNG_SSSP_OPERATORS_VERTEX_NAMES("int32_t");
    const char* v64[14] = DYNG_SSSP_OPERATORS_VERTEX_NAMES("int64_t");
    register_vertex_kernels<std::int32_t>(v32);
    register_vertex_kernels<std::int64_t>(v64);
    const char* g32[3] = DYNG_SSSP_OPERATORS_GRAPH_NAMES("int32_t, int32_t, int32_t");
    const char* g3264[3] = DYNG_SSSP_OPERATORS_GRAPH_NAMES("int32_t, int64_t, int32_t");
    const char* g64[3] = DYNG_SSSP_OPERATORS_GRAPH_NAMES("int64_t, int64_t, int32_t");
    register_graph_kernels<std::int32_t, std::int32_t, std::int32_t>(g32);
    register_graph_kernels<std::int32_t, std::int64_t, std::int32_t>(g3264);
    register_graph_kernels<std::int64_t, std::int64_t, std::int32_t>(g64);
  }
};

#undef DYNG_SSSP_OPERATORS_VERTEX_NAMES
#undef DYNG_SSSP_OPERATORS_GRAPH_NAMES

const operators_kernel_registrar operators_kernels;

}  // namespace

}  // namespace dyng::detail
