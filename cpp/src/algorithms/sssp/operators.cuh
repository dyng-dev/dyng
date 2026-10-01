// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:mosp/src/sospUpdateGpu.cu (packKernel, unpackKernel,
// markRootsKernel, initAncestorsKernel, pointerJumpKernel, invalidateKernel, addCandidatesKernel,
// pullKernel, pushKernel, minDistanceKernel, splitKernel, initFromScratchKernel) and
// MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu (sospPersistentKernel: the exact semantics of each phase)
/**
 * @file operators.cuh
 * @brief The device operators of the sssp operators engine (Tier A, PLAN Sections 4.5.3-4.5.4,
 *        6.4.2; decision O24): one kernel per phase of the fused engine, launched by a host loop.
 *
 * STRUCTURE. MOSP_ESCHER@4b86159's multi-kernel SOSP update (the MP1 structure: pack, roots,
 * pointer-jumping rounds, invalidate, insertion heads, pull, then a near-far host loop of push,
 * minimum and split kernels, then unpack) is the shape; MOSP-CUDA@e220ee2's persistent kernel
 * (fused.cuh) is the semantics, phase by phase, so the two engines return the same bytes:
 *
 *   - every phase of the fused kernel becomes one kernel, with the same body: the same packing
 *     (including the distance-only words and the candidates above the bound that are never
 *     packed), the same stamp generations per list, the same in_far deduplication, the source
 *     never relaxed, and the same unpack (the `affected` counter, the lowest-id parent recovery of
 *     the distance-only mode);
 *   - the grid barriers become kernel boundaries; the decisions the fused kernel takes on the
 *     device are taken here on the device as well where no host value is needed (the early end of
 *     pointer jumping, the near-far threshold, the counts of lists whose length the host does not
 *     know yet), so the host reads the control block once after Step 1, once after the first split,
 *     once per push iteration and threshold raise, and once after the unpack;
 *   - MOSP_ESCHER's simplifications are not taken: its pointer jumping always runs ceil(log2 n)
 *     rounds without detecting a parent cycle, its pull and push pack candidates above the bound,
 *     and its unpack writes every entry.
 *
 * WHY THE BYTES ARE THE SAME. Both engines run the same monotone algorithm and only the order of
 * the atomic operations differs. After the search each word is the minimum of its value after the
 * invalidation and every candidate offered to it; a candidate that survives has the vertex's final
 * distance and comes either from the pull pass (an in-neighbour whose distance did not change) or
 * from a push of an in-neighbour at its own final distance, and every vertex whose distance
 * decreased is pushed at its final distance. That set of candidates, and therefore the minimum,
 * does not depend on the schedule; the parents of the distance-only mode are recomputed from the
 * final distances. The counters `invalidated` and `affected` are counts of those deterministic
 * sets; `iterations`, `epochs` and `pushes` describe the schedule and may differ.
 *
 * OPERATORS (PLAN 4.5.3) and where they live. `invalidate_subtree` (strategy pointer_jumping),
 * the pull `neighbor_reduce` with the packed argmin, the push `advance` with `packed_min`, and the
 * sparse near / far frontiers with their warp-aggregated `push` are written here and in
 * kernels.cuh, inside the sssp folder: no second algorithm uses them yet (the rule of two; the
 * planned second user is hyper_sssp). Every kernel is grid-stride, so a launch never needs more
 * blocks than the device holds at once and a kernel whose work is gone costs one empty launch.
 */
#pragma once

#include "algorithms/sssp/kernels.cuh"
#include "algorithms/sssp/problem.hpp"

#include <cuda_runtime.h>

#include <cstdint>
#include <limits>

namespace dyng::detail::sssp_operators {

using sssp_kernels::append_index;
using sssp_kernels::atomic_min;
using sssp_kernels::block_size;
using sssp_kernels::claim;
using sssp_kernels::count_into;
using sssp_kernels::device_csr;
using sssp_kernels::load;
using sssp_kernels::minimum_into;
using sssp_kernels::packed_inf;
using sssp_kernels::packing;
using sssp_kernels::sum_into;
using sssp_kernels::u64;

/// Slots of the pointer-jumping activity flags: ceil(log2 n) + 1 rounds for n < 2^63.
constexpr int max_rounds = 65;

/**
 * @brief The device-side control block of one run (zeroed by init_control_kernel). The host reads
 *        it back whole at each of its synchronizations.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct control {
  u64 minimum;             ///< min-reduction slot (packed_inf when idle)
  u64 threshold;           ///< the near-far threshold
  u64 affected;            ///< vertices whose distance or parent changed (counted runs only)
  vertex_t candidates;     ///< invalidated vertices + insertion heads (the pull's input)
  vertex_t frontier;       ///< vertices the pull pass improved (or the source)
  vertex_t near[2];        ///< the two near-frontier counts (current, next)
  vertex_t far[2];         ///< the two far-pile counts (current, re-split)
  vertex_t invalidated;    ///< vertices invalidated
  int overflow;            ///< an input distance does not fit the packing
  int active[max_rounds];  ///< pointer jumping: some vertex still jumps in round r
};

/// A count that is either known on the host (`value`) or still on the device (`device`).
template <typename vertex_t>
struct count_of {
  vertex_t value = 0;                ///< the count, if `device` is null; else an upper bound
  const vertex_t* device = nullptr;  ///< the device counter to read, or null

  /// The count (read once per thread; written by an earlier kernel).
  __device__ vertex_t get() const {
    return device != nullptr ? *device : value;
  }
};

/// Index of the calling thread in a grid-stride loop, and the loop stride.
template <typename vertex_t>
__device__ __forceinline__ vertex_t thread_index() {
  return static_cast<vertex_t>(blockIdx.x) * static_cast<vertex_t>(blockDim.x) +
         static_cast<vertex_t>(threadIdx.x);
}
template <typename vertex_t>
__device__ __forceinline__ vertex_t grid_stride() {
  return static_cast<vertex_t>(gridDim.x) * static_cast<vertex_t>(blockDim.x);
}

// ------------------------------------------------------------------------------------------------
// Control block and from-scratch initialization
// ------------------------------------------------------------------------------------------------

/// Zero the control block (the minimum slot idle).
template <typename vertex_t>
__global__ void init_control_kernel(control<vertex_t>* c) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *c = control<vertex_t>{};
    c->minimum = packed_inf;
  }
}

/// compute(): only the source is finite; the source is the frontier.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    init_from_scratch_kernel(vertex_t n, u64* packed, vertex_t source, packing packed_format,
                             vertex_t* frontier, control<vertex_t>* c) {
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    packed[v] = v == source ? packed_format.pack(0, vertex_t{-1}) : packed_inf;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    frontier[0] = source;
    c->frontier = 1;
  }
}

// ------------------------------------------------------------------------------------------------
// Step 1: invalidate_subtree (pointer jumping), insertion heads
// ------------------------------------------------------------------------------------------------

/// Pack the old tree (an input distance outside [0, max_distance] is reported) and start the
/// pointer jumping at the parents.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    pack_tree_kernel(vertex_t n, const long long* distances, const vertex_t* parent, u64* packed,
                     packing packed_format, u64 max_distance, vertex_t* ancestor,
                     control<vertex_t>* c) {
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    const long long d = distances[v];
    if (d >= sssp_infinity / 2) {
      packed[v] = packed_inf;
    } else if (d < 0 || static_cast<u64>(d) > max_distance) {
      c->overflow = 1;
      packed[v] = packed_inf;
    } else {
      packed[v] = packed_format.pack(static_cast<u64>(d), parent[v]);
    }
    ancestor[v] = parent[v];
  }
}

/// Roots: the head of every deleted or weight-increased edge that is its tree edge.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    mark_roots_kernel(const vertex_t* from, const vertex_t* to, vertex_t count,
                      const vertex_t* parent, int* flag) {
  for (vertex_t i = thread_index<vertex_t>(); i < count; i += grid_stride<vertex_t>()) {
    const vertex_t v = to[i];
    if (parent[v] == from[i]) {
      flag[v] = 1;
    }
  }
}

/**
 * One pointer-jumping round (the invariant is fused.cuh's: flag[x] == 1 implies a root among x and
 * its ancestors; flag[x] == 0 implies no root on the tree path from x up to, excluding,
 * ancestor[x]). A round after one in which no vertex jumped returns at once, so the host enqueues
 * ceil(log2 n) + 1 rounds and the device ends them as early as the fused kernel does; a vertex
 * still jumping in the last round is a parent cycle (the host checks active[max_rounds - 1]).
 */
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    pointer_jump_kernel(vertex_t n, vertex_t* ancestor, int* flag, control<vertex_t>* c,
                        int round) {
  if (round > 0 && c->active[round - 1] == 0) {
    return;  // the previous round converged (uniform over the grid)
  }
  int jumping = 0;
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    const vertex_t a = load(&ancestor[v]);
    if (a < 0 || load(&flag[v])) {
      continue;
    }
    if (load(&flag[a])) {
      flag[v] = 1;
      continue;
    }
    const vertex_t next = load(&ancestor[a]);
    ancestor[v] = next;
    jumping |= next >= 0 ? 1 : 0;
  }
  if (__any_sync(0xffffffffu, jumping) && (threadIdx.x & 31) == 0) {
    c->active[round] = 1;
  }
}

/// Invalidate the flagged vertices (INF, stamped, listed as candidates; the flags cleared for the
/// next run) and count them.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    invalidate_kernel(vertex_t n, int* flag, u64* packed, int* stamp, int generation,
                      vertex_t* candidates, control<vertex_t>* c) {
  vertex_t invalidated = 0;
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    if (flag[v]) {
      flag[v] = 0;
      packed[v] = packed_inf;
      stamp[v] = generation;
      candidates[append_index(&c->candidates)] = v;
      ++invalidated;
    }
  }
  count_into(invalidated, &c->invalidated);
}

/// The heads of the insertions join the candidates (the source never; each vertex once).
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    insert_heads_kernel(const vertex_t* heads, vertex_t count, vertex_t source, int* stamp,
                        int generation, vertex_t* candidates, control<vertex_t>* c) {
  for (vertex_t i = thread_index<vertex_t>(); i < count; i += grid_stride<vertex_t>()) {
    const vertex_t v = heads[i];
    if (v != source && claim(stamp, v, generation)) {
      candidates[append_index(&c->candidates)] = v;
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Step 1: the pull pass (neighbor_reduce over in-edges, packed argmin)
// ------------------------------------------------------------------------------------------------

/// Every candidate takes the best (distance, lowest id) word over its in-neighbours; a candidate
/// whose distance decreased joins the frontier.
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    pull_kernel(const vertex_t* candidates, vertex_t count,
                device_csr<vertex_t, edge_t, weight_t> in, u64* packed, packing packed_format,
                u64 max_distance, int* stamp, int generation, vertex_t* frontier,
                control<vertex_t>* c) {
  for (vertex_t i = thread_index<vertex_t>(); i < count; i += grid_stride<vertex_t>()) {
    const vertex_t v = candidates[i];
    const u64 current = load(&packed[v]);
    u64 best = current;
    for (edge_t e = in.row_ptr[v]; e < in.row_ptr[v + 1]; ++e) {
      const vertex_t u = in.col_ind[e];
      const u64 word = load(&packed[u]);
      if (word == packed_inf) {
        continue;
      }
      const u64 nd = packed_format.distance(word) + static_cast<u64>(in.weights[e]);
      if (nd <= max_distance) {  // larger: never shortest, may not fit
        best = min(best, packed_format.pack(nd, u));
      }
    }
    if (best < current) {
      const u64 old = atomicMin(&packed[v], best);
      if (packed_format.distance(best) < packed_format.distance(old) &&
          claim(stamp, v, generation)) {
        frontier[append_index(&c->frontier)] = v;
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Step 2: near-far propagation (bucketed frontier, advance with packed_min)
// ------------------------------------------------------------------------------------------------

/// The smallest distance over a list, folded into c->minimum.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    min_distance_kernel(const vertex_t* list, count_of<vertex_t> count, const u64* packed,
                        packing packed_format, control<vertex_t>* c) {
  const vertex_t length = count.get();
  u64 local = packed_inf;
  for (vertex_t i = thread_index<vertex_t>(); i < length; i += grid_stride<vertex_t>()) {
    const u64 word = load(&packed[list[i]]);
    if (word != packed_inf) {
      local = min(local, packed_format.distance(word));
    }
  }
  minimum_into(local, &c->minimum);
}

/// The threshold: the smallest frontier distance plus delta (first), or past the far pile (a
/// raise: max(threshold, smallest far distance) + delta). The minimum slot is left idle.
template <typename vertex_t>
__global__ void threshold_kernel(control<vertex_t>* c, u64 delta, bool first) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const u64 smallest = c->minimum;
    c->threshold = first ? (smallest == packed_inf ? 0 : smallest) + delta
                         : max(c->threshold, smallest) + delta;
    c->minimum = packed_inf;
  }
}

/**
 * Split a list by the threshold: below joins the near frontier (deduplicated by stamp), the rest
 * the far pile. The first split (from the frontier) deduplicates the far pile with in_far; a
 * re-split of the far pile clears in_far for the vertices that leave it. `reset` (if not null) is
 * zeroed by one thread: the counter of the list being split, free once its length is known.
 */
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    split_kernel(const vertex_t* list, count_of<vertex_t> count, const u64* packed,
                 packing packed_format, int* stamp, int generation, vertex_t* near_list,
                 vertex_t* near_count, int* in_far, vertex_t* far_list, vertex_t* far_count,
                 bool from_far, vertex_t* reset, const control<vertex_t>* c) {
  const vertex_t length = count.get();
  const u64 threshold = c->threshold;
  if (reset != nullptr && blockIdx.x == 0 && threadIdx.x == 0) {
    *reset = 0;
  }
  for (vertex_t i = thread_index<vertex_t>(); i < length; i += grid_stride<vertex_t>()) {
    const vertex_t v = list[i];
    const u64 word = load(&packed[v]);
    const u64 d = word == packed_inf ? packed_inf : packed_format.distance(word);
    if (d < threshold) {
      if (from_far) {
        in_far[v] = 0;
      }
      if (claim(stamp, v, generation)) {
        near_list[append_index(near_count)] = v;
      }
    } else if (from_far || atomicExch(&in_far[v], 1) == 0) {
      far_list[append_index(far_count)] = v;
    }
  }
}

/**
 * One push iteration: the near frontier relaxes its out-edges with packed atomicMin (the source is
 * never relaxed; candidates above the bound are dropped). An improved head joins the next near
 * frontier (below the threshold, deduplicated by stamp) or the far pile (deduplicated by in_far).
 * `reset` is zeroed by one thread: the count of the frontier being pushed, which the host passed
 * by value, becomes the slot of the iteration after next.
 */
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    push_kernel(const vertex_t* near_list, vertex_t near_count,
                device_csr<vertex_t, edge_t, weight_t> out, u64* packed, packing packed_format,
                u64 max_distance, vertex_t source, int* stamp, int generation, vertex_t* next_list,
                vertex_t* next_count, int* in_far, vertex_t* far_list, vertex_t* far_count,
                vertex_t* reset, const control<vertex_t>* c) {
  const u64 threshold = c->threshold;
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    *reset = 0;
  }
  for (vertex_t i = thread_index<vertex_t>(); i < near_count; i += grid_stride<vertex_t>()) {
    const vertex_t u = near_list[i];
    const u64 word = load(&packed[u]);
    if (word == packed_inf) {
      continue;
    }
    const u64 du = packed_format.distance(word);
    for (edge_t e = out.row_ptr[u]; e < out.row_ptr[u + 1]; ++e) {
      const vertex_t w = out.col_ind[e];
      if (w == source) {
        continue;
      }
      const u64 nd = du + static_cast<u64>(out.weights[e]);
      if (nd > max_distance) {
        continue;  // never shortest; would not fit the packed word
      }
      const u64 candidate = packed_format.pack(nd, u);
      if (candidate >= load(&packed[w])) {
        continue;
      }
      const u64 old = atomicMin(&packed[w], candidate);
      if (nd < packed_format.distance(old)) {
        if (nd < threshold) {
          if (claim(stamp, w, generation)) {
            next_list[append_index(next_count)] = w;
          }
        } else if (atomicExch(&in_far[w], 1) == 0) {
          far_list[append_index(far_count)] = w;
        }
      }
    }
  }
}

// ------------------------------------------------------------------------------------------------
// Unpack
// ------------------------------------------------------------------------------------------------

/// Packed words of an update: write only the entries that change, and count them.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    unpack_counted_kernel(vertex_t n, const u64* packed, packing packed_format,
                          long long* distances, vertex_t* parent, control<vertex_t>* c) {
  u64 affected = 0;
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    const u64 word = packed[v];
    const long long distance =
        word == packed_inf ? sssp_infinity : static_cast<long long>(packed_format.distance(word));
    const vertex_t p = word == packed_inf ? vertex_t{-1} : packed_format.parent<vertex_t>(word);
    if (distances[v] != distance || parent[v] != p) {
      distances[v] = distance;
      parent[v] = p;
      ++affected;
    }
  }
  sum_into(affected, &c->affected);
}

/// Packed words of compute(): write every entry.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    unpack_kernel(vertex_t n, const u64* packed, packing packed_format, long long* distances,
                  vertex_t* parent) {
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    const u64 word = packed[v];
    if (word == packed_inf) {
      distances[v] = sssp_infinity;
      parent[v] = -1;
    } else {
      distances[v] = static_cast<long long>(packed_format.distance(word));
      parent[v] = packed_format.parent<vertex_t>(word);
    }
  }
}

/// Distance-only words, first pass: the distances, and every parent reset (-1 for unreachable
/// vertices and the source, the largest id otherwise). An update first keeps the old parent in
/// `old_parent` and whether the distance changed in `changed` (free scratch at this point).
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    unpack_distances_kernel(vertex_t n, const u64* packed, vertex_t source, long long* distances,
                            vertex_t* parent, bool counted, vertex_t* old_parent,
                            vertex_t* changed) {
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    const u64 word = packed[v];
    const long long distance = word == packed_inf ? sssp_infinity : static_cast<long long>(word);
    if (counted) {
      old_parent[v] = parent[v];
      changed[v] = distance != distances[v] ? 1 : 0;
    }
    distances[v] = distance;
    parent[v] =
        word == packed_inf || v == source ? vertex_t{-1} : std::numeric_limits<vertex_t>::max();
  }
}

/// Distance-only words, second pass: every parent is the lowest id over its tight in-edges.
template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    recover_parents_kernel(device_csr<vertex_t, edge_t, weight_t> out, const u64* packed,
                           vertex_t source, vertex_t* parent) {
  const vertex_t n = out.number_of_nodes;
  for (vertex_t u = thread_index<vertex_t>(); u < n; u += grid_stride<vertex_t>()) {
    const u64 du = packed[u];
    if (du == packed_inf) {
      continue;
    }
    for (edge_t e = out.row_ptr[u]; e < out.row_ptr[u + 1]; ++e) {
      const vertex_t w = out.col_ind[e];
      if (w != source && du + static_cast<u64>(out.weights[e]) == packed[w]) {
        atomic_min(&parent[w], u);
      }
    }
  }
}

/// Distance-only words of an update, last pass: count the vertices whose distance or parent
/// changed.
template <typename vertex_t>
__global__ void __launch_bounds__(block_size)
    count_affected_kernel(vertex_t n, const vertex_t* old_parent, const vertex_t* changed,
                          const vertex_t* parent, control<vertex_t>* c) {
  u64 affected = 0;
  for (vertex_t v = thread_index<vertex_t>(); v < n; v += grid_stride<vertex_t>()) {
    affected += (changed[v] != 0 || old_parent[v] != parent[v]) ? 1 : 0;
  }
  sum_into(affected, &c->affected);
}

}  // namespace dyng::detail::sssp_operators
