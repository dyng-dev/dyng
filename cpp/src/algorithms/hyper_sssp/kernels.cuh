// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospInternal.cuh
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu
/**
 * @file kernels.cuh
 * @brief The kernels of hyper_sssp's CUDA engine (MOSP_ESCHER's hsospRecompute kernels, one warp
 *        per candidate, and the packed-word kernels of hsospUpdate: pack and unpack, the roots,
 *        pointer jumping, invalidation, the pulls, the inserted pairs, the push).
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hyper_sssp.toml
 * (M11a, ADR 0091).
 */
#pragma once

#include <cuda_runtime.h>

#include <limits>

namespace dyng::detail::hsosp {

inline int grid_for(long long n, int block) {
  return static_cast<int>((n + block - 1) / block);
}

namespace {

// Same value as HostHypergraph::INF (which is not constexpr).
constexpr long long inf_value = std::numeric_limits<long long>::max() / 4;
static_assert(inf_value > 0, "inf_value must be positive");

// dynG (transplant hunk): the weight hyper_sssp reads for node v: 0 for the virtual-endpoint
// mode's source and target rows (zero_a, zero_b; MOSP_ESCHER's weight-0 virtual hyperedges, which
// the container holds with weight 1), the node weight otherwise (HD2).
__device__ __forceinline__ long long node_weight(const long long* node_w, int v, int zero_a,
                                                 int zero_b) {
  return (v == zero_a || v == zero_b) ? 0 : node_w[v];
}
// ---------------------------------------------------------------------------
// Kernels
// ---------------------------------------------------------------------------

__global__ void fill_ll_kernel(long long* arr, int n, long long value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = value;
}

__global__ void fill_int_kernel(int* arr, int n, int value) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid < n) arr[tid] = value;
}
/**
 * Pull step of the recompute, one warp per candidate v: the best parent
 * over v's (symmetric) neighbour list, read with coalesced strided loads
 * and a shuffle min-reduction; every in-edge of v costs node_w[v], and ties
 * go to the lowest parent id (canonical tree). Candidates are unique
 * (dedup at enqueue), so the dist / parent writes are race-free; reading
 * the neighbours' distances while other warps lower them is the benign
 * chaotic-relaxation race (a changed node re-enqueues its neighbours).
 * (The original ran one thread per candidate over the whole row with
 * dependent loads; rows of the DBLP line graph have up to 3,000 entries.)
 */
__global__ void pull_distances_warp_kernel(const int* cand_list, int num_cand,
                                           const long long* row_start, const int* deg,
                                           const int* col_ind, const long long* node_w,
                                           long long* dist, int* parent, int* is_candidate,
                                           int* is_affected, int* aff_list, int* aff_count,
                                           int source, int zero_a, int zero_b,
                                           const unsigned char* holds) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= num_cand) return;

  const int v = cand_list[warp];
  if (lane == 0) is_candidate[v] = 0;
  if (v == source) return;

  long long best = inf_value;
  int best_p = -1;
#if defined(DYNG_MUTATION_HSOSP_NON_TIGHT_PARENT)
  int any_p = -1;
#endif
  const long long base = row_start[v];
  const int d = deg[v];
  const long long wv = node_weight(node_w, v, zero_a, zero_b);
  // dynG (transplant hunk): the public mode's base offer (a hyperedge holding the source vertex
  // starts at its own weight, parent -1: MOSP_ESCHER's source row).
  if (holds != nullptr && holds[v] != 0) best = wv;
  for (int e = lane; e < d; e += 32) {
    const int p = col_ind[base + e];
    const long long dp = dist[p];
    if (dp >= inf_value / 2) continue;
#if defined(DYNG_MUTATION_HSOSP_PULL_ADDS_NEIGHBOUR_WEIGHT)
    // MUTATION (PLAN 8.4; M11a criterion 16 #1, the CUDA engine): the pull adds the neighbour's
    // weight instead of the node's.
    const long long cd = dp + node_weight(node_w, p, zero_a, zero_b);
#else
    const long long cd = dp + wv;
#endif
    if (cd < best || (cd == best && p < best_p)) {
      best = cd;
      best_p = p;
    }
#if defined(DYNG_MUTATION_HSOSP_NON_TIGHT_PARENT)
    if (p > any_p) any_p = p;
#endif
  }
  for (int o = 16; o > 0; o >>= 1) {
    const long long ob = __shfl_xor_sync(0xffffffffu, best, o);
    const int op = __shfl_xor_sync(0xffffffffu, best_p, o);
    if (ob < best || (ob == best && op < best_p)) {
      best = ob;
      best_p = op;
    }
#if defined(DYNG_MUTATION_HSOSP_NON_TIGHT_PARENT)
    const int oa = __shfl_xor_sync(0xffffffffu, any_p, o);
    if (oa > any_p) any_p = oa;
#endif
  }
#if defined(DYNG_MUTATION_HSOSP_NON_TIGHT_PARENT)
  // MUTATION (PLAN 8.4; M11a criterion 16 #2, the CUDA engine): the pull records the highest
  // reached neighbour as the parent, tight or not; the distances are unchanged.
  if (best_p >= 0) best_p = any_p;
#endif
  if (lane == 0) {
    const bool changed = (best != dist[v]);
    parent[v] = best_p;
    dist[v] = best;
    if (changed && atomicCAS(&is_affected[v], 0, 1) == 0) aff_list[atomicAdd(aff_count, 1)] = v;
  }
}

/** Collect step, one warp per changed node: its neighbours become the next
 *  candidates. A plain read of is_candidate skips the atomicCAS for nodes
 *  that are already listed (the flag only goes 0 -> 1 in this kernel). */
__global__ void collect_candidates_warp_kernel(const int* aff_list, int num_aff,
                                               const long long* row_start, const int* deg,
                                               const int* col_ind, int* is_candidate,
                                               int* cand_list, int* cand_count, int* is_affected,
                                               int source) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= num_aff) return;

  const int u = aff_list[warp];
  if (lane == 0) is_affected[u] = 0;

  const long long base = row_start[u];
  const int d = deg[u];
  for (int e = lane; e < d; e += 32) {
    const int nb = col_ind[base + e];
    if (nb == source) continue;
    if (is_candidate[nb] == 0 && atomicCAS(&is_candidate[nb], 0, 1) == 0)
      cand_list[atomicAdd(cand_count, 1)] = nb;
  }
}
// ---------------------------------------------------------------------------
// Incremental update kernels (update)
// ---------------------------------------------------------------------------
//
// The update works on packed words (dist << 32 | parent): a 64-bit
// atomicMin then keeps the smaller distance and, among equal distances,
// the lower parent id. packed_inf marks an unreachable node; the source is
// (0 << 32 | 0xffffffff). Distances must stay below packed_dist_limit; a
// larger candidate raises the overflow counter and the update falls back
// to the 64-bit recompute.

using u64 = unsigned long long;
constexpr u64 packed_inf = ~0ull;
constexpr long long packed_dist_limit = 0xffffffffLL;

// Update counters (device, mirrored to pinned host memory).
enum upd_counter {
  cnt_list = 0,      // invalidated nodes
  cnt_next = 1,      // size of the next frontier
  cnt_active = 2,    // pointer jumping still active
  cnt_work = 3,      // edge relaxations
  cnt_overflow = 4,  // a distance did not fit 32 bits
};

__device__ __forceinline__ u64 pack_dist(long long d, int parent) {
  return (static_cast<u64>(d) << 32) | static_cast<unsigned>(parent);
}
__device__ __forceinline__ long long packed_dist(u64 x) {
  return static_cast<long long>(x >> 32);
}
__device__ __forceinline__ int packed_parent(u64 x) {
  return static_cast<int>(static_cast<unsigned>(x & 0xffffffffu));
}

__global__ void pack_state_kernel(const long long* dist, const int* parent, u64* packed, int n,
                                  u64* counters) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const long long d = dist[v];
  if (d >= inf_value / 2) {
    packed[v] = packed_inf;
  } else {
    if (d >= packed_dist_limit) atomicAdd(&counters[cnt_overflow], 1ull);
    packed[v] = pack_dist(d, parent[v]);
  }
}

__global__ void unpack_state_kernel(const u64* packed, long long* dist, int* parent, int n) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const u64 x = packed[v];
  if (x == packed_inf) {
    dist[v] = inf_value;
    parent[v] = -1;
  } else {
    dist[v] = packed_dist(x);
    parent[v] = packed_parent(x);
  }
}

// Roots: b of a deleted pair (a, b) whose tree parent was a (and vice
// versa), from the pre-batch packed state.
__global__ void mark_deleted_tree_edges_kernel(const int2* pairs, int n_del, const u64* packed,
                                               int* inv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_del) return;
  const int a = pairs[i].x, b = pairs[i].y;
  const u64 pa = packed[a], pb = packed[b];
  if (pb != packed_inf && packed_parent(pb) == a) inv[b] = 1;
  if (pa != packed_inf && packed_parent(pa) == b) inv[a] = 1;
}

__global__ void mark_ids_kernel(const int* ids, int n, int* inv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) inv[ids[i]] = 1;
}

__global__ void init_jump_kernel(const u64* packed, int* jump, int n) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const u64 x = packed[v];
  const int p = (x == packed_inf) ? -1 : packed_parent(x);
  jump[v] = (p == -1) ? -1 : p;
}

// One pointer-jumping round (double buffered): a node is invalidated if it
// or its current ancestor is; otherwise its ancestor pointer doubles. After
// ceil(log2(depth)) + 1 rounds every descendant of a root is marked.
__global__ void jump_kernel(const int* inv_in, int* inv_out, const int* jump_in, int* jump_out,
                            int n, u64* counters) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const int iv = inv_in[v];
  const int j = jump_in[v];
  if (iv || j < 0) {
    inv_out[v] = iv;
    jump_out[v] = -1;
    return;
  }
  const int ij = inv_in[j];
  const int jj = jump_in[j];
  inv_out[v] = ij;
  jump_out[v] = ij ? -1 : jj;
  if (!ij && jj >= 0) counters[cnt_active] = 1;
}

__global__ void invalidate_kernel(const int* inv, u64* packed, int* list, u64* counters, int n,
                                  int source) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  if (inv[v] && v != source) {
    packed[v] = packed_inf;
    list[atomicAdd(&counters[cnt_list], 1ull)] = v;
  }
}

// Enqueues v for the frontier of epoch ep unless it already is (epoch
// stamps instead of a reset flag, which raced with concurrent pushes).
__device__ __forceinline__ void enqueue(int v, int* stamp, int ep, int* next, u64* counters) {
  if (atomicMax(&stamp[v], ep) < ep) next[atomicAdd(&counters[cnt_next], 1ull)] = v;
}

// Relaxes v with the candidate word c; enqueues v when it improved.
__device__ __forceinline__ void relax(u64* packed, int v, u64 c, int* stamp, int ep, int* next,
                                      u64* counters) {
  if (c < packed[v]) {
    const u64 old = atomicMin(&packed[v], c);
    if (c < old) enqueue(v, stamp, ep, next, counters);
  }
}

// One warp per invalidated node: best (distance, id) over its neighbours.
__global__ void pull_invalidated_kernel(const int* list, int n_list, const long long* row_start,
                                        const int* deg, const int* col_ind, const long long* node_w,
                                        u64* packed, int* stamp, int ep, int* next, u64* counters,
                                        int zero_a, int zero_b, const unsigned char* holds) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_list) return;
  const int v = list[warp];
  const long long base = row_start[v];
  const int d = deg[v];
  const long long wv = node_weight(node_w, v, zero_a, zero_b);
  u64 best = packed_inf;
  // dynG (transplant hunk): the public mode's base offer (see pull_distances_warp_kernel).
  if (holds != nullptr && holds[v] != 0) best = pack_dist(wv, -1);
  bool overflow = false;
  for (int e = lane; e < d; e += 32) {
    const int p = col_ind[base + e];
    const u64 x = packed[p];
    if (x == packed_inf) continue;
#if defined(DYNG_MUTATION_HSOSP_PULL_ADDS_NEIGHBOUR_WEIGHT)
    // MUTATION (PLAN 8.4; M11a criterion 16 #1, the CUDA engine): see pull_distances_warp_kernel.
    const long long cd = packed_dist(x) + node_weight(node_w, p, zero_a, zero_b);
#else
    const long long cd = packed_dist(x) + wv;
#endif
    if (cd >= packed_dist_limit) {
      overflow = true;
      continue;
    }
    const u64 c = pack_dist(cd, p);
    if (c < best) best = c;
  }
  for (int o = 16; o > 0; o >>= 1) {
    const u64 y = __shfl_xor_sync(0xffffffffu, best, o);
    if (y < best) best = y;
  }
  overflow = __any_sync(0xffffffffu, overflow);
  if (lane == 0) {
    atomicAdd(&counters[cnt_work], static_cast<u64>(d));
    if (overflow) atomicAdd(&counters[cnt_overflow], 1ull);
    if (best != packed_inf) relax(packed, v, best, stamp, ep, next, counters);
  }
}

// Both directions of every inserted pair.
__global__ void relax_inserted_kernel(const int2* pairs, int n_ins, const long long* node_w,
                                      u64* packed, int* stamp, int ep, int* next, u64* counters,
                                      int source, int zero_a, int zero_b) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n_ins) return;
  const int ab[2] = {pairs[i].x, pairs[i].y};
  for (int k = 0; k < 2; ++k) {
    const int u = ab[k], v = ab[1 - k];
    if (v == source) continue;
    const u64 x = packed[u];
    if (x == packed_inf) continue;
    const long long cd = packed_dist(x) + node_weight(node_w, v, zero_a, zero_b);
    if (cd >= packed_dist_limit) {
      atomicAdd(&counters[cnt_overflow], 1ull);
      continue;
    }
    relax(packed, v, pack_dist(cd, u), stamp, ep, next, counters);
  }
}

// One warp per frontier node: push its current word to the neighbours
// (decrease-only).
__global__ void push_kernel(const int* front, int n_front, const long long* row_start,
                            const int* deg, const int* col_ind, const long long* node_w,
                            u64* packed, int* stamp, int ep, int* next, u64* counters, int source,
                            int zero_a, int zero_b) {
  const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
  const int lane = threadIdx.x & 31;
  if (warp >= n_front) return;
  const int u = front[warp];
  const u64 x = packed[u];
  if (x == packed_inf) return;
  const long long du = packed_dist(x);
  const long long base = row_start[u];
  const int d = deg[u];
  bool overflow = false;
  for (int e = lane; e < d; e += 32) {
    const int v = col_ind[base + e];
    if (v == source) continue;
    const long long cd = du + node_weight(node_w, v, zero_a, zero_b);
    if (cd >= packed_dist_limit) {
      overflow = true;
      continue;
    }
    relax(packed, v, pack_dist(cd, u), stamp, ep, next, counters);
  }
  overflow = __any_sync(0xffffffffu, overflow);
  if (lane == 0) {
    atomicAdd(&counters[cnt_work], static_cast<u64>(d));
    if (overflow) atomicAdd(&counters[cnt_overflow], 1ull);
  }
}

// dynG (transplant hunk), the public mode: a hyperedge that lost the source vertex is a root when
// it was reached from the source directly (parent -1; MOSP_ESCHER's deleted pair with its source
// row, markDeletedTreeEdgesKernel).
__global__ void mark_lost_source_kernel(const int* ids, int n, const u64* packed, int* inv) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const u64 x = packed[ids[i]];
  if (x != packed_inf && packed_parent(x) == -1) inv[ids[i]] = 1;
}

// dynG (transplant hunk), the public mode: a hyperedge that gained the source vertex is offered
// its own weight with parent -1 (MOSP_ESCHER's inserted pair with its source row,
// relaxInsertedKernel).
__global__ void relax_base_kernel(const int* ids, int n, const long long* node_w, u64* packed,
                                  int* stamp, int ep, int* next, u64* counters) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const int v = ids[i];
  relax(packed, v, pack_dist(node_w[v], -1), stamp, ep, next, counters);
}

// dynG (transplant hunk), the public mode: the recompute's base, every live hyperedge holding the
// source at its own weight (MOSP_ESCHER's source row at distance 0).
__global__ void seed_base_kernel(const int* ids, int n, const long long* node_w, long long* dist) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < n) dist[ids[i]] = node_w[ids[i]];
}

}  // namespace
constexpr int upd_counters = 8;

}  // namespace dyng::detail::hsosp
