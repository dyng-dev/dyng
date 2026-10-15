// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh
/**
 * @file wedge.cuh
 * @brief The wedge-parallel counting kernels: one thread per oriented wedge, the frontier filters
 *        of the exactly-once rule, the block bins and their signed flush, and the set primitives on
 *        sorted rows (ESCHER-GPU's kernel/wedge.cuh).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of parity/escher_port/rename_map_triad.toml (M9a,
 * ADR 0053; the row views' encoding is described in cpp/src/hypergraph/row_views.hpp).
 */
#pragma once

// Wedge-parallel counting kernels.
//
// A work item is one oriented wedge (a, b): a source key a and one entry b
// of its adjacency row. Full counts use the out-wedges of every key (b > a).
// Frontier deltas use every neighbour b of each frontier key f and skip the
// frontier members smaller than f, so that each triangle (or pair) is
// counted once, at its smallest frontier member (see include/frontier.hpp).
// A wedge is closed into triangles (a, b, k) by intersecting the suffixes of
// N(a) after b and of N(b) after b, i.e. k > b.
//
// Mapping one thread per wedge instead of one thread per hyperedge balances
// the work of hub rows across many threads; rows are read through the
// key-indexed escher::row_view (no BST walks in the inner loops). Each block
// accumulates into private 64-bit bins that are flushed once with a sign
// (+1 add, -1 subtract; signed values in two's complement).
//
// An op_t provides:
//   static constexpr int num_bins;
//   triangles: void begin_wedge(int a, int b); void triangle(int k, bins);
//   pairs:     void pair(int a, int b, bins);
//   keys:      void key(int a, bins);
//   void flush(bins)  -- adds register-held partial sums to the block bins.

#include "hypergraph/row_views.hpp"

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>

#include <algorithm>
#include <climits>
#include <cstdlib>
#include <iostream>

namespace dyng::detail::triad {

namespace wedge {

constexpr int block_size = 256;

inline void check(cudaError_t result) {
  if (result != cudaSuccess) {
    std::cerr << "CUDA Runtime Error: " << cudaGetErrorString(result) << std::endl;
    std::exit(-1);
  }
}

// Largest s in [lo, hi] with offsets[s] <= t (offsets is non-decreasing).
__device__ __forceinline__ int find_source(const long long* offsets, int lo, int hi, long long t) {
  while (lo < hi) {
    int mid = (lo + hi + 1) >> 1;
    if (offsets[mid] <= t)
      lo = mid;
    else
      hi = mid - 1;
  }
  return lo;
}

__device__ __forceinline__ int source_key(const int* sources, int s) {
  return sources != nullptr ? sources[s] : s + 1;
}

__device__ __forceinline__ int wedge_begin(const escher::row_view& adj, int a,
                                           bool all_neighbours) {
  return all_neighbours ? adj.loc[a] : adj.out[a];
}

// Number of wedges of each source (all neighbours, or out-neighbours only).
static __global__ void count_wedges_kernel(escher::row_view adj, const int* sources,
                                           int num_sources, bool all_neighbours,
                                           long long* counts) {
  int s = blockIdx.x * blockDim.x + threadIdx.x;
  if (s >= num_sources) return;
  int a = source_key(sources, s);
  counts[s] = adj.loc[a] + adj.len[a] - wedge_begin(adj, a, all_neighbours);
}

// Warp-cooperative grid-stride loop over the wedges: lanes 0 and 1 locate
// the first and last source of the warp's 32 wedges, then every lane binary
// searches its own source within that range. Calls body(a, b, p) for each
// wedge that passes the frontier filter (p = position of b in N(a)).
template <class body_t>
__device__ __forceinline__ void for_each_wedge(const escher::row_view& adj, const int* sources,
                                               int num_sources, const long long* offsets,
                                               long long num_wedges, bool all_neighbours,
                                               const unsigned char* in_frontier, body_t& body) {
  const int lane = threadIdx.x & 31;
  const long long warp = (blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x) >> 5;
  const long long num_warps = (gridDim.x * static_cast<long long>(blockDim.x)) >> 5;
  for (long long base = warp * 32; base < num_wedges; base += num_warps * 32) {
    int lo = 0, hi = 0;
    if (lane == 0) lo = find_source(offsets, 0, num_sources - 1, base);
    if (lane == 1) hi = find_source(offsets, 0, num_sources - 1, min(base + 31, num_wedges - 1));
    lo = __shfl_sync(0xffffffffu, lo, 0);
    hi = __shfl_sync(0xffffffffu, hi, 1);
    long long t = base + lane;
    if (t >= num_wedges) continue;
    int s = find_source(offsets, lo, hi, t);
    int a = source_key(sources, s);
    int p = wedge_begin(adj, a, all_neighbours) + static_cast<int>(t - offsets[s]);
    int b = adj.flat[p];
    if (in_frontier != nullptr && in_frontier[b] && b < a) continue;
    body(a, b, p);
  }
}

template <class op_t>
__device__ __forceinline__ void flush_bins(op_t& op, unsigned long long* bins, long long sign,
                                           unsigned long long* out) {
  op.flush(bins);
  __syncthreads();
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x)
    if (bins[i] != 0)
      atomicAdd(&out[i], static_cast<unsigned long long>(sign * static_cast<long long>(bins[i])));
}

template <class op_t>
struct triangle_body {
  const escher::row_view& adj;
  const unsigned char* in_frontier;
  op_t& op;
  unsigned long long* bins;
  __device__ __forceinline__ void operator()(int a, int b, int p) {
    op.begin_wedge(a, b);
    int x = p + 1, end_a = adj.loc[a] + adj.len[a];
    int y = adj.out[b], end_b = adj.loc[b] + adj.len[b];
    while (x < end_a && y < end_b) {
      int ka = adj.flat[x], kb = adj.flat[y];
      if (ka == kb) {
        if (in_frontier == nullptr || !(in_frontier[ka] && ka < a)) op.triangle(ka, bins);
        ++x;
        ++y;
      } else if (ka < kb) {
        ++x;
      } else {
        ++y;
      }
    }
  }
};

template <class op_t>
struct pair_body {
  op_t& op;
  unsigned long long* bins;
  __device__ __forceinline__ void operator()(int a, int b, int) {
    op.pair(a, b, bins);
  }
};

template <class op_t>
__global__ void triangle_kernel(escher::row_view adj, const int* sources, int num_sources,
                                const long long* offsets, long long num_wedges, bool all_neighbours,
                                const unsigned char* in_frontier, op_t op, long long sign,
                                unsigned long long* out) {
  __shared__ unsigned long long bins[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) bins[i] = 0;
  __syncthreads();
  op_t local = op;
  triangle_body<op_t> body{adj, in_frontier, local, bins};
  for_each_wedge(adj, sources, num_sources, offsets, num_wedges, all_neighbours, in_frontier, body);
  flush_bins(local, bins, sign, out);
}

template <class op_t>
__global__ void pair_kernel(escher::row_view adj, const int* sources, int num_sources,
                            const long long* offsets, long long num_wedges, bool all_neighbours,
                            const unsigned char* in_frontier, op_t op, long long sign,
                            unsigned long long* out) {
  __shared__ unsigned long long bins[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) bins[i] = 0;
  __syncthreads();
  op_t local = op;
  pair_body<op_t> body{local, bins};
  for_each_wedge(adj, sources, num_sources, offsets, num_wedges, all_neighbours, in_frontier, body);
  flush_bins(local, bins, sign, out);
}

template <class op_t>
__global__ void key_kernel(const int* sources, int num_sources, op_t op, long long sign,
                           unsigned long long* out) {
  __shared__ unsigned long long bins[op_t::num_bins];
  for (int i = threadIdx.x; i < op_t::num_bins; i += blockDim.x) bins[i] = 0;
  __syncthreads();
  op_t local = op;
  for (int s = blockIdx.x * blockDim.x + threadIdx.x; s < num_sources; s += gridDim.x * blockDim.x)
    local.key(source_key(sources, s), bins);
  flush_bins(local, bins, sign, out);
}

enum class wedge_shape { triangles, pairs };

// Counts over the wedges of `num_sources` sources: the keys in d_sources
// (frontier mode: all neighbours, frontier filter d_in_frontier) or, when
// d_sources is null, keys 1..num_sources (full count: out-neighbours only).
// Adds sign * (bins of op_t) to d_out[0..num_bins).
template <wedge_shape shape, class op_t>
void count_wedge_shape(const escher::row_view& adj, const int* d_sources, int num_sources,
                       const unsigned char* d_in_frontier, const op_t& op, long long sign,
                       unsigned long long* d_out) {
  if (num_sources <= 0) return;
  const bool all_neighbours = d_sources != nullptr;
  long long* d_offsets = nullptr;
  check(cudaMalloc(&d_offsets, (num_sources + 1) * sizeof(long long)));
  check(cudaMemset(d_offsets + num_sources, 0, sizeof(long long)));
  count_wedges_kernel<<<(num_sources + block_size - 1) / block_size, block_size>>>(
      adj, d_sources, num_sources, all_neighbours, d_offsets);
  thrust::exclusive_scan(thrust::device_ptr<long long>(d_offsets),
                         thrust::device_ptr<long long>(d_offsets + num_sources + 1),
                         thrust::device_ptr<long long>(d_offsets));
  long long num_wedges = 0;
  check(
      cudaMemcpy(&num_wedges, d_offsets + num_sources, sizeof(long long), cudaMemcpyDeviceToHost));
  if (num_wedges > 0) {
    // One thread per wedge: with fewer threads looping over several wedges,
    // threads that draw heavy wedges set the tail of the kernel.
    int grid =
        static_cast<int>(std::min<long long>((num_wedges + block_size - 1) / block_size, INT_MAX));
    if constexpr (shape == wedge_shape::triangles)
      triangle_kernel<op_t><<<grid, block_size>>>(adj, d_sources, num_sources, d_offsets,
                                                  num_wedges, all_neighbours, d_in_frontier, op,
                                                  sign, d_out);
    else
      pair_kernel<op_t><<<grid, block_size>>>(adj, d_sources, num_sources, d_offsets, num_wedges,
                                              all_neighbours, d_in_frontier, op, sign, d_out);
    check(cudaGetLastError());
  }
  check(cudaFree(d_offsets));
}

template <class op_t>
void count_triangles(const escher::row_view& adj, const int* d_sources, int num_sources,
                     const unsigned char* d_in_frontier, const op_t& op, long long sign,
                     unsigned long long* d_out) {
  count_wedge_shape<wedge_shape::triangles>(adj, d_sources, num_sources, d_in_frontier, op, sign,
                                            d_out);
}

template <class op_t>
void count_pairs(const escher::row_view& adj, const int* d_sources, int num_sources,
                 const unsigned char* d_in_frontier, const op_t& op, long long sign,
                 unsigned long long* d_out) {
  count_wedge_shape<wedge_shape::pairs>(adj, d_sources, num_sources, d_in_frontier, op, sign,
                                        d_out);
}

// Per-key counts over the keys in d_sources (or keys 1..num_sources).
template <class op_t>
void count_keys(const int* d_sources, int num_sources, const op_t& op, long long sign,
                unsigned long long* d_out) {
  if (num_sources <= 0) return;
  int grid = (num_sources + block_size - 1) / block_size;
  key_kernel<op_t><<<grid, block_size>>>(d_sources, num_sources, op, sign, d_out);
  check(cudaGetLastError());
}

// ---- Set primitives on sorted, contiguous rows -------------------------

// |A ∩ B|
__device__ __forceinline__ int intersect_size(const int* a, int na, const int* b, int nb) {
  int x = 0, y = 0, c = 0;
  while (x < na && y < nb) {
    int va = a[x], vb = b[y];
    c += (va == vb);
    x += (va <= vb);
    y += (vb <= va);
  }
  return c;
}

// One merge pass over rows a, b, c: |B ∩ C|, |C ∩ A| and |A ∩ B ∩ C|.
__device__ __forceinline__ void venn3(const int* a, int na, const int* b, int nb, const int* c,
                                      int nc, int& cbc, int& cca, int& g) {
  int x = 0, y = 0, z = 0;
  cbc = cca = g = 0;
  while (true) {
    int va = x < na ? a[x] : INT_MAX;
    int vb = y < nb ? b[y] : INT_MAX;
    int vc = z < nc ? c[z] : INT_MAX;
    int m = min(va, min(vb, vc));
    if (m == INT_MAX) break;
    int ia = va == m, ib = vb == m, ic = vc == m;
    cbc += ib & ic;
    cca += ic & ia;
    g += ia & ib & ic;
    x += ia;
    y += ib;
    z += ic;
  }
}

// Whether rows a, b and c share an element.
__device__ __forceinline__ bool intersects3(const int* a, int na, const int* b, int nb,
                                            const int* c, int nc) {
  int x = 0, y = 0, z = 0;
  while (x < na && y < nb && z < nc) {
    int m = max(a[x], max(b[y], c[z]));
    if (a[x] == m && b[y] == m && c[z] == m) return true;
    x += a[x] < m;
    y += b[y] < m;
    z += c[z] < m;
  }
  return false;
}

}  // namespace wedge

}  // namespace dyng::detail::triad
