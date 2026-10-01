// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu (Packing, makePacking, load, claim,
// appendIndex, minimumInto) and headers/sospUpdateGpu.cuh (DeviceCsr)
/**
 * @file kernels.cuh
 * @brief The device building blocks of the two CUDA engines of sssp: the packed (distance,
 *        parent) words (`packed_min`, PLAN Section 4.5.3), the claim of a stamp, the
 *        warp-aggregated list append (the `push` of a sparse frontier) and the warp-wide
 *        reductions into a control block.
 *
 * The fused engine (fused.cuh, Tier B) and the operators engine (operators.cuh, Tier A) both use
 * them (PLAN 4.5.4: a fused kernel uses the shared building blocks where it costs nothing). They
 * stay in the sssp folder until a second algorithm needs them (the rule of two, PLAN 4.5.3; the
 * planned second user is hyper_sssp).
 *
 * The code is MOSP-CUDA's, moved here unchanged from fused.cuh in M7 (with the M1b additions
 * count_into and sum_into): the fused kernel's machine code is the same for every instantiation.
 */
#pragma once

#include "algorithms/sssp/problem.hpp"

#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <cstdint>

namespace dyng::detail::sssp_kernels {

namespace cg = cooperative_groups;

using u64 = unsigned long long;
constexpr u64 packed_inf = ~0ULL;
constexpr int block_size = 256;

/// Largest distance representable in the output (finite distances must stay below
/// DISTANCE_INF / 2).
constexpr u64 output_max_distance = static_cast<u64>(sssp_infinity / 2 - 1);

/// A CSR graph with one weight per edge, in device memory (MOSP's DeviceCsr).
template <typename vertex_t, typename edge_t, typename weight_t>
struct device_csr {
  vertex_t number_of_nodes = 0;  // (MOSP's numberOfEdges is not read by the kernel; dropped)
  const edge_t* row_ptr = nullptr;
  const vertex_t* col_ind = nullptr;
  const weight_t* weights = nullptr;
};

/// Packed (distance, parent) words; parent_bits == 0 means distance only.
struct packing {
  int parent_bits;
  u64 no_parent;  // all-ones parent field (0 without parents)

  __host__ __device__ bool has_parents() const {
    return parent_bits > 0;
  }
  template <typename vertex_t>
  __host__ __device__ u64 pack(u64 distance, vertex_t parent) const {
    if (parent_bits == 0) {
      return distance;
    }
    return (distance << parent_bits) | (parent < 0 ? no_parent : static_cast<u64>(parent));
  }
  __device__ u64 distance(u64 word) const {
    return word >> parent_bits;
  }
  template <typename vertex_t>
  __device__ vertex_t parent(u64 word) const {
    u64 p = word & no_parent;
    return p == no_parent ? vertex_t{-1} : static_cast<vertex_t>(p);
  }
  /// Largest distance that can be stored (the all-ones field is INF).
  u64 max_distance() const {
    return (packed_inf >> parent_bits) - 1;
  }
};

/// Parent bits for n vertices, or distance-only words if the bound (n - 1) * max_weight does not
/// fit next to them.
inline packing make_packing(std::int64_t number_of_nodes, u64 bound) {
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<u64>(number_of_nodes)) {
    ++bits;
  }
  packing packed{bits, (1ULL << bits) - 1};
  if (bound <= packed.max_distance()) {
    return packed;
  }
  return packing{0, 0};
}

__device__ __forceinline__ int load(const int* p) {
  return __ldcg(p);
}
__device__ __forceinline__ std::int64_t load(const std::int64_t* p) {
  return static_cast<std::int64_t>(__ldcg(reinterpret_cast<const long long*>(p)));
}
__device__ __forceinline__ u64 load(const u64* p) {
  return __ldcg(p);
}

__device__ __forceinline__ int atomic_add(int* counter, int value) {
  return atomicAdd(counter, value);
}
__device__ __forceinline__ std::int64_t atomic_add(std::int64_t* counter, std::int64_t value) {
  return static_cast<std::int64_t>(atomicAdd(reinterpret_cast<unsigned long long*>(counter),
                                             static_cast<unsigned long long>(value)));
}

__device__ __forceinline__ void atomic_min(int* address, int value) {
  atomicMin(address, value);
}
__device__ __forceinline__ void atomic_min(std::int64_t* address, std::int64_t value) {
  atomicMin(reinterpret_cast<long long*>(address), static_cast<long long>(value));
}

template <typename vertex_t>
__device__ __forceinline__ bool claim(int* stamp, vertex_t v, int generation) {
  return atomicExch(&stamp[v], generation) != generation;
}

/// Reserve one slot of a shared list: one atomicAdd per group of converged threads
/// (warp-aggregated), each thread gets its own index.
template <typename index_t>
__device__ __forceinline__ index_t append_index(index_t* counter) {
  cg::coalesced_group active = cg::coalesced_threads();
  index_t base = 0;
  if (active.thread_rank() == 0) {
    base = atomic_add(counter, static_cast<index_t>(active.size()));
  }
  return active.shfl(base, 0) + static_cast<index_t>(active.thread_rank());
}

/// Warp-wide minimum of per-thread values, folded into *target.
__device__ inline void minimum_into(u64 value, u64* target) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = min(value, __shfl_down_sync(0xffffffffu, value, offset));
  }
  if ((threadIdx.x & 31) == 0 && value != packed_inf) {
    atomicMin(target, value);
  }
}

/// Added (M1b review): warp-wide sum of per-thread counts of the index type, added to *target
/// (the `invalidated` counter).
template <typename index_t>
__device__ inline void count_into(index_t value, index_t* target) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if ((threadIdx.x & 31) == 0 && value != 0) {
    atomic_add(target, value);
  }
}

/// Added: warp-wide sum of per-thread counts, added to *target (the `affected` counter).
__device__ inline void sum_into(u64 value, u64* target) {
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  if ((threadIdx.x & 31) == 0 && value != 0) {
    atomicAdd(target, value);
  }
}

}  // namespace dyng::detail::sssp_kernels
