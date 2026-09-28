// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/sospUpdateGpu.cu (Packing, makePacking, Control, Params,
// load, claim, appendIndex, minimumInto, sospPersistentKernel) and headers/sospUpdateGpu.cuh
// (DeviceCsr, DeviceChanges)
/**
 * @file fused.cuh
 * @brief The fused CUDA engine of sssp (Tier B, PLAN Section 4.5.4): MOSP-CUDA's persistent
 *        cooperative SOSP kernel, ported verbatim.
 *
 * ALGORITHM (unchanged; MOSP-CUDA's description, abridged)
 *
 * Step 1 (from the change list, grouped by destination):
 *   - Roots: the head v of every deleted or weight-increased edge (u,v) with parent[v] == u.
 *   - Subtree invalidation: pointer jumping over the parent array marks every descendant of a
 *     root. Rounds stop as soon as no vertex is still jumping (at most ceil(log2 n) rounds). The
 *     marked vertices lose their distance (INF) and parent.
 *   - Pull pass: every invalidated vertex and every head of an inserted edge takes the best
 *     (distance, parent id) pair over its in-neighbours (one thread per destination).
 *
 * Step 2 (propagation) is a push-based near-far worklist (a Delta-stepping variant, Davidson et
 * al., IPDPS'14): vertices whose distance improved relax their out-edges with a 64-bit atomicMin
 * on the packed pair (distance << b | parent). A vertex whose new distance is below the current
 * threshold joins the next near frontier, otherwise the far pile; when the near frontier is empty
 * the threshold moves to the smallest far distance plus Delta.
 *
 * Properties: monotone (no iteration cap, no counting to infinity, cut-off vertices keep INF),
 * canonical (the packed atomicMin keeps, among equal distances, the lowest parent id),
 * work-efficient (only improved vertices are expanded). Everything runs inside one cooperative
 * kernel; the phases are separated by grid-wide barriers and the loop decisions are taken on the
 * device, so an update costs one launch and one final copy of the control block. Data written by
 * other blocks is read with __ldcg (L2). List appends reserve their slots with one warp-aggregated
 * atomicAdd per group of converged threads (append_index).
 *
 * Packing: b = number of bits needed for the vertex ids plus a "no parent" value; the remaining
 * 64 - b bits hold the distance, and the all-ones word is INF. If the distance bound
 * (n - 1) * max_weight does not fit in 64 - b bits, the words hold the distance alone and the
 * parents are recovered after the search (lowest id among the in-neighbours u with
 * d[u] + w(u,v) == d[v]). The pull pass and the push loop drop every candidate above the bound
 * before packing it.
 *
 * Mechanical changes only (PLAN Section 6.3 step 5): names in snake_case; templates on the vertex,
 * edge and weight types (the int32 instantiation is the original's code); namespace
 * dyng::detail; DISTANCE_INF is sssp_infinity (the same value). One addition, in the unpack pass
 * of an update (not in compute(), whose input arrays are not a tree): the counter `affected`
 * (vertices whose distance or parent changed; update_stats), summed per warp and added to the
 * control block. With packed words the unpack compares each vertex's new pair with the old one it
 * would overwrite and writes only changed pairs (the same bytes moved as the original's
 * unconditional write, the same output); in the distance-only mode it needs the old parents and
 * the "distance changed" marks, which the unpack pass keeps in the `ancestor` and `candidates`
 * arrays (free at that point) and counts after one more grid barrier.
 *
 * One correction (M1b review, ADR 0017 item 1): the original sets `invalidated` from thread 0's
 * read of the candidate-list counter right after the barrier that ends the invalidation, with no
 * barrier before the insertion heads are appended to the same list by every other thread. If
 * other blocks append first, the counter includes insertion heads (a data race; the trees are not
 * affected, the reported count is). Here every thread counts the vertices it invalidates and the
 * counts are summed per warp into the control block before that barrier (count_into), so the
 * counter is exact on every run. MOSP-CUDA@e220ee2 has the race (compute-sanitizer's scheduling
 * exposes it); its plain runs give the exact count, which is what the golden corpus records.
 */
#pragma once

#include "algorithms/sssp/problem.hpp"

#include <cuda_runtime.h>

#include <cooperative_groups.h>

#include <cstdint>
#include <limits>

namespace dyng::detail::sssp_fused {

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

/// The change batch as seen by one objective, in device memory (MOSP's DeviceChanges).
template <typename vertex_t>
struct device_changes {
  /// Edges (from[i], to[i]) that were deleted or whose weight increased; the head of such an edge
  /// is a root if the edge is its tree edge.
  const vertex_t* changed_from = nullptr;
  const vertex_t* changed_to = nullptr;
  vertex_t number_of_changed = 0;
  /// Heads of inserted edges (their distance may decrease).
  const vertex_t* insert_heads = nullptr;
  vertex_t number_of_insert_heads = 0;
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

/// Device-side control block of one update (initialized before launch).
template <typename vertex_t>
struct control {
  vertex_t list_count;      ///< candidates (invalidated + insert heads)
  vertex_t frontier_count;  ///< vertices improved by the pull pass
  vertex_t near_count;      ///< size of the current near frontier
  vertex_t next_count;      ///< appends to the next near frontier
  vertex_t far_count;       ///< far pile size
  vertex_t far2_count;      ///< re-split far pile size
  int active[3];            ///< pointer jumping: a vertex still jumps
  int overflow;             ///< an input distance does not fit the packing
  vertex_t invalidated;     ///< vertices invalidated (changed: summed, see count_into)
  int rounds;
  int iterations;
  int epochs;
  int generation;  ///< last stamp generation used
  long long pushes;
  u64 minimum;   ///< min-reduction slot (packed_inf when idle)
  u64 affected;  ///< added: vertices whose distance or parent changed (update only)
};

/// Parameters of the persistent kernel.
template <typename vertex_t, typename edge_t, typename weight_t>
struct params {
  device_csr<vertex_t, edge_t, weight_t> out, in;
  device_changes<vertex_t> changes;
  vertex_t source;
  bool from_scratch;
  packing packed_format;
  u64 max_distance;
  u64 delta;
  int max_rounds;
  int generation;  ///< first stamp generation to use
  long long* distances;
  vertex_t* parent;
  u64* packed;
  int *stamp, *in_far, *flag;
  vertex_t* ancestor;
  vertex_t *candidates, *frontier, *near_a, *near_b, *far_a, *far_b;
  control<vertex_t>* ctl;
};

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

template <typename vertex_t, typename edge_t, typename weight_t>
__global__ void __launch_bounds__(block_size)
    sssp_persistent_kernel(params<vertex_t, edge_t, weight_t> p) {
  cg::grid_group grid = cg::this_grid();
  const vertex_t tid = static_cast<vertex_t>(grid.thread_rank());
  const vertex_t threads = static_cast<vertex_t>(grid.size());
  const vertex_t n = p.out.number_of_nodes;
  const packing packed_format = p.packed_format;
  control<vertex_t>* c = p.ctl;
  int generation = p.generation;
  vertex_t frontier_count = 0;

  if (p.from_scratch) {
    // ---- From scratch: only the source is finite. -------------------------------------------
    for (vertex_t v = tid; v < n; v += threads) {
      p.packed[v] = v == p.source ? packed_format.pack(0, vertex_t{-1}) : packed_inf;
    }
    if (tid == 0) {
      p.frontier[0] = p.source;
    }
    frontier_count = 1;
    grid.sync();
  } else {
    // ---- Pack the old tree; roots; ancestors. -----------------------------------------------
    for (vertex_t v = tid; v < n; v += threads) {
      long long d = p.distances[v];
      if (d >= sssp_infinity / 2) {
        p.packed[v] = packed_inf;
      } else if (d < 0 || static_cast<u64>(d) > p.max_distance) {
        c->overflow = 1;
        p.packed[v] = packed_inf;
      } else {
        p.packed[v] = packed_format.pack(static_cast<u64>(d), p.parent[v]);
      }
      p.ancestor[v] = p.parent[v];
    }
    for (vertex_t i = tid; i < p.changes.number_of_changed; i += threads) {
      vertex_t v = p.changes.changed_to[i];
      if (p.parent[v] == p.changes.changed_from[i]) {
        p.flag[v] = 1;
      }
    }
    grid.sync();

    // ---- Subtree invalidation by pointer jumping. -------------------------------------------
    // Invariant for every vertex x: flag[x] == 1 implies a root among x and its ancestors;
    // flag[x] == 0 implies no root on the tree path from x up to (excluding) ancestor[x]. A vertex
    // either inherits its ancestor's flag or jumps to the ancestor's ancestor, at least doubling
    // the covered distance, so ceil(log2 n) rounds suffice; the loop stops earlier once no vertex
    // is still jumping. Only v's thread writes (ancestor[v], flag[v]) and every value another
    // thread can observe satisfies the invariant, so updating in place is safe.
    if (p.changes.number_of_changed > 0) {
      for (int round = 0; round < p.max_rounds; ++round) {
        if (tid == 0) {
          c->active[(round + 1) % 3] = 0;  // slot of the next round
          c->rounds = round + 1;
        }
        int jumping = 0;
        for (vertex_t v = tid; v < n; v += threads) {
          vertex_t a = load(&p.ancestor[v]);
          if (a < 0 || load(&p.flag[v])) {
            continue;
          }
          if (load(&p.flag[a])) {
            p.flag[v] = 1;
            continue;
          }
          vertex_t next = load(&p.ancestor[a]);
          p.ancestor[v] = next;
          jumping |= next >= 0 ? 1 : 0;
        }
        if (__any_sync(0xffffffffu, jumping) && (threadIdx.x & 31) == 0) {
          c->active[round % 3] = 1;
        }
        grid.sync();
        if (load(&c->active[round % 3]) == 0) {
          break;
        }
      }
    }

    // ---- Invalidate; candidates = invalidated + insert heads. -------------------------------
    ++generation;
    vertex_t invalidated = 0;  // changed: counted here (see the file comment, "One correction")
    for (vertex_t v = tid; v < n; v += threads) {
      if (load(&p.flag[v])) {
        p.flag[v] = 0;  // leave the flags clean for the next update
        p.packed[v] = packed_inf;
        p.stamp[v] = generation;
        p.candidates[append_index(&c->list_count)] = v;
        ++invalidated;
      }
    }
    count_into(invalidated, &c->invalidated);
    grid.sync();
    for (vertex_t i = tid; i < p.changes.number_of_insert_heads; i += threads) {
      vertex_t v = p.changes.insert_heads[i];
      if (v != p.source && claim(p.stamp, v, generation)) {
        p.candidates[append_index(&c->list_count)] = v;
      }
    }
    grid.sync();

    // ---- Pull pass. -------------------------------------------------------------------------
    ++generation;
    const vertex_t candidate_count = load(&c->list_count);
    for (vertex_t i = tid; i < candidate_count; i += threads) {
      vertex_t v = load(&p.candidates[i]);
      u64 current = load(&p.packed[v]);
      u64 best = current;
      for (edge_t e = p.in.row_ptr[v]; e < p.in.row_ptr[v + 1]; ++e) {
        vertex_t u = p.in.col_ind[e];
        u64 word = load(&p.packed[u]);
        if (word == packed_inf) {
          continue;
        }
        const u64 nd = packed_format.distance(word) + static_cast<u64>(p.in.weights[e]);
        if (nd <= p.max_distance) {  // larger: never shortest, may not fit
          best = min(best, packed_format.pack(nd, u));
        }
      }
      if (best < current) {
        u64 old = atomicMin(&p.packed[v], best);
        if (packed_format.distance(best) < packed_format.distance(old) &&
            claim(p.stamp, v, generation)) {
          p.frontier[append_index(&c->frontier_count)] = v;
        }
      }
    }
    grid.sync();
    frontier_count = load(&c->frontier_count);
  }

  // ---- Near-far propagation. ------------------------------------------------------------------
  // Threshold = smallest frontier distance + delta.
  u64 local = packed_inf;
  for (vertex_t i = tid; i < frontier_count; i += threads) {
    u64 word = load(&p.packed[load(&p.frontier[i])]);
    if (word != packed_inf) {
      local = min(local, packed_format.distance(word));
    }
  }
  minimum_into(local, &c->minimum);
  grid.sync();
  const u64 smallest = load(&c->minimum);
  u64 threshold = (smallest == packed_inf ? 0 : smallest) + p.delta;

  // The current near frontier has c->near_count entries; a push iteration appends the next one
  // (c->next_count) and thread 0 moves the count over between two barriers.
  vertex_t *current = p.near_a, *next = p.near_b, *far = p.far_a, *far2 = p.far_b;
  ++generation;
  for (vertex_t i = tid; i < frontier_count; i += threads) {
    vertex_t v = load(&p.frontier[i]);
    u64 word = load(&p.packed[v]);
    u64 d = word == packed_inf ? packed_inf : packed_format.distance(word);
    if (d < threshold) {
      if (claim(p.stamp, v, generation)) {
        current[append_index(&c->near_count)] = v;
      }
    } else if (atomicExch(&p.in_far[v], 1) == 0) {
      far[append_index(&c->far_count)] = v;
    }
  }
  grid.sync();
  if (tid == 0) {
    c->minimum = packed_inf;  // everybody read it before the barrier
  }
  grid.sync();
  int iterations = 0, epochs = 0;
  long long pushes = 0;

  while (true) {
    const vertex_t near_count = load(&c->near_count);
    if (near_count > 0) {
      // -- One push iteration over the near frontier. --
      ++generation;
      ++iterations;
      pushes += near_count;
      for (vertex_t i = tid; i < near_count; i += threads) {
        vertex_t u = load(&current[i]);
        u64 word = load(&p.packed[u]);
        if (word == packed_inf) {
          continue;
        }
        u64 du = packed_format.distance(word);
        for (edge_t e = p.out.row_ptr[u]; e < p.out.row_ptr[u + 1]; ++e) {
          vertex_t w = p.out.col_ind[e];
          if (w == p.source) {
            continue;
          }
          u64 nd = du + static_cast<u64>(p.out.weights[e]);
          if (nd > p.max_distance) {
            continue;  // never shortest; would not fit the packed word
          }
          u64 candidate = packed_format.pack(nd, u);
          if (candidate >= load(&p.packed[w])) {
            continue;
          }
          u64 old = atomicMin(&p.packed[w], candidate);
          if (nd < packed_format.distance(old)) {
            if (nd < threshold) {
              if (claim(p.stamp, w, generation)) {
                next[append_index(&c->next_count)] = w;
              }
            } else if (atomicExch(&p.in_far[w], 1) == 0) {
              far[append_index(&c->far_count)] = w;
            }
          }
        }
      }
      grid.sync();
      if (tid == 0) {
        c->near_count = load(&c->next_count);
        c->next_count = 0;
      }
      grid.sync();
      vertex_t* t = current;
      current = next;
      next = t;
      continue;
    }

    // -- Near frontier empty: raise the threshold past the far pile. --
    const vertex_t far_count = load(&c->far_count);
    if (far_count == 0) {
      break;
    }
    ++epochs;
    local = packed_inf;
    for (vertex_t i = tid; i < far_count; i += threads) {
      u64 word = load(&p.packed[load(&far[i])]);
      if (word != packed_inf) {
        local = min(local, packed_format.distance(word));
      }
    }
    minimum_into(local, &c->minimum);
    grid.sync();
    threshold = max(threshold, load(&c->minimum)) + p.delta;
    ++generation;
    for (vertex_t i = tid; i < far_count; i += threads) {
      vertex_t v = load(&far[i]);
      u64 word = load(&p.packed[v]);
      u64 d = word == packed_inf ? packed_inf : packed_format.distance(word);
      if (d < threshold) {
        p.in_far[v] = 0;
        if (claim(p.stamp, v, generation)) {
          current[append_index(&c->near_count)] = v;
        }
      } else {
        far2[append_index(&c->far2_count)] = v;
      }
    }
    grid.sync();
    if (tid == 0) {
      c->far_count = load(&c->far2_count);
      c->far2_count = 0;
      c->minimum = packed_inf;
    }
    grid.sync();
    vertex_t* t = far;
    far = far2;
    far2 = t;
  }

  // ---- Unpack the result. -----------------------------------------------------------------------
  const bool count_affected = !p.from_scratch;  // added: the input arrays of an update are a tree
  u64 affected = 0;
  if (packed_format.has_parents()) {
    if (count_affected) {
      // Added (update only): the arrays still hold the old tree, so only the entries that change
      // are written (and counted). The unpack moves as many bytes as the original's (a read of
      // the old pair instead of a write of an unchanged one) and gives `affected` for free.
      for (vertex_t v = tid; v < n; v += threads) {
        u64 word = load(&p.packed[v]);
        const long long distance = word == packed_inf
                                       ? sssp_infinity
                                       : static_cast<long long>(packed_format.distance(word));
        const vertex_t parent =
            word == packed_inf ? vertex_t{-1} : packed_format.parent<vertex_t>(word);
        if (p.distances[v] != distance || p.parent[v] != parent) {
          p.distances[v] = distance;
          p.parent[v] = parent;
          ++affected;
        }
      }
    } else {
      for (vertex_t v = tid; v < n; v += threads) {
        u64 word = load(&p.packed[v]);
        if (word == packed_inf) {
          p.distances[v] = sssp_infinity;
          p.parent[v] = -1;
        } else {
          p.distances[v] = static_cast<long long>(packed_format.distance(word));
          p.parent[v] = packed_format.parent<vertex_t>(word);
        }
      }
    }
  } else {
    // Distance-only words: recover the lowest-id parent over tight edges.
    for (vertex_t v = tid; v < n; v += threads) {
      u64 word = load(&p.packed[v]);
      const long long distance = word == packed_inf ? sssp_infinity : static_cast<long long>(word);
      if (count_affected) {
        p.ancestor[v] = p.parent[v];                           // added: the old parent
        p.candidates[v] = distance != p.distances[v] ? 1 : 0;  // added: distance changed
      }
      p.distances[v] = distance;
      p.parent[v] =
          word == packed_inf || v == p.source ? vertex_t{-1} : std::numeric_limits<vertex_t>::max();
    }
    grid.sync();
    for (vertex_t u = tid; u < n; u += threads) {
      u64 du = load(&p.packed[u]);
      if (du == packed_inf) {
        continue;
      }
      for (edge_t e = p.out.row_ptr[u]; e < p.out.row_ptr[u + 1]; ++e) {
        vertex_t w = p.out.col_ind[e];
        if (w != p.source && du + static_cast<u64>(p.out.weights[e]) == load(&p.packed[w])) {
          atomic_min(&p.parent[w], u);
        }
      }
    }
    if (count_affected) {  // added: count after every parent is final
      grid.sync();
      for (vertex_t v = tid; v < n; v += threads) {
        affected +=
            (load(&p.candidates[v]) != 0 || load(&p.ancestor[v]) != load(&p.parent[v])) ? 1 : 0;
      }
    }
  }
  if (count_affected) {
    sum_into(affected, &c->affected);
  }
  if (tid == 0) {
    c->iterations = iterations;
    c->epochs = epochs;
    c->pushes = pushes;
    c->generation = generation;
  }
}

}  // namespace dyng::detail::sssp_fused
