// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file operators.cuh
 * @brief hyper_sssp on the shared sssp engine (engine::operators; M11b, decision O12; ADR 0095):
 *        the line graph as the sssp operators read it (graph_access<line_graph_view>: slack rows,
 *        node weights, one symmetric view, the public mode's base offers, the work counter in
 *        MOSP_ESCHER's units) and hyper_sssp's own small kernels (the batch's pair delta and ids
 *        as sssp_changes, the static seeds). CUDA translation units only.
 *
 * The kernels of the update and of compute() are sssp's (algorithms/sssp/operators.cuh: pack the
 * tree, the roots, pointer jumping, invalidation, the insertion heads, the pull, near-far with the
 * push, unpack); the pull and the push are instantiated here over line_graph_view, in the two
 * mappings (one thread or one warp per row). The host loop is in operators.cu.
 */
#pragma once

#include "algorithms/sssp/kernels.cuh"
#include "algorithms/sssp/operators.cuh"
#include "util/slack_rows.cuh"

#include <cuda_runtime.h>

#include <cstdint>

namespace dyng::detail::hyper_ops {

using sssp_kernels::packed_inf;
using sssp_kernels::packing;
using sssp_kernels::u64;
namespace ops = sssp_operators;

/// The control block of a run: sssp's, then the work counter (both read back in one copy).
struct run_control {
  ops::control<int> c;  ///< sssp's control block (int node ids)
  u64 work;             ///< relaxations so far (MOSP_ESCHER's units)
  u64 weight_sum;       ///< compute(): the sum of the entries' weights (the default delta)
};

/**
 * @brief The line graph as the shared engine reads it: slack rows over one symmetric adjacency,
 *        the node weights (0 for the virtual mode's two zero-weight ids), the public mode's source
 *        flags and the work counter.
 */
struct line_graph_view {
  slack_rows<int, long long> rows;       ///< the slack rows (MOSP_ESCHER's rowStart, deg, colInd)
  const long long* node_w = nullptr;     ///< per node: its weight
  int zero_a = -1;                       ///< the virtual mode: an id whose weight reads as 0
  int zero_b = -1;                       ///< (another one)
  const unsigned char* holds = nullptr;  ///< the public mode: 1 if the node holds the source
  u64* work = nullptr;                   ///< the work counter (run_control::work)
  /// The pulled candidates whose rows the work counter counts: the first `counted` ones, the
  /// invalidated nodes (MOSP_ESCHER counts its pulls of the invalidated nodes, not its
  /// relaxations of the inserted pairs, which the insertion heads' pulls stand for).
  int counted = 0;

  /// @brief The weight hyper_sssp reads for node v. @param[in] v The node. @return It.
  __device__ __forceinline__ u64 weight(int v) const {
    return v == zero_a || v == zero_b ? 0 : static_cast<u64>(node_w[v]);
  }
};

}  // namespace dyng::detail::hyper_ops

namespace dyng::detail::sssp_operators {

/**
 * @brief The line graph's policies (the file comment of algorithms/sssp/operators.cuh): slack
 *        rows, node weights (w[col] pushing u -> col, w[v] pulling into v), the base offers of the
 *        public mode, the work counter.
 */
template <>
struct graph_access<hyper_ops::line_graph_view> {
  using graph_type = hyper_ops::line_graph_view;  ///< the graph
  using edge_type = long long;                    ///< slot offsets
  static constexpr bool base_offers = true;       ///< the pull offers base_offer(v)
  static constexpr bool counts_work = true;       ///< the passes count their relaxations

  /// First entry of row v.
  __device__ static __forceinline__ long long begin(const graph_type& g, int v) {
    return g.rows.begin(v);
  }
  /// One past the last entry of row v.
  __device__ static __forceinline__ long long end(const graph_type& g, int v) {
    return g.rows.end(v);
  }
  /// The neighbour of entry e.
  __device__ static __forceinline__ int neighbour(const graph_type& g, long long e) {
    return g.rows.neighbour(e);
  }
  /// Pushing u -> head costs the head's weight.
  __device__ static __forceinline__ u64 push_weight(const graph_type& g, long long /*e*/,
                                                    int head) {
    return g.weight(head);
  }
  /// Pulling into v costs v's weight (MUTATION #1's twin: the neighbour's).
  __device__ static __forceinline__ u64 pull_weight(const graph_type& g, long long e, int v) {
#if defined(DYNG_MUTATION_HSOSP_OPS_PULL_ADDS_NEIGHBOUR_WEIGHT)
    // MUTATION (PLAN 8.4; M11a criterion 16 #1, the shared engine's twin, M11b criterion 4): the
    // pull adds the neighbour's weight instead of the node's.
    (void)v;
    return g.weight(g.rows.neighbour(e));
#else
    (void)e;
    return g.weight(v);
#endif
  }
  /// The public mode: a node holding the source is offered its own weight with no parent.
  __device__ static __forceinline__ u64 base_offer(const graph_type& g, int v, packing pf) {
    return g.holds != nullptr && g.holds[v] != 0 ? pf.pack(g.weight(v), -1) : packed_inf;
  }
  /// The work counter.
  __device__ static __forceinline__ u64* work_counter(const graph_type& g) {
    return g.work;
  }
  /// A neighbour's word, a plain (L1-cached) load: MOSP_ESCHER's kernels read them so; a stale
  /// word only offers a worse candidate (the atomicMin decides; a later decrease is pushed).
  __device__ static __forceinline__ u64 peek(const u64* word) {
    return *word;
  }
  /// Whether the pull of candidate i counts (the invalidated nodes, listed first).
  __device__ static __forceinline__ bool counts_row(const graph_type& g, int i) {
    return i < g.counted;
  }
};

}  // namespace dyng::detail::sssp_operators

namespace dyng::detail::hyper_ops {

using ops::block_size;

// The kernels have internal linkage: one translation unit (operators.cu) instantiates them.
namespace {

/// The pull pass over the line graph, one thread per candidate.
__global__ void __launch_bounds__(block_size)
    pull_thread_kernel(const int* candidates, int count, line_graph_view g, u64* packed,
                       packing packed_format, u64 max_distance, int* stamp, int generation,
                       int* frontier, ops::control<int>* c) {
  ops::pull_pass(candidates, count, g, packed, packed_format, max_distance, stamp, generation,
                 frontier, c);
}

/// The pull pass over the line graph, one warp per candidate.
__global__ void __launch_bounds__(block_size)
    pull_warp_kernel(const int* candidates, int count, line_graph_view g, u64* packed,
                     packing packed_format, u64 max_distance, int* stamp, int generation,
                     int* frontier, ops::control<int>* c) {
  ops::pull_pass_warp(candidates, count, g, packed, packed_format, max_distance, stamp, generation,
                      frontier, c);
}

/// One push iteration over the line graph, one thread per frontier node.
__global__ void __launch_bounds__(block_size)
    push_thread_kernel(const int* near_list, int near_count, line_graph_view g, u64* packed,
                       packing packed_format, u64 max_distance, int source, int* stamp,
                       int generation, int* next_list, int* next_count, int* in_far, int* far_list,
                       int* far_count, int* reset, const ops::control<int>* c) {
  ops::push_pass(near_list, near_count, g, packed, packed_format, max_distance, source, stamp,
                 generation, next_list, next_count, in_far, far_list, far_count, reset, c);
}

/// One push iteration over the line graph, one warp per frontier node.
__global__ void __launch_bounds__(block_size)
    push_warp_kernel(const int* near_list, int near_count, line_graph_view g, u64* packed,
                     packing packed_format, u64 max_distance, int source, int* stamp,
                     int generation, int* next_list, int* next_count, int* in_far, int* far_list,
                     int* far_count, int* reset, const ops::control<int>* c) {
  ops::push_pass_warp(near_list, near_count, g, packed, packed_format, max_distance, source, stamp,
                      generation, next_list, next_count, in_far, far_list, far_count, reset, c);
}

/**
 * The batch as sssp_changes (M11b criterion 3): a deleted pair (a, b) becomes the changed edges
 * a -> b and b -> a (the head is a root if the edge is its tree edge); a new, recreated or dead id
 * v becomes (parent[v] -> v), a root whatever its parent (the source row of the virtual mode
 * excepted, which is never invalidated, as in MOSP_ESCHER's invalidateKernel); a node of the
 * public mode that lost the source becomes (-1 -> v): a root exactly when it was reached from the
 * source directly (parent -1, MOSP_ESCHER's deleted pair with its source row). Slots
 * [0, 2 num_del) hold the pairs, then the ids, then the lost nodes.
 */
__global__ void __launch_bounds__(block_size)
    changes_kernel(const int2* deleted, int num_del, const int* ids, int num_ids, const int* lost,
                   int num_lost, const int* parent, int source, int* from, int* to) {
  const int total = 2 * num_del + num_ids + num_lost;
  for (int i = ops::thread_index<int>(); i < total; i += ops::grid_stride<int>()) {
    int f;
    int t;
    if (i < 2 * num_del) {
      const int2 pair = deleted[i >> 1];
      f = (i & 1) == 0 ? pair.x : pair.y;
      t = (i & 1) == 0 ? pair.y : pair.x;
    } else if (i < 2 * num_del + num_ids) {
      t = ids[i - 2 * num_del];
      // parent[t] == from always holds: a root, unless it is the source (never a root).
      f = t == source ? -2 : parent[t];
    } else {
      t = lost[i - 2 * num_del - num_ids];
      f = -1;
    }
    from[i] = f;
    to[i] = t;
  }
}

/**
 * The seeds of the inserted pairs without pulls (the knob seed_pairs; MOSP_ESCHER's
 * relaxInsertedKernel and dynG's relax_base_kernel on sssp's words): both directions of every
 * inserted pair offer (d(u) + w(v), u) to v, and a node that gained the source its own weight
 * with no parent; a node whose distance decreased joins the frontier of the pull (same generation,
 * so once). After the pull, as in MOSP_ESCHER.
 */
__global__ void __launch_bounds__(block_size)
    pair_seed_kernel(const int2* inserted, int num_ins, const int* gained, int num_gained,
                     line_graph_view g, u64* packed, packing packed_format, u64 max_distance,
                     int source, int* stamp, int generation, int* frontier, ops::control<int>* c) {
  const int total = 2 * num_ins + num_gained;
  for (int i = ops::thread_index<int>(); i < total; i += ops::grid_stride<int>()) {
    int v;
    u64 candidate;
    if (i < 2 * num_ins) {
      const int2 pair = inserted[i >> 1];
      const int u = (i & 1) == 0 ? pair.x : pair.y;
      v = (i & 1) == 0 ? pair.y : pair.x;
      if (v == source) continue;
      const u64 word = ops::load(&packed[u]);
      if (word == packed_inf) continue;
      const u64 nd = packed_format.distance(word) + g.weight(v);
      if (nd > max_distance) continue;
      candidate = packed_format.pack(nd, u);
    } else {
      v = gained[i - 2 * num_ins];
      candidate = packed_format.pack(g.weight(v), -1);
    }
    if (candidate >= ops::load(&packed[v])) continue;
    const u64 old = atomicMin(&packed[v], candidate);
    if (packed_format.distance(candidate) < packed_format.distance(old) &&
        sssp_kernels::claim(stamp, v, generation)) {
      frontier[sssp_kernels::append_index(&c->frontier)] = v;
    }
  }
}

/// The insertion heads: both endpoints of every inserted pair, then the public mode's nodes that
/// gained the source (their pull includes the base offer).
__global__ void __launch_bounds__(block_size)
    heads_kernel(const int2* inserted, int num_ins, const int* gained, int num_gained, int* heads) {
  const int total = 2 * num_ins + num_gained;
  for (int i = ops::thread_index<int>(); i < total; i += ops::grid_stride<int>()) {
    if (i < 2 * num_ins) {
      const int2 pair = inserted[i >> 1];
      heads[i] = (i & 1) == 0 ? pair.x : pair.y;
    } else {
      heads[i] = gained[i - 2 * num_ins];
    }
  }
}

/// compute(): every word infinite; the virtual mode's source row at 0 (a source node at 0).
__global__ void __launch_bounds__(block_size)
    init_kernel(int n, u64* packed, int source, packing packed_format) {
  for (int v = ops::thread_index<int>(); v < n; v += ops::grid_stride<int>()) {
    packed[v] = v == source ? packed_format.pack(0, -1) : packed_inf;
  }
}

/// compute(): the sum over the rows of deg(v) x w(v), the weight sum of the line graph's entries
/// (an entry u -> v costs w(v)), for MOSP's defaultDelta.
__global__ void __launch_bounds__(block_size)
    entry_weight_kernel(int n, line_graph_view g, u64* sum) {
  u64 local = 0;
  for (int v = ops::thread_index<int>(); v < n; v += ops::grid_stride<int>()) {
    local += static_cast<u64>(g.rows.deg[v]) * g.weight(v);
  }
  sssp_kernels::sum_into(local, sum);
}

/// The start of an unbounded near-far loop (delta 0: MOSP_ESCHER's rounds): nothing is far, so the
/// frontier is the first near frontier as it is (the split would copy it) and the threshold is
/// above every distance.
__global__ void unbounded_start_kernel(ops::control<int>* c) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    c->threshold = packed_inf;
    c->near[0] = c->frontier;
  }
}

/// compute(), after init_kernel: the base nodes take their offers and the seeds become the
/// frontier (the source row, or the base nodes).
__global__ void __launch_bounds__(block_size)
    seed_kernel(u64* packed, int source, const int* base, int num_base, line_graph_view g,
                packing packed_format, int* frontier, ops::control<int>* c) {
  if (source >= 0) {
    if (blockIdx.x == 0 && threadIdx.x == 0) {
      frontier[0] = source;
      c->frontier = 1;
    }
    return;
  }
  for (int i = ops::thread_index<int>(); i < num_base; i += ops::grid_stride<int>()) {
    const int v = base[i];
    packed[v] = packed_format.pack(g.weight(v), -1);
    frontier[i] = v;
  }
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    c->frontier = num_base;
  }
}

}  // namespace
}  // namespace dyng::detail::hyper_ops
