// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/iterativePropagation.cu (the whole file: halfSweep,
// appendToQueue, accumulateNeighbors, updateLabel, appendNextFrontier, processVertexGroup,
// blockSum, processVertexBlock, initFrontierKernel, propagateGroupKernel, propagateBlockKernel,
// commitLabelsKernel, smallFrontierKernel, enforceGroundTruthKernel, launchBinSweep,
// createPropagationWorkspace, launchIterativePropagation) and headers/iterativePropagation.h
/**
 * @file propagation.cu
 * @brief Step 3 of DynLP on CUDA, the iterative propagation (fused.hpp; ADR 0075).
 *
 * Verbatim from the original (headers/iterativePropagation.h and its file comment): the frontier
 * is a queue per degree bin and half-sweep (two queue sets alternating between iterations; the
 * second half-sweep filled from the back); 8 lanes per vertex up to 32 neighbours, a warp up to
 * 1024, a 256-thread block above, with warp shuffles and a shared-memory block sum; a vertex whose
 * label changed by more than delta appends itself and its unlabelled active neighbours to the next
 * queues, deduplicated by run-wide stamps (the tag of the iteration whose queue holds it, never
 * cleared) with warp-aggregated appends (__match_any_sync); the persistent smallFrontierKernel
 * <<<1, 1024>>> runs the iterations of frontiers of at most SMALL_FRONTIER_LIMIT vertices back to
 * back; one host synchronization per head iteration (the next queue sizes). in_place reads the
 * neighbours' current labels through L2 (__ldcg) and writes with __stcg; damped writes a side array
 * committed after both half-sweeps. The floating-point arithmetic is the original's, compiled with
 * its flags (--use_fast_math: this target's CUDA TUs only, PLAN 7.4; ADR 0066).
 *
 * Mechanical changes: names, the rows templated on the offset type (the int32 instantiation is
 * the original's code), the statuses with vertex_status's values (active 0 and deleted 1 are the
 * original's codes), the stream of the resources, the queues and counters from the pooled
 * workspace and the stamps and nextTag from the result (fused.hpp), CUDA errors thrown, the prof::
 * counters returned (propagation_report), and a callback after the first synchronization (the
 * settle of the graph's report, merged into the original's own read of the first queue sizes).
 */
#include "algorithms/label_propagation/fused.hpp"
#include "core/budget_counters.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace dyng::detail::lp_cuda {

namespace {

constexpr int VERTEX_ACTIVE = 0;   // vertex_status::active
constexpr int VERTEX_DELETED = 1;  // vertex_status::deleted

cudaStream_t as_stream(void* s) noexcept {
  return static_cast<cudaStream_t>(s);
}

const int SMALL_BIN_LANES = 8;        // threads per vertex, small bin
const int MEDIUM_BIN_LANES = 32;      // threads per vertex, medium bin
const int BLOCK_THREADS = 256;        // threads per block (multi-block kernels)
const int PERSISTENT_THREADS = 1024;  // threads of the persistent kernel

/// Degree bin of a vertex with @p degree current neighbors (degreeBin).
__host__ __device__ __forceinline__ int degreeBin(int degree) {
  if (degree <= small_bin_max_degree) return bin_small;
  if (degree <= medium_bin_max_degree) return bin_medium;
  return bin_large;
}

/// Read-only view of the graph plus the label array. The neighbors of v are the entries
/// [rowStart[v], rowEnd[v]).
template <typename edge_t>
struct GraphView {
  const edge_t* rowStart;
  const edge_t* rowEnd;
  const std::int32_t* colInd;
  const float* values;
  const std::int32_t* status;
  const std::int32_t* gtClass;
  float* labels;
};

/// Destination of appends: the next queue set.
struct QueueOut {
  std::int32_t* queue[num_degree_bins];
  int capacity[num_degree_bins];
  std::int32_t* counts;  // counters_per_set counters of the next set
  std::int32_t* stamp;
  int tag;  // tag of the next iteration
};

/// Both queue sets, for the persistent kernel.
struct QueueSets {
  std::int32_t* queue[2][num_degree_bins];
  int capacity[num_degree_bins];
  float* newLabels[num_degree_bins];
  std::int32_t* counts;  // 2 * counters_per_set
  std::int32_t* stamp;
};

/// Half-sweep (0 or 1) of vertex v in the iteration with tag `tag`: a hash, so that adjacent
/// vertices are separated with probability 1/2 in every iteration.
__device__ __forceinline__ int halfSweep(int v, int tag) {
  unsigned x = static_cast<unsigned>(v) ^ (static_cast<unsigned>(tag) * 0x9E3779B9u);
  x ^= x >> 16;
  x *= 0x7FEB352Du;
  x ^= x >> 15;
  x *= 0x846CA68Bu;
  x ^= x >> 16;
  return static_cast<int>(x & 1u);
}

/// Offset of a half-sweep inside a bin's queue array.
__host__ __device__ __forceinline__ int halfOffset(int half, int capacity, int count) {
  return half == 0 ? 0 : capacity - count;
}

// ============================================================================
// Per-vertex building blocks (shared by all thread mappings)
// ============================================================================

/// Append v to the next queue of its degree bin, once per iteration (appendToQueue).
template <typename edge_t>
__device__ __forceinline__ void appendToQueue(int v, const GraphView<edge_t>& g,
                                              const QueueOut& out) {
  if (out.stamp[v] == out.tag) return;
  if (atomicExch(&out.stamp[v], out.tag) == out.tag) return;
  const int bin = degreeBin(static_cast<int>(g.rowEnd[v] - g.rowStart[v]));
  const int half = halfSweep(v, out.tag);
  const int key = bin * 2 + half;
  const unsigned active = __activemask();
  const unsigned peers = __match_any_sync(active, key);
  const int lane = threadIdx.x & 31;
  const int leader = __ffs(peers) - 1;
  int base = 0;
  if (lane == leader) base = atomicAdd(&out.counts[key], __popc(peers));
  base = __shfl_sync(peers, base, leader);
  const int idx = base + __popc(peers & ((1u << lane) - 1u));
  out.queue[bin][half == 0 ? idx : out.capacity[bin] - 1 - idx] = v;
}

/// Partial sums over the entries e = rowStart + lane, + stride, ... (accumulateNeighbors). In-place
/// mode reads neighbor labels through L2 (__ldcg).
template <typename edge_t>
__device__ __forceinline__ void accumulateNeighbors(const GraphView<edge_t>& g, edge_t rowStart,
                                                    edge_t rowEnd, int lane, int stride, int damped,
                                                    float& wAll, float& weightedLabel) {
  wAll = 0.0f;
  weightedLabel = 0.0f;
  for (edge_t e = rowStart + lane; e < rowEnd; e += stride) {
    int v = g.colInd[e];
    if (v < 0) continue;                          // soft-deleted edge
    if (g.status[v] == VERTEX_DELETED) continue;  // deleted vertex
    float w = g.values[e];
    float Fv = damped ? g.labels[v] : __ldcg(&g.labels[v]);
    wAll += w;
    weightedLabel += w * Fv;
  }
}

/// Compute and store the new label of u; true if it changed by > delta (updateLabel).
template <typename edge_t>
__device__ __forceinline__ bool updateLabel(const GraphView<edge_t>& g, int u, float wAll,
                                            float weightedLabel, float* newLabelSlot, float delta,
                                            int damped) {
  float oldLabel = g.labels[u];
  // F'_u = sum alpha_{u,v} * F_v = (sum w(u,v) * F_v) / (sum w(u,v))
  float newLabel = (wAll > 1e-10f) ? weightedLabel / wAll : oldLabel;
  if (damped) newLabel = 0.5f * (oldLabel + newLabel);
  // Clamp to [0, 1] for numerical stability
  newLabel = fminf(fmaxf(newLabel, 0.0f), 1.0f);
  if (damped) {
    *newLabelSlot = newLabel;  // committed after the iteration
  } else {
    __stcg(&g.labels[u], newLabel);  // in place, visible to other threads
  }
  return fabsf(newLabel - oldLabel) > delta;
}

/// Append u's unlabeled active neighbors (lane-strided) and, from lane 0, u itself to the next
/// queues (appendNextFrontier).
template <typename edge_t>
__device__ __forceinline__ void appendNextFrontier(const GraphView<edge_t>& g, int u,
                                                   edge_t rowStart, edge_t rowEnd, int lane,
                                                   int stride, const QueueOut& out) {
  for (edge_t e = rowStart + lane; e < rowEnd; e += stride) {
    int v = g.colInd[e];
    if (v >= 0 && g.status[v] == VERTEX_ACTIVE && g.gtClass[v] < 0) {
      appendToQueue(v, g, out);
    }
  }
  if (lane == 0) appendToQueue(u, g, out);
}

/// Full update of the vertex at queue position i by a group of LANES threads
/// (processVertexGroup).
template <int LANES, typename edge_t>
__device__ __forceinline__ void processVertexGroup(const GraphView<edge_t>& g, const int* list,
                                                   float* newLabels, int i, int lane,
                                                   const QueueOut& out, float delta, int damped) {
  const unsigned mask =
      (LANES == 32) ? 0xffffffffu : (((1u << LANES) - 1u) << ((threadIdx.x & 31) & ~(LANES - 1)));
  const int u = list[i];
  const edge_t rowStart = g.rowStart[u];
  const edge_t rowEnd = g.rowEnd[u];

  float wAll, weightedLabel;
  accumulateNeighbors(g, rowStart, rowEnd, lane, LANES, damped, wAll, weightedLabel);
#pragma unroll
  for (int offset = LANES / 2; offset > 0; offset >>= 1) {
    wAll += __shfl_down_sync(mask, wAll, offset, LANES);
    weightedLabel += __shfl_down_sync(mask, weightedLabel, offset, LANES);
  }

  int changed = 0;
  if (lane == 0) {
    changed =
        updateLabel(g, u, wAll, weightedLabel, newLabels ? &newLabels[i] : nullptr, delta, damped);
  }
  changed = __shfl_sync(mask, changed, 0, LANES);
  if (changed) appendNextFrontier(g, u, rowStart, rowEnd, lane, LANES, out);
}

/// Sum a value over the whole block (blockSum).
__device__ __forceinline__ float blockSum(float value, float* s_partial) {
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int numWarps = (blockDim.x + 31) >> 5;
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffffu, value, offset);
  }
  __syncthreads();  // s_partial may still be read by a previous call
  if (lane == 0) s_partial[warp] = value;
  __syncthreads();
  float total = 0.0f;
  for (int w = 0; w < numWarps; ++w) total += s_partial[w];
  return total;
}

/// Full update of the vertex at queue position i by the whole block (processVertexBlock).
template <typename edge_t>
__device__ __forceinline__ void processVertexBlock(const GraphView<edge_t>& g, const int* list,
                                                   float* newLabels, int i, const QueueOut& out,
                                                   float delta, int damped, float* s_partial,
                                                   int* s_flag) {
  const int u = list[i];
  const edge_t rowStart = g.rowStart[u];
  const edge_t rowEnd = g.rowEnd[u];

  float wAll, weightedLabel;
  accumulateNeighbors(g, rowStart, rowEnd, threadIdx.x, blockDim.x, damped, wAll, weightedLabel);
  wAll = blockSum(wAll, s_partial);
  weightedLabel = blockSum(weightedLabel, s_partial);

  if (threadIdx.x == 0) {
    *s_flag =
        updateLabel(g, u, wAll, weightedLabel, newLabels ? &newLabels[i] : nullptr, delta, damped);
  }
  __syncthreads();
  const int changed = *s_flag;
  if (changed) {
    appendNextFrontier(g, u, rowStart, rowEnd, threadIdx.x, blockDim.x, out);
  }
  __syncthreads();  // s_flag is reused by the next vertex
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/// Build the first queue of a batch from the affected flags (initFrontierKernel).
template <typename edge_t>
__global__ void init_frontier_kernel(const std::int32_t* d_isAffected, GraphView<edge_t> g,
                                     QueueOut out, int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;
  // Only active, non-GT, affected vertices
  if (d_isAffected[v] == 1 && g.status[v] == VERTEX_ACTIVE && g.gtClass[v] < 0) {
    appendToQueue(v, g, out);
  }
}

/// Update the vertices of one queue with LANES threads per vertex (propagateGroupKernel).
template <int LANES, typename edge_t>
__global__ void propagate_group_kernel(GraphView<edge_t> g, const int* list, float* newLabels,
                                       int count, QueueOut out, float delta, int damped) {
  const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  const int i = static_cast<int>(t / LANES);
  const int lane = static_cast<int>(t % LANES);
  if (i >= count) return;
  processVertexGroup<LANES>(g, list, newLabels, i, lane, out, delta, damped);
}

/// Update the vertices of one queue with one block per vertex (propagateBlockKernel).
template <typename edge_t>
__global__ void propagate_block_kernel(GraphView<edge_t> g, const int* list, float* newLabels,
                                       int count, QueueOut out, float delta, int damped) {
  __shared__ float s_partial[BLOCK_THREADS / 32];
  __shared__ int s_flag;
  if (static_cast<int>(blockIdx.x) >= count) return;
  processVertexBlock(g, list, newLabels, blockIdx.x, out, delta, damped, s_partial, &s_flag);
}

/// Damped mode: commit the new labels of one queue (commitLabelsKernel).
__global__ void commit_labels_kernel(float* d_labels, const float* d_newLabels, const int* d_list,
                                     int count) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < count) d_labels[d_list[i]] = d_newLabels[i];
}

/// Persistent single-block propagation for small frontiers (smallFrontierKernel).
template <typename edge_t>
__global__ void __launch_bounds__(PERSISTENT_THREADS)
    small_frontier_kernel(GraphView<edge_t> g, QueueSets qs, int cur, int tag, int limit,
                          int maxIter, float delta, int damped, long long* control) {
  __shared__ float s_partial[PERSISTENT_THREADS / 32];
  __shared__ int s_flag;
  __shared__ int s_counts[counters_per_set];
  __shared__ int s_total;

  const int lane8 = threadIdx.x % SMALL_BIN_LANES;
  const int group8 = threadIdx.x / SMALL_BIN_LANES;
  const int groups8 = blockDim.x / SMALL_BIN_LANES;
  const int lane32 = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  const int warps = blockDim.x >> 5;

  if (threadIdx.x < counters_per_set) {
    s_counts[threadIdx.x] = qs.counts[cur * counters_per_set + threadIdx.x];
  }
  __syncthreads();
  if (threadIdx.x == 0) {
    int total = 0;
    for (int c = 0; c < counters_per_set; ++c) total += s_counts[c];
    s_total = total;
  }
  __syncthreads();

  int iterations = 0;
  long long visits = 0;
  while (s_total > 0 && s_total <= limit && iterations < maxIter) {
    const int next = cur ^ 1;
    QueueOut out;
    for (int b = 0; b < num_degree_bins; ++b) {
      out.queue[b] = qs.queue[next][b];
      out.capacity[b] = qs.capacity[b];
    }
    out.counts = qs.counts + next * counters_per_set;
    out.stamp = qs.stamp;
    out.tag = tag + 1;
    if (threadIdx.x < counters_per_set) atomicExch(&out.counts[threadIdx.x], 0);
    visits += s_total;
    __syncthreads();

    for (int half = 0; half < 2; ++half) {
      for (int b = 0; b < num_degree_bins; ++b) {
        const int count = s_counts[b * 2 + half];
        if (count == 0) continue;
        const int offset = halfOffset(half, qs.capacity[b], count);
        const int* list = qs.queue[cur][b] + offset;
        float* newLabels = damped ? qs.newLabels[b] + offset : nullptr;
        if (b == bin_small) {
          for (int i = group8; i < count; i += groups8) {
            processVertexGroup<SMALL_BIN_LANES>(g, list, newLabels, i, lane8, out, delta, damped);
          }
        } else if (b == bin_medium) {
          for (int i = warp; i < count; i += warps) {
            processVertexGroup<MEDIUM_BIN_LANES>(g, list, newLabels, i, lane32, out, delta, damped);
          }
        } else {
          for (int i = 0; i < count; ++i) {
            processVertexBlock(g, list, newLabels, i, out, delta, damped, s_partial, &s_flag);
          }
        }
      }
      // The second half-sweep reads the labels written by the first.
      __syncthreads();
    }

    if (damped) {
      for (int b = 0; b < num_degree_bins; ++b) {
        for (int half = 0; half < 2; ++half) {
          const int count = s_counts[b * 2 + half];
          const int offset = halfOffset(half, qs.capacity[b], count);
          for (int i = threadIdx.x; i < count; i += blockDim.x) {
            g.labels[qs.queue[cur][b][offset + i]] = qs.newLabels[b][offset + i];
          }
        }
      }
    }
    __syncthreads();

    // Next frontier sizes (appended with atomics during this iteration).
    if (threadIdx.x < counters_per_set) {
      s_counts[threadIdx.x] = atomicAdd(&out.counts[threadIdx.x], 0);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      int total = 0;
      for (int c = 0; c < counters_per_set; ++c) total += s_counts[c];
      s_total = total;
    }
    cur = next;
    ++tag;
    ++iterations;
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    control[0] = iterations;
    control[1] = cur;
    control[2] = visits;
  }
}

/// Ensure GT vertices retain their fixed labels (enforceGroundTruthKernel).
__global__ void enforce_ground_truth_kernel(float* d_labels, const std::int32_t* d_gtClass,
                                            int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;

  if (d_gtClass[v] == 0) {
    d_labels[v] = 0.0f;
  } else if (d_gtClass[v] == 1) {
    d_labels[v] = 1.0f;
  }
}

/// The labels' bits before the update (dynG: the stats' affected).
__global__ void keep_labels_kernel(const float* labels, std::int32_t* bits, int n) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v < n) bits[v] = __float_as_int(labels[v]);
}

/// The vertices whose label bits changed (dynG: the stats' affected).
__global__ void count_changed_kernel(const float* labels, const std::int32_t* bits, int n,
                                     std::int32_t* count) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  const bool changed = v < n && __float_as_int(labels[v]) != bits[v];
  const unsigned votes = __ballot_sync(0xffffffffu, changed);
  if ((threadIdx.x & 31) == 0 && votes != 0) atomicAdd(count, __popc(votes));
}

inline int blocksFor(long long threads) {
  return static_cast<int>((threads + BLOCK_THREADS - 1) / BLOCK_THREADS);
}

QueueOut queueOut(state& st, workspace& ws, int set, int tag) {
  QueueOut out;
  for (int b = 0; b < num_degree_bins; ++b) {
    out.queue[b] = ws.queue[set][b].data();
    out.capacity[b] = st.capacity[b];
  }
  out.counts = ws.counts.data() + set * counters_per_set;
  out.stamp = st.stamp.data();
  out.tag = tag;
  return out;
}

/// Launch the update kernel of one degree bin for one half-sweep (launchBinSweep).
template <typename edge_t>
void launchBinSweep(int bin, const GraphView<edge_t>& g, const int* list, float* newLabels,
                    int count, const QueueOut& out, float delta, int damped, cudaStream_t stream) {
  if (count == 0) return;
  if (bin == bin_small) {
    propagate_group_kernel<SMALL_BIN_LANES, edge_t>
        <<<blocksFor(static_cast<long long>(count) * SMALL_BIN_LANES), BLOCK_THREADS, 0, stream>>>(
            g, list, newLabels, count, out, delta, damped);
  } else if (bin == bin_medium) {
    propagate_group_kernel<MEDIUM_BIN_LANES, edge_t>
        <<<blocksFor(static_cast<long long>(count) * MEDIUM_BIN_LANES), BLOCK_THREADS, 0, stream>>>(
            g, list, newLabels, count, out, delta, damped);
  } else {
    propagate_block_kernel<edge_t>
        <<<count, BLOCK_THREADS, 0, stream>>>(g, list, newLabels, count, out, delta, damped);
  }
  DYNG_CHECK_KERNEL(stream);
}

}  // namespace

void launch_keep_labels(const float* labels, std::int32_t* before_bits, int n, void* s) {
  if (n == 0) return;
  keep_labels_kernel<<<blocksFor(n), BLOCK_THREADS, 0, as_stream(s)>>>(labels, before_bits, n);
  DYNG_CHECK_KERNEL(as_stream(s));
}

void launch_count_changed(const float* labels, const std::int32_t* before_bits, int n,
                          std::int32_t* count, void* s) {
  DYNG_CUDA_TRY(cudaMemsetAsync(count, 0, sizeof(std::int32_t), as_stream(s)));
  if (n == 0) return;
  count_changed_kernel<<<blocksFor(n), BLOCK_THREADS, 0, as_stream(s)>>>(labels, before_bits, n,
                                                                         count);
  DYNG_CHECK_KERNEL(as_stream(s));
}

// ============================================================================
// Host Launch Function (launchIterativePropagation)
// ============================================================================

template <typename edge_t>
propagation_report launch_iterative_propagation(const rows<edge_t>& view, float* d_labels,
                                                const std::int32_t* d_gtClass,
                                                const std::int32_t* d_isAffected, float delta,
                                                int maxIterations, bool is_damped, state& st,
                                                workspace& ws, void* s,
                                                void (*after_first_sync)(void*), void* arg) {
  const cudaStream_t stream = as_stream(s);
  const int numNodes = static_cast<int>(view.n);
  const int damped = is_damped ? 1 : 0;
  const GraphView<edge_t> g = {view.row_start, view.row_end, view.col_ind, view.values,
                               view.status,    d_gtClass,    d_labels};
  propagation_report report;
  report.tag_before = st.next_tag;
  std::int32_t* h_counts = ws.host_counts.data();
  long long* h_control = ws.host_control.data();

  // --- First frontier: the affected vertices, in queue set 0 ---
  int cur = 0;
  int tag = st.next_tag++;
  DYNG_CUDA_TRY(cudaMemsetAsync(ws.counts.data(), 0, counters_per_set * sizeof(int), stream));
  init_frontier_kernel<edge_t><<<blocksFor(numNodes), BLOCK_THREADS, 0, stream>>>(
      d_isAffected, g, queueOut(st, ws, cur, tag), numNodes);
  DYNG_CHECK_KERNEL(stream);
  DYNG_CUDA_TRY(cudaMemcpyAsync(h_counts, ws.counts.data(), counters_per_set * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  ++report.host_syncs;
  if (after_first_sync != nullptr) {
    after_first_sync(arg);
  }

  auto frontierSize = [&](int set) {
    long long total = 0;
    for (int c = 0; c < counters_per_set; ++c) {
      total += h_counts[set * counters_per_set + c];
    }
    return total;
  };

  int iterationsUsed = 0;
  long long frontierVisits = 0;
  long long persistentIterations = 0;
  long long total = frontierSize(cur);
  report.initial_frontier = total;

  // Converged when the frontier is empty: every vertex that changed by more than delta
  // re-appends itself, so an empty queue means no change.
  while (total > 0 && iterationsUsed < maxIterations) {
    if (total <= small_frontier_limit) {
      // --- Tail: persistent single-block iterations ---
      QueueSets qs;
      for (int b = 0; b < num_degree_bins; ++b) {
        qs.queue[0][b] = ws.queue[0][b].data();
        qs.queue[1][b] = ws.queue[1][b].data();
        qs.capacity[b] = st.capacity[b];
        qs.newLabels[b] = damped ? ws.new_labels[b].data() : nullptr;
      }
      qs.counts = ws.counts.data();
      qs.stamp = st.stamp.data();
      small_frontier_kernel<edge_t><<<1, PERSISTENT_THREADS, 0, stream>>>(
          g, qs, cur, tag, small_frontier_limit, maxIterations - iterationsUsed, delta, damped,
          ws.control.data());
      DYNG_CHECK_KERNEL(stream);
      DYNG_CUDA_TRY(cudaMemcpyAsync(h_control, ws.control.data(), 3 * sizeof(long long),
                                    cudaMemcpyDeviceToHost, stream));
      DYNG_CUDA_TRY(cudaMemcpyAsync(h_counts, ws.counts.data(), 2 * counters_per_set * sizeof(int),
                                    cudaMemcpyDeviceToHost, stream));
      DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
      note_host_sync();
      ++report.host_syncs;
      const int done = static_cast<int>(h_control[0]);
      cur = static_cast<int>(h_control[1]);
      frontierVisits += h_control[2];
      persistentIterations += done;
      iterationsUsed += done;
      tag += done;
      total = frontierSize(cur);
      continue;
    }

    // --- Head: one launch per (half-sweep, degree bin) ---
    const int next = cur ^ 1;
    const QueueOut out = queueOut(st, ws, next, tag + 1);
    DYNG_CUDA_TRY(cudaMemsetAsync(out.counts, 0, counters_per_set * sizeof(int), stream));
    frontierVisits += total;
    const std::int32_t* counts = h_counts + cur * counters_per_set;
    for (int half = 0; half < 2; ++half) {
      for (int b = 0; b < num_degree_bins; ++b) {
        const int count = counts[b * 2 + half];
        const int offset = halfOffset(half, st.capacity[b], count);
        launchBinSweep(b, g, ws.queue[cur][b].data() + offset,
                       damped ? ws.new_labels[b].data() + offset : nullptr, count, out, delta,
                       damped, stream);
      }
    }
    // Damped: all new values were computed from this iteration's labels; commit them now.
    if (damped) {
      for (int half = 0; half < 2; ++half) {
        for (int b = 0; b < num_degree_bins; ++b) {
          const int count = counts[b * 2 + half];
          if (count == 0) continue;
          const int offset = halfOffset(half, st.capacity[b], count);
          commit_labels_kernel<<<blocksFor(count), BLOCK_THREADS, 0, stream>>>(
              d_labels, ws.new_labels[b].data() + offset, ws.queue[cur][b].data() + offset, count);
          DYNG_CHECK_KERNEL(stream);
        }
      }
    }
    // The only host synchronization of the iteration: next queue sizes.
    DYNG_CUDA_TRY(cudaMemcpyAsync(h_counts + next * counters_per_set, out.counts,
                                  counters_per_set * sizeof(int), cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    note_host_sync();
    ++report.host_syncs;
    cur = next;
    ++tag;
    ++iterationsUsed;
    total = frontierSize(cur);
  }
  st.next_tag = tag + 1;
  report.iterations = iterationsUsed;
  report.frontier_visits = frontierVisits;
  report.persistent_iterations = persistentIterations;
  report.capped = total > 0;

  // --- Enforce ground-truth labels (safety net) ---
  enforce_ground_truth_kernel<<<blocksFor(numNodes), BLOCK_THREADS, 0, stream>>>(
      d_labels, d_gtClass, numNodes);
  DYNG_CHECK_KERNEL(stream);
  return report;
}

template propagation_report launch_iterative_propagation<std::int32_t>(
    const rows<std::int32_t>&, float*, const std::int32_t*, const std::int32_t*, float, int, bool,
    state&, workspace&, void*, void (*)(void*), void*);
template propagation_report launch_iterative_propagation<std::int64_t>(
    const rows<std::int64_t>&, float*, const std::int32_t*, const std::int32_t*, float, int, bool,
    state&, workspace&, void*, void (*)(void*), void*);

DYNG_REGISTER_KERNEL(init_frontier_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(init_frontier_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(propagate_group_kernel<8, std::int32_t>);
DYNG_REGISTER_KERNEL(propagate_group_kernel<32, std::int32_t>);
DYNG_REGISTER_KERNEL(propagate_group_kernel<8, std::int64_t>);
DYNG_REGISTER_KERNEL(propagate_group_kernel<32, std::int64_t>);
DYNG_REGISTER_KERNEL(propagate_block_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(propagate_block_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(commit_labels_kernel);
DYNG_REGISTER_KERNEL(small_frontier_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(small_frontier_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(enforce_ground_truth_kernel);
DYNG_REGISTER_KERNEL(keep_labels_kernel);
DYNG_REGISTER_KERNEL(count_changed_kernel);

}  // namespace dyng::detail::lp_cuda
