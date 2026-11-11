// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/validation.cu (irLPKernel, irLPEnforceGTKernel,
// launchIrLP, runIrLPValidation, runIrLPSnapshot)
/**
 * @file irlp.cu
 * @brief compute() on CUDA: IrLP, the published baseline (fused.hpp; ADR 0075).
 *
 * The kernels and the iteration are verbatim from the original (its file comment: one 256-thread
 * block per vertex, block-strided sums reduced in shared memory by a tree, the 1e-10 guard of an
 * isolated vertex, the clamp to [0, 1], double buffering, ground truth restored after every
 * iteration, one device-to-host copy of the changed flag per iteration). The snapshot the original
 * builds on the host from its full graph (runIrLPSnapshot: every row's entries between present
 * vertices, in the full graph's row order, which is sorted by column) is built on the device from
 * the current graph's rows instead: the live entries counted and scanned, copied in slot order,
 * and every row sorted by column with CUB's stable segmented sort (the rows of the original's full
 * graph are sorted by column and hold distinct columns, so this is its order). Mechanical changes:
 * names, the offsets templated, the stream of the resources, the buffers from the pooled workspace
 * (the original allocates them per call), CUDA errors thrown.
 */
#include "algorithms/label_propagation/fused.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <cub/device/device_scan.cuh>
#include <cub/device/device_segmented_sort.cuh>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>

namespace dyng::detail::lp_cuda {

namespace {

constexpr int VERTEX_ACTIVE = 0;   // vertex_status::active
constexpr int VERTEX_DELETED = 1;  // vertex_status::deleted

static const int IRLP_BLOCK_SIZE = 256;

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief IrLP iteration kernel: weighted average of neighbor labels (irLPKernel).
 *
 * One block per vertex. Double-buffered: reads from d_labelsOld, writes to d_labelsNew.
 */
template <typename offset_t>
__global__ void ir_lp_kernel(const offset_t* d_rowPtr, const std::int32_t* d_colInd,
                             const float* d_values, const float* d_labelsOld, float* d_labelsNew,
                             const std::int32_t* d_vertexStatus, const std::int32_t* d_gtClass,
                             int numNodes, std::int32_t* d_changed, float delta) {
  int u = blockIdx.x;
  if (u >= numNodes) return;

  // Ground-truth vertices keep their labels
  if (d_gtClass[u] >= 0) {
    if (threadIdx.x == 0) {
      d_labelsNew[u] = d_labelsOld[u];
    }
    return;
  }

  // Deleted/inactive vertices keep neutral labels
  if (d_vertexStatus[u] != VERTEX_ACTIVE) {
    if (threadIdx.x == 0) {
      d_labelsNew[u] = d_labelsOld[u];
    }
    return;
  }

  offset_t rowStart = d_rowPtr[u];
  offset_t rowEnd = d_rowPtr[u + 1];

  // Shared memory for reduction
  __shared__ float s_wAll[IRLP_BLOCK_SIZE];
  __shared__ float s_weightedLabel[IRLP_BLOCK_SIZE];

  float localWAll = 0.0f;
  float localWeightedLabel = 0.0f;

  // Block-strided neighbor traversal
  for (offset_t e = rowStart + threadIdx.x; e < rowEnd; e += blockDim.x) {
    int v = d_colInd[e];
    if (v < 0) continue;
    if (d_vertexStatus[v] == VERTEX_DELETED) continue;

    float w = d_values[e];
    float Fv = d_labelsOld[v];
    localWAll += w;
    localWeightedLabel += w * Fv;
  }

  s_wAll[threadIdx.x] = localWAll;
  s_weightedLabel[threadIdx.x] = localWeightedLabel;
  __syncthreads();

  // Block-level reduction
  for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if (threadIdx.x < stride) {
      s_wAll[threadIdx.x] += s_wAll[threadIdx.x + stride];
      s_weightedLabel[threadIdx.x] += s_weightedLabel[threadIdx.x + stride];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    float wAll = s_wAll[0];
    float weightedLabel = s_weightedLabel[0];
    float oldLabel = d_labelsOld[u];
    float newLabel;

    if (wAll > 1e-10f) {
      newLabel = weightedLabel / wAll;
    } else {
      newLabel = oldLabel;  // Isolated vertex
    }

    // Clamp to [0, 1]
    newLabel = fminf(fmaxf(newLabel, 0.0f), 1.0f);

    d_labelsNew[u] = newLabel;

    if (fabsf(newLabel - oldLabel) > delta) {
      atomicMax(d_changed, 1);
    }
  }
}

/// Enforce ground-truth labels after each iteration (irLPEnforceGTKernel).
__global__ void ir_lp_enforce_gt_kernel(float* d_labels, const std::int32_t* d_gtClass,
                                        int numNodes) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= numNodes) return;
  if (d_gtClass[v] == 0)
    d_labels[v] = 0.0f;
  else if (d_gtClass[v] == 1)
    d_labels[v] = 1.0f;
}

// ---- the snapshot (runIrLPSnapshot's count, scan and fill, on the device) ----

__device__ __forceinline__ bool present(const std::int32_t* status, int v) {
  return status == nullptr || status[v] == VERTEX_ACTIVE;
}

/// The snapshot's ground truth: the seeds that are active (vertexStatus == GROUND_TRUTH), and the
/// starting labels: 0.5, a seed its class.
__global__ void snapshot_seeds_kernel(const std::int32_t* gt_class, const std::int32_t* status,
                                      int n, std::int32_t* gt_active, float* labels) {
  int v = blockIdx.x * blockDim.x + threadIdx.x;
  if (v >= n) return;
  const int c = present(status, v) ? gt_class[v] : -1;
  gt_active[v] = c;
  labels[v] = c >= 0 ? static_cast<float>(c) : 0.5f;
}

template <typename edge_t>
__global__ void snapshot_count_kernel(const edge_t* row_start, const edge_t* row_end,
                                      const std::int32_t* col_ind, const std::int32_t* status,
                                      int n, edge_t* count) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u > n) return;
  edge_t c = 0;
  if (u < n && present(status, u)) {
    for (edge_t e = row_start[u]; e < row_end[u]; ++e) {
      const int v = col_ind[e];
      c += (v >= 0 && present(status, v)) ? 1 : 0;
    }
  }
  count[u] = c;  // count[n] = 0: the scan's last element is the total
}

template <typename edge_t>
__global__ void snapshot_fill_kernel(const edge_t* row_start, const edge_t* row_end,
                                     const std::int32_t* col_ind, const float* values,
                                     const std::int32_t* status, int n, const edge_t* offsets,
                                     std::int32_t* out_col, float* out_values) {
  int u = blockIdx.x * blockDim.x + threadIdx.x;
  if (u >= n || !present(status, u)) return;
  edge_t at = offsets[u];
  for (edge_t e = row_start[u]; e < row_end[u]; ++e) {
    const int v = col_ind[e];
    if (v >= 0 && present(status, v)) {
      out_col[at] = v;
      out_values[at] = values[e];
      ++at;
    }
  }
}

template <typename value_t>
value_t* grow(const resources& res, scratch_buffer<value_t>& b, std::size_t count) {
  return b.reserve_with_headroom(res, std::max<std::size_t>(count, 1));
}

}  // namespace

std::size_t irlp_workspace::bytes() const noexcept {
  return row_ptr.bytes() + count.bytes() + col_ind.bytes() + values.bytes() + col_sorted.bytes() +
         values_sorted.bytes() + temp.bytes() + labels_buffer.bytes() + gt_class.bytes() +
         status.bytes() + words.bytes() + host_words.size() * sizeof(long long);
}

// ============================================================================
// Host function (runIrLPSnapshot, runIrLPValidation, launchIrLP)
// ============================================================================

template <typename edge_t>
int run_irlp(const resources& res, const rows<edge_t>& g, const std::int32_t* d_gtClass,
             float* d_labels, float delta, int maxIterations, irlp_workspace& ws, bool& capped) {
  const scoped_device guard(res.device());
  const auto stream = static_cast<cudaStream_t>(res.stream().get());
  const int numNodes = static_cast<int>(g.n);
  const auto n = static_cast<std::size_t>(numNodes);
  capped = false;
  if (ws.host_words.size() < 2) {
    ws.host_words =
        buffer<long long>(2, res.stream(), resources_access::staging_memory(res), res.device());
  }
  long long* h_words = ws.host_words.data();
  std::int32_t* gt_active = grow(res, ws.gt_class, n);
  long long* d_words = grow(res, ws.words, 2);
  const int threadsPerBlock = 256;
  const int numBlocksAll = (numNodes + threadsPerBlock - 1) / threadsPerBlock;
  if (numNodes <= 0) {
    return 0;
  }
  // --- The starting labels: 0.5, an active seed its class (runIrLPSnapshot) ---
  snapshot_seeds_kernel<<<numBlocksAll, threadsPerBlock, 0, stream>>>(d_gtClass, g.status, numNodes,
                                                                      gt_active, d_labels);
  DYNG_CHECK_KERNEL(stream);
  // --- The snapshot: count, scan, fill, sort each row by column ---
  auto* count = reinterpret_cast<edge_t*>(grow(res, ws.count, (n + 1) * sizeof(edge_t)));
  auto* offsets = reinterpret_cast<edge_t*>(grow(res, ws.row_ptr, (n + 1) * sizeof(edge_t)));
  snapshot_count_kernel<edge_t>
      <<<(numNodes + threadsPerBlock) / threadsPerBlock, threadsPerBlock, 0, stream>>>(
          g.row_start, g.row_end, g.col_ind, g.status, numNodes, count);
  DYNG_CHECK_KERNEL(stream);
  std::size_t scan_bytes = 0;
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(nullptr, scan_bytes, count, offsets, numNodes + 1, stream));
  unsigned char* temp = grow(res, ws.temp, scan_bytes);
  DYNG_CUDA_TRY(
      cub::DeviceScan::ExclusiveSum(temp, scan_bytes, count, offsets, numNodes + 1, stream));
  DYNG_CUDA_TRY(cudaMemcpyAsync(h_words + 1, offsets + numNodes, sizeof(edge_t),
                                cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  edge_t total = 0;
  std::memcpy(&total, h_words + 1, sizeof(edge_t));
  if (total <= 0) {
    // runIrLPValidation: "IrLP validation skipped (empty graph)"; the labels stay as they start.
    return 0;
  }
  const auto m = static_cast<std::size_t>(total);
  std::int32_t* fill_col = grow(res, ws.col_ind, m);
  float* fill_values = grow(res, ws.values, m);
  std::int32_t* d_colInd = grow(res, ws.col_sorted, m);
  float* d_values = grow(res, ws.values_sorted, m);
  snapshot_fill_kernel<edge_t><<<numBlocksAll, threadsPerBlock, 0, stream>>>(
      g.row_start, g.row_end, g.col_ind, g.values, g.status, numNodes, offsets, fill_col,
      fill_values);
  DYNG_CHECK_KERNEL(stream);
  std::size_t sort_bytes = 0;
  DYNG_CUDA_TRY(cub::DeviceSegmentedSort::StableSortPairs(
      nullptr, sort_bytes, fill_col, d_colInd, fill_values, d_values, static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(numNodes), offsets, offsets + 1, stream));
  temp = grow(res, ws.temp, sort_bytes);
  DYNG_CUDA_TRY(cub::DeviceSegmentedSort::StableSortPairs(
      temp, sort_bytes, fill_col, d_colInd, fill_values, d_values, static_cast<std::int64_t>(m),
      static_cast<std::int64_t>(numNodes), offsets, offsets + 1, stream));
  const edge_t* d_rowPtr = offsets;
  // The states the kernel reads (a compact graph has none: every vertex active).
  const std::int32_t* d_vertexStatus = g.status;
  if (d_vertexStatus == nullptr) {
    std::int32_t* all_active = grow(res, ws.status, n);
    DYNG_CUDA_TRY(cudaMemsetAsync(all_active, 0, n * sizeof(std::int32_t), stream));
    d_vertexStatus = all_active;
  }

  // --- launchIrLP ---
  float* d_labelsBuffer = grow(res, ws.labels_buffer, n);
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_labelsBuffer, d_labels, n * sizeof(float),
                                cudaMemcpyDeviceToDevice, stream));
  auto* d_changed = reinterpret_cast<std::int32_t*>(d_words);
  float* d_labelsOld = d_labels;
  float* d_labelsNew = d_labelsBuffer;
  int iterationsUsed = 0;
  int changed = 0;
  for (int iter = 0; iter < maxIterations; ++iter) {
    DYNG_CUDA_TRY(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));
    // One block per vertex
    ir_lp_kernel<edge_t><<<numNodes, IRLP_BLOCK_SIZE, 0, stream>>>(
        d_rowPtr, d_colInd, d_values, d_labelsOld, d_labelsNew, d_vertexStatus, gt_active, numNodes,
        d_changed, delta);
    DYNG_CHECK_KERNEL(stream);
    // Enforce GT labels
    ir_lp_enforce_gt_kernel<<<numBlocksAll, threadsPerBlock, 0, stream>>>(d_labelsNew, gt_active,
                                                                          numNodes);
    DYNG_CHECK_KERNEL(stream);
    // Swap buffers
    std::swap(d_labelsOld, d_labelsNew);
    iterationsUsed = iter + 1;
    // Check convergence
    DYNG_CUDA_TRY(cudaMemcpyAsync(h_words, d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    note_host_sync();
    std::memcpy(&changed, h_words, sizeof(int));
    if (changed == 0) break;
  }
  capped = changed != 0;
  // Ensure final labels are in d_labels
  if (d_labelsOld != d_labels) {
    DYNG_CUDA_TRY(cudaMemcpyAsync(d_labels, d_labelsOld, n * sizeof(float),
                                  cudaMemcpyDeviceToDevice, stream));
  }
  return iterationsUsed;
}

template int run_irlp<std::int32_t>(const resources&, const rows<std::int32_t>&,
                                    const std::int32_t*, float*, float, int, irlp_workspace&,
                                    bool&);
template int run_irlp<std::int64_t>(const resources&, const rows<std::int64_t>&,
                                    const std::int32_t*, float*, float, int, irlp_workspace&,
                                    bool&);

DYNG_REGISTER_KERNEL(ir_lp_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(ir_lp_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(ir_lp_enforce_gt_kernel);
DYNG_REGISTER_KERNEL(snapshot_seeds_kernel);
DYNG_REGISTER_KERNEL(snapshot_count_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(snapshot_count_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(snapshot_fill_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(snapshot_fill_kernel<std::int64_t>);

}  // namespace dyng::detail::lp_cuda
