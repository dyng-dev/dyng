// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/connectedComponents.cu (initParentKernel,
// hookKernel, compressKernel, markRootsKernel, numberRootsKernel, assignComponentIdsKernel,
// createCCWorkspace, launchConnectedComponents)
/**
 * @file components.cu
 * @brief Shiloach-Vishkin connected components of G' on CUDA (fused.hpp; ADR 0075).
 *
 * Verbatim from the original (its file comment: hook with atomicMin of the larger root to the
 * smaller, compress by following the parents to the root, alternate until a hook changes nothing,
 * with one device-to-host copy of the changed flag per round; then the roots are flagged, scanned
 * with CUB and numbered, and every vertex takes its root's id). Mechanical changes: names, the
 * stream of the resources, the workspace from the pool (fused.hpp), CUDA errors thrown, and the
 * 64-round cap (a malfunction guard) an internal_error instead of exit(). Two host
 * synchronizations per batch with insertions beyond the rounds: none (the original's).
 */
#include "algorithms/label_propagation/fused.hpp"
#include "core/budget_counters.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cub/device/device_scan.cuh>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dyng::detail::lp_cuda {

namespace {

cudaStream_t as_stream(void* s) noexcept {
  return static_cast<cudaStream_t>(s);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/// Initialize parent array: par[v] = v for all new vertices (initParentKernel).
__global__ void init_parent_kernel(std::int32_t* d_par, const std::int32_t* d_insertedNodes,
                                   int numInserted) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int v = d_insertedNodes[tid];
  d_par[v] = v;
}

/// Hook phase: merge trees by hooking larger root to smaller root (hookKernel). The edge count is
/// read on the device.
__global__ void hook_kernel(const std::int32_t* d_edgeSrc, const std::int32_t* d_edgeDst,
                            const std::int32_t* d_numEdges, std::int32_t* d_par,
                            std::int32_t* d_changed) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= *d_numEdges) return;
  int u = d_edgeSrc[tid];
  int v = d_edgeDst[tid];
  // After a Compress phase every parent is a root
  int rootU = d_par[u];
  int rootV = d_par[v];
  if (rootU != rootV) {
    // Hook the larger root to the smaller root
    int largeRoot = max(rootU, rootV);
    int smallRoot = min(rootU, rootV);
    int old = atomicMin(&d_par[largeRoot], smallRoot);
    if (old != smallRoot) {
      *d_changed = 1;  // Signal that a change occurred
    }
  }
}

/// Compress phase: point every vertex directly at its root (compressKernel).
__global__ void compress_kernel(std::int32_t* d_par, const std::int32_t* d_insertedNodes,
                                int numInserted) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int v = d_insertedNodes[tid];
  int root = d_par[v];
  while (d_par[root] != root) root = d_par[root];
  d_par[v] = root;
}

/// Flag the inserted vertices that are component roots (markRootsKernel).
__global__ void mark_roots_kernel(const std::int32_t* d_par, const std::int32_t* d_insertedNodes,
                                  int numInserted, std::int32_t* d_rootFlag) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int v = d_insertedNodes[tid];
  d_rootFlag[tid] = (d_par[v] == v) ? 1 : 0;
}

/// Give every root its compact ID (its rank among the roots) (numberRootsKernel).
__global__ void number_roots_kernel(const std::int32_t* d_insertedNodes, int numInserted,
                                    const std::int32_t* d_rootFlag, const std::int32_t* d_rootScan,
                                    std::int32_t* d_rootId, std::int32_t* d_numComponents) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  if (d_rootFlag[tid]) d_rootId[d_insertedNodes[tid]] = d_rootScan[tid];
  if (tid == numInserted - 1) {
    *d_numComponents = d_rootScan[tid] + d_rootFlag[tid];
  }
}

/// Map each vertex to the compact ID of its root (assignComponentIdsKernel).
__global__ void assign_component_ids_kernel(const std::int32_t* d_par, const std::int32_t* d_rootId,
                                            std::int32_t* d_componentId,
                                            const std::int32_t* d_insertedNodes, int numInserted) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int v = d_insertedNodes[tid];
  d_componentId[v] = d_rootId[d_par[v]];
}

}  // namespace

// ============================================================================
// Workspace (createCCWorkspace: the scan's temporary storage)
// ============================================================================

std::size_t components_scan_bytes(int max_inserted) {
  std::size_t bytes = 0;
  const int n = max_inserted > 0 ? max_inserted : 1;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(nullptr, bytes,
                                              static_cast<const std::int32_t*>(nullptr),
                                              static_cast<std::int32_t*>(nullptr), n));
  return bytes;
}

// ============================================================================
// Host Launch Function
// ============================================================================

int launch_connected_components(const std::int32_t* d_edgeSrc, const std::int32_t* d_edgeDst,
                                const std::int32_t* d_numEdges, int maxEdges,
                                const std::int32_t* d_insertedNodes, int numInserted,
                                std::int32_t* d_componentId, int& numComponents, workspace& ws,
                                void* s) {
  const cudaStream_t stream = as_stream(s);
  numComponents = 0;
  if (numInserted == 0) return 0;
  const int threadsPerBlock = 256;
  const int numBlocksV = (numInserted + threadsPerBlock - 1) / threadsPerBlock;
  const int numBlocksE = maxEdges > 0 ? (maxEdges + threadsPerBlock - 1) / threadsPerBlock : 0;
  std::int32_t* d_par = ws.par.data();
  std::int32_t* d_changed = ws.words.data() + 0;
  std::int32_t* d_numComponents = ws.words.data() + 1;
  std::int32_t* h_result = ws.host_words.data();

  // --- par[v] = v for the vertices of G' ---
  init_parent_kernel<<<numBlocksV, threadsPerBlock, 0, stream>>>(d_par, d_insertedNodes,
                                                                 numInserted);
  DYNG_CHECK_KERNEL(stream);

  // --- Alternating Hook + Compress until a Hook changes nothing ---
  // SV needs O(log n) rounds; the cap only guards against a malfunction.
  const int maxRounds = 64;
  int rounds = 0;
  while (numBlocksE > 0 && rounds < maxRounds) {
    ++rounds;
    DYNG_CUDA_TRY(cudaMemsetAsync(d_changed, 0, sizeof(int), stream));
    hook_kernel<<<numBlocksE, threadsPerBlock, 0, stream>>>(d_edgeSrc, d_edgeDst, d_numEdges, d_par,
                                                            d_changed);
    DYNG_CHECK_KERNEL(stream);
    compress_kernel<<<numBlocksV, threadsPerBlock, 0, stream>>>(d_par, d_insertedNodes,
                                                                numInserted);
    DYNG_CHECK_KERNEL(stream);
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(&h_result[0], d_changed, sizeof(int), cudaMemcpyDeviceToHost, stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
    note_host_sync();
    if (h_result[0] == 0) break;  // Converged
  }
  if (rounds == maxRounds) {
    DYNG_FAIL("label_propagation: the connected components of G' did not converge in ", maxRounds,
              " Shiloach-Vishkin rounds (a malfunction guard of the original)");
  }

  // --- Compact component IDs on the device: flag roots, scan, gather ---
  mark_roots_kernel<<<numBlocksV, threadsPerBlock, 0, stream>>>(d_par, d_insertedNodes, numInserted,
                                                                ws.root_flag.data());
  DYNG_CHECK_KERNEL(stream);
  std::size_t tempBytes = ws.scan_temp_bytes;
  DYNG_CUDA_TRY(cub::DeviceScan::ExclusiveSum(ws.scan_temp.data(), tempBytes, ws.root_flag.data(),
                                              ws.root_scan.data(), numInserted, stream));
  number_roots_kernel<<<numBlocksV, threadsPerBlock, 0, stream>>>(
      d_insertedNodes, numInserted, ws.root_flag.data(), ws.root_scan.data(), ws.root_id.data(),
      d_numComponents);
  DYNG_CHECK_KERNEL(stream);
  assign_component_ids_kernel<<<numBlocksV, threadsPerBlock, 0, stream>>>(
      d_par, ws.root_id.data(), d_componentId, d_insertedNodes, numInserted);
  DYNG_CHECK_KERNEL(stream);
  // The component count and (dynG) G''s edge count, in the one read of the original.
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(&h_result[1], d_numComponents, sizeof(int), cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(&h_result[2], d_numEdges, sizeof(int), cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
  note_host_sync();
  numComponents = h_result[1];
  return rounds;
}

DYNG_REGISTER_KERNEL(init_parent_kernel);
DYNG_REGISTER_KERNEL(hook_kernel);
DYNG_REGISTER_KERNEL(compress_kernel);
DYNG_REGISTER_KERNEL(mark_roots_kernel);
DYNG_REGISTER_KERNEL(number_roots_kernel);
DYNG_REGISTER_KERNEL(assign_component_ids_kernel);

}  // namespace dyng::detail::lp_cuda
