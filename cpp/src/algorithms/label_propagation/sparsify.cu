// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/sparsifySubgraph.cu (markDeletionsKernel,
// buildSubgraphKernel, markInsertedKernel, launchMarkInserted, launchMarkDeletions,
// launchBuildSparsifiedSubgraph) and src/dynamicGraph.cu (setGroundTruthKernel's d_gtClass)
/**
 * @file sparsify.cu
 * @brief Step 1 of DynLP on CUDA (fused.hpp; ADR 0075): the deletion marks on the rows before the
 *        batch, the inserted vertices, G' and the marks of the batch's edges.
 *
 * Verbatim from the original (its file comment: one thread per deleted vertex walks its row before
 * the batch; one thread per new edge builds G' with an atomic counter and marks both endpoints;
 * the writes of d_isAffected are idempotent). Mechanical changes: names, the rows templated on the
 * offset type, the stream of the resources, CUDA errors thrown. Two kernels are dynG's: the seeds'
 * classes (the d_gtClass half of the original's setGroundTruthKernel, which runs in its graph
 * apply; here the classes are the result's) and the marks of a batch without insertions (the
 * original skips its G' kernel and so the marks; the documented fix of ADR 0072, shared with the
 * sequential backend).
 */
#include "algorithms/label_propagation/fused.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <cuda_runtime.h>

#include <cstdint>

namespace dyng::detail::lp_cuda {

namespace {

constexpr std::int32_t vertex_deleted = 1;  // VERTEX_DELETED (vertex_status::deleted)

cudaStream_t as_stream(void* s) noexcept {
  return static_cast<cudaStream_t>(s);
}

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Mark the neighbors of deleted vertices as affected (markDeletionsKernel).
 *
 * One thread per deleted vertex. The rows must be those of the graph *before* the deletions
 * (G_t).
 */
template <typename edge_t>
__global__ void mark_deletions_kernel(const edge_t* d_rowStart, const edge_t* d_rowEnd,
                                      const std::int32_t* d_colInd, std::int32_t* d_isAffected,
                                      const std::int32_t* d_deletedNodes, int numDeleted,
                                      int numNodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numDeleted) return;
  int u = d_deletedNodes[tid];
  if (u < 0 || u >= numNodes) return;
  edge_t rowStart = d_rowStart[u];
  edge_t rowEnd = d_rowEnd[u];
  for (edge_t e = rowStart; e < rowEnd; ++e) {
    int v = d_colInd[e];
    if (v >= 0 && v < numNodes) d_isAffected[v] = 1;  // idempotent
  }
}

/**
 * @brief Build the new-vertex subgraph G' and mark affected vertices (buildSubgraphKernel).
 *
 * One thread per new edge. Both endpoints of every new edge are marked, whatever the weight; only
 * the strong new-to-new edges enter G'.
 */
__global__ void build_subgraph_kernel(const std::int32_t* d_newEdgeSrc,
                                      const std::int32_t* d_newEdgeDst,
                                      const float* d_newEdgeWeight, int numNewEdges,
                                      const std::int32_t* d_vertexStatus,
                                      const std::int32_t* d_newStamp, int batchStamp,
                                      std::int32_t* d_isAffected, float tau,
                                      std::int32_t* d_subEdgeSrc, std::int32_t* d_subEdgeDst,
                                      std::int32_t* d_subEdgeCount, int numNodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numNewEdges) return;
  int u = d_newEdgeSrc[tid];
  int v = d_newEdgeDst[tid];
  float w = d_newEdgeWeight[tid];
  if (u < 0 || u >= numNodes || v < 0 || v >= numNodes) return;
  if (d_vertexStatus[u] == vertex_deleted || d_vertexStatus[v] == vertex_deleted) {
    return;  // edge is not in the updated graph
  }
  // Sparsification: only strong new-to-new edges enter G'
  if (w > tau && d_newStamp[u] == batchStamp && d_newStamp[v] == batchStamp) {
    // Both new -> add to subgraph G'
    int idx = atomicAdd(d_subEdgeCount, 1);
    d_subEdgeSrc[idx] = u;
    d_subEdgeDst[idx] = v;
  }
  // Mark endpoints as affected (both new and existing neighbors)
  d_isAffected[u] = 1;
  d_isAffected[v] = 1;
}

/**
 * @brief Mark the inserted vertices as affected and as new in this batch (markInsertedKernel).
 */
__global__ void mark_inserted_kernel(const std::int32_t* d_vertices, int count,
                                     std::int32_t* d_isAffected, std::int32_t* d_newStamp,
                                     int batchStamp) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= count) return;
  int v = d_vertices[tid];
  d_isAffected[v] = 1;
  d_newStamp[v] = batchStamp;
}

/// The seeds' classes of the batch (setGroundTruthKernel: d_gtClass[v] = the class).
__global__ void set_seed_classes_kernel(const std::int32_t* d_ids, const std::int8_t* d_labels,
                                        int count, std::int32_t* d_gtClass) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= count) return;
  const int label = d_labels[i];
  if (label >= 0) d_gtClass[d_ids[i]] = label;
}

/// A batch without insertions: the endpoints of its edges in the updated graph are affected
/// (buildSubgraphKernel's marks without G').
__global__ void mark_edge_endpoints_kernel(const std::int32_t* d_newEdgeSrc,
                                           const std::int32_t* d_newEdgeDst, int numNewEdges,
                                           const std::int32_t* d_vertexStatus,
                                           std::int32_t* d_isAffected, int numNodes) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numNewEdges) return;
  int u = d_newEdgeSrc[tid];
  int v = d_newEdgeDst[tid];
  if (u < 0 || u >= numNodes || v < 0 || v >= numNodes) return;
  if (d_vertexStatus[u] == vertex_deleted || d_vertexStatus[v] == vertex_deleted) return;
  d_isAffected[u] = 1;
  d_isAffected[v] = 1;
}

}  // namespace

// ============================================================================
// Host Launch Functions
// ============================================================================

void launch_mark_inserted(const std::int32_t* d_insertedNodes, int numInserted,
                          std::int32_t* d_isAffected, std::int32_t* d_newStamp, int batchStamp,
                          void* stream) {
  if (numInserted == 0) return;
  int threadsPerBlock = 256;
  int numBlocks = (numInserted + threadsPerBlock - 1) / threadsPerBlock;
  mark_inserted_kernel<<<numBlocks, threadsPerBlock, 0, as_stream(stream)>>>(
      d_insertedNodes, numInserted, d_isAffected, d_newStamp, batchStamp);
  DYNG_CHECK_KERNEL(as_stream(stream));
}

void launch_set_seed_classes(const std::int32_t* inserted, const std::int8_t* labels,
                             int num_inserted, std::int32_t* gt_class, void* stream) {
  if (num_inserted == 0 || labels == nullptr) return;
  const int threadsPerBlock = 256;
  const int numBlocks = (num_inserted + threadsPerBlock - 1) / threadsPerBlock;
  set_seed_classes_kernel<<<numBlocks, threadsPerBlock, 0, as_stream(stream)>>>(
      inserted, labels, num_inserted, gt_class);
  DYNG_CHECK_KERNEL(as_stream(stream));
}

template <typename edge_t>
void launch_mark_deletions(const edge_t* d_rowStart, const edge_t* d_rowEnd,
                           const std::int32_t* d_colInd, std::int32_t* d_isAffected,
                           const std::int32_t* d_deletedNodes, int numDeleted, int numNodes,
                           void* stream) {
  if (numDeleted == 0) return;
  int threadsPerBlock = 256;
  int numBlocks = (numDeleted + threadsPerBlock - 1) / threadsPerBlock;
  mark_deletions_kernel<edge_t><<<numBlocks, threadsPerBlock, 0, as_stream(stream)>>>(
      d_rowStart, d_rowEnd, d_colInd, d_isAffected, d_deletedNodes, numDeleted, numNodes);
  DYNG_CHECK_KERNEL(as_stream(stream));
}

void launch_build_sparsified_subgraph(std::int32_t* d_isAffected, const std::int32_t* d_newEdgeSrc,
                                      const std::int32_t* d_newEdgeDst,
                                      const float* d_newEdgeWeight, int numNewEdges, int numNodes,
                                      const std::int32_t* d_vertexStatus,
                                      const std::int32_t* d_newStamp, int batchStamp, float tau,
                                      std::int32_t* d_subEdgeSrc, std::int32_t* d_subEdgeDst,
                                      std::int32_t* d_numSubEdges, void* stream) {
  DYNG_CUDA_TRY(cudaMemsetAsync(d_numSubEdges, 0, sizeof(int), as_stream(stream)));
  if (numNewEdges == 0) return;
  int threadsPerBlock = 256;
  int numBlocks = (numNewEdges + threadsPerBlock - 1) / threadsPerBlock;
  // --- Build G' (edge list) and mark affected ---
  build_subgraph_kernel<<<numBlocks, threadsPerBlock, 0, as_stream(stream)>>>(
      d_newEdgeSrc, d_newEdgeDst, d_newEdgeWeight, numNewEdges, d_vertexStatus, d_newStamp,
      batchStamp, d_isAffected, tau, d_subEdgeSrc, d_subEdgeDst, d_numSubEdges, numNodes);
  DYNG_CHECK_KERNEL(as_stream(stream));
}

void launch_mark_edge_endpoints(std::int32_t* affected, const std::int32_t* src,
                                const std::int32_t* dst, int num_edges, int num_nodes,
                                const std::int32_t* status, void* stream) {
  if (num_edges == 0) return;
  const int threadsPerBlock = 256;
  const int numBlocks = (num_edges + threadsPerBlock - 1) / threadsPerBlock;
  mark_edge_endpoints_kernel<<<numBlocks, threadsPerBlock, 0, as_stream(stream)>>>(
      src, dst, num_edges, status, affected, num_nodes);
  DYNG_CHECK_KERNEL(as_stream(stream));
}

template void launch_mark_deletions<std::int32_t>(const std::int32_t*, const std::int32_t*,
                                                  const std::int32_t*, std::int32_t*,
                                                  const std::int32_t*, int, int, void*);
template void launch_mark_deletions<std::int64_t>(const std::int64_t*, const std::int64_t*,
                                                  const std::int32_t*, std::int32_t*,
                                                  const std::int32_t*, int, int, void*);

DYNG_REGISTER_KERNEL(mark_deletions_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(mark_deletions_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(build_subgraph_kernel);
DYNG_REGISTER_KERNEL(mark_inserted_kernel);
DYNG_REGISTER_KERNEL(set_seed_classes_kernel);
DYNG_REGISTER_KERNEL(mark_edge_endpoints_kernel);

}  // namespace dyng::detail::lp_cuda
