// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/labelInitialization.cu
// (computeSupernodeWeightsKernel, initializeLabelsKernel, launchLabelInitialization)
/**
 * @file label_init.cu
 * @brief Step 2 of DynLP on CUDA, the label initialization (fused.hpp; ADR 0075).
 *
 * Verbatim from the original (its file comment: one thread per new vertex walks its row and adds
 * the weights to ground-truth neighbours into its component's accumulators, in 64-bit fixed point
 * weight * 2^32 so that the sums do not depend on the order of the atomics; then
 * F_u = W_L1 / (W_L0 + W_L1), 0.5 when the denominator is at most 1e-10, a seed its class).
 * Mechanical changes: names, the rows templated on the offset type, the stream of the resources,
 * the accumulators from the pooled workspace, CUDA errors thrown, and the workspace overflow (more
 * components than accumulators, impossible after reserve) an internal_error instead of exit().
 */
#include "algorithms/label_propagation/fused.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/error.hpp>

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dyng::detail::lp_cuda {

namespace {

cudaStream_t as_stream(void* s) noexcept {
  return static_cast<cudaStream_t>(s);
}

// Fixed-point scale of the supernode weight accumulators (see Phase 1). Edge weights are in
// [0, 1], so a component sum stays below 2^64 for any graph with fewer than 2^32 edges.
const double WEIGHT_SCALE = 4294967296.0;  // 2^32

// ============================================================================
// CUDA Kernels
// ============================================================================

/**
 * @brief Compute aggregate edge weights from each component to L0 and L1
 *        (computeSupernodeWeightsKernel).
 *
 * d_compWeightL0[compId] = sum_{u in comp, v in L0} w(u, v)
 * d_compWeightL1[compId] = sum_{u in comp, v in L1} w(u, v)
 */
template <typename edge_t>
__global__ void compute_supernode_weights_kernel(
    const edge_t* d_rowStart, const edge_t* d_rowEnd, const std::int32_t* d_colInd,
    const float* d_values, const std::int32_t* /*d_vertexStatus*/,
    const std::int32_t* d_componentId, const std::int32_t* d_gtClass,
    const std::int32_t* d_insertedNodes, int numInserted, unsigned long long* d_compWeightL0,
    unsigned long long* d_compWeightL1) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int u = d_insertedNodes[tid];
  int compId = d_componentId[u];
  if (compId < 0) return;  // Vertex not assigned to a component
  // Walk the adjacency list of u in the main graph
  edge_t rowStart = d_rowStart[u];
  edge_t rowEnd = d_rowEnd[u];
  for (edge_t e = rowStart; e < rowEnd; ++e) {
    int v = d_colInd[e];
    if (v < 0) continue;  // Soft-deleted edge
    // Check if neighbor v is a ground-truth vertex
    int gtClass = d_gtClass[v];
    if (gtClass < 0) continue;
    unsigned long long w = __double2ull_rn(static_cast<double>(d_values[e]) * WEIGHT_SCALE);
    if (gtClass == 0) {
      atomicAdd(&d_compWeightL0[compId], w);
    } else {
      atomicAdd(&d_compWeightL1[compId], w);
    }
  }
}

/**
 * @brief Initialize labels for new vertices based on component weights (initializeLabelsKernel).
 *
 * F_u = W_L1 / (W_L0 + W_L1) if denominator > 0, else 0.5.
 */
__global__ void initialize_labels_kernel(float* d_labels, const std::int32_t* d_componentId,
                                         const std::int32_t* d_gtClass,
                                         const unsigned long long* d_compWeightL0,
                                         const unsigned long long* d_compWeightL1,
                                         const std::int32_t* d_insertedNodes, int numInserted) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  if (tid >= numInserted) return;
  int u = d_insertedNodes[tid];
  // Ground-truth vertices get fixed labels
  int gtClass = d_gtClass[u];
  if (gtClass == 0) {
    d_labels[u] = 0.0f;
    return;
  }
  if (gtClass == 1) {
    d_labels[u] = 1.0f;
    return;
  }
  // Non-GT vertex: use component-based initialization
  int compId = d_componentId[u];
  if (compId < 0) {
    // Not assigned to a component (shouldn't happen, but handle gracefully)
    d_labels[u] = 0.5f;
    return;
  }
  double wL0 = static_cast<double>(d_compWeightL0[compId]) / WEIGHT_SCALE;
  double wL1 = static_cast<double>(d_compWeightL1[compId]) / WEIGHT_SCALE;
  double denominator = wL0 + wL1;
  if (denominator > 1e-10) {
    // Simplified formula: F_u = W_L1 / (W_L0 + W_L1)
    d_labels[u] = static_cast<float>(wL1 / denominator);
  } else {
    // No edges to any GT vertex -> neutral initialization
    d_labels[u] = 0.5f;
  }
}

}  // namespace

// ============================================================================
// Host Launch Function
// ============================================================================

template <typename edge_t>
void launch_label_initialization(const rows<edge_t>& g, float* d_labels,
                                 const std::int32_t* d_componentId, int numComponents,
                                 const std::int32_t* d_insertedNodes, int numInserted,
                                 const std::int32_t* d_gtClass, workspace& ws, void* s) {
  const cudaStream_t stream = as_stream(s);
  if (numInserted == 0) return;
  if (static_cast<std::size_t>(numComponents) > ws.weight_l0.capacity()) {
    DYNG_FAIL("label_propagation: ", numComponents, " components exceed the label-init workspace (",
              ws.weight_l0.capacity(), ")");
  }
  int threadsPerBlock = 256;
  int numBlocks = (numInserted + threadsPerBlock - 1) / threadsPerBlock;
  // --- Reset the component weight accumulators ---
  unsigned long long* d_compWeightL0 = ws.weight_l0.data();
  unsigned long long* d_compWeightL1 = ws.weight_l1.data();
  std::size_t bytes = static_cast<std::size_t>(numComponents) * sizeof(unsigned long long);
  DYNG_CUDA_TRY(cudaMemsetAsync(d_compWeightL0, 0, bytes, stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(d_compWeightL1, 0, bytes, stream));
  // --- Phase 1: Compute supernode weights ---
  compute_supernode_weights_kernel<edge_t><<<numBlocks, threadsPerBlock, 0, stream>>>(
      g.row_start, g.row_end, g.col_ind, g.values, g.status, d_componentId, d_gtClass,
      d_insertedNodes, numInserted, d_compWeightL0, d_compWeightL1);
  DYNG_CHECK_KERNEL(stream);
  // --- Phase 2: Initialize labels ---
  initialize_labels_kernel<<<numBlocks, threadsPerBlock, 0, stream>>>(
      d_labels, d_componentId, d_gtClass, d_compWeightL0, d_compWeightL1, d_insertedNodes,
      numInserted);
  DYNG_CHECK_KERNEL(stream);
}

template void launch_label_initialization<std::int32_t>(const rows<std::int32_t>&, float*,
                                                        const std::int32_t*, int,
                                                        const std::int32_t*, int,
                                                        const std::int32_t*, workspace&, void*);
template void launch_label_initialization<std::int64_t>(const rows<std::int64_t>&, float*,
                                                        const std::int32_t*, int,
                                                        const std::int32_t*, int,
                                                        const std::int32_t*, workspace&, void*);

DYNG_REGISTER_KERNEL(compute_supernode_weights_kernel<std::int32_t>);
DYNG_REGISTER_KERNEL(compute_supernode_weights_kernel<std::int64_t>);
DYNG_REGISTER_KERNEL(initialize_labels_kernel);

}  // namespace dyng::detail::lp_cuda
