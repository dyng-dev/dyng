// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:tests/cbst_test.cu
/**
 * @file cbst_layout_test.cu
 * @brief ESCHER-GPU's cbst_test on the dynG store's construction kernels (M8a step 3; PLAN
 *        6.4.5): complete binary search trees of n = 1..1024 keys and of the 17 sizes up to 2^24
 *        (2,467,389, coauth-DBLP's hyperedge count, among them) are built with the store's
 *        build_empty_tree_kernel and store_items_kernel (launch_build_layout), and
 *         1. a BST search from the root finds every key,
 *         2. the node found for key k holds the payload offset of row k, and
 *         3. the inverse layout formula (extract_keys_from_positions_kernel) maps every node
 *            position back to the key stored there.
 * The test's own kernels and the launches run on the legacy default stream (handle 0), as in the
 * original test; the library code it calls takes the stream as an argument.
 */
#include "hypergraph/escher/kernels.cuh"

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include <string>
#include <vector>

namespace dyng::detail::escher {
namespace {

__global__ void search_all_keys_kernel(const cbst_node* nodes, int n, int* found,
                                       int* wrong_value) {
  const int k = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x) + 1;  // keys 1..n
  if (k > n) return;
  const cbst_node* c = nodes;
  while (c != nullptr && c->index != k) c = (c->index > k) ? c->left : c->right;
  if (c == nullptr) return;
  atomicAdd(found, 1);
  if (c->value != 4 * (k - 1)) atomicAdd(wrong_value, 1);
}

__global__ void compare_keys_kernel(const cbst_node* nodes, const int* recovered, int n,
                                    int* mismatches) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < n && nodes[t].index != recovered[t]) atomicAdd(mismatches, 1);
}

__global__ void iota_kernel(int* a, int n, int base, int step) {
  const int t = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
  if (t < n) a[t] = base + step * t;
}

/// Runs the three checks for n keys; returns "" when they pass, else a description.
std::string check_size(int n) {
  cbst_node* nodes = nullptr;
  int *keys = nullptr, *starts = nullptr, *positions = nullptr, *recovered = nullptr,
      *counters = nullptr;
  const auto bytes = [](int count, std::size_t size) {
    return static_cast<std::size_t>(count) * size;
  };
  EXPECT_EQ(cudaMalloc(&nodes, bytes(n, sizeof(cbst_node))), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&keys, bytes(n, sizeof(int))), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&starts, bytes(n, sizeof(int))), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&positions, bytes(n, sizeof(int))), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&recovered, bytes(n, sizeof(int))), cudaSuccess);
  EXPECT_EQ(cudaMalloc(&counters, 3 * sizeof(int)), cudaSuccess);
  EXPECT_EQ(cudaMemset(counters, 0, 3 * sizeof(int)), cudaSuccess);
  EXPECT_EQ(cudaMemset(nodes, 0, bytes(n, sizeof(cbst_node))), cudaSuccess);
  const int block = 256;
  const int grid = (n + block - 1) / block;
  iota_kernel<<<grid, block>>>(keys, n, 1, 1);       // sorted keys 1..n
  iota_kernel<<<grid, block>>>(starts, n, 0, 4);     // row k starts at 4(k-1)
  iota_kernel<<<grid, block>>>(positions, n, 0, 1);  // every node position
  launch_build_layout(nullptr, nodes, keys, starts, n, 4 * n);
  search_all_keys_kernel<<<grid, block>>>(nodes, n, counters, counters + 1);
  launch_extract_keys_from_positions(nullptr, keys, positions, recovered, n, n);
  compare_keys_kernel<<<grid, block>>>(nodes, recovered, n, counters + 2);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
  int h[3] = {-1, -1, -1};
  EXPECT_EQ(cudaMemcpy(h, counters, sizeof h, cudaMemcpyDeviceToHost), cudaSuccess);
  cudaFree(nodes);
  cudaFree(keys);
  cudaFree(starts);
  cudaFree(positions);
  cudaFree(recovered);
  cudaFree(counters);
  if (h[0] == n && h[1] == 0 && h[2] == 0) return "";
  return "n=" + std::to_string(n) + ": found " + std::to_string(h[0]) + ", wrong offsets " +
         std::to_string(h[1]) + ", inverse-layout mismatches " + std::to_string(h[2]);
}

bool have_device() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess) {
    (void)cudaGetLastError();
    return false;
  }
  return count > 0;
}

TEST(CbstLayout, EverySizeUpTo1024) {
  if (!have_device()) GTEST_SKIP() << "no CUDA device";
  for (int n = 1; n <= 1024; ++n) EXPECT_EQ(check_size(n), "");
}

TEST(CbstLayout, TheSeventeenLargeSizes) {
  if (!have_device()) GTEST_SKIP() << "no CUDA device";
  const std::vector<int> sizes = {4095,    4096,    4097,     32767,    32768,   65535,
                                  65536,   65537,   70000,    100000,   1000000, 2467389,
                                  8388607, 8388608, 16000000, 16777215, 16777216};
  ASSERT_EQ(sizes.size(), 17U);
  for (int n : sizes) EXPECT_EQ(check_size(n), "");
}

}  // namespace
}  // namespace dyng::detail::escher
