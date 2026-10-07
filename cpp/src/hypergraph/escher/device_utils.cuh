// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/device_utils.cuh
/**
 * @file device_utils.cuh
 * @brief Device and host helpers of the CBST kernels: padded sizes and the 64-bit in-order rank of
 *        a heap position (the CBST layout).
 *
 * Ported from ESCHER-GPU@abcf9b2 with the renames of
 * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in
 * cpp/src/hypergraph/escher/README.md).
 */
#pragma once

namespace dyng::detail::escher {

// Common device/host helpers used by kernels

static inline __host__ __device__ int next_multiple_of_32(int num) {
  return ((num + 32) / 32) * 32;
}

static inline __host__ __device__ int next_multiple_of_4(int num) {
  if (num == 0) return 0;
  return ((num + 4) / 4) * 4;
}

static inline __device__ int ceil_log2(int x) {
  int log = 0;
  while ((1 << log) < x) ++log;
  return log;
}

static inline __device__ int floor_log2(int x) {
  int log = 0;
  while (x >>= 1) ++log;
  return log;
}

// In-order rank (0-based index into the sorted keys) of the node stored at
// heap position `pos` of a complete binary search tree with n nodes
// (the ESCHER CBST layout). (2*(pos+1-2^k)+1) * 2^log2_n / 2^k is evaluated as a
// 64-bit shift: the original int32 product overflowed for n > 65,535, which
// broke the BST order and made most keys unreachable.
static inline __device__ int cbst_rank_of_position(int pos, int n) {
  int k = floor_log2(pos + 1);
  int log2_n = floor_log2(n);
  long long index = (2LL * (pos + 1 - (1LL << k)) + 1) << (log2_n - k);
  long long index2 = min(index, index - (index / 2) + (n + 1 - (1LL << log2_n)));
  return static_cast<int>(index2 - 1);
}

}  // namespace dyng::detail::escher
