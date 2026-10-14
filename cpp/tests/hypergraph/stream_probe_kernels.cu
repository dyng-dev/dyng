// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file stream_probe_kernels.cu
 * @brief A delayed reader for the stream-ordering test of the hypergraph container
 *        (hypergraph_cuda_test.cpp): a kernel that spins, then copies an int32 array, queued on
 *        a stream of the caller's choice (M8b review: apply() on another stream than the
 *        hypergraph's).
 */
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace dyng::test::hyper {

namespace {

__global__ void delayed_copy_kernel(const std::int32_t* from, std::int32_t* to, std::size_t n,
                                    long long spin_cycles) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    const long long start = clock64();
    while (clock64() - start < spin_cycles) {
    }
  }
  __syncthreads();
  for (std::size_t i = threadIdx.x; i < n; i += blockDim.x) to[i] = from[i];
}

}  // namespace

/// Queue on `stream`: spin for about `spin_cycles` clock cycles, then copy `n` values (one block).
cudaError_t queue_delayed_copy(const std::int32_t* from, std::int32_t* to, std::size_t n,
                               long long spin_cycles, cudaStream_t stream) {
  delayed_copy_kernel<<<1, 256, 0, stream>>>(from, to, n, spin_cycles);
  return cudaGetLastError();
}

}  // namespace dyng::test::hyper
