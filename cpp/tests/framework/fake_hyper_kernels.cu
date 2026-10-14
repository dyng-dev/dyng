// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fake_hyper_kernels.cu
 * @brief The CUDA counts of the hypergraph fakes (fake_hyper_problems.hpp): the owned pairs of
 *        the frontier's rows of h2h and the squared lengths of its rows of h2v, read from the
 *        internal row views in device memory. One host synchronization per count (the read-back).
 */
#include "core/budget_counters.hpp"
#include "framework/fake_hyper_counts.hpp"
#include "util/cuda_check.hpp"

#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::test::framework_fakes {

namespace {

cudaStream_t native(const resources& res) {
  return static_cast<cudaStream_t>(res.stream().get());
}

constexpr int block = 256;

unsigned int grid_for(std::int64_t n) {
  return static_cast<unsigned int>(std::max<std::int64_t>(1, (n + block - 1) / block));
}

__global__ void mark_kernel(const std::int64_t* ids, std::int64_t n, int max_key,
                            std::uint8_t* flags) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < n) {
    const std::int64_t key = ids[i] + 1;
    if (key >= 1 && key <= max_key) flags[key] = 1;
  }
}

// One thread per frontier row: the entries it owns (min_member).
__global__ void owned_pairs_kernel(detail::escher::row_view h2h, const std::int64_t* ids,
                                   std::int64_t n, const std::uint8_t* flags,
                                   unsigned long long* counter) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const std::int64_t key = ids[i] + 1;
  if (key < 1 || key > h2h.max_key) return;
  const int* row = h2h.flat + h2h.loc[key];
  unsigned long long owned = 0;
  for (int j = 0; j < h2h.len[key]; ++j) {
    const int neighbour = row[j];
    if (flags[neighbour] == 0 || key < neighbour) ++owned;
  }
  atomicAdd(counter, owned);
}

__global__ void squares_kernel(detail::escher::row_view h2v, const std::int64_t* ids,
                               std::int64_t n, unsigned long long* counter) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i >= n) return;
  const std::int64_t key = ids[i] + 1;
  if (key < 1 || key > h2v.max_key) return;
  const auto len = static_cast<unsigned long long>(h2v.len[key]);
  atomicAdd(counter, len * len);
}

/// The frontier in device memory (uploaded into the workspace when it is on the host).
const std::int64_t* device_ids(const resources& res, const id_frontier& f,
                               fake_hyper_workspace& ws) {
  if (f.on_device) return f.ids;
  std::int64_t* out = ws.ids.reserve(res, static_cast<std::size_t>(f.size));
  DYNG_CUDA_TRY(cudaMemcpyAsync(out, f.ids, sizeof(std::int64_t) * static_cast<std::size_t>(f.size),
                                cudaMemcpyHostToDevice, native(res)));
  return out;
}

/// Read the counter back (the count's one host synchronization).
std::int64_t read_counter(const resources& res, const unsigned long long* counter) {
  unsigned long long value = 0;
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(&value, counter, sizeof(value), cudaMemcpyDeviceToHost, native(res)));
  DYNG_CUDA_TRY(cudaStreamSynchronize(native(res)));
  detail::note_host_sync();  // the fake's budget (I9) counts it
  return static_cast<std::int64_t>(value);
}

}  // namespace

std::int64_t device_owned_pairs(const resources& res, const detail::escher::row_view& h2h,
                                const id_frontier& f, fake_hyper_workspace& ws) {
  const cudaStream_t stream = native(res);
  const std::int64_t* ids = device_ids(res, f, ws);
  const auto keys = static_cast<std::size_t>(h2h.max_key) + 1;
  std::uint8_t* flags = ws.flags.reserve(res, keys);
  unsigned long long* counter = ws.counter.reserve(res, 1);
  DYNG_CUDA_TRY(cudaMemsetAsync(flags, 0, keys, stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(counter, 0, sizeof(unsigned long long), stream));
  mark_kernel<<<grid_for(f.size), block, 0, stream>>>(ids, f.size, h2h.max_key, flags);
  DYNG_CUDA_TRY(cudaGetLastError());
  owned_pairs_kernel<<<grid_for(f.size), block, 0, stream>>>(h2h, ids, f.size, flags, counter);
  DYNG_CUDA_TRY(cudaGetLastError());
  return read_counter(res, counter);
}

std::int64_t device_squares(const resources& res, const detail::escher::row_view& h2v,
                            const id_frontier& f, fake_hyper_workspace& ws) {
  const cudaStream_t stream = native(res);
  const std::int64_t* ids = device_ids(res, f, ws);
  unsigned long long* counter = ws.counter.reserve(res, 1);
  DYNG_CUDA_TRY(cudaMemsetAsync(counter, 0, sizeof(unsigned long long), stream));
  squares_kernel<<<grid_for(f.size), block, 0, stream>>>(h2v, ids, f.size, counter);
  DYNG_CUDA_TRY(cudaGetLastError());
  return read_counter(res, counter);
}

}  // namespace dyng::test::framework_fakes
