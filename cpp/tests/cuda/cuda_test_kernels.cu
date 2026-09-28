// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_test_kernels.cu
 * @brief Test kernels: device error flags, launch checks, and the CCCL adapter checks.
 */
#include "cuda/cuda_test_kernels.hpp"
#include "util/cccl_memory_resource.cuh"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::test {

namespace {

cudaStream_t native(stream_ref stream) {
  return static_cast<cudaStream_t>(stream.get());
}

__global__ void raise_kernel(std::uint32_t* word, std::uint32_t bits) {
  // Each thread raises the kinds separately, as independent violations would.
  for (std::uint32_t bit = 1; bit != 0; bit <<= 1U) {
    if ((bits & bit) != 0U) {
      detail::raise_device_error(word, static_cast<detail::device_error>(bit));
    }
  }
}

__global__ void empty_kernel() {}

__global__ void sequence_kernel(std::int64_t* data, std::int64_t count, std::int64_t value) {
  const std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < count) {
    data[i] = value + i;
  }
}

}  // namespace

void launch_raise_device_errors(stream_ref stream, std::uint32_t* word, std::uint32_t bits,
                                int threads) {
  raise_kernel<<<1, threads, 0, native(stream)>>>(word, bits);
  DYNG_CHECK_KERNEL(native(stream));
}

void launch_invalid_configuration(stream_ref stream) {
  empty_kernel<<<1, 2048, 0, native(stream)>>>();
  DYNG_CHECK_KERNEL(native(stream));
}

void launch_write_sequence(stream_ref stream, std::int64_t* data, std::int64_t count,
                           std::int64_t value) {
  if (count == 0) {
    return;
  }
  const auto blocks = static_cast<unsigned int>((count + 255) / 256);
  sequence_kernel<<<blocks, 256, 0, native(stream)>>>(data, count, value);
  DYNG_CHECK_KERNEL(native(stream));
}

#if DYNG_HAS_CCCL3_MEMORY_RESOURCE

/// A resource written against the CCCL concept only (cuda::stream_ref), as a user would. Not in an
/// unnamed namespace: its get_property() friend is found by the CCCL concept only (nvcc would
/// report an unreferenced internal function).
class cccl_only_resource {
 public:
  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t /*alignment*/) {
    void* ptr = nullptr;
    DYNG_CUDA_TRY(cudaMallocAsync(&ptr, bytes, stream.get()));
    return ptr;
  }
  void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t /*bytes*/,
                  std::size_t /*alignment*/) noexcept {
    (void)cudaFreeAsync(ptr, stream.get());
  }
  void* allocate_sync(std::size_t bytes, std::size_t /*alignment*/) {
    void* ptr = nullptr;
    DYNG_CUDA_TRY(cudaMalloc(&ptr, bytes));
    return ptr;
  }
  void deallocate_sync(void* ptr, std::size_t /*bytes*/, std::size_t /*alignment*/) noexcept {
    (void)cudaFree(ptr);
  }
  friend bool operator==(const cccl_only_resource&, const cccl_only_resource&) noexcept {
    return true;
  }
  friend bool operator!=(const cccl_only_resource&, const cccl_only_resource&) noexcept {
    return false;
  }
  friend void get_property(const cccl_only_resource&, ::cuda::mr::device_accessible) noexcept {}
};

static_assert(::cuda::mr::resource<cccl_only_resource>);
static_assert(::cuda::mr::resource_with<cccl_only_resource, ::cuda::mr::device_accessible>);
static_assert(
    ::cuda::mr::resource_with<detail::cccl_resource_adapter, ::cuda::mr::device_accessible>);

namespace {

__global__ void touch_kernel(unsigned char* data, std::size_t bytes) {
  for (std::size_t i = threadIdx.x; i < bytes; i += blockDim.x) {
    data[i] = static_cast<unsigned char>(i & 0xFFU);
  }
}

bool touch(void* ptr, std::size_t bytes, cudaStream_t stream) {
  touch_kernel<<<1, 128, 0, stream>>>(static_cast<unsigned char*>(ptr), bytes);
  if (cudaGetLastError() != cudaSuccess) {
    return false;
  }
  std::vector<unsigned char> host(bytes);
  if (cudaMemcpyAsync(host.data(), ptr, bytes, cudaMemcpyDeviceToHost, stream) != cudaSuccess ||
      cudaStreamSynchronize(stream) != cudaSuccess) {
    return false;
  }
  for (std::size_t i = 0; i < bytes; ++i) {
    if (host[i] != static_cast<unsigned char>(i & 0xFFU)) {
      return false;
    }
  }
  return true;
}

}  // namespace

bool cccl_adapters_round_trip(memory_resource_ref device_memory, stream_ref stream,
                              std::size_t bytes) {
  const cudaStream_t s = native(stream);
  // dynG -> CCCL: a dynG resource used where CCCL expects a cuda::mr::resource.
  detail::cccl_resource_adapter as_cccl(device_memory);
  void* a = as_cccl.allocate(::cuda::stream_ref(s), bytes, 256);
  const bool a_ok = touch(a, bytes, s);
  as_cccl.deallocate(::cuda::stream_ref(s), a, bytes, 256);

  // CCCL -> dynG: a CCCL resource behind a memory_resource_ref.
  cccl_only_resource user;
  detail::from_cccl_resource<cccl_only_resource> adapted(user, memory_space::device);
  memory_resource_ref as_dyng(adapted);
  void* b = as_dyng.allocate(stream, bytes, 256);
  const bool b_ok = touch(b, bytes, s);
  as_dyng.deallocate(stream, b, bytes, 256);
  return a_ok && b_ok && cudaStreamSynchronize(s) == cudaSuccess;
}

#else

bool cccl_adapters_round_trip(memory_resource_ref /*device_memory*/, stream_ref /*stream*/,
                              std::size_t /*bytes*/) {
  return false;
}

#endif

bool cccl3_memory_resource_available() {
  return DYNG_HAS_CCCL3_MEMORY_RESOURCE != 0;
}

}  // namespace dyng::test
