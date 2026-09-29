// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file memory_cuda.cpp
 * @brief The CUDA memory resources: cuda_async_memory_resource (a stream-ordered pool, the default
 *        of the CUDA backend) and pinned_host_memory_resource (PLAN Section 4.7.2). Without CUDA the
 *        resources exist but cannot be created or used (not_supported_error).
 */
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>
#include <vector>

#if DYNG_HAS_CUDA
#include "util/cuda_check.hpp"

#include <cuda_runtime_api.h>
#endif

namespace dyng {

namespace {

// The alignment every CUDA allocation (pool, cudaHostAlloc) guarantees.
constexpr std::size_t cuda_allocation_alignment = 256;

void expect_cuda_alignment(std::size_t alignment) {
  DYNG_EXPECTS(alignment != 0 && (alignment & (alignment - 1)) == 0 &&
                   alignment <= cuda_allocation_alignment,
               "alignment ", alignment, " is not a power of two of at most ",
               cuda_allocation_alignment, " bytes");
}

}  // namespace

#if DYNG_HAS_CUDA

namespace {

cudaStream_t native(stream_ref stream) noexcept {
  return static_cast<cudaStream_t>(stream.get());
}

cudaMemPool_t native_pool(void* pool) noexcept {
  return static_cast<cudaMemPool_t>(pool);
}

std::size_t pool_attribute(void* pool, cudaMemPoolAttr attribute, int device) {
  const detail::scoped_device guard(device);
  std::uint64_t value = 0;
  DYNG_CUDA_TRY(cudaMemPoolGetAttribute(native_pool(pool), attribute, &value));
  return static_cast<std::size_t>(value);
}

[[noreturn]] void throw_device_oom(std::size_t bytes, int device, cudaError_t status) {
  (void)cudaGetLastError();
  throw out_of_memory_error(detail::concat_message("dyng: device allocation of ", bytes,
                                                   " bytes on CUDA device ", device, " failed (",
                                                   cudaGetErrorString(status), ")"));
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// cuda_async_memory_resource
// ----------------------------------------------------------------------------------------------

cuda_async_memory_resource::cuda_async_memory_resource(int device) : device_(device) {
  const detail::cuda_device_properties props = detail::query_cuda_device(device);
  if (!props.memory_pools) {
    throw not_supported_error(detail::concat_message(
        "dyng: CUDA device ", device, " (", props.name,
        ") does not support stream-ordered allocation (cudaMallocAsync); pass another memory "
        "resource to resources::set_memory_resource()"));
  }
  const detail::scoped_device guard(device);
  cudaMemPoolProps pool_props{};
  pool_props.allocType = cudaMemAllocationTypePinned;
  pool_props.handleTypes = cudaMemHandleTypeNone;
  pool_props.location.type = cudaMemLocationTypeDevice;
  pool_props.location.id = device;
  cudaMemPool_t pool = nullptr;
  DYNG_CUDA_TRY(cudaMemPoolCreate(&pool, &pool_props));
  // Keep freed memory in the pool: a steady-state workload then never returns to the driver.
  std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
  const cudaError_t status =
      cudaMemPoolSetAttribute(pool, cudaMemPoolAttrReleaseThreshold, &threshold);
  if (status != cudaSuccess) {
    DYNG_CUDA_TRY_NO_THROW(cudaMemPoolDestroy(pool));
    detail::throw_cuda_error(status, "cudaMemPoolSetAttribute(cudaMemPoolAttrReleaseThreshold)",
                             __FILE__, __LINE__);
  }
  pool_ = pool;
}

cuda_async_memory_resource::~cuda_async_memory_resource() {
  if (pool_ != nullptr) {
    DYNG_CUDA_TRY_NO_THROW(cudaMemPoolDestroy(native_pool(pool_)));
  }
}

void* cuda_async_memory_resource::allocate(stream_ref stream, std::size_t bytes,
                                           std::size_t alignment) {
  expect_cuda_alignment(alignment);
  if (bytes == 0) {
    return nullptr;
  }
  const detail::scoped_device guard(device_);
  detail::note_allocation(bytes);  // invariant I9 (counts only with DYNG_DEBUG_BUDGETS)
  void* ptr = nullptr;
  const cudaError_t status =
      cudaMallocFromPoolAsync(&ptr, bytes, native_pool(pool_), native(stream));
  if (status == cudaErrorMemoryAllocation) {
    throw_device_oom(bytes, device_, status);
  }
  if (status != cudaSuccess) {
    detail::throw_cuda_error(status, "cudaMallocFromPoolAsync", __FILE__, __LINE__);
  }
  return ptr;
}

void cuda_async_memory_resource::deallocate(stream_ref stream, void* ptr, std::size_t /*bytes*/,
                                            std::size_t /*alignment*/) noexcept {
  if (ptr == nullptr) {
    return;
  }
  try {
    const detail::scoped_device guard(device_);
    DYNG_CUDA_TRY_NO_THROW(cudaFreeAsync(ptr, native(stream)));
  } catch (...) {
    // The device could not be made current (the error was reported by the guard's exception);
    // the memory returns to the driver with the pool.
  }
}

void* cuda_async_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment) {
  const stream_ref stream{};
  void* ptr = allocate(stream, bytes, alignment);
  if (ptr != nullptr) {
    try {
      detail::cuda_synchronize(device_, stream);
    } catch (...) {
      deallocate(stream, ptr, bytes, alignment);
      throw;
    }
  }
  return ptr;
}

void cuda_async_memory_resource::deallocate_sync(void* ptr, std::size_t bytes,
                                                 std::size_t alignment) noexcept {
  if (ptr == nullptr) {
    return;
  }
  const stream_ref stream{};
  deallocate(stream, ptr, bytes, alignment);
  try {
    detail::cuda_synchronize(device_, stream);
  } catch (...) {
    // A sticky error of the context; the next checked call reports it.
  }
}

std::size_t cuda_async_memory_resource::used_bytes() const {
  return pool_attribute(pool_, cudaMemPoolAttrUsedMemCurrent, device_);
}

std::size_t cuda_async_memory_resource::reserved_bytes() const {
  return pool_attribute(pool_, cudaMemPoolAttrReservedMemCurrent, device_);
}

cuda_async_memory_resource& default_device_memory_resource(int device) {
  // One resource per device, created on first use and deliberately never destroyed: buffers and
  // workspaces may outlive every resources handle, and destroying a pool during static
  // destruction would race the CUDA runtime's own teardown.
  static std::mutex mutex;
  static std::vector<cuda_async_memory_resource*>* table = nullptr;
  const int count = detail::cuda_device_count();
  if (count == 0) {
    throw not_supported_error(
        "dyng: the cuda backend is built but no CUDA device is visible (check the driver and "
        "CUDA_VISIBLE_DEVICES)");
  }
  DYNG_EXPECTS(device >= 0 && device < count, "CUDA device ", device, " does not exist; ", count,
               " device(s) visible");
  const std::lock_guard<std::mutex> lock(mutex);
  if (table == nullptr) {
    table =
        new std::vector<cuda_async_memory_resource*>(  // NOLINT(cppcoreguidelines-owning-memory)
            static_cast<std::size_t>(count), nullptr);
  }
  cuda_async_memory_resource*& slot = (*table)[static_cast<std::size_t>(device)];
  if (slot == nullptr) {
    slot = new cuda_async_memory_resource(device);  // NOLINT(cppcoreguidelines-owning-memory)
  }
  return *slot;
}

// ----------------------------------------------------------------------------------------------
// pinned_host_memory_resource
// ----------------------------------------------------------------------------------------------

void* pinned_host_memory_resource::allocate(stream_ref /*stream*/, std::size_t bytes,
                                            std::size_t alignment) {
  return allocate_sync(bytes, alignment);
}

void pinned_host_memory_resource::deallocate(stream_ref stream, void* ptr, std::size_t bytes,
                                             std::size_t alignment) noexcept {
  if (ptr == nullptr) {
    return;
  }
  // cudaFreeHost does not wait for asynchronous copies that still read or write the memory.
  detail::note_host_sync();  // invariant I9 (counts only with DYNG_DEBUG_BUDGETS)
  DYNG_CUDA_TRY_NO_THROW(cudaStreamSynchronize(native(stream)));
  deallocate_sync(ptr, bytes, alignment);
}

void* pinned_host_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment) {
  expect_cuda_alignment(alignment);
  if (bytes == 0) {
    return nullptr;
  }
  detail::note_allocation(bytes);  // invariant I9 (counts only with DYNG_DEBUG_BUDGETS)
  void* ptr = nullptr;
  // Portable: pinned for every device's context, not only the current one.
  const cudaError_t status = cudaHostAlloc(&ptr, bytes, cudaHostAllocPortable);
  if (status == cudaErrorMemoryAllocation) {
    (void)cudaGetLastError();
    throw out_of_memory_error(
        detail::concat_message("dyng: pinned host allocation of ", bytes, " bytes failed"));
  }
  if (status != cudaSuccess) {
    detail::throw_cuda_error(status, "cudaHostAlloc", __FILE__, __LINE__);
  }
  return ptr;
}

void pinned_host_memory_resource::deallocate_sync(void* ptr, std::size_t /*bytes*/,
                                                  std::size_t /*alignment*/) noexcept {
  if (ptr != nullptr) {
    DYNG_CUDA_TRY_NO_THROW(cudaFreeHost(ptr));
  }
}

#else  // !DYNG_HAS_CUDA

namespace {

[[noreturn]] void cuda_not_built(const char* what) {
  throw not_supported_error(
      detail::concat_message("dyng: ", what,
                             " needs the cuda backend, which is not built (configure with "
                             "DYNG_ENABLE_CUDA=ON)"));
}

}  // namespace

cuda_async_memory_resource::cuda_async_memory_resource(int device) : device_(device) {
  cuda_not_built("cuda_async_memory_resource");
}

cuda_async_memory_resource::~cuda_async_memory_resource() = default;

void* cuda_async_memory_resource::allocate(stream_ref /*stream*/, std::size_t /*bytes*/,
                                           std::size_t /*alignment*/) {
  cuda_not_built("cuda_async_memory_resource");
}

void cuda_async_memory_resource::deallocate(stream_ref /*stream*/, void* /*ptr*/,
                                            std::size_t /*bytes*/,
                                            std::size_t /*alignment*/) noexcept {}

void* cuda_async_memory_resource::allocate_sync(std::size_t /*bytes*/, std::size_t /*alignment*/) {
  cuda_not_built("cuda_async_memory_resource");
}

void cuda_async_memory_resource::deallocate_sync(void* /*ptr*/, std::size_t /*bytes*/,
                                                 std::size_t /*alignment*/) noexcept {}

std::size_t cuda_async_memory_resource::used_bytes() const {
  return 0;
}

std::size_t cuda_async_memory_resource::reserved_bytes() const {
  return 0;
}

cuda_async_memory_resource& default_device_memory_resource(int /*device*/) {
  cuda_not_built("default_device_memory_resource()");
}

void* pinned_host_memory_resource::allocate(stream_ref /*stream*/, std::size_t /*bytes*/,
                                            std::size_t alignment) {
  expect_cuda_alignment(alignment);
  cuda_not_built("pinned_host_memory_resource");
}

void pinned_host_memory_resource::deallocate(stream_ref /*stream*/, void* /*ptr*/,
                                             std::size_t /*bytes*/,
                                             std::size_t /*alignment*/) noexcept {}

void* pinned_host_memory_resource::allocate_sync(std::size_t /*bytes*/, std::size_t alignment) {
  expect_cuda_alignment(alignment);
  cuda_not_built("pinned_host_memory_resource");
}

void pinned_host_memory_resource::deallocate_sync(void* /*ptr*/, std::size_t /*bytes*/,
                                                  std::size_t /*alignment*/) noexcept {}

#endif  // DYNG_HAS_CUDA

pinned_host_memory_resource& default_pinned_host_memory_resource() noexcept {
  static pinned_host_memory_resource resource;
  return resource;
}

}  // namespace dyng
