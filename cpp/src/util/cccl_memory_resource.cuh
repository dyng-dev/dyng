// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cccl_memory_resource.cuh
 * @brief Adapters between dynG's memory_resource_ref and the CCCL 3.x memory-resource concept
 *        (cuda::mr::resource of the CCCL 3.1 in CUDA 13.1; PLAN Section 4.7.2, ADR 0003).
 *
 * The two interfaces have the same member names and argument order (allocate(stream, bytes,
 * alignment), deallocate(stream, ptr, bytes, alignment), allocate_sync, deallocate_sync); only the
 * stream type differs (cuda::stream_ref vs dyng::stream_ref, both a cudaStream_t underneath), so
 * each adapter is one line per member:
 *
 * - cccl_resource_adapter: a dynG resource where CCCL expects a cuda::mr::resource (for example
 *   a CCCL algorithm's temporary storage); it satisfies
 *   cuda::mr::resource_with<cccl_resource_adapter, cuda::mr::device_accessible>.
 * - from_cccl_resource<R>: a CCCL resource (for example RMM's, once RMM follows CCCL 3.x) behind
 *   a dyng::memory_resource_ref, e.g. for resources::set_memory_resource().
 *
 * Private and for nvcc translation units (the CCCL headers are on nvcc's include path).
 * cuda::mr::resource_ref is still experimental in CCCL 3.1 (LIBCUDACXX_ENABLE_EXPERIMENTAL_-
 * MEMORY_RESOURCE); the concepts used here are not. The CCCL 2.x of CUDA 12 has the older shape
 * (allocate_async(bytes, alignment, stream)); there the adapters are not defined and
 * DYNG_HAS_CCCL3_MEMORY_RESOURCE is 0.
 */
#pragma once

#include <cuda/version>

#if CCCL_MAJOR_VERSION >= 3

/// 1 when the CCCL 3.x memory-resource concept (and the adapters below) is available.
#define DYNG_HAS_CCCL3_MEMORY_RESOURCE 1

#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <cuda/memory_resource>
#include <cuda/stream_ref>
#include <cuda_runtime_api.h>

#include <cstddef>

namespace dyng::detail {

/**
 * @brief A dynG memory resource seen through the CCCL 3.x resource concept.
 */
class cccl_resource_adapter {
 public:
  /**
   * @brief Wrap a resource (device-accessible memory: device, managed or pinned host).
   * @param[in] resource The resource; must outlive the adapter.
   */
  explicit cccl_resource_adapter(memory_resource_ref resource) noexcept : resource_(resource) {}

  /**
   * @brief Allocate, ordered on a stream.
   * @param[in] stream    The CCCL stream reference.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes.
   * @return Pointer to the memory.
   */
  void* allocate(::cuda::stream_ref stream, std::size_t bytes, std::size_t alignment) {
    return resource_.allocate(stream_ref(stream.get()), bytes, alignment);
  }

  /**
   * @brief Release, ordered on a stream.
   * @param[in] stream    The CCCL stream reference.
   * @param[in] ptr       Pointer from allocate().
   * @param[in] bytes     Size passed to allocate().
   * @param[in] alignment Alignment passed to allocate().
   */
  void deallocate(::cuda::stream_ref stream, void* ptr, std::size_t bytes,
                  std::size_t alignment) noexcept {
    resource_.deallocate(stream_ref(stream.get()), ptr, bytes, alignment);
  }

  /**
   * @brief Allocate memory usable as soon as the call returns.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes.
   * @return Pointer to the memory.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return resource_.allocate_sync(bytes, alignment);
  }

  /**
   * @brief Release memory from allocate_sync().
   * @param[in] ptr       Pointer from allocate_sync().
   * @param[in] bytes     Size passed to allocate_sync().
   * @param[in] alignment Alignment passed to allocate_sync().
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    resource_.deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief Same underlying resource.
   * @param[in] lhs First adapter.
   * @param[in] rhs Second adapter.
   * @return True if both wrap the same resource object.
   */
  friend bool operator==(const cccl_resource_adapter& lhs,
                         const cccl_resource_adapter& rhs) noexcept {
    return lhs.resource_ == rhs.resource_;
  }

  /**
   * @brief Different underlying resources.
   * @param[in] lhs First adapter.
   * @param[in] rhs Second adapter.
   * @return True if the wrapped resource objects differ.
   */
  friend bool operator!=(const cccl_resource_adapter& lhs,
                         const cccl_resource_adapter& rhs) noexcept {
    return !(lhs == rhs);
  }

  /// The memory is device-accessible (CCCL property).
  friend void get_property(const cccl_resource_adapter& /*resource*/,
                           ::cuda::mr::device_accessible) noexcept {}

 private:
  memory_resource_ref resource_;
};

/**
 * @brief A CCCL 3.x resource seen through dynG's memory_resource_ref interface.
 * @tparam cccl_resource_t A type satisfying cuda::mr::resource.
 */
template <typename cccl_resource_t>
class from_cccl_resource {
  static_assert(::cuda::mr::resource<cccl_resource_t>,
                "from_cccl_resource: the type must satisfy cuda::mr::resource");

 public:
  /**
   * @brief Wrap a CCCL resource.
   * @param[in] resource The resource; must outlive the adapter.
   * @param[in] space    The memory space of its allocations.
   */
  from_cccl_resource(cccl_resource_t& resource, memory_space space) noexcept
      : resource_(&resource), space_(space) {}

  /**
   * @brief Allocate, ordered on a stream.
   * @param[in] stream    The stream.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes.
   * @return Pointer to the memory.
   */
  void* allocate(stream_ref stream, std::size_t bytes, std::size_t alignment) {
    return resource_->allocate(::cuda::stream_ref(stream.get()), bytes, alignment);
  }

  /**
   * @brief Release, ordered on a stream.
   * @param[in] stream    The stream.
   * @param[in] ptr       Pointer from allocate().
   * @param[in] bytes     Size passed to allocate().
   * @param[in] alignment Alignment passed to allocate().
   */
  void deallocate(stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    resource_->deallocate(::cuda::stream_ref(stream.get()), ptr, bytes, alignment);
  }

  /**
   * @brief Allocate memory usable as soon as the call returns.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes.
   * @return Pointer to the memory.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return resource_->allocate_sync(bytes, alignment);
  }

  /**
   * @brief Release memory from allocate_sync().
   * @param[in] ptr       Pointer from allocate_sync().
   * @param[in] bytes     Size passed to allocate_sync().
   * @param[in] alignment Alignment passed to allocate_sync().
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    resource_->deallocate_sync(ptr, bytes, alignment);
  }

  /**
   * @brief The memory space given at construction.
   * @return The space.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return space_;
  }

 private:
  cccl_resource_t* resource_;
  memory_space space_;
};

}  // namespace dyng::detail

#else  // CCCL 2.x (CUDA 12)

/// 1 when the CCCL 3.x memory-resource concept (and the adapters of this header) is available.
#define DYNG_HAS_CCCL3_MEMORY_RESOURCE 0

#endif
