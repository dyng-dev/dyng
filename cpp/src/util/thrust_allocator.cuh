// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file thrust_allocator.cuh
 * @brief Thrust on a resources handle (PLAN 4.6 rule 7): the execution policy
 *        `thrust::cuda::par_nosync(allocator).on(stream)`, whose temporary storage comes from the
 *        memory resource of the resources, ordered on their stream.
 *
 *     thrust_allocator alloc(res);
 *     thrust::inclusive_scan(thrust_policy(res, alloc), first, last, out);
 *
 * par_nosync does not synchronize after an algorithm; an algorithm that returns a value to the
 * host (count_if, reduce, the end iterator of copy_if) still waits for it, which the caller counts
 * as a host synchronization. Private, for nvcc translation units.
 */
#pragma once

#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <cuda_runtime_api.h>
#include <thrust/execution_policy.h>

#include <cstddef>

namespace dyng::detail {

/**
 * @brief A Thrust allocator (value_type char) on a memory resource and a stream.
 */
class thrust_allocator {
 public:
  using value_type = char;  ///< Thrust allocates bytes

  /**
   * @brief Allocate from the memory resource of `res`, ordered on its stream.
   * @param[in] res The resources (must outlive the allocator).
   */
  explicit thrust_allocator(const resources& res) noexcept
      : resource_(res.memory()), stream_(res.stream()) {}

  /**
   * @brief Allocate temporary storage.
   * @param[in] bytes Size in bytes.
   * @return The storage.
   */
  char* allocate(std::ptrdiff_t bytes) {
    return static_cast<char*>(
        resource_.allocate(stream_, static_cast<std::size_t>(bytes), alignment));
  }

  /**
   * @brief Release temporary storage.
   * @param[in] ptr   Storage from allocate().
   * @param[in] bytes Its size.
   */
  void deallocate(char* ptr, std::size_t bytes) noexcept {
    resource_.deallocate(stream_, ptr, bytes, alignment);
  }

 private:
  static constexpr std::size_t alignment = 256;
  memory_resource_ref resource_;
  stream_ref stream_;
};

/**
 * @brief The Thrust execution policy of a resources handle.
 * @param[in] res   The resources.
 * @param[in] alloc An allocator of `res` (must outlive the algorithm call).
 * @return `thrust::cuda::par_nosync(alloc).on(stream)`.
 */
inline auto thrust_policy(const resources& res, thrust_allocator& alloc) {
  return thrust::cuda::par_nosync(alloc).on(static_cast<cudaStream_t>(res.stream().get()));
}

}  // namespace dyng::detail
