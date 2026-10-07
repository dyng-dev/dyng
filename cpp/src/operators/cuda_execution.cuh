// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_execution.cuh
 * @brief cuda_exec: the CUDA executor of the framework operators (operators/execution.hpp has the
 *        contract and the host executors). CUDA translation units only.
 *
 * for_each() launches one grid-stride kernel per call on the stream of the resources handle, so
 * the work of consecutive calls is ordered and a later call sees what an earlier one wrote. read()
 * is the only host synchronization: it copies one scalar to the host, waits for the stream and
 * reports the synchronization to the budget counters (invariant I9). Every kernel instantiation
 * registers itself for resources::warm_up() (util/kernel_registry.hpp).
 */
#pragma once

#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "operators/execution.hpp"
#include "util/cuda_check.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/resources.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

namespace dyng::detail::operators {

/// Threads per block of for_each_kernel.
inline constexpr int for_each_block_size = 256;
/// Most blocks of one launch; the grid-stride loop covers any n (a few waves on current GPUs).
inline constexpr std::int64_t for_each_max_blocks = 4096;

/**
 * @brief The kernel of cuda_exec::for_each(): `f(i)` for every i in [0, n), grid-stride.
 * @tparam function_t The functor (trivially copyable; `DYNG_HD void operator()(std::int64_t)`).
 * @param[in] n Number of elements.
 * @param[in] f The work on one element.
 */
template <typename function_t>
__global__ void __launch_bounds__(for_each_block_size)
    for_each_kernel(std::int64_t n, function_t f) {
  const std::int64_t stride =
      static_cast<std::int64_t>(gridDim.x) * static_cast<std::int64_t>(blockDim.x);
  for (std::int64_t i = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < n;
       i += stride) {
    f(i);
  }
}

/// Registers for_each_kernel<function_t> when the library is loaded (one per instantiation).
template <typename function_t>
inline const kernel_registrar for_each_registrar(
    reinterpret_cast<const void*>(&for_each_kernel<function_t>), "operators::for_each_kernel");

/**
 * @brief The functor of cuda_exec::fill() (at namespace scope: a kernel's template argument
 *        cannot be a private member type).
 * @tparam value_t Element type.
 */
template <typename value_t>
struct fill_one {
  value_t* p;     ///< the array
  value_t value;  ///< the value

  /// Set element i.
  DYNG_HD void operator()(std::int64_t i) const {
    p[i] = value;
  }
};

/**
 * @brief Runs the work on the GPU of a CUDA resources handle (see the file comment).
 *
 * Makes the handle's device current for its lifetime (not copyable: one per hook call).
 */
class cuda_exec {
 public:
  /// The memory the work reads and writes is device memory.
  static constexpr bool on_device = true;

  /**
   * @brief An executor on the device and stream of `res`.
   * @param[in] res CUDA resources.
   */
  explicit cuda_exec(const resources& res)
      : guard_(res.device()), stream_(static_cast<cudaStream_t>(res.stream().get())) {}

  cuda_exec(const cuda_exec&) = delete;             ///< not copyable
  cuda_exec& operator=(const cuda_exec&) = delete;  ///< not copyable

  /**
   * @brief Launch `f(i)` for every i in [0, n) (asynchronous, ordered on the stream).
   * @tparam function_t A trivially copyable functor with `DYNG_HD void operator()(std::int64_t)`.
   * @param[in] n Number of elements.
   * @param[in] f The work on one element.
   */
  template <typename function_t>
  void for_each(std::int64_t n, function_t f) const {
    (void)&for_each_registrar<function_t>;  // instantiates the registration
    if (n <= 0) {
      return;
    }
    const std::int64_t blocks =
        std::min(for_each_max_blocks, (n + for_each_block_size - 1) / for_each_block_size);
    for_each_kernel<function_t>
        <<<static_cast<unsigned int>(blocks), for_each_block_size, 0, stream_>>>(n, f);
    DYNG_CHECK_KERNEL(stream_);
  }

  /**
   * @brief Read a scalar the work wrote: copy it to the host and wait for the stream (one host
   *        synchronization, counted for invariant I9).
   * @tparam value_t Element type.
   * @param[in] p The scalar (device memory).
   * @return Its value.
   */
  template <typename value_t>
  value_t read(const value_t* p) const {
    value_t value{};
    DYNG_CUDA_TRY(cudaMemcpyAsync(&value, p, sizeof(value_t), cudaMemcpyDeviceToHost, stream_));
    DYNG_CUDA_TRY(cudaStreamSynchronize(stream_));
    note_host_sync();
    return value;
  }

  /**
   * @brief Write a scalar the work will read (asynchronous; `value` is copied before the call
   *        returns).
   * @tparam value_t Element type.
   * @param[out] p     The scalar (device memory).
   * @param[in]  value The value.
   */
  template <typename value_t>
  void write(value_t* p, value_t value) const {
    upload(p, &value, 1);
  }

  /**
   * @brief Set `n` elements to `value` (asynchronous).
   * @tparam value_t Element type.
   * @param[out] p     The array (device memory).
   * @param[in]  n     Number of elements.
   * @param[in]  value The value.
   */
  template <typename value_t>
  void fill(value_t* p, std::int64_t n, value_t value) const {
    for_each(n, fill_one<value_t>{p, value});
  }

  /**
   * @brief Copy `n` elements within device memory (asynchronous).
   * @tparam value_t Element type.
   * @param[out] dst The destination.
   * @param[in]  src The source.
   * @param[in]  n   Number of elements.
   */
  template <typename value_t>
  void copy(value_t* dst, const value_t* src, std::int64_t n) const {
    if (n > 0) {
      DYNG_CUDA_TRY(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n) * sizeof(value_t),
                                    cudaMemcpyDeviceToDevice, stream_));
    }
  }

  /**
   * @brief Copy `n` elements of host memory into device memory (asynchronous; pageable host
   *        memory is staged before the call returns, so `src` may be reused at once).
   * @tparam value_t Element type.
   * @param[out] dst The destination (device memory).
   * @param[in]  src The source (host memory).
   * @param[in]  n   Number of elements.
   */
  template <typename value_t>
  void upload(value_t* dst, const value_t* src, std::int64_t n) const {
    if (n > 0) {
      DYNG_CUDA_TRY(cudaMemcpyAsync(dst, src, static_cast<std::size_t>(n) * sizeof(value_t),
                                    cudaMemcpyHostToDevice, stream_));
    }
  }

 private:
  scoped_device guard_;
  cudaStream_t stream_ = nullptr;
};

}  // namespace dyng::detail::operators
