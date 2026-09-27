// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_runtime.hpp
 * @brief Private host-side helpers of the CUDA backend: device queries, the device guard, stream
 *        synchronization, warm-up and cross-space copies (PLAN Sections 4.6 and 4.7.1).
 *
 * Host-only (no CUDA headers), so every build compiles the callers; the definitions live in
 * cuda_runtime.cpp, whose body is compiled only with DYNG_HAS_CUDA. Without CUDA, the functions
 * that need a device throw not_supported_error and cuda_device_count() returns 0.
 */
#pragma once

#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <cstddef>
#include <string>

namespace dyng::detail {

/**
 * @brief What the CUDA backend records about its device once, when resources::cuda() is created
 *        (PLAN Section 4.6 rule 7: engine::automatic reads the cooperative-launch capability and
 *        the co-resident block count from here instead of querying on every call).
 */
struct cuda_device_properties {
  int ordinal = -1;                           ///< device ordinal
  std::string name;                           ///< marketing name (e.g. "NVIDIA RTX A5000")
  int major = 0;                              ///< compute capability, major
  int minor = 0;                              ///< compute capability, minor
  int multiprocessor_count = 0;               ///< SMs
  int max_threads_per_multiprocessor = 0;     ///< resident threads per SM
  int max_threads_per_block = 0;              ///< threads per block
  int max_shared_memory_per_block_optin = 0;  ///< opt-in shared memory per block (bytes)
  std::size_t total_global_memory = 0;        ///< bytes of device memory
  bool cooperative_launch = false;            ///< cudaLaunchCooperativeKernel is supported
  bool memory_pools = false;                  ///< stream-ordered allocation is supported
};

/**
 * @brief The number of CUDA devices this process can use.
 * @return 0 if CUDA is not built, no driver is loaded or no device is visible.
 */
[[nodiscard]] int cuda_device_count() noexcept;

/**
 * @brief Query a device's properties.
 * @param[in] device Device ordinal.
 * @return The properties.
 * @throws invalid_argument_error if `device` is out of range.
 * @throws not_supported_error    if CUDA is not built or no device is visible.
 * @throws cuda_error             if the runtime reports an error.
 */
[[nodiscard]] cuda_device_properties query_cuda_device(int device);

/**
 * @brief Make a device current for the lifetime of the guard and restore the previous one.
 *
 * Cheap when the device is already current (one cudaGetDevice). The library calls it at every
 * public entry that enqueues CUDA work, so a caller's current device never matters and is never
 * changed behind its back.
 */
class scoped_device {
 public:
  /**
   * @brief Make `device` current.
   * @param[in] device Device ordinal (>= 0).
   * @throws cuda_error if the device cannot be made current.
   */
  explicit scoped_device(int device);
  scoped_device(const scoped_device&) = delete;             ///< not copyable
  scoped_device& operator=(const scoped_device&) = delete;  ///< not copyable
  scoped_device(scoped_device&&) = delete;                  ///< not movable
  scoped_device& operator=(scoped_device&&) = delete;       ///< not movable
  /// @brief Restore the previously current device.
  ~scoped_device();

 private:
  [[maybe_unused]] int previous_ = -1;  // unused in builds without CUDA
};

/**
 * @brief Wait for a stream of a device.
 * @param[in] device Device ordinal (the per-thread default stream is per device), or -1 for the
 *                   current device.
 * @param[in] stream The stream.
 * @throws cuda_error if the runtime reports an error.
 */
void cuda_synchronize(int device, stream_ref stream);

/**
 * @brief Initialise a device ahead of timed work: create its primary context, load every
 *        registered library kernel (lazy module loading would otherwise load each one inside the
 *        first timed call; this replaces the originals' setenv("CUDA_MODULE_LOADING", "EAGER")),
 *        touch the stream and prime the memory resource with one small allocation.
 * @param[in] device Device ordinal.
 * @param[in] stream The stream of the resources.
 * @param[in] memory The memory resource of the resources.
 * @return The number of kernels loaded.
 * @throws cuda_error if the runtime reports an error.
 */
std::size_t cuda_warm_up(int device, stream_ref stream, memory_resource_ref memory);

/**
 * @brief Copy bytes between any two memory spaces, ordered on `stream` (cudaMemcpyAsync with
 *        cudaMemcpyDefault; unified addressing tells the runtime where each pointer lives).
 * @param[out] dst    Destination.
 * @param[in]  src    Source.
 * @param[in]  bytes  Byte count (> 0).
 * @param[in]  stream Stream the copy is ordered on.
 * @param[in]  device Device of the stream (made current for the call), or -1 for the current one.
 * @throws cuda_error if the runtime reports an error.
 */
void cuda_copy_bytes(void* dst, const void* src, std::size_t bytes, stream_ref stream, int device);

}  // namespace dyng::detail
