// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_check.hpp
 * @brief Checked CUDA runtime calls: DYNG_CUDA_TRY, DYNG_CUDA_TRY_NO_THROW, DYNG_CHECK_KERNEL
 *        (PLAN Section 4.7.3). Private; include only from files built with DYNG_HAS_CUDA.
 *
 * These replace the five CUDA check helpers of the originals (and their exit() / abort() calls):
 * library code never exits, it throws.
 *
 * - DYNG_CUDA_TRY(call): throws cuda_error (out_of_memory_error for cudaErrorMemoryAllocation)
 *   with the error name, its description, the failed call and file:line.
 * - DYNG_CUDA_TRY_NO_THROW(call): logs the same message at error level; for destructors and
 *   deallocation (noexcept paths). Errors caused by process teardown (cudaErrorCudartUnloading)
 *   are not logged: the runtime is gone and the memory with it.
 * - DYNG_CHECK_KERNEL(stream): after a kernel launch, checks the launch (cudaGetLastError); with
 *   DYNG_CUDA_DEBUG_SYNC (Debug builds and the sanitize-cuda preset) it also synchronizes the
 *   stream, so an asynchronous fault is reported at the kernel that caused it.
 *
 * The file is .hpp and not .cuh: it needs only the runtime API header, so host translation units
 * (compiled by the host compiler) use it as well as .cu files.
 */
#pragma once

#include <cuda_runtime_api.h>

namespace dyng::detail {

/**
 * @brief Throw the exception that describes a failed CUDA call.
 *
 * Clears the runtime's last-error state first (a non-sticky error would otherwise be reported
 * again by the next DYNG_CHECK_KERNEL).
 * @param[in] status The error (not cudaSuccess).
 * @param[in] call   The call's source text.
 * @param[in] file   Source file.
 * @param[in] line   Source line.
 * @throws out_of_memory_error for cudaErrorMemoryAllocation, cuda_error otherwise.
 */
[[noreturn]] void throw_cuda_error(cudaError_t status, const char* call, const char* file,
                                   int line);

/**
 * @brief Log a failed CUDA call at error level (noexcept paths).
 * @param[in] status The error (not cudaSuccess).
 * @param[in] call   The call's source text.
 * @param[in] file   Source file.
 * @param[in] line   Source line.
 */
void log_cuda_error(cudaError_t status, const char* call, const char* file, int line) noexcept;

}  // namespace dyng::detail

/// Run a CUDA runtime call; throw cuda_error (or out_of_memory_error) if it fails.
#define DYNG_CUDA_TRY(call)                                                           \
  do {                                                                                \
    const cudaError_t dyng_cuda_status_ = (call);                                     \
    if (dyng_cuda_status_ != cudaSuccess) {                                           \
      ::dyng::detail::throw_cuda_error(dyng_cuda_status_, #call, __FILE__, __LINE__); \
    }                                                                                 \
  } while (false)

/// Run a CUDA runtime call; log (never throw) if it fails. For destructors and noexcept paths.
#define DYNG_CUDA_TRY_NO_THROW(call)                                                \
  do {                                                                              \
    const cudaError_t dyng_cuda_status_ = (call);                                   \
    if (dyng_cuda_status_ != cudaSuccess) {                                         \
      ::dyng::detail::log_cuda_error(dyng_cuda_status_, #call, __FILE__, __LINE__); \
    }                                                                               \
  } while (false)

#if defined(DYNG_CUDA_DEBUG_SYNC) && DYNG_CUDA_DEBUG_SYNC
/// Check the last kernel launch; in debug-sync builds also wait for `stream` and check again.
#define DYNG_CHECK_KERNEL(stream)                   \
  do {                                              \
    DYNG_CUDA_TRY(cudaGetLastError());              \
    DYNG_CUDA_TRY(cudaStreamSynchronize((stream))); \
  } while (false)
#else
/// Check the last kernel launch; in debug-sync builds also wait for `stream` and check again.
#define DYNG_CHECK_KERNEL(stream)      \
  do {                                 \
    (void)(stream);                    \
    DYNG_CUDA_TRY(cudaGetLastError()); \
  } while (false)
#endif
