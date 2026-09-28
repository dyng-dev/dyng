// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file stream.hpp
 * @brief stream_ref: a non-owning reference to a CUDA stream that needs no CUDA headers.
 * @ingroup core
 */
#pragma once

#include <cstdint>

/// Opaque CUDA stream type (the same forward declaration as in cuda_runtime_api.h).
struct CUstream_st;

namespace dyng {

/**
 * @brief The CUDA stream handle type (identical to cudaStream_t).
 * @ingroup core
 */
using cuda_stream_handle = CUstream_st*;

/**
 * @brief A non-owning reference to a CUDA stream.
 *
 * A default stream_ref is the explicit per-thread default stream (`cudaStreamPerThread`), never
 * the legacy stream (handle 0); pass handle 0 explicitly to use the legacy stream. The per-thread
 * default stream is a different stream on every host thread (and per device): the same stream_ref
 * value used on two threads names two streams. On builds without CUDA, and for host backends, a
 * stream_ref is carried along but no work is enqueued on it.
 * @ingroup core
 */
class stream_ref {
 public:
  /**
   * @brief The per-thread default stream (`cudaStreamPerThread`).
   */
  stream_ref() noexcept;

  /**
   * @brief Refer to an existing stream.
   * @param[in] stream The stream handle (a cudaStream_t); not owned.
   */
  explicit stream_ref(cuda_stream_handle stream) noexcept : handle_(stream) {}

  /**
   * @brief The underlying handle.
   * @return The cudaStream_t this object refers to.
   */
  [[nodiscard]] cuda_stream_handle get() const noexcept {
    return handle_;
  }

  /**
   * @brief Whether this is the per-thread default stream.
   * @return True if the handle is `cudaStreamPerThread`.
   */
  [[nodiscard]] bool is_per_thread_default() const noexcept;

  /**
   * @brief Wait until all work enqueued on the stream has completed.
   *
   * For the per-thread default stream: the calling thread's default stream of the CURRENT device
   * (resources::synchronize() makes the handle's device current first). A no-op in builds without
   * CUDA.
   * @throws cuda_error if the CUDA runtime reports an error.
   * @sync
   */
  void synchronize() const;

  /**
   * @brief Compare two stream references.
   * @param[in] lhs First stream.
   * @param[in] rhs Second stream.
   * @return True if both refer to the same handle.
   */
  friend bool operator==(stream_ref lhs, stream_ref rhs) noexcept {
    return lhs.handle_ == rhs.handle_;
  }

  /**
   * @brief Compare two stream references.
   * @param[in] lhs First stream.
   * @param[in] rhs Second stream.
   * @return True if the handles differ.
   */
  friend bool operator!=(stream_ref lhs, stream_ref rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  cuda_stream_handle handle_;
};

}  // namespace dyng
