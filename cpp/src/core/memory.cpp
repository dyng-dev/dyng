// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file memory.cpp
 * @brief Host memory resource, stream_ref and cross-space byte copies (host part).
 */
#include <dyng/config.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/stream.hpp>

#include <cstdint>
#include <cstring>
#include <new>

namespace dyng {

namespace {

// cudaStreamPerThread is defined by the CUDA runtime as ((cudaStream_t)0x2).
constexpr std::uintptr_t per_thread_stream_value = 0x2;

bool is_power_of_two(std::size_t x) noexcept {
  return x != 0 && (x & (x - 1)) == 0;
}

}  // namespace

// ----------------------------------------------------------------------------------------------
// stream_ref
// ----------------------------------------------------------------------------------------------

stream_ref::stream_ref() noexcept
    : handle_(reinterpret_cast<cuda_stream_handle>(  // NOLINT(performance-no-int-to-ptr)
          per_thread_stream_value)) {}

bool stream_ref::is_per_thread_default() const noexcept {
  return reinterpret_cast<std::uintptr_t>(handle_) == per_thread_stream_value;
}

void stream_ref::synchronize() const {
#if DYNG_HAS_CUDA
#error "stream_ref::synchronize() for CUDA builds is implemented in milestone M1b"
#endif
}

// ----------------------------------------------------------------------------------------------
// host_memory_resource
// ----------------------------------------------------------------------------------------------

void* host_memory_resource::allocate(stream_ref /*stream*/, std::size_t bytes,
                                     std::size_t alignment) {
  return allocate_sync(bytes, alignment);
}

void host_memory_resource::deallocate(stream_ref /*stream*/, void* ptr, std::size_t bytes,
                                      std::size_t alignment) noexcept {
  deallocate_sync(ptr, bytes, alignment);
}

void* host_memory_resource::allocate_sync(std::size_t bytes, std::size_t alignment) {
  DYNG_EXPECTS(is_power_of_two(alignment), "alignment ", alignment, " is not a power of two");
  try {
    return ::operator new(bytes == 0 ? 1 : bytes, std::align_val_t{alignment});
  } catch (const std::bad_alloc&) {
    throw out_of_memory_error(
        detail::concat_message("dyng: host allocation of ", bytes, " bytes failed"));
  }
}

void host_memory_resource::deallocate_sync(void* ptr, std::size_t /*bytes*/,
                                           std::size_t alignment) noexcept {
  ::operator delete(ptr, std::align_val_t{alignment});
}

host_memory_resource& default_host_memory_resource() noexcept {
  static host_memory_resource resource;
  return resource;
}

// ----------------------------------------------------------------------------------------------
// copies
// ----------------------------------------------------------------------------------------------

namespace detail {

void copy_bytes(void* dst, memory_space dst_space, const void* src, memory_space src_space,
                std::size_t bytes, stream_ref /*stream*/) {
  if (bytes == 0) {
    return;
  }
  if (is_host_accessible(dst_space) && is_host_accessible(src_space)) {
    std::memmove(dst, src, bytes);
    return;
  }
  throw not_supported_error(
      "dyng: copies involving device memory need the CUDA backend, which is not built");
}

}  // namespace detail
}  // namespace dyng
