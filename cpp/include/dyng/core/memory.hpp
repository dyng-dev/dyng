// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file memory.hpp
 * @brief Memory spaces, the type-erased memory_resource_ref, and the built-in resources.
 * @ingroup core
 *
 * memory_resource_ref has the member names and argument order of the CCCL 3.x memory-resource
 * concept (cuda::mr::resource: allocate(stream, bytes, alignment), deallocate(stream, ptr, bytes,
 * alignment), allocate_sync, deallocate_sync; checked against the CCCL 3.1 of CUDA 13.1, ADR
 * 0003), so adapters in both directions are trivial (PLAN Section 4.7.2). The built-in resources
 * are host_memory_resource (host backends), cuda_async_memory_resource (the default of the CUDA
 * backend: a stream-ordered cudaMallocAsync pool) and pinned_host_memory_resource (page-locked
 * host memory for staging). The header needs no CUDA headers.
 */
#pragma once

#include <dyng/core/stream.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <utility>

namespace dyng {

/**
 * @brief Where a block of memory lives.
 * @ingroup core
 */
enum class memory_space : std::uint8_t {
  host,         ///< pageable host memory
  pinned_host,  ///< page-locked host memory (device-accessible for DMA)
  device,       ///< device memory of one GPU
  managed,      ///< CUDA managed (unified) memory
};

/**
 * @brief Whether host code may dereference memory of a space.
 * @param[in] space The memory space.
 * @return True for host, pinned_host and managed.
 * @ingroup core
 */
[[nodiscard]] constexpr bool is_host_accessible(memory_space space) noexcept {
  return space != memory_space::device;
}

/**
 * @brief Non-owning, type-erased reference to a stream-ordered memory resource.
 *
 * Any object with the members
 * `void* allocate(stream_ref, std::size_t bytes, std::size_t alignment)`,
 * `void deallocate(stream_ref, void*, std::size_t bytes, std::size_t alignment) noexcept`,
 * `void* allocate_sync(std::size_t bytes, std::size_t alignment)`,
 * `void deallocate_sync(void*, std::size_t bytes, std::size_t alignment) noexcept` and
 * `memory_space space() const noexcept` can be referenced. The referenced resource must outlive
 * the reference and every allocation made through it.
 * @ingroup core
 */
class memory_resource_ref {
 public:
  /**
   * @brief Refer to a resource.
   * @tparam resource_t A type providing the members listed in the class description.
   * @param[in] resource The resource; not owned.
   */
  template <
      typename resource_t,
      typename = std::enable_if_t<!std::is_same_v<std::decay_t<resource_t>, memory_resource_ref>>>
  memory_resource_ref(resource_t& resource) noexcept  // NOLINT(google-explicit-constructor)
      : object_(static_cast<void*>(&resource)), vtable_(&vtable_for<resource_t>) {}

  /**
   * @brief Allocate stream-ordered memory.
   * @param[in] stream    The stream the allocation is ordered on.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes (a power of two).
   * @return Pointer to at least `bytes` bytes.
   * @throws out_of_memory_error if the allocation fails.
   */
  void* allocate(stream_ref stream, std::size_t bytes, std::size_t alignment) {
    return vtable_->allocate(object_, stream, bytes, alignment);
  }

  /**
   * @brief Release stream-ordered memory.
   * @param[in] stream    The stream the release is ordered on.
   * @param[in] ptr       Pointer returned by allocate() of the same resource.
   * @param[in] bytes     The size passed to allocate().
   * @param[in] alignment The alignment passed to allocate().
   */
  void deallocate(stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    vtable_->deallocate(object_, stream, ptr, bytes, alignment);
  }

  /**
   * @brief Allocate memory that is usable as soon as the call returns.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes (a power of two).
   * @return Pointer to at least `bytes` bytes.
   * @throws out_of_memory_error if the allocation fails.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment) {
    return vtable_->allocate_sync(object_, bytes, alignment);
  }

  /**
   * @brief Release memory obtained from allocate_sync().
   * @param[in] ptr       Pointer returned by allocate_sync() of the same resource.
   * @param[in] bytes     The size passed to allocate_sync().
   * @param[in] alignment The alignment passed to allocate_sync().
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept {
    vtable_->deallocate_sync(object_, ptr, bytes, alignment);
  }

  /**
   * @brief The memory space of every allocation of the resource.
   * @return The resource's memory space.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return vtable_->space(object_);
  }

  /**
   * @brief Whether two references refer to the same resource object.
   * @param[in] lhs First reference.
   * @param[in] rhs Second reference.
   * @return True if both refer to the same object.
   */
  friend bool operator==(const memory_resource_ref& lhs, const memory_resource_ref& rhs) noexcept {
    return lhs.object_ == rhs.object_;
  }

  /**
   * @brief Whether two references refer to different resource objects.
   * @param[in] lhs First reference.
   * @param[in] rhs Second reference.
   * @return True if the objects differ.
   */
  friend bool operator!=(const memory_resource_ref& lhs, const memory_resource_ref& rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  struct vtable {
    void* (*allocate)(void*, stream_ref, std::size_t, std::size_t);
    void (*deallocate)(void*, stream_ref, void*, std::size_t, std::size_t) noexcept;
    void* (*allocate_sync)(void*, std::size_t, std::size_t);
    void (*deallocate_sync)(void*, void*, std::size_t, std::size_t) noexcept;
    memory_space (*space)(const void*) noexcept;
  };

  template <typename resource_t>
  static constexpr vtable vtable_for{
      [](void* r, stream_ref s, std::size_t b, std::size_t a) -> void* {
        return static_cast<resource_t*>(r)->allocate(s, b, a);
      },
      [](void* r, stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
        static_cast<resource_t*>(r)->deallocate(s, p, b, a);
      },
      [](void* r, std::size_t b, std::size_t a) -> void* {
        return static_cast<resource_t*>(r)->allocate_sync(b, a);
      },
      [](void* r, void* p, std::size_t b, std::size_t a) noexcept {
        static_cast<resource_t*>(r)->deallocate_sync(p, b, a);
      },
      [](const void* r) noexcept -> memory_space {
        return static_cast<const resource_t*>(r)->space();
      },
  };

  void* object_;
  const vtable* vtable_;
};

/**
 * @brief Pageable host memory from the global aligned operator new.
 *
 * Stateless: every instance is interchangeable. The stream arguments are ignored (host
 * allocations are immediately usable).
 * @ingroup core
 */
class host_memory_resource {
 public:
  /**
   * @brief Allocate host memory.
   * @param[in] stream    Ignored.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes (a power of two).
   * @return Pointer to at least `bytes` bytes (non-null, also for 0 bytes).
   * @throws out_of_memory_error if the allocation fails.
   * @throws invalid_argument_error if `alignment` is not a power of two.
   */
  void* allocate(stream_ref stream, std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release host memory.
   * @param[in] stream    Ignored.
   * @param[in] ptr       Pointer returned by allocate() or allocate_sync().
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   */
  void deallocate(stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief Allocate host memory.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes (a power of two).
   * @return Pointer to at least `bytes` bytes.
   * @throws out_of_memory_error if the allocation fails.
   * @throws invalid_argument_error if `alignment` is not a power of two.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release host memory.
   * @param[in] ptr       Pointer returned by allocate() or allocate_sync().
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief The memory space of the allocations.
   * @return memory_space::host.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return memory_space::host;
  }

  /**
   * @brief Host resources are interchangeable (stateless).
   * @return True.
   */
  friend bool operator==(const host_memory_resource& /*lhs*/,
                         const host_memory_resource& /*rhs*/) noexcept {
    return true;
  }

  /**
   * @brief Host resources are interchangeable (stateless).
   * @return False.
   */
  friend bool operator!=(const host_memory_resource& /*lhs*/,
                         const host_memory_resource& /*rhs*/) noexcept {
    return false;
  }
};

/**
 * @brief The process-wide stateless host resource used by host backends by default.
 * @return A reference to a static host_memory_resource.
 * @ingroup core
 */
[[nodiscard]] host_memory_resource& default_host_memory_resource() noexcept;

/**
 * @brief Device memory from a stream-ordered memory pool (cudaMallocFromPoolAsync); the default
 *        resource of the CUDA backend.
 *
 * Each object owns its own pool on one device (it never changes the device's default pool, which
 * belongs to the application). Freed memory stays in the pool (release threshold: unlimited), so
 * once a workload has reached its peak, further allocations and releases do not call into the
 * driver and do not synchronize (invariant I9). Allocations are aligned to 256 bytes.
 *
 * The resource is neither copyable nor movable: memory_resource_ref and buffer refer to it by
 * address. Destroying it returns the pool's memory to the device once every allocation made from
 * it has been released.
 * @ingroup core
 */
class cuda_async_memory_resource {
 public:
  /**
   * @brief Create a pool on a device.
   * @param[in] device CUDA device ordinal.
   * @throws not_supported_error    if the library was built without CUDA, or the device does not
   *                                support stream-ordered allocation.
   * @throws invalid_argument_error if `device` is not a visible device.
   * @throws cuda_error             if the runtime reports an error.
   */
  explicit cuda_async_memory_resource(int device = 0);

  cuda_async_memory_resource(const cuda_async_memory_resource&) = delete;  ///< not copyable
  cuda_async_memory_resource& operator=(const cuda_async_memory_resource&) = delete;  ///< no copy
  cuda_async_memory_resource(cuda_async_memory_resource&&) = delete;             ///< not movable
  cuda_async_memory_resource& operator=(cuda_async_memory_resource&&) = delete;  ///< not movable

  /// @brief Destroy the pool (deferred by the driver until every allocation is released).
  ~cuda_async_memory_resource();

  /**
   * @brief Allocate device memory, ordered on a stream.
   *
   * The memory may be used by work enqueued on `stream` after this call; other streams must be
   * ordered after it (events or synchronization).
   * @param[in] stream    The stream (of this resource's device).
   * @param[in] bytes     Size in bytes (0 gives nullptr).
   * @param[in] alignment Alignment in bytes: a power of two, at most 256.
   * @return Device pointer.
   * @throws out_of_memory_error    if the device is out of memory.
   * @throws invalid_argument_error if `alignment` is not a power of two or exceeds 256.
   * @throws cuda_error             if the runtime reports another error.
   * @async
   */
  void* allocate(stream_ref stream, std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release memory, ordered on a stream: it is reused only after the work enqueued on
   *        `stream` before this call has completed.
   * @param[in] stream    The stream.
   * @param[in] ptr       Pointer returned by this resource (nullptr is ignored).
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   * @async
   */
  void deallocate(stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief Allocate device memory that every stream may use as soon as the call returns.
   * @param[in] bytes     Size in bytes.
   * @param[in] alignment Alignment in bytes: a power of two, at most 256.
   * @return Device pointer.
   * @throws out_of_memory_error    if the device is out of memory.
   * @throws invalid_argument_error if `alignment` is not a power of two or exceeds 256.
   * @throws cuda_error             if the runtime reports another error.
   * @sync
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release memory obtained from allocate_sync(); all work using it must have completed.
   * @param[in] ptr       Pointer returned by this resource (nullptr is ignored).
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   * @sync
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief The memory space of the allocations.
   * @return memory_space::device.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return memory_space::device;
  }

  /**
   * @brief The device of the pool.
   * @return The device ordinal.
   */
  [[nodiscard]] int device() const noexcept {
    return device_;
  }

  /**
   * @brief Bytes currently allocated from the pool and not yet released.
   * @return The pool's used memory.
   * @throws cuda_error if the runtime reports an error.
   */
  [[nodiscard]] std::size_t used_bytes() const;

  /**
   * @brief Bytes the pool holds from the device (used plus cached for reuse).
   * @return The pool's reserved memory.
   * @throws cuda_error if the runtime reports an error.
   */
  [[nodiscard]] std::size_t reserved_bytes() const;

  /**
   * @brief Whether two references denote the same pool.
   * @param[in] lhs First resource.
   * @param[in] rhs Second resource.
   * @return True if they are the same object.
   */
  friend bool operator==(const cuda_async_memory_resource& lhs,
                         const cuda_async_memory_resource& rhs) noexcept {
    return &lhs == &rhs;
  }

  /**
   * @brief Whether two references denote different pools.
   * @param[in] lhs First resource.
   * @param[in] rhs Second resource.
   * @return True if they are different objects.
   */
  friend bool operator!=(const cuda_async_memory_resource& lhs,
                         const cuda_async_memory_resource& rhs) noexcept {
    return !(lhs == rhs);
  }

 private:
  int device_ = -1;
  [[maybe_unused]] void* pool_ = nullptr;  // cudaMemPool_t (unused in builds without CUDA)
};

/**
 * @brief The default memory resource of the CUDA backend on a device: a process-wide
 *        cuda_async_memory_resource, created on first use.
 *
 * It is never destroyed (its memory returns to the driver when the process exits), so buffers and
 * workspaces may outlive every resources handle that used it.
 * @param[in] device CUDA device ordinal.
 * @return The device's default resource.
 * @throws not_supported_error    if the library was built without CUDA.
 * @throws invalid_argument_error if `device` is not a visible device.
 * @throws cuda_error             if the runtime reports an error.
 * @ingroup core
 */
[[nodiscard]] cuda_async_memory_resource& default_device_memory_resource(int device);

/**
 * @brief Page-locked (pinned) host memory, usable by asynchronous copies from and to every device.
 *
 * Stateless: every instance is interchangeable. Allocation is synchronous and costly (it maps the
 * pages for DMA), so the library allocates staging buffers once per workspace and reuses them.
 * @ingroup core
 */
class pinned_host_memory_resource {
 public:
  /**
   * @brief Allocate pinned host memory.
   * @param[in] stream    Ignored: the memory is usable when the call returns.
   * @param[in] bytes     Size in bytes (0 gives nullptr).
   * @param[in] alignment Alignment in bytes: a power of two, at most 256.
   * @return Host pointer, also valid in device code through unified addressing.
   * @throws out_of_memory_error    if the allocation fails.
   * @throws invalid_argument_error if `alignment` is not a power of two or exceeds 256.
   * @throws not_supported_error    if the library was built without CUDA.
   */
  void* allocate(stream_ref stream, std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release pinned host memory after the work enqueued on `stream` has completed (the call
   *        waits for the stream).
   * @param[in] stream    The stream the memory was last used on.
   * @param[in] ptr       Pointer returned by this resource (nullptr is ignored).
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   * @sync
   */
  void deallocate(stream_ref stream, void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief Allocate pinned host memory.
   * @param[in] bytes     Size in bytes (0 gives nullptr).
   * @param[in] alignment Alignment in bytes: a power of two, at most 256.
   * @return Host pointer.
   * @throws out_of_memory_error    if the allocation fails.
   * @throws invalid_argument_error if `alignment` is not a power of two or exceeds 256.
   * @throws not_supported_error    if the library was built without CUDA.
   */
  void* allocate_sync(std::size_t bytes, std::size_t alignment);

  /**
   * @brief Release pinned host memory; all work using it must have completed.
   * @param[in] ptr       Pointer returned by this resource (nullptr is ignored).
   * @param[in] bytes     The size passed at allocation.
   * @param[in] alignment The alignment passed at allocation.
   */
  void deallocate_sync(void* ptr, std::size_t bytes, std::size_t alignment) noexcept;

  /**
   * @brief The memory space of the allocations.
   * @return memory_space::pinned_host.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return memory_space::pinned_host;
  }

  /**
   * @brief Pinned resources are interchangeable (stateless).
   * @return True.
   */
  friend bool operator==(const pinned_host_memory_resource& /*lhs*/,
                         const pinned_host_memory_resource& /*rhs*/) noexcept {
    return true;
  }

  /**
   * @brief Pinned resources are interchangeable (stateless).
   * @return False.
   */
  friend bool operator!=(const pinned_host_memory_resource& /*lhs*/,
                         const pinned_host_memory_resource& /*rhs*/) noexcept {
    return false;
  }
};

/**
 * @brief The process-wide pinned host resource (host staging of the CUDA backend).
 * @return A reference to a static pinned_host_memory_resource.
 * @ingroup core
 */
[[nodiscard]] pinned_host_memory_resource& default_pinned_host_memory_resource() noexcept;

}  // namespace dyng
