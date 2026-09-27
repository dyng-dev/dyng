// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file memory.hpp
 * @brief Memory spaces, the type-erased memory_resource_ref, and the built-in host resource.
 * @ingroup core
 *
 * memory_resource_ref has the member names and argument order of the CCCL 3.x memory-resource
 * concept (cuda::mr::resource_ref: allocate(stream, bytes, alignment), deallocate(stream, ptr,
 * bytes, alignment), allocate_sync, deallocate_sync), so adapters in both directions are trivial
 * (PLAN Section 4.7.2). The device resources (cuda_async_memory_resource, pinned_host) arrive
 * with the CUDA backend.
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
};

/**
 * @brief The process-wide stateless host resource used by host backends by default.
 * @return A reference to a static host_memory_resource.
 * @ingroup core
 */
[[nodiscard]] host_memory_resource& default_host_memory_resource() noexcept;

}  // namespace dyng
