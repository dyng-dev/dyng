// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file buffer.hpp
 * @brief buffer<T>: an owning, move-only, uninitialized, stream-ordered array.
 * @ingroup core
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>

#include <algorithm>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace dyng {

namespace detail {

/**
 * @brief Copy bytes between two memory spaces, ordered on `stream`.
 * @param[out] dst       Destination pointer.
 * @param[in]  dst_space Memory space of `dst`.
 * @param[in]  src       Source pointer.
 * @param[in]  src_space Memory space of `src`.
 * @param[in]  bytes     Number of bytes.
 * @param[in]  stream    Stream the copy is ordered on (device copies).
 * @param[in]  device    CUDA device the stream belongs to (made current for the copy; the
 *                       per-thread default stream is per device), or -1 for the current device.
 * @throws not_supported_error if a device space is involved and CUDA is not built.
 * @throws cuda_error          if the CUDA runtime reports an error.
 */
void copy_bytes(void* dst, memory_space dst_space, const void* src, memory_space src_space,
                std::size_t bytes, stream_ref stream, int device = -1);

}  // namespace detail

/**
 * @brief An owning, move-only, UNINITIALIZED array allocated from a memory_resource_ref.
 *
 * Semantics follow rmm::device_uvector: the elements are not constructed or zeroed, so
 * `value_t` must be trivially copyable. Allocation and release are ordered on the buffer's
 * stream. The memory resource must outlive the buffer.
 *
 * @tparam value_t Trivially copyable element type.
 * @ingroup core
 */
template <typename value_t>
class buffer {
  static_assert(std::is_trivially_copyable_v<value_t>, "buffer<T> needs a trivially copyable T");
  static_assert(!std::is_const_v<value_t>, "buffer<T> owns its elements; T must not be const");

 public:
  using value_type = value_t;            ///< element type
  using size_type = std::size_t;         ///< size type
  using pointer = value_t*;              ///< pointer to an element
  using const_pointer = const value_t*;  ///< pointer to a const element

  /**
   * @brief An empty host buffer that owns no memory.
   */
  buffer() noexcept : mr_(default_host_memory_resource()) {}

  /**
   * @brief Allocate `size` uninitialized elements.
   * @param[in] size   Number of elements.
   * @param[in] stream Stream the allocation is ordered on.
   * @param[in] mr     Memory resource; must outlive the buffer.
   * @param[in] device CUDA device of the memory for device / managed spaces, else -1.
   * @throws out_of_memory_error if the allocation fails.
   */
  buffer(size_type size, stream_ref stream, memory_resource_ref mr, int device = -1)
      : stream_(stream), mr_(mr), device_(device) {
    allocate_storage(size);
  }

  /**
   * @brief Allocate `size` uninitialized elements with the memory resource and stream of `res`.
   * @param[in] res  Resources providing the memory resource, stream and device.
   * @param[in] size Number of elements.
   * @throws out_of_memory_error if the allocation fails.
   * @async The allocation is ordered on the stream of `res` (host memory resources allocate
   *        before returning).
   */
  buffer(const resources& res, size_type size)
      : buffer(size, res.stream(), res.memory(), res.device()) {}

  buffer(const buffer&) = delete;
  buffer& operator=(const buffer&) = delete;

  /**
   * @brief Take over the memory of another buffer.
   * @param[in,out] other The source; left empty.
   */
  buffer(buffer&& other) noexcept
      : data_(std::exchange(other.data_, nullptr)),
        size_(std::exchange(other.size_, 0)),
        stream_(other.stream_),
        mr_(other.mr_),
        device_(other.device_) {}

  /**
   * @brief Release the current memory and take over the memory of another buffer.
   * @param[in,out] other The source; left empty.
   * @return *this.
   */
  buffer& operator=(buffer&& other) noexcept {
    if (this != &other) {
      release_storage();
      data_ = std::exchange(other.data_, nullptr);
      size_ = std::exchange(other.size_, 0);
      stream_ = other.stream_;
      mr_ = other.mr_;
      device_ = other.device_;
    }
    return *this;
  }

  /**
   * @brief Release the memory (stream-ordered).
   */
  ~buffer() {
    release_storage();
  }

  /**
   * @brief Pointer to the first element.
   * @return The data pointer (nullptr if empty).
   */
  [[nodiscard]] pointer data() noexcept {
    return data_;
  }

  /**
   * @brief Pointer to the first element.
   * @return The data pointer (nullptr if empty).
   */
  [[nodiscard]] const_pointer data() const noexcept {
    return data_;
  }

  /**
   * @brief Number of elements.
   * @return The element count.
   */
  [[nodiscard]] size_type size() const noexcept {
    return size_;
  }

  /**
   * @brief Whether the buffer holds no elements.
   * @return True if size() == 0.
   */
  [[nodiscard]] bool empty() const noexcept {
    return size_ == 0;
  }

  /**
   * @brief The memory space of the elements.
   * @return The space of the memory resource.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return mr_.space();
  }

  /**
   * @brief The CUDA device of the elements.
   * @return The device ordinal, or -1 for host memory.
   */
  [[nodiscard]] int device() const noexcept {
    return device_;
  }

  /**
   * @brief The stream allocation and release are ordered on.
   * @return The buffer's stream.
   */
  [[nodiscard]] stream_ref stream() const noexcept {
    return stream_;
  }

  /**
   * @brief The memory resource the buffer allocates from.
   * @return A reference to the resource.
   */
  [[nodiscard]] memory_resource_ref memory_resource() const noexcept {
    return mr_;
  }

  /**
   * @brief A mutable view of the elements.
   * @return array_view over data(), size() in space().
   */
  [[nodiscard]] array_view<value_t> view() noexcept {
    return array_view<value_t>(data_, size_, space(), device_);
  }

  /**
   * @brief A read-only view of the elements.
   * @return array_view<const value_t> over data(), size() in space().
   */
  [[nodiscard]] array_view<const value_t> view() const noexcept {
    return array_view<const value_t>(data_, size_, space(), device_);
  }

  /**
   * @brief Unchecked element access (host-accessible spaces only).
   * @param[in] i Index in [0, size()).
   * @return Reference to element `i`.
   */
  value_t& operator[](size_type i) noexcept {
    return data_[i];
  }

  /**
   * @brief Unchecked element access (host-accessible spaces only).
   * @param[in] i Index in [0, size()).
   * @return Reference to element `i`.
   */
  const value_t& operator[](size_type i) const noexcept {
    return data_[i];
  }

  /**
   * @brief Change the size; the first min(old, new) elements are kept, new ones are uninitialized.
   * @param[in] new_size The new element count.
   * @throws out_of_memory_error if the allocation fails.
   * @throws not_supported_error if the memory is device memory and CUDA is not built.
   */
  void resize(size_type new_size) {
    if (new_size == size_) {
      return;
    }
    buffer next(new_size, stream_, mr_, device_);
    const size_type keep = std::min(size_, new_size);
    if (keep > 0) {
      detail::copy_bytes(next.data_, next.space(), data_, space(), keep * sizeof(value_t), stream_,
                         device_);
    }
    *this = std::move(next);
  }

 private:
  static constexpr std::size_t alignment = alignof(value_t) > alignof(std::max_align_t)
                                               ? alignof(value_t)
                                               : alignof(std::max_align_t);

  void allocate_storage(size_type size) {
    if (size > 0) {
      data_ = static_cast<pointer>(mr_.allocate(stream_, size * sizeof(value_t), alignment));
      size_ = size;
    }
  }

  void release_storage() noexcept {
    if (data_ != nullptr) {
      mr_.deallocate(stream_, data_, size_ * sizeof(value_t), alignment);
      data_ = nullptr;
      size_ = 0;
    }
  }

  pointer data_ = nullptr;
  size_type size_ = 0;
  stream_ref stream_{};
  memory_resource_ref mr_;
  int device_ = -1;
};

}  // namespace dyng
