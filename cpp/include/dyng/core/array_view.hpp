// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file array_view.hpp
 * @brief array_view<T>: a non-owning view of a typed array in any memory space.
 * @ingroup core
 */
#pragma once

#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace dyng {

/**
 * @brief A non-owning, trivially copyable description of an array: {data, size, space, device}.
 *
 * One type describes host and device arrays, which keeps public signatures (and the Python
 * bindings, through DLPack) free of memory-space templates. Element access through operator[],
 * begin() and end() is only valid for host-accessible spaces (see is_host_accessible()).
 *
 * @tparam value_t Element type; `const value_t` for read-only views.
 * @ingroup core
 */
template <typename value_t>
class array_view {
 public:
  using value_type = std::remove_cv_t<value_t>;  ///< the element type without const
  using element_type = value_t;                  ///< the element type as viewed
  using size_type = std::size_t;                 ///< the size type
  using pointer = value_t*;                      ///< pointer to an element
  using reference = value_t&;                    ///< reference to an element (host spaces only)
  using iterator = value_t*;                     ///< host iterator

  /**
   * @brief An empty host view.
   */
  constexpr array_view() noexcept = default;

  /**
   * @brief View `size` elements at `data`.
   * @param[in] data   Pointer to the first element (may be null if `size` is 0).
   * @param[in] size   Number of elements.
   * @param[in] space  Memory space of the elements.
   * @param[in] device CUDA device ordinal for device / managed memory; -1 for host memory.
   */
  constexpr array_view(pointer data, size_type size, memory_space space = memory_space::host,
                       int device = -1) noexcept
      : data_(data), size_(size), space_(space), device_(device) {}

  /**
   * @brief Implicit conversion from a mutable view to a read-only view.
   * @tparam other_t The element type of the source view (`value_type`).
   * @param[in] other The mutable view.
   */
  template <typename other_t, typename = std::enable_if_t<std::is_const_v<value_t> &&
                                                          std::is_same_v<const other_t, value_t>>>
  constexpr array_view(
      const array_view<other_t>& other) noexcept  // NOLINT(google-explicit-constructor)
      : data_(other.data()), size_(other.size()), space_(other.space()), device_(other.device()) {}

  /**
   * @brief Pointer to the first element.
   * @return The data pointer (may be a device pointer).
   */
  [[nodiscard]] constexpr pointer data() const noexcept {
    return data_;
  }

  /**
   * @brief Number of elements.
   * @return The element count.
   */
  [[nodiscard]] constexpr size_type size() const noexcept {
    return size_;
  }

  /**
   * @brief Size of the viewed memory in bytes.
   * @return size() * sizeof(value_t).
   */
  [[nodiscard]] constexpr size_type size_bytes() const noexcept {
    return size_ * sizeof(value_t);
  }

  /**
   * @brief Whether the view has no elements.
   * @return True if size() == 0.
   */
  [[nodiscard]] constexpr bool empty() const noexcept {
    return size_ == 0;
  }

  /**
   * @brief The memory space of the elements.
   * @return The space given at construction.
   */
  [[nodiscard]] constexpr memory_space space() const noexcept {
    return space_;
  }

  /**
   * @brief The CUDA device of the elements.
   * @return The device ordinal, or -1 for host memory.
   */
  [[nodiscard]] constexpr int device() const noexcept {
    return device_;
  }

  /**
   * @brief Unchecked element access (host-accessible spaces only).
   * @param[in] i Index in [0, size()).
   * @return Reference to element `i`.
   */
  constexpr reference operator[](size_type i) const noexcept {
    return data_[i];
  }

  /**
   * @brief Checked element access (host-accessible spaces only).
   * @param[in] i Index.
   * @return Reference to element `i`.
   * @throws invalid_argument_error if `i >= size()` or the space is not host-accessible.
   */
  reference at(size_type i) const {
    DYNG_EXPECTS(is_host_accessible(space_), "array_view::at() on device memory");
    DYNG_EXPECTS(i < size_, "array_view index ", i, " out of range [0, ", size_, ")");
    return data_[i];
  }

  /**
   * @brief Host iterator to the first element.
   * @return data().
   */
  [[nodiscard]] constexpr iterator begin() const noexcept {
    return data_;
  }

  /**
   * @brief Host iterator past the last element.
   * @return data() + size().
   */
  [[nodiscard]] constexpr iterator end() const noexcept {
    return data_ + size_;
  }

  /**
   * @brief A view of a contiguous sub-range.
   * @param[in] offset First element of the sub-range.
   * @param[in] count  Number of elements.
   * @return The view of [offset, offset + count), in the same space and device.
   * @throws invalid_argument_error if the range exceeds the view.
   */
  [[nodiscard]] array_view subview(size_type offset, size_type count) const {
    DYNG_EXPECTS(offset <= size_ && count <= size_ - offset, "subview [", offset, ", ",
                 offset + count, ") exceeds the view size ", size_);
    return array_view(data_ + offset, count, space_, device_);
  }

 private:
  pointer data_ = nullptr;
  size_type size_ = 0;
  memory_space space_ = memory_space::host;
  int device_ = -1;
};

/**
 * @brief A host view of a pointer range.
 * @tparam value_t Element type.
 * @param[in] data Pointer to the first element.
 * @param[in] size Number of elements.
 * @return array_view in memory_space::host.
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] constexpr array_view<value_t> host_view(value_t* data, std::size_t size) noexcept {
  return array_view<value_t>(data, size, memory_space::host, -1);
}

/**
 * @brief A mutable host view of a std::vector.
 * @tparam value_t Element type.
 * @tparam alloc_t Allocator type.
 * @param[in] v The vector; must outlive the view and not be resized while viewed.
 * @return array_view over v.data(), v.size().
 * @ingroup core
 */
template <typename value_t, typename alloc_t>
[[nodiscard]] array_view<value_t> host_view(std::vector<value_t, alloc_t>& v) noexcept {
  return array_view<value_t>(v.data(), v.size(), memory_space::host, -1);
}

/**
 * @brief A read-only host view of a std::vector.
 * @tparam value_t Element type.
 * @tparam alloc_t Allocator type.
 * @param[in] v The vector; must outlive the view and not be resized while viewed.
 * @return array_view<const value_t> over v.data(), v.size().
 * @ingroup core
 */
template <typename value_t, typename alloc_t>
[[nodiscard]] array_view<const value_t> host_view(const std::vector<value_t, alloc_t>& v) noexcept {
  return array_view<const value_t>(v.data(), v.size(), memory_space::host, -1);
}

/**
 * @brief A device view of a pointer range.
 * @tparam value_t Element type.
 * @param[in] data   Device pointer to the first element.
 * @param[in] size   Number of elements.
 * @param[in] device CUDA device ordinal owning the memory.
 * @return array_view in memory_space::device.
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] constexpr array_view<value_t> device_view(value_t* data, std::size_t size,
                                                        int device = 0) noexcept {
  return array_view<value_t>(data, size, memory_space::device, device);
}

}  // namespace dyng
