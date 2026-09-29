// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file scratch_buffer.hpp
 * @brief scratch_buffer<T>: a growable array for pooled workspaces, in the memory space of the
 *        resources it is sized with (host for host backends, device for cuda; ADR 0015).
 *
 * A workspace type (derived from pooled_workspace) holds its arrays as scratch_buffer members and
 * sizes them at the start of every run with reserve(res, n): a no-op once the capacity suffices,
 * so a steady-state run allocates nothing (invariant I9). The contents are scratch: they are not
 * kept when the buffer grows. Memory comes from res.memory() on res.stream() and returns there
 * when the workspace is destroyed, which is why resources::set_memory_resource() releases the idle
 * workspaces first.
 */
#pragma once

#include <dyng/core/buffer.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cstddef>

namespace dyng::detail {

/**
 * @brief A growable, uninitialized scratch array allocated from a resources handle.
 * @tparam value_t Trivially copyable element type.
 */
template <typename value_t>
class scratch_buffer {
 public:
  /**
   * @brief Make room for at least `count` elements (contents are not kept when it grows).
   * @param[in] res   The resources of the run (memory resource, stream, device).
   * @param[in] count Required number of elements.
   * @return Pointer to the first element (nullptr if `count` and the capacity are 0).
   * @throws out_of_memory_error if the allocation fails.
   * @async On the CUDA backend the allocation is ordered on the stream of `res`.
   */
  value_t* reserve(const resources& res, std::size_t count) {
    if (count > storage_.size() || storage_.memory_resource() != res.memory()) {
      storage_ = buffer<value_t>();  // release first: the peak holds one array, not two
      storage_ = buffer<value_t>(res, count);
    }
    return storage_.data();
  }

  /**
   * @brief Return the memory (the next reserve() allocates again).
   * @async On the CUDA backend the memory returns to its pool in stream order.
   */
  void release() noexcept {
    storage_ = buffer<value_t>();
  }

  /**
   * @brief The array.
   * @return Pointer to the first element.
   */
  [[nodiscard]] value_t* data() noexcept {
    return storage_.data();
  }

  /**
   * @brief The array.
   * @return Pointer to the first element.
   */
  [[nodiscard]] const value_t* data() const noexcept {
    return storage_.data();
  }

  /**
   * @brief Elements the buffer can hold without growing.
   * @return The capacity.
   */
  [[nodiscard]] std::size_t capacity() const noexcept {
    return storage_.size();
  }

  /**
   * @brief Bytes held (for workspace memory reports).
   * @return capacity() * sizeof(value_t).
   */
  [[nodiscard]] std::size_t bytes() const noexcept {
    return storage_.size() * sizeof(value_t);
  }

  /**
   * @brief The memory space of the array.
   * @return The space of the memory resource it was allocated from.
   */
  [[nodiscard]] memory_space space() const noexcept {
    return storage_.space();
  }

 private:
  buffer<value_t> storage_;
};

}  // namespace dyng::detail
