// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file copy.hpp
 * @brief Copies between memory spaces: to_vector(), copy() and to_space().
 * @ingroup core
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cstddef>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace dyng {

namespace detail {

/// @brief Wraps a type so that it does not take part in template argument deduction.
/// @tparam type_t The type.
template <typename type_t>
struct identity {
  using type = type_t;  ///< `type_t` itself
};

/// @brief `type_t` in a non-deduced context (C++20's std::type_identity_t).
/// @tparam type_t The type.
template <typename type_t>
using non_deduced = typename identity<type_t>::type;

}  // namespace detail

/**
 * @brief Copy an array (in any memory space) into a new host std::vector.
 * @tparam value_t Element type (possibly const).
 * @param[in] res  Resources whose stream orders the copy.
 * @param[in] src  The array.
 * @return A host vector with the elements of `src`.
 * @throws not_supported_error if `src` is device memory and CUDA is not built.
 * @throws out_of_memory_error if the vector cannot be allocated.
 * @throws cuda_error          if the CUDA runtime reports an error.
 * @sync Waits for the stream of `res` when `src` is not host-accessible.
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] std::vector<std::remove_const_t<value_t>> to_vector(const resources& res,
                                                                  array_view<value_t> src) {
  std::vector<std::remove_const_t<value_t>> out;
  try {
    out.resize(src.size());
  } catch (const std::bad_alloc& e) {
    detail::throw_host_allocation_failure(
        detail::concat_message("to_vector (", src.size(), " elements)"), e.what());
  } catch (const std::length_error& e) {
    detail::throw_host_allocation_failure(
        detail::concat_message("to_vector (", src.size(), " elements)"), e.what());
  }
  if (!src.empty()) {
    detail::copy_bytes(out.data(), memory_space::host, src.data(), src.space(), src.size_bytes(),
                       res.stream(), res.device());
    if (!is_host_accessible(src.space())) {
      res.synchronize();
    }
  }
  return out;
}

/**
 * @brief Copy the elements of one array into another of the same size.
 *
 * The element type is deduced from `dst` alone, so a mutable source view converts:
 * `copy(res, host_view(a), host_view(b))` works for two std::vector<T>.
 * @tparam value_t Element type (deduced from `dst`).
 * @param[in]  res Resources whose stream orders the copy.
 * @param[in]  src Source array (a view of `value_t` or `const value_t`).
 * @param[out] dst Destination array (same size as `src`).
 * @throws invalid_argument_error if the sizes differ.
 * @throws not_supported_error    if a device space is involved and CUDA is not built.
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @async Ordered on the stream of `res` when a device space is involved (host-to-host copies
 *        complete before the call returns).
 * @ingroup core
 */
template <typename value_t>
void copy(const resources& res, detail::non_deduced<array_view<const value_t>> src,
          array_view<value_t> dst) {
  static_assert(!std::is_const_v<value_t>, "copy(): the destination must be mutable");
  DYNG_EXPECTS(src.size() == dst.size(), "copy(): source has ", src.size(),
               " elements but destination has ", dst.size());
  if (!src.empty()) {
    detail::copy_bytes(dst.data(), dst.space(), src.data(), src.space(), src.size_bytes(),
                       res.stream(), res.device());
  }
}

/**
 * @brief Copy an array into a new buffer in the memory space `space`.
 *
 * The memory resource follows the space: res.memory() when it allocates in `space`; otherwise
 * default_host_memory_resource() for memory_space::host, default_pinned_host_memory_resource()
 * for memory_space::pinned_host (the host staging resource of a CUDA handle), and
 * default_device_memory_resource(res.device()) for memory_space::device on a CUDA handle. Managed
 * memory needs a handle whose memory resource is managed.
 * @tparam value_t Element type (possibly const).
 * @param[in] res   Resources providing the stream (and the memory resource, see above).
 * @param[in] src   The array.
 * @param[in] space The memory space of the new buffer.
 * @return A buffer holding a copy of `src`.
 * @throws invalid_argument_error if no memory resource of `res` allocates in `space` (device
 *                                memory with host resources; managed memory without a managed
 *                                resource).
 * @throws not_supported_error    if a device or pinned space is involved and CUDA is not built.
 * @throws out_of_memory_error    if the allocation fails.
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @async Allocation and copy are ordered on the stream of `res` (host-to-host copies complete
 *        before the call returns).
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] buffer<std::remove_const_t<value_t>> to_space(const resources& res,
                                                            array_view<value_t> src,
                                                            memory_space space) {
  memory_resource_ref mr = res.memory();
  int device = res.device();
  if (mr.space() != space) {
    switch (space) {
      case memory_space::host:
        mr = default_host_memory_resource();
        device = -1;
        break;
      case memory_space::pinned_host:
        mr = default_pinned_host_memory_resource();
        break;
      case memory_space::device:
        DYNG_EXPECTS(res.get_backend() == backend::cuda,
                     "to_space(): device memory needs CUDA resources (resources::cuda())");
        mr = default_device_memory_resource(res.device());
        break;
      case memory_space::managed:
        DYNG_EXPECTS(false,
                     "to_space(): managed memory needs resources whose memory resource is "
                     "managed (set_memory_resource())");
        break;
    }
  }
  buffer<std::remove_const_t<value_t>> out(src.size(), res.stream(), mr, device);
  if (!src.empty()) {
    detail::copy_bytes(out.data(), out.space(), src.data(), src.space(), src.size_bytes(),
                       res.stream(), res.device());
  }
  return out;
}

}  // namespace dyng
