// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file copy.hpp
 * @brief Copies between memory spaces: to_vector(), copy() and to_space().
 * @ingroup core
 */
#pragma once

#include <dyng/core/array_view.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <cstddef>
#include <type_traits>
#include <vector>

namespace dyng {

/**
 * @brief Copy an array (in any memory space) into a new host std::vector.
 * @tparam value_t Element type (possibly const).
 * @param[in] res  Resources whose stream orders the copy.
 * @param[in] src  The array.
 * @return A host vector with the elements of `src`.
 * @throws not_supported_error if `src` is device memory and CUDA is not built.
 * @sync
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] std::vector<std::remove_const_t<value_t>> to_vector(const resources& res,
                                                                  array_view<value_t> src) {
  std::vector<std::remove_const_t<value_t>> out(src.size());
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
 * @tparam value_t Element type.
 * @param[in]  res Resources whose stream orders the copy.
 * @param[in]  src Source array.
 * @param[out] dst Destination array (same size as `src`).
 * @throws invalid_argument_error if the sizes differ.
 * @throws not_supported_error    if a device space is involved and CUDA is not built.
 * @async
 * @ingroup core
 */
template <typename value_t>
void copy(const resources& res, array_view<const value_t> src, array_view<value_t> dst) {
  static_assert(!std::is_const_v<value_t>, "copy(): the destination must be mutable");
  DYNG_EXPECTS(src.size() == dst.size(), "copy(): source has ", src.size(),
               " elements but destination has ", dst.size());
  if (!src.empty()) {
    detail::copy_bytes(dst.data(), dst.space(), src.data(), src.space(), src.size_bytes(),
                       res.stream(), res.device());
  }
}

/**
 * @brief Copy an array into a new buffer allocated from the memory resource of `res`.
 * @tparam value_t Element type (possibly const).
 * @param[in] res   Resources providing the memory resource and stream.
 * @param[in] src   The array.
 * @param[in] space Required memory space; must equal res.memory().space().
 * @return A buffer holding a copy of `src`.
 * @throws invalid_argument_error if `space` differs from the space of res.memory().
 * @throws not_supported_error    if a device space is involved and CUDA is not built.
 * @async
 * @ingroup core
 */
template <typename value_t>
[[nodiscard]] buffer<std::remove_const_t<value_t>> to_space(const resources& res,
                                                            array_view<value_t> src,
                                                            memory_space space) {
  DYNG_EXPECTS(res.memory().space() == space,
               "to_space(): the memory resource of the resources object does not allocate in "
               "the requested space");
  buffer<std::remove_const_t<value_t>> out(res, src.size());
  if (!src.empty()) {
    detail::copy_bytes(out.data(), out.space(), src.data(), src.space(), src.size_bytes(),
                       res.stream(), res.device());
  }
  return out;
}

}  // namespace dyng
