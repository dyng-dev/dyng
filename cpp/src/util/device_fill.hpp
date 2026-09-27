// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_fill.hpp
 * @brief fill_async(): set every element of a device array to one value, ordered on a stream.
 *
 * Host-only declaration; the kernel and its explicit instantiations (std::uint8_t, std::int32_t,
 * std::uint32_t, std::int64_t, std::uint64_t, float, double) live in device_fill.cu, which is
 * compiled only with DYNG_HAS_CUDA. Only CUDA code paths may call it.
 */
#pragma once

#include <dyng/core/stream.hpp>

#include <cstddef>

namespace dyng::detail {

/**
 * @brief Set `count` elements at `data` to `value`, ordered on `stream`.
 * @tparam value_t One of the instantiated element types.
 * @param[in]  stream Stream the kernel is enqueued on (of the current device).
 * @param[out] data   Device-accessible array.
 * @param[in]  count  Number of elements (0 enqueues nothing).
 * @param[in]  value  The value.
 * @throws cuda_error if the launch fails.
 * @async
 */
template <typename value_t>
void fill_async(stream_ref stream, value_t* data, std::size_t count, value_t value);

}  // namespace dyng::detail
