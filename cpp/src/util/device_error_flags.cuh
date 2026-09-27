// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_error_flags.cuh
 * @brief Device side of the sticky error word: raise_device_error() and device_error_raised().
 */
#pragma once

#include "util/device_error_flags.hpp"

#include <cstdint>

namespace dyng::detail {

/**
 * @brief Set error bits in the word (atomic; the first violation does not stop the kernel, the
 *        host decides after the launch).
 * @param[in,out] word The device word of a device_error_flags.
 * @param[in]     e    The error kind.
 */
__device__ __forceinline__ void raise_device_error(std::uint32_t* word, device_error e) {
  atomicOr(reinterpret_cast<unsigned int*>(word), static_cast<unsigned int>(bits_of(e)));
}

/**
 * @brief Whether any error has been raised (for kernels that stop early after an error).
 * @param[in] word The device word.
 * @return True if a bit is set.
 */
__device__ __forceinline__ bool device_error_raised(const std::uint32_t* word) {
  return *reinterpret_cast<const volatile std::uint32_t*>(word) != 0U;
}

}  // namespace dyng::detail
