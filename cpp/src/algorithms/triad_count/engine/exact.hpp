// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file exact.hpp
 * @brief The exact 64-bit arithmetic of the counting operations (ADR 0057): C(n, 3) without an
 *        intermediate overflow, and products and sums that report a result past 2^64 - 1 instead
 *        of wrapping, on the host and the device.
 *
 * ESCHER-GPU@abcf9b2 computes C(|e|, 3), C(a, 2) * b * c and x * y * z in signed 64-bit
 * (Choose3Op, Type2Op, Type3Op) and adds them with plain unsigned additions: C(|e|, 3) goes wrong
 * from |e| = 2,097,152 (n (n - 1) (n - 2) passes 2^63 - 1, undefined behaviour on the host), and a
 * product or a sum past 2^64 - 1 wraps. Here every product is unsigned and checked, C(n, 3) is
 * divided before it is multiplied, and the operations keep an overflow flag beside their register
 * sum that their flush() reports through the bins handle (bins.overflow(): count_overflow on the
 * host, the capacity bit of the device error word on cuda; both become capacity_error). The values
 * are the original's wherever the original's arithmetic does not overflow.
 */
#pragma once

#include "algorithms/triad_count/engine/sets.hpp"

namespace dyng::detail::triad {

/**
 * @brief a * b.
 * @param[in]  a   A factor.
 * @param[in]  b   A factor.
 * @param[out] out The product modulo 2^64.
 * @return Whether the product exceeds 2^64 - 1.
 */
DYNG_TRIAD_INLINE bool mul_overflows(unsigned long long a, unsigned long long b,
                                     unsigned long long& out) {
#if defined(__CUDA_ARCH__)
  out = a * b;
  return __umul64hi(a, b) != 0;
#else
  return __builtin_mul_overflow(a, b, &out);
#endif
}

/**
 * @brief a + b.
 * @param[in]  a   A term.
 * @param[in]  b   A term.
 * @param[out] out The sum modulo 2^64.
 * @return Whether the sum exceeds 2^64 - 1.
 */
DYNG_TRIAD_INLINE bool add_overflows(unsigned long long a, unsigned long long b,
                                     unsigned long long& out) {
  out = a + b;
  return out < a;
}

/**
 * @brief C(n, 3), exact: of n, n - 1 and n - 2, the even one of the first two is halved and the
 *        multiple of 3 divided by 3 before any product, so only the last product can pass 2^64 - 1
 *        (from n = 4,801,281 on; n < 2^31, so the first product is below 2^62).
 * @param[in]  n   The members of a hyperedge (0 <= n < 2^31).
 * @param[out] out C(n, 3) (0 for n < 3) when it fits.
 * @return Whether C(n, 3) exceeds 2^64 - 1.
 */
DYNG_TRIAD_INLINE bool choose3_overflows(int n, unsigned long long& out) {
  if (n < 3) {
    out = 0;
    return false;
  }
  unsigned long long a = static_cast<unsigned long long>(n);
  unsigned long long b = a - 1;
  unsigned long long c = a - 2;
  if (a % 2 == 0) {
    a /= 2;
  } else {
    b /= 2;
  }
  // One of n, n - 1, n - 2 is a multiple of 3, and halving keeps it one.
  if (a % 3 == 0) {
    a /= 3;
  } else if (b % 3 == 0) {
    b /= 3;
  } else {
    c /= 3;
  }
  return mul_overflows(a * b, c, out);
}

}  // namespace dyng::detail::triad
