// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:escher/kernel/find.cu (readFlat, the bounded chain walk)
// Derived from ESCHER-GPU@abcf9b2:kernel/motif_utils.cuh (readFlat)
/**
 * @file chain.cuh
 * @brief The one bounded walk of a CBST row along its chain of payload segments (M8a step 4,
 *        ADR 0039), used by every reader of the store (the row reads of find, the row locator,
 *        the integrity checker) and by the unfill writer.
 *
 * A row is read from its first slot: a value > 0 belongs to the row, 0 or INT_MIN ends it, any
 * other negative value -p continues it at slot p (README.md, "The payload"). ESCHER-GPU's readFlat
 * walked without a bound (a cyclic chain hung the kernel); MOSP_ESCHER's bounded it to 1,024 hops
 * and then returned INT_MIN, which silently truncates a longer, legitimate row (ESCHER-GPU's V2H
 * rows gain a segment per overflowing fill and are never compacted). dynG bounds the walk by the
 * used payload instead: every slot of a well-formed row is read once, so a walk that reads more
 * slots than the payload uses, or leaves it, is a broken chain. The caller then raises the store's
 * device error word (the host throws internal_error at the next read point and poisons the store);
 * a row is never truncated.
 *
 * Host and device (the integrity checker walks host copies).
 */
#pragma once

#include <climits>

namespace dyng::detail::escher {

/// How a bounded walk of a row ended.
enum class chain_end : int {
  terminated = 0,  ///< at the row's terminator (0 or INT_MIN)
  overrun = 1,     ///< the walk left the used payload or read more slots than it has
};

/**
 * @brief Walks the row that starts at slot `base` of `flat`, of which `used` slots are in use.
 * @param flat    The payload (a pointer, or any array-like with operator[]).
 * @param used    The slots in use (the bump pointer): the walk stays in [0, used).
 * @param base    The row's first slot.
 * @param value   Called as value(slot, v) for every value v of the row, in read order.
 * @param segment Called as segment(begin, end, v) at the end of every segment: `begin` is its
 *                first slot, `end` the slot holding v, its end (0 or INT_MIN, or a chain pointer
 *                to the next segment). A writer may change the slots of the segment before
 *                `end` and the slot `end` itself in the callbacks; the walk has read them.
 * @return chain_end::terminated, or chain_end::overrun for a broken chain.
 */
// The callbacks run where the walk runs (host lambdas from host code, device lambdas in kernels).
#pragma nv_exec_check_disable
template <class flat_t, class value_t, class segment_t>
__host__ __device__ inline chain_end walk_chain(flat_t&& flat, long long used, long long base,
                                                value_t&& value, segment_t&& segment) {
  long long loc = base, begin = base;
  for (long long steps = 0; steps < used; ++steps) {
    if (loc < 0 || loc >= used) return chain_end::overrun;
    const int v = flat[loc];
    if (v < 0 && v != INT_MIN) {
      segment(begin, loc, v);
      loc = -static_cast<long long>(v);
      begin = loc;
      continue;
    }
    if (v == 0 || v == INT_MIN) {
      segment(begin, loc, v);
      return chain_end::terminated;
    }
    value(loc, v);
    ++loc;
  }
  return chain_end::overrun;
}

/// The same without a segment callback.
#pragma nv_exec_check_disable
template <class flat_t, class value_t>
__host__ __device__ inline chain_end walk_chain(flat_t&& flat, long long used, long long base,
                                                value_t&& value) {
  return walk_chain(flat, used, base, value, [](long long, long long, int) {});
}

}  // namespace dyng::detail::escher
