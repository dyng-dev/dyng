// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu (alignUp32, rowCapacityFor, buildCsr)
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDelta.cu (heSlotFor, vSlotFor, the re-uploads)
/**
 * @file slack_layout.hpp
 * @brief The layout rules of the slack_csr storage and line graph (MOSP_ESCHER@4b86159's H-SOSP
 *        DeviceH2H and DeviceIncidence), shared by the host store (M11a step 2) and the CUDA store
 *        (M11a step 3), and the internal knobs that set them (ADR 0086).
 *
 * The line graph is a slack CSR of node rows: row r (hyperedge id r) holds deg[r] neighbour ids at
 * col_ind[row_start[r] ..), in a slot of cap[r] entries. A deletion swap-removes, an insertion
 * appends; a row that outgrows its slot moves to the tail of col_ind with a new slot; a full tail
 * rebuilds the whole CSR from the incidence lists. None of this is public: capacity_policy has no
 * field for it (HD9, decided by the author on 2026-10-28); the parity harness sets the knobs to
 * the original's capacities through detail::hypergraph_access.
 */
#pragma once

#include <algorithm>
#include <climits>
#include <cstdint>

namespace dyng::detail::hyper::slack {

/// @brief x rounded up to a multiple of 32 (alignUp32). @param[in] x The value. @return It.
inline std::int64_t align_up_32(std::int64_t x) noexcept {
  return (x + 31) & ~std::int64_t{31};
}

/// @brief The slot of a line-graph row of `deg` entries (rowCapacityFor: at least 4 spare slots,
/// at least 25 % headroom, 32-aligned; a relocated row uses the same rule for its new size).
/// @param[in] deg The entries. @return The slot.
inline std::int64_t row_capacity_for(std::int64_t deg) noexcept {
  return align_up_32(deg + std::max<std::int64_t>(4, deg / 4));
}

/// @brief The slot of a hyperedge row of the incidence mirror (heSlotFor). @param[in] len Its
/// members. @return The slot.
inline std::int64_t he_slot_for(std::int64_t len) noexcept {
  return len + std::max<std::int64_t>(2, len / 2);
}

/// @brief The slot of a vertex row of the incidence mirror (vSlotFor). @param[in] len Its
/// hyperedges. @return The slot.
inline std::int64_t v_slot_for(std::int64_t len) noexcept {
  return len + std::max<std::int64_t>(4, len / 2);
}

/// @brief The entry capacity of a CSR whose rows take `cursor` slots (buildCsr:
/// max(cursor, cursor * headroom) + 4096). @param[in] cursor The slots. @param[in] headroom The
/// entry headroom (>= 1). @return The capacity.
inline std::int64_t entry_capacity(std::int64_t cursor, double headroom) noexcept {
  return std::max(cursor, static_cast<std::int64_t>(static_cast<double>(cursor) * headroom)) + 4096;
}

/// @brief The capacity of a mirror whose rows take `cursor` slots (uploadHeRows /
/// uploadVertexRows: max(cursor, cursor * growth) + 64). @param[in] cursor The slots.
/// @param[in] growth The headroom (the build's entry headroom, 1.5 on a re-upload).
/// @return The capacity.
inline std::int64_t mirror_capacity(std::int64_t cursor, double growth) noexcept {
  return std::max(cursor, static_cast<std::int64_t>(static_cast<double>(cursor) * growth)) + 64;
}

/// The growth of a mirror re-uploaded after its tail filled up (deriveDelta).
inline constexpr double mirror_reupload_growth = 1.5;

/// The internal knobs of a slack_csr hypergraph (none of them is public; ADR 0086).
struct knobs {
  /// The line graph's entry headroom (H-SOSP's entryHeadroom: the build, every rebuild; its
  /// hsospBench passes 1.6). Below 1 or NaN is refused (buildCsr).
  double entry_headroom = 1.6;
  /// The node capacity of the node-indexed arrays (H-SOSP's maxNodes = Caps::maxHyperedges);
  /// 0: the id bound at construction plus room, grown by half when an id passes it.
  std::int64_t node_capacity = 0;
  /// The largest candidate pair count of one batch's line-graph delta (deriveDelta refuses more
  /// than INT_MAX); lowered by the planted-limit test.
  std::int64_t max_candidates = INT_MAX;
  /// The largest id bound (the line graph's int32 node ids, MOSP_ESCHER's int hyperedge ids): a
  /// batch whose ids pass it is a capacity_error; lowered by the over-limit test.
  std::int64_t max_ids = INT_MAX;
  /// The largest entry count of the line graph and the incidence mirror's value arrays (their
  /// int32 counts): a build, rebuild or re-upload past it is a capacity_error; lowered by the
  /// over-limit test.
  std::int64_t max_entries = INT_MAX;
};

}  // namespace dyng::detail::hyper::slack
