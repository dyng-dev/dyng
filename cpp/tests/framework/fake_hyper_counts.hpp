// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fake_hyper_counts.hpp
 * @brief The counts of the hypergraph fakes (fake_hyper_problems.hpp) over the internal row
 *        views: on the host backends here, on CUDA in fake_hyper_kernels.cu (one host
 *        synchronization per count). Kept apart from the problems so that the .cu file needs no
 *        framework header.
 */
#pragma once

#include "core/budget_counters.hpp"
#include "framework/scratch_buffer.hpp"
#include "framework/workspace.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/resources.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::test::framework_fakes {

/// The scratch of the hypergraph fakes (pooled, as a real workspace).
struct fake_hyper_workspace final : detail::pooled_workspace {
  std::vector<std::uint8_t> host_flags;                ///< host backends: frontier keys marked
  std::vector<std::int64_t> host_ids;                  ///< the planted bug's frontier (host)
  detail::scratch_buffer<std::uint8_t> flags;          ///< cuda: frontier keys marked
  detail::scratch_buffer<std::int64_t> ids;            ///< cuda: an uploaded frontier
  detail::scratch_buffer<unsigned long long> counter;  ///< cuda: the count

  [[nodiscard]] std::size_t bytes() const noexcept override {
    return host_flags.capacity() + host_ids.capacity() * sizeof(std::int64_t) + flags.bytes() +
           ids.bytes() + counter.bytes();
  }
};

/// A frontier of hyperedge ids: a pointer into the staged batch, the applied record or the
/// workspace (never owning, so it is default-constructible without allocating).
struct id_frontier {
  const std::int64_t* ids = nullptr;  ///< the ids
  std::int64_t size = 0;              ///< how many
  bool on_device = false;             ///< in device memory (cuda)

  /// Whether the frontier holds no id.
  [[nodiscard]] bool empty() const noexcept {
    return size == 0;
  }
};

/// cuda: the pairs of the rows of `frontier` in `h2h` owned by them (min_member); the frontier is
/// uploaded first when it is on the host (fake_hyper_kernels.cu).
std::int64_t device_owned_pairs(const resources& res, const detail::escher::row_view& h2h,
                                const id_frontier& frontier, fake_hyper_workspace& ws);
/// cuda: the sum of the squared row lengths of `frontier` in `h2v`.
std::int64_t device_squares(const resources& res, const detail::escher::row_view& h2v,
                            const id_frontier& frontier, fake_hyper_workspace& ws);

/// Host backends: the pairs of the rows of `frontier` in `h2h` owned by them (min_member).
inline std::int64_t host_owned_pairs(const detail::escher::row_view& h2h,
                                     const id_frontier& frontier, fake_hyper_workspace& ws) {
  const auto keys = static_cast<std::size_t>(h2h.max_key) + 1;
  if (ws.host_flags.capacity() < keys) {
    detail::note_reservation();  // a deliberate growth of a reusable array (I9)
    ws.host_flags.reserve(keys);
  }
  ws.host_flags.assign(keys, 0);
  for (std::int64_t i = 0; i < frontier.size; ++i) {
    const std::int64_t key = frontier.ids[i] + 1;
    if (key >= 1 && key <= h2h.max_key) ws.host_flags[static_cast<std::size_t>(key)] = 1;
  }
  std::int64_t count = 0;
  for (std::int64_t i = 0; i < frontier.size; ++i) {
    const std::int64_t key = frontier.ids[i] + 1;
    if (key < 1 || key > h2h.max_key) continue;
    const int* row = h2h.flat + h2h.loc[key];
    for (int j = 0; j < h2h.len[key]; ++j) {
      const int neighbour = row[j];
      // min_member: a pair of two frontier ids is counted by the smaller one.
      if (ws.host_flags[static_cast<std::size_t>(neighbour)] == 0 || key < neighbour) ++count;
    }
  }
  return count;
}

/// Host backends: the sum of the squared row lengths of `frontier` in `h2v`.
inline std::int64_t host_squares(const detail::escher::row_view& h2v, const id_frontier& frontier) {
  std::int64_t sum = 0;
  for (std::int64_t i = 0; i < frontier.size; ++i) {
    const std::int64_t key = frontier.ids[i] + 1;
    if (key < 1 || key > h2v.max_key) continue;
    const auto len = static_cast<std::int64_t>(h2v.len[key]);
    sum += len * len;
  }
  return sum;
}

}  // namespace dyng::test::framework_fakes
