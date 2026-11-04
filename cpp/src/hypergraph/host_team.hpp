// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file host_team.hpp
 * @brief The OpenMP team of the slack_csr stores' host regions (M11a step 3, ADR 0089).
 *
 * The host regions of slack_csr storage (the line-graph builds, Step 0 of the incidence edits,
 * the H2V and v2h edits, the host store's pair delta and CSR apply) run on one fixed team of at
 * most host_team_cap threads, or on one thread when the work is small:
 * - a batch's regions are short: on the shared 28-core host a team of every hardware thread (56)
 *   spent more in its fork and join than in the work (a 25K-edit batch's Step 0 took 12 ms on 56
 *   threads, under 1 ms on 16), and even the line-graph build of geology.hg was faster on 16
 *   (0.75 s) than on 56 (0.92 s);
 * - the size is fixed because libgomp ends the surplus threads of its pool when a smaller team
 *   starts and starts new ones for a larger team: mixing team sizes cost about 20-30 ms in the
 *   first batch after a 56-thread build.
 * A region whose `if` clause is false runs on the calling thread and leaves the pool as it is.
 */
#pragma once

#include <algorithm>
#include <cstdint>

namespace dyng::detail::hyper {

/// The largest team of a slack_csr store's host regions.
inline constexpr int host_team_cap = 16;
/// The work below which a batch's host region runs on one thread.
inline constexpr std::int64_t host_team_min_work = 4096;

/// @brief The team of the slack_csr host regions. @param[in] threads The resources' host threads.
/// @return min(threads, host_team_cap), at least 1.
inline int host_team_size(int threads) noexcept {
  return std::max(1, std::min(threads, host_team_cap));
}

/// @brief The team of one batch region over `work` items: the host team, or 1 for small work.
/// @param[in] threads The resources' host threads. @param[in] work The items.
/// @return The team size.
inline int host_team(int threads, std::int64_t work) noexcept {
  return work < host_team_min_work ? 1 : host_team_size(threads);
}

}  // namespace dyng::detail::hyper
