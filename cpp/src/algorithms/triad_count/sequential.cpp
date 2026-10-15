// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sequential.cpp
 * @brief triad_count: the sequential backend's full count (host_count.hpp), the recount of
 *        compute() and, in this release, of update() (PLAN Section 4.8, file 6).
 */
#include "algorithms/triad_count/host_count.hpp"
#include "algorithms/triad_count/problem.hpp"

namespace dyng::detail {

void triad_count_sequential(triad_count::pattern kind, const hyper::row_views& views,
                            triad_bins& counts) {
  counts.fill(0);
  const int n = triad_host::num_sources(kind, views);
  if (n > 0) {
    triad_host::count_sources(kind, views, 1, n, counts);
  }
}

}  // namespace dyng::detail
