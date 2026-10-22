// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sequential.cpp
 * @brief triad_count: the sequential backend (PLAN Section 4.8, file 6), the host engine
 *        (host_engine.hpp) on operators::sequential_exec: the full count of compute() and the
 *        frontier phases of update(), one thread, in source order (ADR 0054).
 */
#include "algorithms/triad_count/host_engine.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "operators/execution.hpp"

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

triad::wedge::launch triad_sequential_full(const resources& res, const local_patterns::kernels& k,
                                           const void* op, const hyper::row_views& views,
                                           framework::ownership::min_member rule,
                                           std::uint64_t* scratch, std::uint64_t* counts) {
  return triad_host::count_full(operators::sequential_exec(res), k, op, views, rule, scratch,
                                counts);
}

triad::wedge::launch triad_sequential_phase(const resources& res, const local_patterns::kernels& k,
                                            const void* op, const hyper::row_views& views,
                                            const int* keys, int n, const unsigned char* flags,
                                            framework::sign s,
                                            framework::ownership::min_member rule,
                                            std::uint64_t* scratch, triad_phase_bins& out) {
  return triad_host::count_phase(operators::sequential_exec(res), k, op, views, keys, n, flags, s,
                                 rule, scratch, out);
}

}  // namespace dyng::detail
