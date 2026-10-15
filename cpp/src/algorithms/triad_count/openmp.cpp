// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file openmp.cpp
 * @brief triad_count: the OpenMP backend (PLAN Section 4.8, file 7), the host engine
 *        (host_engine.hpp) on operators::openmp_exec: a static partition of the sources over the
 *        handle's threads, private bins per thread added in thread order, so every thread count
 *        gives the same counts (ADR 0054). Without OpenMP the executor runs one thread.
 */
#include "algorithms/triad_count/host_engine.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "operators/execution.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>

#include <cstddef>
#include <cstdint>

namespace dyng::detail {

triad::wedge::launch triad_openmp_full(const resources& res, triad_count::pattern kind,
                                       const hyper::row_views& views,
                                       framework::ownership::min_member rule,
                                       std::uint64_t* scratch, triad_bins& counts) {
  return triad_host::count_full(operators::openmp_exec(res), kind, views, rule, scratch, counts);
}

triad::wedge::launch triad_openmp_phase(const resources& res, triad_count::pattern kind,
                                        const hyper::row_views& views, const int* keys, int n,
                                        const unsigned char* flags, framework::sign s,
                                        framework::ownership::min_member rule,
                                        std::uint64_t* scratch, triad_signed_bins& out) {
  return triad_host::count_phase(operators::openmp_exec(res), kind, views, keys, n, flags, s, rule,
                                 scratch, out);
}

std::size_t triad_host_scratch_words(const resources& res) noexcept {
  if (res.get_backend() == backend::openmp) {
    return triad_host::scratch_words(operators::openmp_exec(res));
  }
  return triad_host::scratch_words(operators::sequential_exec(res));
}

}  // namespace dyng::detail
