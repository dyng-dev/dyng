// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file openmp.cpp
 * @brief triad_count: the OpenMP backend's full count (host_count.hpp): the source keys in
 *        parallel (dynamic schedule), one set of bins per thread, summed after the region (PLAN
 *        Section 4.8, file 7).
 */
#include "algorithms/triad_count/host_count.hpp"
#include "algorithms/triad_count/problem.hpp"
#include "core/resources_access.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>

#include <algorithm>
#include <cstddef>
#include <exception>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail {

void triad_count_openmp(const resources& res, triad_count::pattern kind,
                        const hyper::row_views& views, triad_bins& counts) {
#if DYNG_HAS_OPENMP
  const int threads = std::max(resources_access::host_threads(res), 1);
  const int n = triad_host::num_sources(kind, views);
  counts.fill(0);
  // Each thread counts into bins on its stack (nothing is allocated) and adds them to `counts`
  // once. An exception must not leave a parallel region: the first one is rethrown after it.
  std::exception_ptr failure;
#pragma omp parallel num_threads(threads)
  {
    triad_bins mine{};
    bool ok = true;
#pragma omp for schedule(dynamic, 64) nowait
    for (int a = 1; a <= n; ++a) {
      if (!ok) {
        continue;
      }
      try {
        triad_host::count_source(kind, views, a, mine);
      } catch (...) {
        ok = false;
#pragma omp critical(dyng_triad_count_openmp)
        if (!failure) {
          failure = std::current_exception();
        }
      }
    }
#pragma omp critical(dyng_triad_count_openmp)
    {
      try {
        for (std::size_t k = 0; ok && k < counts.size(); ++k) {
          triad_host::add_checked(counts[k], mine[k]);
        }
      } catch (...) {
        if (!failure) {
          failure = std::current_exception();
        }
      }
    }
  }
  if (failure) {
    std::rethrow_exception(failure);
  }
#else
  (void)res;
  triad_count_sequential(kind, views, counts);
#endif
}

}  // namespace dyng::detail
