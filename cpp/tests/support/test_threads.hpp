// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file test_threads.hpp
 * @brief The OpenMP thread count of a test's openmp resources: the test's default, or
 *        DYNG_TEST_OPENMP_THREADS (a positive integer) when it is set (the informational timing
 *        of ADR 0083 runs the tutorial algorithms' tests at 1 and 8 threads).
 */
#pragma once

#include <cstdlib>
#include <string>

namespace dyng::test {

/**
 * @brief The thread count of a test's openmp resources.
 * @param[in] fallback The test's own count.
 * @return DYNG_TEST_OPENMP_THREADS when it is a positive integer, else `fallback`.
 */
inline int openmp_test_threads(int fallback) {
  const char* value = std::getenv("DYNG_TEST_OPENMP_THREADS");
  if (value == nullptr || *value == '\0') return fallback;
  try {
    const int n = std::stoi(value);
    return n > 0 ? n : fallback;
  } catch (...) {
    return fallback;
  }
}

}  // namespace dyng::test
