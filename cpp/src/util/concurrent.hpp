// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/csrGraph.cpp (runConcurrently)
/**
 * @file concurrent.hpp
 * @brief Run a few independent jobs (e.g. reading the files of one input) concurrently.
 *
 * Like MOSP's runConcurrently(), the jobs run on the threads of an OpenMP parallel region when
 * OpenMP is enabled: with OMP_PROC_BIND / OMP_PLACES set, libgomp pins the initial thread to one
 * place, and std::thread / std::async threads inherit that single-place mask, so they would all
 * share one core; OpenMP threads are placed on distinct places. Without OpenMP the jobs run on
 * std::async threads. Inside an enclosing parallel region (nested parallelism off) the jobs run
 * one after the other, with the same results.
 */
#pragma once

#include <cstddef>
#include <exception>
#include <functional>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#else
#include <future>
#endif

namespace dyng::detail {

/**
 * @brief Run every job, concurrently where possible, and wait for all of them.
 * @param[in] jobs The jobs; each must be safe to run concurrently with the others.
 * @throws The exception of the FIRST job (in the order of `jobs`) that threw, after every job has
 *         finished, so the reported error does not depend on thread timing.
 */
inline void run_concurrently(const std::vector<std::function<void()>>& jobs) {
  const auto count = static_cast<std::ptrdiff_t>(jobs.size());
  std::vector<std::exception_ptr> errors(jobs.size());
  const auto run_one = [&](std::ptrdiff_t i) {
    try {
      jobs[static_cast<std::size_t>(i)]();
    } catch (...) {
      errors[static_cast<std::size_t>(i)] = std::current_exception();
    }
  };
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 1) num_threads(static_cast<int>(count))
  for (std::ptrdiff_t i = 0; i < count; ++i) {
    run_one(i);
  }
#else
  std::vector<std::future<void>> running;
  running.reserve(jobs.size());
  for (std::ptrdiff_t i = 1; i < count; ++i) {
    running.push_back(std::async(std::launch::async, run_one, i));
  }
  if (count > 0) {
    run_one(0);
  }
  for (auto& f : running) {
    f.get();
  }
#endif
  for (const std::exception_ptr& e : errors) {
    if (e) {
      std::rethrow_exception(e);
    }
  }
}

}  // namespace dyng::detail
