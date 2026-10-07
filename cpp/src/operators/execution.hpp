// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file execution.hpp
 * @brief The host executors and the atomics of the framework operators (PLAN Section 4.5.3):
 *        write the work on one element once, as a functor, and run it on any backend.
 *
 * An executor runs `f(i)` for every i in [0, n) on one backend and moves scalars and arrays
 * between the host and the backend's memory:
 *
 *     template <typename exec_t>
 *     void double_all(const exec_t& exec, std::int64_t* values, std::int64_t n) {
 *       exec.for_each(n, double_one{values});      // struct double_one { std::int64_t* values;
 *     }                                            //   DYNG_HD void operator()(std::int64_t i)
 *                                                  //   const { values[i] *= 2; } };
 *
 * | Executor | Header | for_each |
 * |---|---|---|
 * | `sequential_exec` | this header | a plain loop (the reference order) |
 * | `openmp_exec` | this header | `#pragma omp parallel for` with the handle's threads |
 * | `cuda_exec` | operators/cuda_execution.cuh (CUDA translation units only) | a grid-stride kernel on the handle's stream |
 *
 * Every executor has the same members: `for_each(n, f)`, `read(p)` (a scalar the work wrote, read
 * on the host: a host synchronization on CUDA), `write(p, v)`, `fill(p, n, v)`, `copy(dst, src,
 * n)` (within the backend's memory) and `upload(dst, host, n)` (host memory into the backend's
 * memory). A functor is a small struct of raw pointers whose call operator is `DYNG_HD`; it may
 * use the atomics below, which compile to the CUDA atomics on the device and to the GCC/Clang
 * `__atomic` builtins on the host.
 *
 * The rule of two (PLAN Section 4.5.3): the executors and atomics are here because both tutorial
 * algorithms (dynamic_bfs and triangle_delta) are written with them. Operators with one user stay
 * in that algorithm's folder (dynamic_bfs's frontier push, triangle_delta's sorted intersection).
 */
#pragma once

#include <dyng/config.hpp>
#include <dyng/core/resources.hpp>

#include <algorithm>
#include <cstdint>
#include <cstring>

#if DYNG_HAS_OPENMP && defined(_OPENMP)
#include <omp.h>
#endif

#if defined(__CUDACC__)
/// Marks a function (a functor's call operator) that runs on the host and on the device.
#define DYNG_HD __host__ __device__
#else
/// Marks a function (a functor's call operator) that runs on the host and on the device.
#define DYNG_HD
#endif

namespace dyng::detail::operators {

// ------------------------------------------------------------------------------------------------
// Atomics (relaxed; usable in DYNG_HD functors on every backend)
// ------------------------------------------------------------------------------------------------

/**
 * @brief Atomically lower `*p` to `value` if `value` is smaller (unsigned 64-bit comparison).
 * @param[in,out] p     The word.
 * @param[in]     value The candidate.
 * @return The value of `*p` before the call.
 */
DYNG_HD inline std::uint64_t atomic_min(std::uint64_t* p, std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(
      atomicMin(reinterpret_cast<unsigned long long*>(p), static_cast<unsigned long long>(value)));
#else
  std::uint64_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (value < old &&
         !__atomic_compare_exchange_n(p, &old, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
  return old;
#endif
}

/**
 * @brief Atomically add `value` to `*p`.
 * @param[in,out] p     The counter.
 * @param[in]     value The increment.
 * @return The value of `*p` before the call.
 */
DYNG_HD inline std::uint64_t atomic_add(std::uint64_t* p, std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  return static_cast<std::uint64_t>(
      atomicAdd(reinterpret_cast<unsigned long long*>(p), static_cast<unsigned long long>(value)));
#else
  return __atomic_fetch_add(p, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Atomically replace `*p` by `value`.
 * @param[in,out] p     The word.
 * @param[in]     value The new value.
 * @return The value of `*p` before the call.
 */
DYNG_HD inline std::uint32_t atomic_exchange(std::uint32_t* p, std::uint32_t value) {
#if defined(__CUDA_ARCH__)
  return atomicExch(reinterpret_cast<unsigned int*>(p), static_cast<unsigned int>(value));
#else
  return __atomic_exchange_n(p, value, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Read a word that other threads may change in the same pass (a relaxed atomic load).
 * @param[in] p The word.
 * @return Its value.
 */
DYNG_HD inline std::uint64_t atomic_load(const std::uint64_t* p) {
#if defined(__CUDA_ARCH__)
  return *static_cast<const volatile std::uint64_t*>(p);
#else
  return __atomic_load_n(p, __ATOMIC_RELAXED);
#endif
}

/**
 * @brief Store a word that other threads may read in the same pass (a relaxed atomic store).
 * @param[out] p     The word.
 * @param[in]  value The value.
 */
DYNG_HD inline void atomic_store(std::uint64_t* p, std::uint64_t value) {
#if defined(__CUDA_ARCH__)
  *static_cast<volatile std::uint64_t*>(p) = value;
#else
  __atomic_store_n(p, value, __ATOMIC_RELAXED);
#endif
}

// ------------------------------------------------------------------------------------------------
// Host executors
// ------------------------------------------------------------------------------------------------

/**
 * @brief Runs the work on the calling thread, in index order (the sequential backend: the
 *        reference every other backend is compared with).
 */
class sequential_exec {
 public:
  /// The memory the work reads and writes is host memory.
  static constexpr bool on_device = false;

  /**
   * @brief An executor for the resources of a call (unused: one thread).
   * @param[in] res The resources.
   */
  explicit sequential_exec(const resources& res) noexcept {
    (void)res;
  }

  /**
   * @brief Run `f(i)` for every i in [0, n), in order.
   * @tparam function_t A callable `void(std::int64_t)`.
   * @param[in] n Number of elements.
   * @param[in] f The work on one element.
   */
  template <typename function_t>
  void for_each(std::int64_t n, function_t f) const {
    for (std::int64_t i = 0; i < n; ++i) {
      f(i);
    }
  }

  /**
   * @brief Read a scalar the work wrote.
   * @tparam value_t Element type.
   * @param[in] p The scalar.
   * @return Its value.
   */
  template <typename value_t>
  value_t read(const value_t* p) const {
    return *p;
  }

  /**
   * @brief Write a scalar the work will read.
   * @tparam value_t Element type.
   * @param[out] p     The scalar.
   * @param[in]  value The value.
   */
  template <typename value_t>
  void write(value_t* p, value_t value) const {
    *p = value;
  }

  /**
   * @brief Set `n` elements to `value`.
   * @tparam value_t Element type.
   * @param[out] p     The array.
   * @param[in]  n     Number of elements.
   * @param[in]  value The value.
   */
  template <typename value_t>
  void fill(value_t* p, std::int64_t n, value_t value) const {
    std::fill(p, p + n, value);
  }

  /**
   * @brief Copy `n` elements within the backend's memory.
   * @tparam value_t Element type.
   * @param[out] dst The destination.
   * @param[in]  src The source.
   * @param[in]  n   Number of elements.
   */
  template <typename value_t>
  void copy(value_t* dst, const value_t* src, std::int64_t n) const {
    if (n > 0) {
      std::memcpy(dst, src, static_cast<std::size_t>(n) * sizeof(value_t));
    }
  }

  /**
   * @brief Copy `n` elements of host memory into the backend's memory.
   * @tparam value_t Element type.
   * @param[out] dst The destination (the backend's memory).
   * @param[in]  src The source (host memory).
   * @param[in]  n   Number of elements.
   */
  template <typename value_t>
  void upload(value_t* dst, const value_t* src, std::int64_t n) const {
    copy(dst, src, n);
  }
};

/**
 * @brief Runs the work on the OpenMP threads of the resources handle (the sequential executor
 *        when the library is built without OpenMP).
 */
class openmp_exec : public sequential_exec {
 public:
  /**
   * @brief An executor with the thread count of `res`.
   * @param[in] res The resources (resources::openmp(threads)).
   */
  explicit openmp_exec(const resources& res) noexcept
      : sequential_exec(res), threads_(std::max(res.num_threads(), 1)) {}

  /**
   * @brief Run `f(i)` for every i in [0, n) on the handle's threads (dynamic schedule: the
   *        elements of graph work differ in cost).
   * @tparam function_t A callable `void(std::int64_t)` that is safe to run concurrently for
   *                    distinct i (shared words only through the atomics of this header).
   * @param[in] n Number of elements.
   * @param[in] f The work on one element.
   */
  template <typename function_t>
  void for_each(std::int64_t n, function_t f) const {
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 64) num_threads(threads_)
    for (std::int64_t i = 0; i < n; ++i) {
      f(i);
    }
#else
    sequential_exec::for_each(n, f);
#endif
  }

 private:
  int threads_ = 1;
};

}  // namespace dyng::detail::operators
