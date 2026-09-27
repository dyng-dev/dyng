// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file gtest_helpers.hpp
 * @brief Shared GoogleTest helpers: backend lists and skip macros.
 */
#pragma once

#include <dyng/config.hpp>
#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <string>
#include <type_traits>
#include <vector>

#if defined(__SANITIZE_ADDRESS__) || defined(__SANITIZE_THREAD__)
#define DYNG_TEST_SANITIZED 1
#elif defined(__has_feature)
#if __has_feature(address_sanitizer) || __has_feature(thread_sanitizer)
#define DYNG_TEST_SANITIZED 1
#endif
#endif
#ifndef DYNG_TEST_SANITIZED
/// 1 when the test is built with AddressSanitizer or ThreadSanitizer, else 0.
#define DYNG_TEST_SANITIZED 0
#endif

namespace dyng::test {

/// The host backends compiled into this build (sequential always, openmp if built).
inline std::vector<backend> host_backends() {
  std::vector<backend> out{backend::sequential};
  if (backend_available(backend::openmp)) {
    out.push_back(backend::openmp);
  }
  return out;
}

/// The backends of the parameterized algorithm suites. A suite source is compiled twice: into the
/// host test executable (label cpu: the host backends) and, with DYNG_TEST_CUDA=1, into the CUDA
/// test executable (label gpu: the cuda backend, if a device is visible; empty otherwise, so the
/// suite is skipped). Suites that compare backends take reference_backends() first.
inline std::vector<backend> suite_backends() {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  if (backend_available(backend::cuda)) {
    return {backend::cuda};
  }
  return {};
#else
  return host_backends();
#endif
}

/// The backends a cross-backend comparison runs on: the host backends (sequential first, the
/// reference), plus cuda in the CUDA test executable when a device is visible.
inline std::vector<backend> comparison_backends() {
  std::vector<backend> out = host_backends();
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  if (backend_available(backend::cuda)) {
    out.push_back(backend::cuda);
  }
#endif
  return out;
}

/// A host copy of an array in any memory space (device arrays are copied on the default stream of
/// their device and waited for).
template <typename value_t>
std::vector<std::remove_const_t<value_t>> host_copy(array_view<value_t> v) {
  if (is_host_accessible(v.space())) {
    return std::vector<std::remove_const_t<value_t>>(v.begin(), v.end());
  }
  return to_vector(resources::cuda(v.device() < 0 ? 0 : v.device()), v);
}

/// Resources for a backend with its default settings.
inline resources make_resources(backend b, int num_threads = 0) {
  switch (b) {
    case backend::sequential:
      return resources::sequential();
    case backend::openmp:
      return resources::openmp(num_threads);
    case backend::cuda:
      return resources::cuda();
  }
  return resources::sequential();
}

/// Name generator for typed test suites: the type's index, as GoogleTest's default. Passing it
/// explicitly keeps TYPED_TEST_SUITE(suite, types, name_generator) valid C++17: with only two
/// arguments the macro's variadic part is empty, which Clang reports as a C++20 extension.
struct type_index_name {
  template <typename type_t>
  static std::string GetName(int index) {  // NOLINT(readability-identifier-naming): gtest API
    return std::to_string(index);
  }
};

/// Name printer for value-parameterized tests over backends.
struct backend_name {
  std::string operator()(const ::testing::TestParamInfo<backend>& info) const {
    return std::string(to_string(info.param));
  }
};

}  // namespace dyng::test

/// Skip the current test if the CUDA backend is not available.
#define DYNG_SKIP_IF_NO_CUDA()                                    \
  do {                                                            \
    if (!::dyng::backend_available(::dyng::backend::cuda)) {      \
      GTEST_SKIP() << "CUDA backend not available in this build"; \
    }                                                             \
  } while (false)

/// Skip the current test if the OpenMP backend is not available.
#define DYNG_SKIP_IF_NO_OPENMP()                                    \
  do {                                                              \
    if (!::dyng::backend_available(::dyng::backend::openmp)) {      \
      GTEST_SKIP() << "OpenMP backend not available in this build"; \
    }                                                               \
  } while (false)
