// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file gtest_helpers.hpp
 * @brief Shared GoogleTest helpers: backend lists and skip macros.
 */
#pragma once

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <string>
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
