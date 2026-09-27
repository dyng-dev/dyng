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
