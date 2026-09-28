// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>

#include <gtest/gtest.h>

TEST(Backend, SequentialIsAlwaysAvailable) {
  EXPECT_TRUE(dyng::backend_available(dyng::backend::sequential));
}

TEST(Backend, OpenmpAvailabilityMatchesConfig) {
  EXPECT_EQ(dyng::backend_available(dyng::backend::openmp), DYNG_HAS_OPENMP != 0);
}

TEST(Backend, CudaUnavailableWithoutCudaBuild) {
  if (DYNG_HAS_CUDA) {
    GTEST_SKIP() << "CUDA is built; availability depends on the visible devices (gpu tests)";
  }
  EXPECT_FALSE(dyng::backend_available(dyng::backend::cuda));
}

TEST(Backend, DefaultPrefersCudaThenOpenmpThenSequential) {
  dyng::backend expected = DYNG_HAS_OPENMP ? dyng::backend::openmp : dyng::backend::sequential;
  if (dyng::backend_available(dyng::backend::cuda)) {
    expected = dyng::backend::cuda;
  }
  EXPECT_EQ(dyng::default_backend(), expected);
}

TEST(Backend, Names) {
  EXPECT_EQ(dyng::to_string(dyng::backend::sequential), "sequential");
  EXPECT_EQ(dyng::to_string(dyng::backend::openmp), "openmp");
  EXPECT_EQ(dyng::to_string(dyng::backend::cuda), "cuda");
}
