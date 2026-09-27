# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Third-party dependencies (PLAN Section 7.5). Build-time dependencies are fetched with CPM and
# pinned in package-lock.cmake; with CPM_USE_LOCAL_PACKAGES=ON an installed package is tried
# first (find_package). Dependencies of the public interface are re-found by dyng-config.cmake.

# OpenMP: the OpenMP backends. Linked PRIVATELY into dyng.
if(DYNG_ENABLE_OPENMP)
  find_package(OpenMP REQUIRED COMPONENTS CXX)
endif()

# Threads: the text readers parse files concurrently (std::async). Linked PRIVATELY into dyng.
set(THREADS_PREFER_PTHREAD_FLAG ON)
find_package(Threads REQUIRED)

# Test-only dependencies.
if(DYNG_BUILD_TESTS)
  include("${CMAKE_CURRENT_LIST_DIR}/CPM.cmake")
  CPMUsePackageLock("${CMAKE_CURRENT_LIST_DIR}/package-lock.cmake")
  CPMGetPackage(GTest)
  if(NOT TARGET GTest::gtest_main AND TARGET gtest_main)
    add_library(GTest::gtest ALIAS gtest)
    add_library(GTest::gtest_main ALIAS gtest_main)
  endif()
endif()
