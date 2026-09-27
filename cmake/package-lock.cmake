# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Pinned versions of the dependencies fetched with CPM (CPMUsePackageLock format).
# Change a version here, never at the CPMAddPackage call site.

# GoogleTest (BSD-3-Clause): tests only; never part of the exported targets.
CPMDeclarePackage(
  GTest
  NAME GTest
  VERSION 1.17.0
  GITHUB_REPOSITORY google/googletest
  GIT_TAG v1.17.0
  EXCLUDE_FROM_ALL YES
  SYSTEM YES
  OPTIONS "INSTALL_GTEST OFF" "BUILD_GMOCK ON" "gtest_force_shared_crt ON"
)
