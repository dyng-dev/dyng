# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of dynamic_bfs (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_dynamic_bfs_tests
  SOURCES dynamic_bfs_test.cpp
  LABELS cpu dynamic_bfs
)
