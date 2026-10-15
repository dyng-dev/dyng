# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of triad_count (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_triad_count_tests
  SOURCES triad_count_test.cpp
  LABELS cpu triad_count
)
