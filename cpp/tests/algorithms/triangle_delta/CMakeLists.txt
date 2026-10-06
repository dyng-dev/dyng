# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of triangle_delta (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_triangle_delta_tests
  SOURCES triangle_delta_test.cpp
  LABELS cpu triangle_delta
)
