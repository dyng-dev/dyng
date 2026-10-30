# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of hyper_sssp (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_hyper_sssp_tests
  SOURCES hyper_sssp_test.cpp
  LABELS cpu hyper_sssp
)
