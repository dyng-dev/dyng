# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The hand cases of algorithm_template (PLAN Section 4.8, file 12); its conformance suite is built
# by cpp/tests/CMakeLists.txt.

dyng_add_test(
  NAME dyng_algorithm_template_tests
  SOURCES algorithm_template_test.cpp
  LABELS cpu algorithm_template
)
