// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file strict_budgets_environment.hpp
 * @brief A GoogleTest environment that arms strict budgets (invariant I9) for a whole test
 *        executable: in a DYNG_DEBUG_BUDGETS build any update whose algorithm work exceeds its
 *        budget throws internal_error in every test, not only in conformance check C8 and the
 *        tests that open a strict_budgets_scope (a Debug build of the library only logs an excess,
 *        which ctest hides). The M7 review found mosp's CUDA updates one synchronization over their
 *        budget whenever the vertex set grew, logged as warnings by dozens of tests.
 *
 * Register it once per executable, at namespace scope of one of its sources:
 *
 *     const auto* const strict = ::testing::AddGlobalTestEnvironment(
 *         new dyng::test::strict_budgets_environment);
 */
#pragma once

#include "framework/budgets.hpp"

#include <gtest/gtest.h>

namespace dyng::test {

/// Arms strict budgets before the first test and restores the previous setting after the last.
class strict_budgets_environment : public ::testing::Environment {
 public:
  /// Arm strict budgets.
  void SetUp() override {
    previous_ = detail::framework::strict_budgets();
    detail::framework::set_strict_budgets(true);
  }
  /// Restore the previous setting.
  void TearDown() override {
    detail::framework::set_strict_budgets(previous_);
  }

 private:
  bool previous_ = false;
};

}  // namespace dyng::test
