// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// The conformance kit for mosp (PLAN Section 8.2): C0-C12 on every backend and graph type.
#include "conformance/conformance.hpp"
#include "support/strict_budgets_environment.hpp"

DYNG_CONFORMANCE_SUITE(mosp);

// Every check runs with strict budgets (I9), not only C8 (the M7 review).
namespace {
const auto* const strict_budgets =
    ::testing::AddGlobalTestEnvironment(new dyng::test::strict_budgets_environment);
}  // namespace
