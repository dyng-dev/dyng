// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// The conformance kit for label_propagation (PLAN Section 8.2). Its axis of float slotted graphs
// with vertex batches arrives in M10a step 5 (label_propagation_traits.hpp); until then this suite
// states that it is not run.
#include <gtest/gtest.h>

TEST(LabelPropagationConformance, KitAxisArrivesInStep5) {
  GTEST_SKIP() << "the conformance kit's float slotted axis with vertex batches arrives in M10a "
                  "step 5 (ADR 0072); the hand cases and the replays check label_propagation";
}
