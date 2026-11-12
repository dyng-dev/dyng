// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// The conformance kit for hyper_sssp (PLAN Section 8.2): C0-C12 on every backend and
// graph type.
#include "conformance/conformance.hpp"

DYNG_CONFORMANCE_SUITE(hyper_sssp);

#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
// M11b criterion 3: the kit on both CUDA engines (C4 compares them in either suite).
DYNG_CONFORMANCE_SUITE(hyper_sssp_shared_engine);
#endif
