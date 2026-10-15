// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// The conformance kit for triad_count (PLAN Section 8.2): C0-C12 on every backend and hypergraph
// type for pattern h_motif (the registered tag); StatHyper's types 1-3 run under test-only tags
// with the manifest's entry in triad_count_types_conformance_test.cpp (triad_count_traits.hpp).
// Two translation units: GCC 12 at -O3 with _FORTIFY_SOURCE=3 reports a false -Wnonnull in
// std::vector's assignment when the four suites share one.
#include "conformance/conformance.hpp"

DYNG_CONFORMANCE_SUITE(triad_count);
