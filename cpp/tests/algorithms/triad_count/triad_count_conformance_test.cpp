// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// The conformance kit for triad_count (PLAN Section 8.2): C0-C12 on every backend and hypergraph
// type, one suite per pattern (triad_count_traits.hpp: h_motif under the registered tag, StatHyper's
// types 1-3 under test-only tags with the manifest's entry).
#include "conformance/conformance.hpp"

DYNG_CONFORMANCE_SUITE(triad_count);
DYNG_CONFORMANCE_SUITE(triad_count_type1);
DYNG_CONFORMANCE_SUITE(triad_count_type2);
DYNG_CONFORMANCE_SUITE(triad_count_type3);
