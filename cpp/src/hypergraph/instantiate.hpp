// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file instantiate.hpp
 * @brief The explicitly instantiated (vertex_t, weight_t) combinations of the hypergraph module
 *        (PLAN Section 4.4.3: hypergraph, triad_count, hyper_sssp).
 */
#pragma once

#include <dyng/core/types.hpp>

#include <cstdint>

/// Calls `X(vertex_t, weight_t)` once per instantiated hypergraph type combination:
/// (int32, unweighted) and (int32, int32), in every DYNG_INDEX_TYPES configuration.
#define DYNG_FOR_EACH_HYPERGRAPH_TYPE(X) \
  X(std::int32_t, ::dyng::unweighted)    \
  X(std::int32_t, std::int32_t)
