// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file instantiate.hpp
 * @brief The explicitly instantiated (vertex_t, edge_t, weight_t) combinations of the graph
 *        module (PLAN Section 4.4.3).
 */
#pragma once

#include <dyng/core/types.hpp>

#include <cstdint>

/// Calls `X(vertex_t, edge_t, weight_t)` once per instantiated graph type combination.
#define DYNG_FOR_EACH_GRAPH_TYPE(X)           \
  X(std::int32_t, std::int32_t, std::int32_t) \
  X(std::int32_t, std::int64_t, std::int32_t) \
  X(std::int64_t, std::int64_t, std::int32_t)

/// Calls `X(vertex_t, edge_t, weight_t)` once per instantiated graph type WITHOUT weights (the
/// `unweighted` rows of PLAN Section 4.4.3; cycle_count). The graph container, its host apply and
/// the algorithms that accept unweighted graphs instantiate these in addition to
/// DYNG_FOR_EACH_GRAPH_TYPE.
#define DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(X)      \
  X(std::int32_t, std::int32_t, ::dyng::unweighted) \
  X(std::int32_t, std::int64_t, ::dyng::unweighted)

/// Calls `X(vertex_t, weight_t)` once per instantiated (vertex, weight) combination.
#define DYNG_FOR_EACH_VERTEX_WEIGHT_TYPE(X) \
  X(std::int32_t, std::int32_t)             \
  X(std::int64_t, std::int32_t)
