// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file io_instantiate.hpp
 * @brief Type lists for the explicit instantiations of the io module.
 */
#pragma once

#include "graph/instantiate.hpp"

#include <cstdint>

/// Calls `X(vertex_t)` once per instantiated vertex id type.
#define DYNG_FOR_EACH_VERTEX_TYPE(X) \
  X(std::int32_t)                    \
  X(std::int64_t)

/// Calls `X(vertex_t, weight_t)` once per instantiated (vertex, unweighted) combination.
#define DYNG_FOR_EACH_VERTEX_TYPE_UNWEIGHTED(X) \
  X(std::int32_t, ::dyng::unweighted)           \
  X(std::int64_t, ::dyng::unweighted)

/// Calls `X(distance_t)` once per instantiated distance type.
#define DYNG_FOR_EACH_DISTANCE_TYPE(X) X(std::int64_t)
