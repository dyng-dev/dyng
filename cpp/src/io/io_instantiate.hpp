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

/// Calls `X(distance_t)` once per instantiated distance type.
#define DYNG_FOR_EACH_DISTANCE_TYPE(X) X(std::int64_t)
