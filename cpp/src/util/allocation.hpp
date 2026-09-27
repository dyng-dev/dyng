// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocation.hpp
 * @brief DYNG_TRANSLATE_ALLOCATION_FAILURE: the handlers of a function-try-block that report host
 *        allocation failures as out_of_memory_error (PLAN Section 4.7.3).
 *
 * Usage, on a public entry point:
 *
 *     void graph::reserve(const resources& res, edge_t edge_capacity) try {
 *       ...
 *     } DYNG_TRANSLATE_ALLOCATION_FAILURE("graph::reserve (", edge_capacity, " edges)")
 *
 * std::bad_alloc and std::length_error (a container sized beyond max_size()) become
 * dyng::out_of_memory_error with the context in the message; every other exception passes
 * through unchanged. The parameters of the function are in scope in the context arguments.
 */
#pragma once

#include <dyng/core/error.hpp>

#include <new>
#include <stdexcept>

// NOLINTBEGIN(cppcoreguidelines-macro-usage)
#define DYNG_TRANSLATE_ALLOCATION_FAILURE(...)                                                 \
  catch (const std::bad_alloc& dyng_bad_alloc) {                                               \
    ::dyng::detail::throw_host_allocation_failure(::dyng::detail::concat_message(__VA_ARGS__), \
                                                  dyng_bad_alloc.what());                      \
  }                                                                                            \
  catch (const std::length_error& dyng_length_error) {                                         \
    ::dyng::detail::throw_host_allocation_failure(::dyng::detail::concat_message(__VA_ARGS__), \
                                                  dyng_length_error.what());                   \
  }
// NOLINTEND(cppcoreguidelines-macro-usage)
