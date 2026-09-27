// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file backend.hpp
 * @brief The execution backends and their run-time availability.
 * @ingroup core
 */
#pragma once

#include <cstdint>
#include <string_view>

namespace dyng {

/**
 * @brief An execution backend, chosen at run time through dyng::resources.
 *
 * `sequential` is the mandatory reference backend of every algorithm.
 * @ingroup core
 */
enum class backend : std::uint8_t {
  sequential,  ///< one host thread; the reference implementation
  openmp,      ///< host threads through OpenMP
  cuda,        ///< an NVIDIA GPU
};

/**
 * @brief Whether a backend can be used in this process.
 * @param[in] b The backend.
 * @return True if `b` was compiled in and, for cuda, a device is visible.
 * @ingroup core
 */
[[nodiscard]] bool backend_available(backend b) noexcept;

/**
 * @brief The backend a default-constructed dyng::resources uses.
 * @return cuda if available, else openmp if available, else sequential.
 * @ingroup core
 */
[[nodiscard]] backend default_backend() noexcept;

/**
 * @brief The lower-case name of a backend ("sequential", "openmp", "cuda").
 * @param[in] b The backend.
 * @return A static string naming `b`.
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(backend b) noexcept;

}  // namespace dyng
