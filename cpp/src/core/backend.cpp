// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file backend.cpp
 * @brief Backend availability.
 */
#include "core/cuda_runtime.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>

namespace dyng {

bool backend_available(backend b) noexcept {
  switch (b) {
    case backend::sequential:
      return true;
    case backend::openmp:
      return DYNG_HAS_OPENMP != 0;
    case backend::cuda:
      return DYNG_HAS_CUDA != 0 && detail::cuda_device_count() > 0;
  }
  return false;
}

backend default_backend() noexcept {
  if (backend_available(backend::cuda)) {
    return backend::cuda;
  }
  if (backend_available(backend::openmp)) {
    return backend::openmp;
  }
  return backend::sequential;
}

std::string_view to_string(backend b) noexcept {
  switch (b) {
    case backend::sequential:
      return "sequential";
    case backend::openmp:
      return "openmp";
    case backend::cuda:
      return "cuda";
  }
  return "unknown";
}

}  // namespace dyng
