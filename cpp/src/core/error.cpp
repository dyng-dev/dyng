// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file error.cpp
 * @brief Helpers of the error macros.
 */
#include <dyng/core/error.hpp>

#include <cstring>

namespace dyng::detail {

const char* source_basename(const char* path) noexcept {
  if (path == nullptr) {
    return "";
  }
  const char* slash = std::strrchr(path, '/');
  const char* backslash = std::strrchr(path, '\\');
  const char* last = slash > backslash ? slash : backslash;
  return last != nullptr ? last + 1 : path;
}

}  // namespace dyng::detail
