// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file error.cpp
 * @brief Helpers of the error macros.
 */
#include <dyng/core/error.hpp>

#include <cstring>
#include <string>

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

void throw_host_allocation_failure(const std::string& context, const char* cause) {
  std::string message;
  try {
    message = "dyng: " + context + ": host memory allocation failed (" +
              (cause != nullptr ? cause : "unknown") + ")";
  } catch (...) {
    throw out_of_memory_error("dyng: host memory allocation failed");
  }
  throw out_of_memory_error(message);
}

}  // namespace dyng::detail
