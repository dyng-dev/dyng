// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file version.cpp
 * @brief The version of the compiled library.
 */
#include <dyng/version.hpp>

namespace dyng {

version_info library_version() noexcept {
  return header_version();
}

}  // namespace dyng
