// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file names.cpp
 * @brief to_string() of the core enumerations (engine, determinism, memory_space, copy_policy):
 *        the enumerators' own names, which the manifests, the CLI and the Python layer use.
 */
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>

#include <string_view>

namespace dyng {

std::string_view to_string(engine e) noexcept {
  switch (e) {
    case engine::automatic:
      return "automatic";
    case engine::fused:
      return "fused";
    case engine::operators:
      return "operators";
  }
  return "unknown";
}

std::string_view to_string(determinism d) noexcept {
  switch (d) {
    case determinism::bitwise:
      return "bitwise";
    case determinism::exact_value:
      return "exact_value";
    case determinism::tolerance:
      return "tolerance";
  }
  return "unknown";
}

std::string_view to_string(memory_space space) noexcept {
  switch (space) {
    case memory_space::host:
      return "host";
    case memory_space::pinned_host:
      return "pinned_host";
    case memory_space::device:
      return "device";
    case memory_space::managed:
      return "managed";
  }
  return "unknown";
}

std::string_view to_string(copy_policy policy) noexcept {
  switch (policy) {
    case copy_policy::allow:
      return "allow";
    case copy_policy::warn:
      return "warn";
    case copy_policy::error:
      return "error";
  }
  return "unknown";
}

}  // namespace dyng
