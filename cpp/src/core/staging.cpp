// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file staging.cpp
 * @brief The copy policy of implicit input copies (PLAN Section 4.7.1; ADR 0016 item 11).
 */
#include "core/staging.hpp"

#include "core/cuda_runtime.hpp"

#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>

namespace dyng::detail {

namespace {

const char* space_name(memory_space space) noexcept {
  switch (space) {
    case memory_space::host:
      return "host";
    case memory_space::pinned_host:
      return "pinned host";
    case memory_space::device:
      return "device";
    case memory_space::managed:
      return "managed";
  }
  return "unknown";
}

}  // namespace

copy_policy effective_copy_policy(const resources& res) noexcept {
  const copy_policy policy = res.get_copy_policy();
  if (policy == copy_policy::allow && res.get_profiler() != nullptr) {
    return copy_policy::warn;
  }
  return policy;
}

void note_implicit_copy(const resources& res, std::string_view what, memory_space from,
                        memory_space to, std::size_t bytes) {
  const copy_policy policy = effective_copy_policy(res);
  if (policy == copy_policy::error) {
    throw invalid_argument_error(concat_message(
        "dyng: ", what, " is in ", space_name(from), " memory but is read in ", space_name(to),
        " memory, and the copy policy of the resources is copy_policy::error; copy it first "
        "(dyng::to_vector(), dyng::to_space()) or allow the copy with res.set_copy_policy()"));
  }
  const log_level level = policy == copy_policy::warn ? log_level::warn : log_level::debug;
  if (log_enabled(level)) {
    log_message(level, concat_message("dyng: ", what, ": implicit copy of ", bytes, " bytes from ",
                                      space_name(from), " to ", space_name(to), " memory"));
  }
}

void copy_to_host(const resources& res, void* dst, const void* src, memory_space space, int device,
                  std::size_t bytes) {
  if (bytes == 0) {
    return;
  }
  const bool cuda = res.get_backend() == backend::cuda;
  const int on = device >= 0 ? device : res.device();
  const stream_ref stream = cuda ? res.stream() : stream_ref{};
  copy_bytes(dst, memory_space::host, src, space, bytes, stream, on);
  cuda_synchronize(on, stream);
}

}  // namespace dyng::detail
