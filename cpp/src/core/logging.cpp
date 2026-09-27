// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file logging.cpp
 * @brief Process-wide log level and sink (the only global state of the library).
 */
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>

#include <atomic>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>
#include <utility>

namespace dyng {

namespace {

log_level initial_level() noexcept {
  const char* env = std::getenv("DYNG_LOG_LEVEL");
  if (env == nullptr) {
    return log_level::warn;
  }
  try {
    return parse_log_level(env);
  } catch (...) {
    return log_level::warn;
  }
}

std::atomic<log_level>& level_storage() noexcept {
  static std::atomic<log_level> level{initial_level()};
  return level;
}

struct sink_storage {
  std::mutex mutex;
  log_sink sink;
};

sink_storage& sink_state() {
  static sink_storage state;
  return state;
}

void default_sink(log_level level, std::string_view message) {
  std::cerr << "[dyng] " << to_string(level) << ": " << message << '\n';
}

}  // namespace

void set_log_level(log_level level) noexcept {
  level_storage().store(level, std::memory_order_relaxed);
}

log_level get_log_level() noexcept {
  return level_storage().load(std::memory_order_relaxed);
}

void set_log_sink(log_sink sink) {
  auto& state = sink_state();
  std::scoped_lock lock(state.mutex);
  state.sink = std::move(sink);
}

bool log_enabled(log_level level) noexcept {
  return level != log_level::off && static_cast<int>(level) <= static_cast<int>(get_log_level());
}

void log_message(log_level level, std::string_view message) {
  if (!log_enabled(level)) {
    return;
  }
  auto& state = sink_state();
  std::scoped_lock lock(state.mutex);
  if (state.sink) {
    state.sink(level, message);
  } else {
    default_sink(level, message);
  }
}

log_level parse_log_level(std::string_view name) {
  std::string lower(name);
  for (char& c : lower) {
    c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  }
  if (lower == "off") return log_level::off;
  if (lower == "error") return log_level::error;
  if (lower == "warn" || lower == "warning") return log_level::warn;
  if (lower == "info") return log_level::info;
  if (lower == "debug") return log_level::debug;
  if (lower == "trace") return log_level::trace;
  throw invalid_argument_error(detail::concat_message(
      "dyng: unknown log level '", name, "' (expected off, error, warn, info, debug or trace)"));
}

std::string_view to_string(log_level level) noexcept {
  switch (level) {
    case log_level::off:
      return "off";
    case log_level::error:
      return "error";
    case log_level::warn:
      return "warn";
    case log_level::info:
      return "info";
    case log_level::debug:
      return "debug";
    case log_level::trace:
      return "trace";
  }
  return "unknown";
}

}  // namespace dyng
