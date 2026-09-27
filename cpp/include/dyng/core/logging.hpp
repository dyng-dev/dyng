// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file logging.hpp
 * @brief Diagnostic logging: levels and a replaceable sink.
 * @ingroup core
 *
 * The library logs only diagnostics (fallbacks taken, capacity growth, implicit copies,
 * deprecations) and never results. The log level and the sink are the only process-global state
 * of the library (PLAN Section 4.7.5). The default sink writes one line per message to stderr;
 * the Python layer installs a sink that forwards to logging.getLogger("dyng").
 */
#pragma once

#include <cstdint>
#include <functional>
#include <string_view>

namespace dyng {

/**
 * @brief Severity of a log message; a message is emitted if its level <= the current level.
 * @ingroup core
 */
enum class log_level : std::uint8_t {
  off,    ///< nothing is logged
  error,  ///< failures reported as exceptions that are also worth a log line (e.g. in destructors)
  warn,   ///< the default: suspicious but handled situations
  info,   ///< coarse progress information
  debug,  ///< details such as implicit copies and chosen engines
  trace,  ///< very verbose
};

/**
 * @brief A log sink: receives the level and the formatted message (without trailing newline).
 * @ingroup core
 */
using log_sink = std::function<void(log_level, std::string_view)>;

/**
 * @brief Set the process-wide log level.
 * @param[in] level The new level.
 * @ingroup core
 */
void set_log_level(log_level level) noexcept;

/**
 * @brief The process-wide log level.
 *
 * Initially the value of the environment variable DYNG_LOG_LEVEL (off, error, warn, info, debug,
 * trace; read once), or warn if it is unset or invalid.
 * @return The current level.
 * @ingroup core
 */
[[nodiscard]] log_level get_log_level() noexcept;

/**
 * @brief Replace the log sink.
 * The sink is called without any library lock held, so it may itself log or call other dynG
 * functions, and it may be called concurrently from several threads (it must be thread-safe, or
 * serialize itself). A call that is running when the sink is replaced finishes with the old sink.
 * @param[in] sink The new sink; an empty function restores the default stderr sink (which writes
 *                 whole lines under its own lock).
 * @ingroup core
 */
void set_log_sink(log_sink sink);

/**
 * @brief Whether a message of `level` would be emitted.
 * @param[in] level The message level.
 * @return True if level != off and level <= get_log_level().
 * @ingroup core
 */
[[nodiscard]] bool log_enabled(log_level level) noexcept;

/**
 * @brief Emit a message if its level is enabled.
 *
 * Named log_message (not log) so that unqualified calls of the math function log() inside
 * namespace dyng keep working.
 * @param[in] level   The message level (must not be log_level::off).
 * @param[in] message The message.
 * @ingroup core
 */
void log_message(log_level level, std::string_view message);

/**
 * @brief Parse a level name.
 * @param[in] name One of "off", "error", "warn", "info", "debug", "trace" (case-insensitive).
 * @return The level.
 * @throws invalid_argument_error if `name` is not a level name.
 * @ingroup core
 */
[[nodiscard]] log_level parse_log_level(std::string_view name);

/**
 * @brief The lower-case name of a level.
 * @param[in] level The level.
 * @return A static string ("off", "error", "warn", "info", "debug" or "trace").
 * @ingroup core
 */
[[nodiscard]] std::string_view to_string(log_level level) noexcept;

}  // namespace dyng
