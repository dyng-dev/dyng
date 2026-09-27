// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file error.hpp
 * @brief Exception hierarchy and the DYNG_EXPECTS / DYNG_FAIL macros.
 * @ingroup core
 *
 * Every exception thrown by dynG derives from dyng::error, which derives from
 * std::runtime_error. Host allocation failures (std::bad_alloc, and std::length_error from sizing
 * a container) are reported as out_of_memory_error at the library's entry points. Library code never calls exit() or abort() and never reports errors by
 * printing; it throws one of the types below (PLAN Section 4.7.3).
 */
#pragma once

#include <cstdint>
#include <sstream>
#include <stdexcept>
#include <string>
#include <utility>

namespace dyng {

/**
 * @brief Base class of every exception thrown by dynG.
 * @ingroup core
 */
class error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

/**
 * @brief Bad input: ids out of range, wrong layout or backend, violated preconditions.
 * @ingroup core
 */
class invalid_argument_error : public error {
 public:
  using error::error;
};

/**
 * @brief A result was used with a graph whose version no longer matches it.
 * @ingroup core
 */
class stale_result_error : public invalid_argument_error {
 public:
  using invalid_argument_error::invalid_argument_error;
};

/**
 * @brief A file could not be read or written, or its content is malformed.
 * @ingroup core
 */
class io_error : public error {
 public:
  /**
   * @brief Construct an I/O error with its location.
   * @param[in] message Human-readable description.
   * @param[in] path    The file concerned (may be empty).
   * @param[in] line    1-based line number, or 0 if unknown.
   * @param[in] column  1-based column number, or 0 if unknown.
   */
  explicit io_error(const std::string& message, std::string path = {}, std::int64_t line = 0,
                    std::int64_t column = 0)
      : error(message), path_(std::move(path)), line_(line), column_(column) {}

  /**
   * @brief The file concerned.
   * @return The path given at construction (may be empty).
   */
  [[nodiscard]] const std::string& path() const noexcept {
    return path_;
  }

  /**
   * @brief The line of the error.
   * @return 1-based line number, or 0 if unknown.
   */
  [[nodiscard]] std::int64_t line() const noexcept {
    return line_;
  }

  /**
   * @brief The column of the error.
   * @return 1-based column number, or 0 if unknown.
   */
  [[nodiscard]] std::int64_t column() const noexcept {
    return column_;
  }

 private:
  std::string path_;
  std::int64_t line_;
  std::int64_t column_;
};

/**
 * @brief A fixed capacity was exceeded (slotted row overflow, 2^31 payload, too many objectives).
 * @ingroup core
 */
class capacity_error : public error {
 public:
  using error::error;
};

/**
 * @brief A backend, type, engine or feature is not built or not implemented.
 * @ingroup core
 */
class not_supported_error : public error {
 public:
  using error::error;
};

/**
 * @brief An iterative algorithm hit its iteration cap and was asked to fail.
 * @ingroup core
 */
class convergence_error : public error {
 public:
  using error::error;
};

/**
 * @brief A CUDA runtime or driver call failed.
 * @ingroup core
 */
class cuda_error : public error {
 public:
  /**
   * @brief Construct a CUDA error.
   * @param[in] message Description, including the failing call, file:line and the CUDA error string.
   * @param[in] code    The numeric cudaError_t value.
   */
  cuda_error(const std::string& message, int code) : error(message), code_(code) {}

  /**
   * @brief The CUDA error code.
   * @return The numeric cudaError_t value.
   */
  [[nodiscard]] int code() const noexcept {
    return code_;
  }

 private:
  int code_;
};

/**
 * @brief A host or device allocation failed.
 * @ingroup core
 */
class out_of_memory_error : public error {
 public:
  using error::error;
};

/**
 * @brief A broken internal invariant: a library bug. Please report it.
 * @ingroup core
 */
class internal_error : public error {
 public:
  using error::error;
};

namespace detail {

/**
 * @brief Strip the directories from a source path (for short file:line messages).
 * @param[in] path A path such as __FILE__.
 * @return Pointer to the file name inside `path`.
 */
const char* source_basename(const char* path) noexcept;

/**
 * @brief Concatenate the arguments with operator<<.
 * @tparam args_t Streamable argument types.
 * @param[in] args The values to concatenate.
 * @return The concatenation.
 */
template <typename... args_t>
std::string concat_message(const args_t&... args) {
  std::ostringstream out;
  (out << ... << args);
  return out.str();
}

/**
 * @brief Throw `error_t` with a message and its source location.
 * @tparam error_t The exception type (derived from dyng::error, constructible from a string).
 * @tparam args_t  Streamable argument types of the message.
 * @param[in] file      __FILE__ of the check.
 * @param[in] line      __LINE__ of the check.
 * @param[in] condition The failed condition as text (may be nullptr).
 * @param[in] args      The message parts.
 * @throws error_t Always.
 */
template <typename error_t, typename... args_t>
[[noreturn]] void throw_with_location(const char* file, int line, const char* condition,
                                      const args_t&... args) {
  std::ostringstream out;
  out << "dyng: ";
  (out << ... << args);
  if (condition != nullptr) {
    out << " [expected: " << condition << "]";
  }
  out << " (" << source_basename(file) << ":" << line << ")";
  throw error_t(out.str());
}

/**
 * @brief Report a failed host allocation (std::bad_alloc, or std::length_error from sizing a
 *        standard container beyond its max_size()) as out_of_memory_error.
 *
 * The public entry points of the library catch those two standard exceptions and call this, so
 * every exception that leaves dynG derives from dyng::error (PLAN Section 4.7.3).
 * @param[in] context What was being done, with the sizes involved.
 * @param[in] cause   The standard exception's what().
 * @throws out_of_memory_error Always.
 */
[[noreturn]] void throw_host_allocation_failure(const std::string& context, const char* cause);

}  // namespace detail
}  // namespace dyng

/**
 * @brief Check a precondition; throw dyng::invalid_argument_error if it does not hold.
 *
 * Usage: `DYNG_EXPECTS(source < n, "source ", source, " is out of range [0, ", n, ")");`
 * At least one message argument is required. The message receives the condition text and
 * file:line.
 * @ingroup core
 */
#define DYNG_EXPECTS(cond, ...)                                                                \
  do {                                                                                         \
    if (!(cond)) {                                                                             \
      ::dyng::detail::throw_with_location<::dyng::invalid_argument_error>(__FILE__, __LINE__,  \
                                                                          #cond, __VA_ARGS__); \
    }                                                                                          \
  } while (false)

/**
 * @brief Report a broken internal invariant: throws dyng::internal_error.
 *
 * Usage: `DYNG_FAIL("unexpected row layout ", static_cast<int>(layout));`
 * At least one message argument is required.
 * @ingroup core
 */
#define DYNG_FAIL(...)                                                                     \
  ::dyng::detail::throw_with_location<::dyng::internal_error>(__FILE__, __LINE__, nullptr, \
                                                              "internal error: ", __VA_ARGS__)
