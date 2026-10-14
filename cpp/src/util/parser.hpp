// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/csrGraph.cu (TextScanner, readIntFile, parseInteger)
/**
 * @file parser.hpp
 * @brief Strict text scanning for the readers: whole-file reads, tokens with line and column,
 *        range-checked integer parsing, io_error with path:line:column.
 *
 * The scanning follows the MOSP readers (the TextScanner / readIntFile / parseInteger helpers
 * named in the provenance line above), generalized: a token is a maximal run of
 * characters other than space, tab, carriage return and newline, and must parse COMPLETELY as a
 * decimal integer (optional leading '-', no '+', no trailing characters), so "12abc" or "1.5" is
 * an error instead of being split into two values.
 */
#pragma once

#include <dyng/core/error.hpp>

#include <charconv>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>

namespace dyng::detail {

/**
 * @brief Read a whole file into memory.
 * @param[in] path The file.
 * @return Its bytes.
 * @throws io_error if the file cannot be opened or read.
 */
std::string read_file(const std::string& path);

/**
 * @brief Throw io_error with the location prepended as "path:line:column: message".
 * @param[in] path    The file.
 * @param[in] line    1-based line (0: unknown).
 * @param[in] column  1-based column (0: unknown).
 * @param[in] message The description.
 * @throws io_error always.
 */
[[noreturn]] void throw_io_error(const std::string& path, std::int64_t line, std::int64_t column,
                                 const std::string& message);

/**
 * @brief One whitespace-separated token and where it starts.
 */
struct token {
  std::string_view text;    ///< the characters of the token
  std::int64_t line = 0;    ///< 1-based line of the first character
  std::int64_t column = 0;  ///< 1-based column of the first character
};

/**
 * @brief What text_scanner::next_number() and text_scanner::next_number_in_line() found.
 */
enum class scanned {
  end,     ///< no further token (of the text, or of the line)
  number,  ///< a token of 1 to 18 decimal digits, already parsed
  other,   ///< any other token: the caller parses it with parse_integer() (value or error)
};

/**
 * @brief Tokenizer over an in-memory text with line tracking.
 *
 * Two modes: next_token() returns the next token anywhere (newlines are separators), and
 * next_line() + next_in_line() walk the text line by line. next_number() and
 * next_number_in_line() are the same walks with the common case of the large readers parsed on
 * the way (a token of decimal digits only), so its characters are read once; every other token is
 * returned as it is for parse_integer(), so values and errors are exactly those of next_token() /
 * next_in_line() followed by parse_integer() (see integer_value()).
 */
class text_scanner {
 public:
  /**
   * @brief Scan `text`, reporting errors against `path`.
   * @param[in] text The text (must outlive the scanner).
   * @param[in] path The file name used in error messages.
   */
  text_scanner(std::string_view text, std::string path) : text_(text), path_(std::move(path)) {}

  /**
   * @brief The next token, crossing line breaks.
   * @param[out] out The token.
   * @return False at the end of the text.
   */
  bool next_token(token& out) noexcept {
    skip_blanks();
    while (pos_ < text_.size() && text_[pos_] == '\n') {
      new_line();
      skip_blanks();
    }
    return read_token(out);
  }

  /**
   * @brief The next token, crossing line breaks, parsed if it is a plain decimal number.
   * @param[out] out   The token (for scanned::number and scanned::other).
   * @param[out] value Its value (for scanned::number only).
   * @return scanned::end at the end of the text, otherwise what the token is.
   */
  scanned next_number(token& out, std::int64_t& value) noexcept {
    skip_blanks();
    while (pos_ < text_.size() && text_[pos_] == '\n') {
      new_line();
      skip_blanks();
    }
    return read_number(out, value);
  }

  /**
   * @brief The next token of the current line, parsed if it is a plain decimal number.
   * @param[out] out   The token (for scanned::number and scanned::other).
   * @param[out] value Its value (for scanned::number only).
   * @return scanned::end at the end of the line, otherwise what the token is.
   */
  scanned next_number_in_line(token& out, std::int64_t& value) noexcept {
    skip_blanks();
    return read_number(out, value);
  }

  /**
   * @brief Move to the start of the next line (the first call starts at line 1).
   * @return False if there is no further line.
   */
  bool next_line() noexcept {
    if (started_) {
      while (pos_ < text_.size() && text_[pos_] != '\n') {
        ++pos_;
      }
      if (pos_ >= text_.size()) {
        return false;
      }
      new_line();
      return true;
    }
    started_ = true;
    return pos_ < text_.size();
  }

  /**
   * @brief The next token of the current line.
   * @param[out] out The token.
   * @return False at the end of the line.
   */
  bool next_in_line(token& out) noexcept {
    skip_blanks();
    return read_token(out);
  }

  /**
   * @brief The rest of the current line (from the current position, without the newline).
   * @return The characters up to the next newline or the end of the text.
   */
  [[nodiscard]] std::string_view rest_of_line() const noexcept {
    std::size_t end = pos_;
    while (end < text_.size() && text_[end] != '\n') {
      ++end;
    }
    return text_.substr(pos_, end - pos_);
  }

  /**
   * @brief The current 1-based line.
   * @return The line number.
   */
  [[nodiscard]] std::int64_t line() const noexcept {
    return line_;
  }

  /**
   * @brief The file name used in error messages.
   * @return The path given at construction.
   */
  [[nodiscard]] const std::string& path() const noexcept {
    return path_;
  }

  /**
   * @brief Throw io_error at a token.
   * @param[in] at      The token the error refers to.
   * @param[in] message The description.
   * @throws io_error always.
   */
  [[noreturn]] void fail(const token& at, const std::string& message) const {
    throw_io_error(path_, at.line, at.column, message);
  }

  /**
   * @brief Throw io_error at the current line.
   * @param[in] message The description.
   * @throws io_error always.
   */
  [[noreturn]] void fail_line(const std::string& message) const {
    throw_io_error(path_, line_, 0, message);
  }

 private:
  static constexpr bool is_separator(char c) noexcept {
    return c == ' ' || c == '\t' || c == '\r' || c == '\n';
  }
  void skip_blanks() noexcept {
    while (pos_ < text_.size() &&
           (text_[pos_] == ' ' || text_[pos_] == '\t' || text_[pos_] == '\r')) {
      ++pos_;
    }
  }
  void new_line() noexcept {
    ++pos_;
    ++line_;
    line_start_ = pos_;
  }
  bool read_token(token& out) noexcept {
    if (pos_ >= text_.size() || text_[pos_] == '\n') {
      return false;
    }
    const std::size_t begin = pos_;
    while (pos_ < text_.size() && !is_separator(text_[pos_])) {
      ++pos_;
    }
    out.text = text_.substr(begin, pos_ - begin);
    out.line = line_;
    out.column = static_cast<std::int64_t>(begin - line_start_) + 1;
    return true;
  }
  // read_token(), and the value of a token of 1 to 18 digits (at most 999...9 < 2^63: no
  // overflow, and std::from_chars would give the same value, leading zeros included).
  scanned read_number(token& out, std::int64_t& value) noexcept {
    if (pos_ >= text_.size() || text_[pos_] == '\n') {
      return scanned::end;
    }
    const std::size_t begin = pos_;
    std::uint64_t digits_value = 0;
    while (pos_ < text_.size()) {
      const auto digit = static_cast<unsigned char>(text_[pos_] - '0');
      if (digit > 9) {
        break;
      }
      digits_value = digits_value * 10 + digit;  // wraps harmlessly past 18 digits (not used)
      ++pos_;
    }
    const std::size_t digits = pos_ - begin;
    const bool number =
        digits > 0 && digits <= 18 && (pos_ == text_.size() || is_separator(text_[pos_]));
    while (pos_ < text_.size() && !is_separator(text_[pos_])) {  // the rest of another token
      ++pos_;
    }
    out.text = text_.substr(begin, pos_ - begin);
    out.line = line_;
    out.column = static_cast<std::int64_t>(begin - line_start_) + 1;
    if (!number) {
      return scanned::other;
    }
    value = static_cast<std::int64_t>(digits_value);
    return scanned::number;
  }

  std::string_view text_;
  std::string path_;
  std::size_t pos_ = 0;
  std::size_t line_start_ = 0;
  std::int64_t line_ = 1;
  bool started_ = false;
};

/**
 * @brief Parse a complete token as a decimal integer.
 * @param[in]  text  The token.
 * @param[out] value The value.
 * @return False if the token is not exactly one integer representable in std::int64_t.
 */
inline bool parse_int64(std::string_view text, std::int64_t& value) noexcept {
  if (text.empty()) {
    return false;
  }
  const char* first = text.data();
  const char* last = text.data() + text.size();
  const auto result = std::from_chars(first, last, value);
  return result.ec == std::errc() && result.ptr == last;
}

/**
 * @brief Parse a token as an integer in [minimum, maximum]; throws io_error at the token if not.
 * @tparam int_t   The integer type of the result.
 * @param[in] scanner The scanner (for the error location).
 * @param[in] at      The token.
 * @param[in] minimum Smallest accepted value.
 * @param[in] maximum Largest accepted value.
 * @param[in] what    What the value is ("vertex id", "weight", ...), for the message.
 * @return The value.
 * @throws io_error if the token is not an integer or is out of range.
 */
template <typename int_t>
int_t parse_integer(const text_scanner& scanner, const token& at, std::int64_t minimum,
                    std::int64_t maximum, const char* what) {
  static_assert(std::is_integral_v<int_t>, "parse_integer needs an integer type");
  std::int64_t value = 0;
  if (!parse_int64(at.text, value)) {
    scanner.fail(at, std::string("invalid ") + what + " '" + std::string(at.text) +
                         "' (expected a decimal integer)");
  }
  if (value < minimum || value > maximum) {
    scanner.fail(at, std::string(what) + " " + std::to_string(value) + " out of range [" +
                         std::to_string(minimum) + ", " + std::to_string(maximum) + "]");
  }
  return static_cast<int_t>(value);
}

/**
 * @brief The value of a token from text_scanner::next_number() / next_number_in_line() in
 *        [minimum, maximum]: exactly parse_integer()'s value or error.
 * @tparam int_t   The integer type of the result.
 * @param[in] scanner The scanner (for the error location).
 * @param[in] at      The token.
 * @param[in] kind    What the scanner found (scanned::number or scanned::other).
 * @param[in] value   The value the scanner parsed (read for scanned::number only).
 * @param[in] minimum Smallest accepted value.
 * @param[in] maximum Largest accepted value.
 * @param[in] what    What the value is, for the message.
 * @return The value.
 * @throws io_error if the token is not an integer or is out of range (parse_integer()'s errors).
 */
template <typename int_t>
int_t integer_value(const text_scanner& scanner, const token& at, scanned kind, std::int64_t value,
                    std::int64_t minimum, std::int64_t maximum, const char* what) {
  if (kind == scanned::number && value >= minimum && value <= maximum) {
    return static_cast<int_t>(value);
  }
  return parse_integer<int_t>(scanner, at, minimum, maximum, what);  // the general path
}

/**
 * @brief The largest value of an integer type as std::int64_t (saturating for wider types).
 * @tparam int_t An integer type.
 * @return min(numeric_limits<int_t>::max(), INT64_MAX).
 */
template <typename int_t>
constexpr std::int64_t max_as_int64() noexcept {
  if constexpr (sizeof(int_t) >= sizeof(std::int64_t)) {
    return std::numeric_limits<std::int64_t>::max();
  } else {
    return static_cast<std::int64_t>(std::numeric_limits<int_t>::max());
  }
}

}  // namespace dyng::detail
