// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/csrGraph.cpp (the TextWriter class)
// Derived from LabelPropagation-CUDA@a276a3a:src/utils.cu (TextWriter::putFixed6)
/**
 * @file text_writer.hpp
 * @brief text_writer: a buffered text file writer with fast integer formatting.
 *
 * Straight port of MOSP's TextWriter (same bytes: std::to_chars decimal integers), changed to
 * throw io_error instead of reporting through ok()/close() return values. The parent directory
 * of the file is created if it does not exist (as the MOSP writers do). put_fixed6() adds
 * LabelPropagation-CUDA's fixed 6-decimal float format (its TextWriter of the same design).
 */
#pragma once

#include <charconv>
#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>

namespace dyng::detail {

/**
 * @brief A buffered writer; throws io_error when the file cannot be opened or written.
 */
class text_writer {
 public:
  /**
   * @brief Open (truncate) `path` for writing; creates its parent directory.
   * @param[in] path The file.
   * @throws io_error if the file cannot be opened.
   */
  explicit text_writer(std::string path);
  ~text_writer();  ///< closes the file; errors are ignored here (call close() to see them)
  text_writer(const text_writer&) = delete;             ///< not copyable
  text_writer& operator=(const text_writer&) = delete;  ///< not copyable

  /**
   * @brief Append a decimal integer.
   * @param[in] value The value.
   */
  void put(std::int64_t value) {
    char digits[24];
    const auto result = std::to_chars(digits, digits + sizeof(digits), value);
    buffer_.append(digits, result.ptr);
    maybe_flush();
  }

  /**
   * @brief Append a float in fixed notation with 6 decimals, exactly as LabelPropagation-CUDA's
   *        TextWriter::putFixed6 (src/utils.cu): std::to_chars(double(x), chars_format::fixed, 6),
   *        the bytes of iostream's `std::fixed << std::setprecision(6)` (correctly rounded).
   * @param[in] value The value (finite or not: to_chars prints inf / nan as O does).
   */
  void put_fixed6(float value) {
    char digits[64];  // |value| <= FLT_MAX: at most 39 integer digits, the point and 6 decimals
    const auto result = std::to_chars(digits, digits + sizeof(digits), static_cast<double>(value),
                                      std::chars_format::fixed, 6);
    buffer_.append(digits, result.ptr);
    maybe_flush();
  }

  /**
   * @brief Append text.
   * @param[in] text The characters.
   */
  void put(std::string_view text) {
    buffer_.append(text);
    maybe_flush();
  }

  /**
   * @brief Append one character.
   * @param[in] c The character.
   */
  void put_char(char c) {
    buffer_.push_back(c);
    maybe_flush();
  }

  /**
   * @brief Flush and close the file.
   * @throws io_error if a write or the close failed.
   */
  void close();

 private:
  static constexpr std::size_t flush_size = std::size_t{1} << 22;
  void maybe_flush() {
    if (buffer_.size() >= flush_size) {
      flush();
    }
  }
  void flush();

  std::string path_;
  std::FILE* file_ = nullptr;
  std::string buffer_;
  bool failed_ = false;
};

}  // namespace dyng::detail
