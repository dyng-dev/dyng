// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file parser.cpp
 * @brief Whole-file reads and located I/O errors for the readers.
 */
#include "util/parser.hpp"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <system_error>

namespace dyng::detail {

std::string read_file(const std::string& path) {
  // What the path names decides how it is read (M8b review: ftell() of a directory is LONG_MAX
  // here, and the reader allocated that size; a character device such as /dev/zero never ends).
  // A regular file is read with one allocation of its size; a pipe or socket chunk by chunk;
  // anything else is an io_error.
  std::error_code ec;
  const std::filesystem::file_status st = std::filesystem::status(path, ec);
  std::uintmax_t size = 0;
  if (!ec && std::filesystem::is_directory(st)) {
    throw io_error("dyng: " + path + ": is a directory, not a file", path);
  }
  if (!ec && std::filesystem::is_regular_file(st)) {
    size = std::filesystem::file_size(path, ec);
    if (ec) size = 0;
  } else if (!ec && std::filesystem::exists(st) && !std::filesystem::is_fifo(st) &&
             !std::filesystem::is_socket(st)) {
    throw io_error("dyng: " + path + ": not a regular file (a device or another special file)",
                   path);
  }
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw io_error("dyng: " + path + ": cannot open file for reading", path);
  }
  std::string text;
  // One allocation and one read for a regular file (the size is known), as MOSP's readWholeFile()
  // does; growing the string chunk by chunk would copy a large file several times. The chunk loop
  // below then reads whatever is left (a file that grew, or a stream without a size).
  if (size > 0) {
    try {
      text.resize(static_cast<std::size_t>(size));
    } catch (...) {
      std::fclose(file);
      throw;
    }
    text.resize(std::fread(text.data(), 1, text.size(), file));
  }
  char chunk[1 << 16];
  bool failed = false;
  while (true) {
    const std::size_t got = std::fread(chunk, 1, sizeof(chunk), file);
    text.append(chunk, got);
    if (got < sizeof(chunk)) {
      failed = std::ferror(file) != 0;
      break;
    }
  }
  std::fclose(file);
  if (failed) {
    throw io_error("dyng: " + path + ": read error", path);
  }
  return text;
}

void throw_io_error(const std::string& path, std::int64_t line, std::int64_t column,
                    const std::string& message) {
  std::string where = path;
  if (line > 0) {
    where += ":" + std::to_string(line);
    if (column > 0) {
      where += ":" + std::to_string(column);
    }
  }
  throw io_error("dyng: " + where + ": " + message, path, line, column);
}

}  // namespace dyng::detail
