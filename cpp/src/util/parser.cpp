// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file parser.cpp
 * @brief Whole-file reads and located I/O errors for the readers.
 */
#include "util/parser.hpp"

#include <cstdio>
#include <string>

namespace dyng::detail {

std::string read_file(const std::string& path) {
  std::FILE* file = std::fopen(path.c_str(), "rb");
  if (file == nullptr) {
    throw io_error("dyng: " + path + ": cannot open file for reading", path);
  }
  std::string text;
  // One allocation and one read for a regular file (the size is known), as MOSP's readWholeFile()
  // does; growing the string chunk by chunk would copy a large file several times. The chunk loop
  // below then reads whatever is left (a file that grew, or a stream without a size).
  if (std::fseek(file, 0, SEEK_END) == 0) {
    const long size = std::ftell(file);
    std::rewind(file);
    if (size > 0) {
      text.resize(static_cast<std::size_t>(size));
      text.resize(std::fread(text.data(), 1, text.size(), file));
    }
  } else {
    std::clearerr(file);
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
