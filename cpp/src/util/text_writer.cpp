// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/csrGraph.cpp (the TextWriter class)
/**
 * @file text_writer.cpp
 * @brief text_writer implementation.
 */
#include "util/text_writer.hpp"

#include "util/parser.hpp"

#include <filesystem>
#include <system_error>
#include <utility>

namespace dyng::detail {

text_writer::text_writer(std::string path) : path_(std::move(path)) {
  const std::filesystem::path parent = std::filesystem::path(path_).parent_path();
  if (!parent.empty()) {
    std::error_code ec;
    std::filesystem::create_directories(parent, ec);  // on failure fopen fails below
  }
  file_ = std::fopen(path_.c_str(), "wb");
  if (file_ == nullptr) {
    throw_io_error(path_, 0, 0, "cannot open file for writing");
  }
  buffer_.reserve(flush_size + 64);
}

text_writer::~text_writer() {
  if (file_ != nullptr) {
    flush();
    std::fclose(file_);
  }
}

void text_writer::flush() {
  if (file_ != nullptr && !buffer_.empty() &&
      std::fwrite(buffer_.data(), 1, buffer_.size(), file_) != buffer_.size()) {
    failed_ = true;
  }
  buffer_.clear();
}

void text_writer::close() {
  if (file_ == nullptr) {
    return;
  }
  flush();
  if (std::fclose(file_) != 0) {
    failed_ = true;
  }
  file_ = nullptr;
  if (failed_) {
    throw_io_error(path_, 0, 0, "write error");
  }
}

}  // namespace dyng::detail
