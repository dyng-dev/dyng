// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file data_paths.hpp
 * @brief Test data locations, per-test temporary directories and file helpers.
 */
#pragma once

#include <gtest/gtest.h>

#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <vector>

#ifndef DYNG_TEST_DATA_DIR
#error "DYNG_TEST_DATA_DIR must be defined by the build (cpp/tests/data)"
#endif

namespace dyng::test {

/// Absolute path of a file or directory under cpp/tests/data.
inline std::string data_path(const std::string& relative) {
  return std::string(DYNG_TEST_DATA_DIR) + "/" + relative;
}

/// The whole content of a file ("" if it does not exist).
inline std::string read_text(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
}

/// Write `text` to `path` (creating the parent directory).
inline void write_text(const std::string& path, const std::string& text) {
  std::filesystem::create_directories(std::filesystem::path(path).parent_path());
  std::ofstream out(path, std::ios::binary);
  out << text;
}

/// The non-empty lines of a file.
inline std::vector<std::string> read_lines(const std::string& path) {
  std::istringstream in(read_text(path));
  std::vector<std::string> lines;
  for (std::string line; std::getline(in, line);) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

/// A fresh temporary directory for the current test, removed when the object is destroyed.
class temp_dir {
 public:
  temp_dir() {
    const auto* info = ::testing::UnitTest::GetInstance()->current_test_info();
    std::string name = "dyng_test_" + std::to_string(::getpid());
    if (info != nullptr) {
      name += std::string("_") + info->test_suite_name() + "_" + info->name();
    }
    for (char& c : name) {
      if (c == '/') {
        c = '_';
      }
    }
    path_ = std::filesystem::temp_directory_path() / name;
    std::filesystem::remove_all(path_);
    std::filesystem::create_directories(path_);
  }
  ~temp_dir() {
    std::error_code ec;
    std::filesystem::remove_all(path_, ec);
  }
  temp_dir(const temp_dir&) = delete;
  temp_dir& operator=(const temp_dir&) = delete;

  /// Path of `relative` inside the directory.
  [[nodiscard]] std::string path(const std::string& relative) const {
    return (path_ / relative).string();
  }

 private:
  std::filesystem::path path_;
};

}  // namespace dyng::test
