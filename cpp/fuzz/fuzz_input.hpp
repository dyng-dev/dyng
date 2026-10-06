// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_input.hpp
 * @brief The shared pieces of the reader fuzzers (cpp/fuzz): how an input is split into reader
 *        options and files, the temporary files the readers open, and the checks.
 *
 * Every fuzz target defines LLVMFuzzerTestOneInput(). Its input is laid out as
 *
 *     <option line>\n<file 1>[\n@@\n<file 2>[\n@@\n<file 3>]]
 *
 * The first line selects the reader options: its byte i (0 if the line is shorter) selects
 * option i, by its value modulo the number of choices; a digit '0'..'9' counts as its value
 * (so the seed line `0213` selects choices 0, 2, 1 and 3). The rest is the content of the files the reader opens; a
 * reader of several files (the CSR triplet, the MOSP batch) splits it at lines that are exactly
 * `@@` (missing files are empty). The files are written to a private temporary directory
 * ($TMPDIR, default /tmp) because the readers take paths.
 *
 * A reader may reject an input only with an exception its documentation names (io_error,
 * invalid_argument_error, out_of_memory_error; read_or_reject()); any other exception escapes the
 * target, which libFuzzer reports as a crash (std::terminate). Where a writer exists, an accepted input must
 * also survive write -> read unchanged (DYNG_FUZZ_CHECK aborts otherwise).
 *
 * The same sources build two ways (cpp/fuzz/CMakeLists.txt): with Clang's libFuzzer
 * (DYNG_BUILD_FUZZERS, preset `fuzz`) and, in every test build, linked with replay_main.cpp into
 * `dyng_fuzz_replay_<target>`, which runs the committed seed corpus and crash reproducers once
 * (CTest `fuzz.replay.<target>`).
 */
#pragma once

#include <dyng/core/error.hpp>

#include <unistd.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

/// Aborts with a message when a property of an accepted input does not hold.
#define DYNG_FUZZ_CHECK(cond)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "%s:%d: fuzz check failed: %s\n", __FILE__, __LINE__, #cond); \
      std::abort();                                                                      \
    }                                                                                    \
  } while (false)

namespace dyng::fuzz {

/// An input split into its option line and its files.
class fuzz_input {
 public:
  /**
   * @brief Split `size` bytes at `data` into the option line and up to `num_files` files.
   * @param[in] data      The input.
   * @param[in] size      Its size.
   * @param[in] num_files Files of the reader (>= 1); the last one takes the rest.
   */
  fuzz_input(const std::uint8_t* data, std::size_t size, int num_files) {
    const std::string_view all(reinterpret_cast<const char*>(data), size);
    const std::size_t eol = all.find('\n');
    options_ = std::string(all.substr(0, eol));
    std::string_view rest =
        eol == std::string_view::npos ? std::string_view{} : all.substr(eol + 1);
    for (int f = 0; f + 1 < num_files; ++f) {
      if (rest.substr(0, 3) == "@@\n") {  // an empty file
        files_.emplace_back();
        rest = rest.substr(3);
        continue;
      }
      const std::size_t cut = rest.find("\n@@\n");
      if (cut == std::string_view::npos) {
        files_.emplace_back(rest);
        rest = {};
      } else {
        files_.emplace_back(rest.substr(0, cut));
        rest = rest.substr(cut + 4);
      }
    }
    files_.emplace_back(rest);
    files_.resize(static_cast<std::size_t>(num_files));
  }

  /**
   * @brief Option i: byte i of the option line (a digit as its value) modulo `choices` (0 if the
   *        line is shorter).
   * @param[in] i       The option's index.
   * @param[in] choices Number of choices (>= 1).
   * @return A value in [0, choices).
   */
  [[nodiscard]] int option(std::size_t i, int choices) const noexcept {
    unsigned value = i < options_.size() ? static_cast<unsigned char>(options_[i]) : 0U;
    if (value >= '0' && value <= '9') {
      value -= '0';  // a digit selects its own value (readable seeds)
    }
    return static_cast<int>(value % static_cast<unsigned>(choices));
  }

  /**
   * @brief The content of file f.
   * @param[in] f The file's index.
   * @return Its bytes.
   */
  [[nodiscard]] const std::string& file(std::size_t f) const noexcept {
    return files_[f];
  }

 private:
  std::string options_;
  std::vector<std::string> files_;
};

/// The private temporary directory of this process and the files written into it.
class temp_files {
 public:
  temp_files() {
    const char* base = std::getenv("TMPDIR");
    std::string pattern =
        std::string(base != nullptr && *base != '\0' ? base : "/tmp") + "/dyng-fuzz-XXXXXX";
    std::vector<char> buffer(pattern.begin(), pattern.end());
    buffer.push_back('\0');
    if (::mkdtemp(buffer.data()) == nullptr) {
      std::perror("dyng fuzz: mkdtemp");
      std::abort();
    }
    dir_ = buffer.data();
  }

  temp_files(const temp_files&) = delete;
  temp_files& operator=(const temp_files&) = delete;
  temp_files(temp_files&&) = delete;
  temp_files& operator=(temp_files&&) = delete;

  ~temp_files() {
    for (const std::string& path : written_) {
      std::remove(path.c_str());
    }
    ::rmdir(dir_.c_str());
  }

  /**
   * @brief A path in the directory (the file need not exist).
   * @param[in] name The file name.
   * @return dir/name.
   */
  [[nodiscard]] std::string path(const std::string& name) const {
    return dir_ + "/" + name;
  }

  /**
   * @brief Write `content` to dir/name (replacing it) and return the path.
   * @param[in] name    The file name.
   * @param[in] content The bytes.
   * @return The path.
   */
  std::string write(const std::string& name, const std::string& content) {
    const std::string p = path(name);
    std::ofstream out(p, std::ios::binary | std::ios::trunc);
    out.write(content.data(), static_cast<std::streamsize>(content.size()));
    out.close();
    if (!out) {
      std::fprintf(stderr, "dyng fuzz: cannot write %s\n", p.c_str());
      std::abort();
    }
    remember(p);
    return p;
  }

  /**
   * @brief Note a path a writer created, so that it is removed at exit.
   * @param[in] p The path.
   */
  void remember(const std::string& p) {
    for (const std::string& known : written_) {
      if (known == p) {
        return;
      }
    }
    written_.push_back(p);
  }

 private:
  std::string dir_;
  std::vector<std::string> written_;
};

/**
 * @brief The temporary files of this process (created on first use, removed at exit).
 * @return The instance.
 */
inline temp_files& files() {
  static temp_files instance;
  return instance;
}

/**
 * @brief Run a reader (and the checks of its result); the exceptions a reader documents end the
 *        input quietly, any other one propagates (a crash under libFuzzer).
 * @param[in] body The reader call and its checks.
 */
template <typename body_t>
void read_or_reject(body_t&& body) {
  try {
    body();
  } catch (const ::dyng::io_error&) {
  } catch (const ::dyng::invalid_argument_error&) {
  } catch (const ::dyng::out_of_memory_error&) {
  }
}

}  // namespace dyng::fuzz
