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
 * invalid_argument_error, out_of_memory_error; read_or_reject(), which wraps the FIRST read
 * only); any other exception escapes the target, which libFuzzer reports as a crash
 * (std::terminate). Where a writer exists, an accepted input must also survive write -> read
 * unchanged: round_trip() runs the writer and the second read, and ANY exception there, a rejected
 * copy included, is a finding, as is a copy that reads back different (DYNG_FUZZ_CHECK).
 *
 * Built with DYNG_FUZZ_CORRUPT_WRITES=1 (the CTest fuzz.selftest.<target>), written() appends a
 * line of garbage to every file a writer produced, so the replay of the corpus must report a
 * round-trip failure: the check is proven able to fail.
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
#include <exception>
#include <fstream>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

/// Stops the run with a message when a property of an accepted input does not hold.
#define DYNG_FUZZ_CHECK(cond)                                                            \
  do {                                                                                   \
    if (!(cond)) {                                                                       \
      std::fprintf(stderr, "%s:%d: fuzz check failed: %s\n", __FILE__, __LINE__, #cond); \
      ::dyng::fuzz::fail();                                                              \
    }                                                                                    \
  } while (false)

namespace dyng::fuzz {

/**
 * @brief End the run after a failed check: abort (a crash for libFuzzer and the replay), or, in
 *        the self-test build (DYNG_FUZZ_CORRUPT_WRITES), exit with status 3 (CTest matches the
 *        message).
 */
[[noreturn]] inline void fail() {
#if defined(DYNG_FUZZ_CORRUPT_WRITES) && DYNG_FUZZ_CORRUPT_WRITES
  std::exit(3);  // runs the destructors: the temporary files are removed
#else
  std::abort();
#endif
}

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
 * @brief Run the first read of an input: the exceptions a reader documents reject the input
 *        quietly (std::nullopt), any other one propagates (a crash under libFuzzer). Only the
 *        read belongs here: the checks of its result and the round trip run outside.
 * @tparam read_t A callable without arguments that returns the reader's result.
 * @param[in] read The reader call.
 * @return The result, or std::nullopt if the reader rejected the input.
 */
template <typename read_t>
auto read_or_reject(read_t&& read) -> std::optional<std::decay_t<decltype(read())>> {
  try {
    return read();
  } catch (const ::dyng::io_error&) {
  } catch (const ::dyng::invalid_argument_error&) {
  } catch (const ::dyng::out_of_memory_error&) {
  }
  return std::nullopt;
}

/**
 * @brief Run the write -> read round trip of an accepted input: every exception is a finding
 *        (a writer that throws, or a copy its own reader rejects), reported and ended with fail().
 * @tparam body_t A callable without arguments.
 * @param[in] body The writer call, the second read and its checks.
 */
template <typename body_t>
void round_trip(body_t&& body) {
  try {
    body();
  } catch (const std::exception& e) {
    std::fprintf(stderr, "fuzz check failed: the write -> read round trip threw: %s\n", e.what());
    fail();
  } catch (...) {
    std::fprintf(stderr, "fuzz check failed: the write -> read round trip threw\n");
    fail();
  }
}

/**
 * @brief Note a file a writer produced (removed at exit). In the self-test build
 *        (DYNG_FUZZ_CORRUPT_WRITES) also append a line of garbage to it, a planted writer bug.
 * @param[in] path The file.
 */
inline void written(const std::string& path) {
  files().remember(path);
#if defined(DYNG_FUZZ_CORRUPT_WRITES) && DYNG_FUZZ_CORRUPT_WRITES
  std::ofstream out(path, std::ios::binary | std::ios::app);
  out << "\nx y z\n";
#endif
}

}  // namespace dyng::fuzz
