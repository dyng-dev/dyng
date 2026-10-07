// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file replay_main.cpp
 * @brief The driver of the replay builds `dyng_fuzz_replay_<target>` (any compiler, no
 *        libFuzzer): runs LLVMFuzzerTestOneInput() once on every file named on the command line,
 *        and on every regular file of every directory named there (sorted by name).
 *
 * Exit status 0 when every input ran (a failing check or an unexpected exception aborts), 1 when
 * an argument cannot be read or names no input at all.
 */
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <system_error>
#include <vector>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size);

int main(int argc, char** argv) {
  namespace fs = std::filesystem;
  std::vector<fs::path> inputs;
  for (int i = 1; i < argc; ++i) {
    const fs::path arg(argv[i]);
    std::error_code ec;
    if (fs::is_directory(arg, ec)) {
      std::vector<fs::path> found;
      for (const auto& entry : fs::directory_iterator(arg, ec)) {
        if (entry.is_regular_file()) {
          found.push_back(entry.path());
        }
      }
      std::sort(found.begin(), found.end());
      inputs.insert(inputs.end(), found.begin(), found.end());
    } else if (fs::is_regular_file(arg, ec)) {
      inputs.push_back(arg);
    } else {
      std::fprintf(stderr, "replay: cannot read %s\n", argv[i]);
      return 1;
    }
  }
  if (inputs.empty()) {
    std::fprintf(stderr, "replay: no input\n");
    return 1;
  }
  for (const fs::path& path : inputs) {
    std::ifstream in(path, std::ios::binary);
    const std::vector<char> bytes((std::istreambuf_iterator<char>(in)),
                                  std::istreambuf_iterator<char>());
    std::fprintf(stderr, "replay: %s (%zu bytes)\n", path.string().c_str(), bytes.size());
    (void)LLVMFuzzerTestOneInput(reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size());
  }
  std::fprintf(stderr, "replay: %zu inputs passed\n", inputs.size());
  return 0;
}
