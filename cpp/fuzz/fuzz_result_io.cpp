// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_result_io.cpp
 * @brief libFuzzer target of the result readers dyng::io::read_distances() and
 *        dyng::io::read_parents() (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the reader (distances, parents), 1 the vertex count (0..7, 1000). One file. An
 * accepted input must give one value per vertex in range, and the matching writer followed by the
 * reader must give the same values (distances of at least infinite_distance() / 2 come back as
 * infinite_distance(): the writer prints them as INF).
 */
#include "fuzz_input.hpp"

#include <dyng/core/types.hpp>
#include <dyng/io/result_io.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace {

void distances(const std::string& path, std::int64_t n) {
  const auto read =
      dyng::fuzz::read_or_reject([&] { return dyng::io::read_distances<std::int64_t>(path, n); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& d = *read;
  DYNG_FUZZ_CHECK(d.size() == static_cast<std::size_t>(n));
  constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();
  std::vector<std::int64_t> expected = d;
  for (std::int64_t& value : expected) {
    DYNG_FUZZ_CHECK(value >= 0);
    if (value >= inf / 2) {
      value = inf;
    }
  }
  dyng::fuzz::round_trip([&] {
    const std::string copy = dyng::fuzz::files().path("copy_distances.txt");
    dyng::io::write_distances<std::int64_t>(copy, dyng::host_view(d));
    dyng::fuzz::written(copy);
    DYNG_FUZZ_CHECK(dyng::io::read_distances<std::int64_t>(copy, n) == expected);
  });
}

void parents(const std::string& path, std::int64_t n) {
  const auto read =
      dyng::fuzz::read_or_reject([&] { return dyng::io::read_parents<std::int32_t>(path, n); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& p = *read;
  DYNG_FUZZ_CHECK(p.size() == static_cast<std::size_t>(n));
  for (const std::int32_t v : p) {
    DYNG_FUZZ_CHECK(v >= -1 && v < n);
  }
  dyng::fuzz::round_trip([&] {
    const std::string copy = dyng::fuzz::files().path("copy_parents.txt");
    dyng::io::write_parents<std::int32_t>(copy, dyng::host_view(p));
    dyng::fuzz::written(copy);
    DYNG_FUZZ_CHECK(dyng::io::read_parents<std::int32_t>(copy, n) == p);
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.txt", in.file(0));
  const int choice = in.option(1, 9);
  const std::int64_t n = choice == 8 ? 1000 : choice;
  if (in.option(0, 2) == 0) {
    distances(path, n);
  } else {
    parents(path, n);
  }
  return 0;
}
