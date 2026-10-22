// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_triad_bins.cpp
 * @brief libFuzzer target of the triad-bins JSON reader dyng::io::read_triad_bins_json()
 *        (cpp/fuzz/fuzz_input.hpp).
 *
 * No options (the first line is ignored). One file. An accepted input must name a known kind,
 * have that kind's number of counts and of values in every delta row, and the writer followed by
 * the reader must give the same bins (unknown keys are dropped by the reader, so the copy is
 * compared with what was read, not with the input's text).
 */
#include "fuzz_input.hpp"

#include <dyng/io/result_io.hpp>

#include <cstddef>
#include <cstdint>
#include <string>

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.json", in.file(0));
  const auto read =
      dyng::fuzz::read_or_reject([&] { return dyng::io::read_triad_bins_json(path); });
  if (!read) {
    return 0;  // rejected with a documented exception
  }
  const dyng::io::triad_bins& b = *read;
  const auto n = static_cast<std::size_t>(dyng::io::triad_bins_size(b.kind));
  DYNG_FUZZ_CHECK(b.counts.size() == n);
  for (const auto& row : b.deltas) {
    DYNG_FUZZ_CHECK(row.size() == n);
  }
  dyng::fuzz::round_trip([&] {
    const std::string copy = dyng::fuzz::files().path("copy.json");
    dyng::io::write_triad_bins_json(copy, b);
    dyng::fuzz::written(copy);
    const dyng::io::triad_bins again = dyng::io::read_triad_bins_json(copy);
    DYNG_FUZZ_CHECK(again.kind == b.kind && again.graph_version == b.graph_version &&
                    again.counts == b.counts && again.deltas == b.deltas);
  });
  return 0;
}
