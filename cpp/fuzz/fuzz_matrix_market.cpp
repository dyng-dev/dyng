// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_matrix_market.cpp
 * @brief libFuzzer target of dyng::io::read_matrix_market() (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the weight source (automatic, none, from_file, random), 1 drop_self_loops,
 * 2 sort_and_dedupe, 3 the random weight count K (1..3), 4 the vertex id type (int32, int64).
 * One file. An accepted input must give a consistent edge list (ids in range, K weights per edge,
 * sorted and duplicate-free with sort_and_dedupe, no self-loops with drop_self_loops).
 */
#include "fuzz_input.hpp"

#include <dyng/io/matrix_market.hpp>

#include <cstddef>
#include <cstdint>

namespace {

template <typename vertex_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& path) {
  dyng::io::matrix_market_options opt;
  opt.weights = static_cast<dyng::io::matrix_market_weights>(in.option(0, 4));
  opt.drop_self_loops = in.option(1, 2) == 1;
  opt.sort_and_dedupe = in.option(2, 2) == 1;
  opt.random.num_weights = 1 + in.option(3, 3);
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_matrix_market<vertex_t, std::int32_t>(path, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& edges = *read;
  const std::size_t m = edges.num_edges();
  DYNG_FUZZ_CHECK(edges.dst.size() == m);
  DYNG_FUZZ_CHECK(edges.weights.size() == m * static_cast<std::size_t>(edges.num_weights));
  for (std::size_t e = 0; e < m; ++e) {
    DYNG_FUZZ_CHECK(edges.src[e] >= 0 && edges.src[e] < edges.num_vertices);
    DYNG_FUZZ_CHECK(edges.dst[e] >= 0 && edges.dst[e] < edges.num_vertices);
    DYNG_FUZZ_CHECK(!opt.drop_self_loops || edges.src[e] != edges.dst[e]);
    if (opt.sort_and_dedupe && e > 0) {
      const bool increasing = edges.src[e - 1] < edges.src[e] ||
                              (edges.src[e - 1] == edges.src[e] && edges.dst[e - 1] < edges.dst[e]);
      DYNG_FUZZ_CHECK(increasing);
    }
  }
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.mtx", in.file(0));
  if (in.option(4, 2) == 0) {
    run<std::int32_t>(in, path);
  } else {
    run<std::int64_t>(in, path);
  }
  return 0;
}
