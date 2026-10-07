// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_csr_triplet.cpp
 * @brief libFuzzer target of dyng::io::read_csr_triplet() (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the expected weight columns K (0 = inferred, 1..3), 1 the graph type (int32/int32,
 * int32/int64, int64/int64 vertex/edge ids). Three files: RowPtr, ColInd, Values. An accepted
 * input must give a valid CSR, and write_csr_triplet() followed by read_csr_triplet() must give
 * the same CSR.
 */
#include "fuzz_input.hpp"

#include <dyng/io/csr_triplet.hpp>

#include <cstddef>
#include <cstdint>

namespace {

template <typename vertex_t, typename edge_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& prefix) {
  dyng::io::csr_triplet_options opt;
  opt.num_weights = in.option(0, 4);
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_csr_triplet<vertex_t, edge_t, std::int32_t>(prefix, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& g = *read;
  const vertex_t n = g.num_vertices();
  DYNG_FUZZ_CHECK(n >= 1);
  DYNG_FUZZ_CHECK(g.row_ptr.front() == 0);
  DYNG_FUZZ_CHECK(static_cast<std::size_t>(g.row_ptr.back()) == g.col_ind.size());
  DYNG_FUZZ_CHECK(g.weights.size() == g.col_ind.size() * static_cast<std::size_t>(g.num_weights));
  DYNG_FUZZ_CHECK(opt.num_weights == 0 || g.num_weights == opt.num_weights);
  for (std::size_t v = 1; v < g.row_ptr.size(); ++v) {
    DYNG_FUZZ_CHECK(g.row_ptr[v - 1] <= g.row_ptr[v]);
  }
  for (const vertex_t c : g.col_ind) {
    DYNG_FUZZ_CHECK(c >= 0 && c < n);
  }
  for (const std::int32_t w : g.weights) {
    DYNG_FUZZ_CHECK(w >= 1);
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string copy = dyng::fuzz::files().path("copy");
    dyng::io::write_csr_triplet(copy, g.view());
    for (const char* part : {"RowPtr.txt", "ColInd.txt", "Values.txt"}) {
      dyng::fuzz::written(copy + part);
    }
    dyng::io::csr_triplet_options back_opt;
    back_opt.num_weights = g.num_weights;
    const auto back = dyng::io::read_csr_triplet<vertex_t, edge_t, std::int32_t>(copy, back_opt);
    DYNG_FUZZ_CHECK(back.row_ptr == g.row_ptr);
    DYNG_FUZZ_CHECK(back.col_ind == g.col_ind);
    DYNG_FUZZ_CHECK(back.weights == g.weights);
    DYNG_FUZZ_CHECK(back.num_weights == g.num_weights);
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 3);
  auto& files = dyng::fuzz::files();
  files.write("graphRowPtr.txt", in.file(0));
  files.write("graphColInd.txt", in.file(1));
  files.write("graphValues.txt", in.file(2));
  const std::string prefix = files.path("graph");
  switch (in.option(1, 3)) {
    case 0:
      run<std::int32_t, std::int32_t>(in, prefix);
      break;
    case 1:
      run<std::int32_t, std::int64_t>(in, prefix);
      break;
    default:
      run<std::int64_t, std::int64_t>(in, prefix);
      break;
  }
  return 0;
}
