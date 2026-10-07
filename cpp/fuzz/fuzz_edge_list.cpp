// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_edge_list.cpp
 * @brief libFuzzer target of dyng::io::read_edge_list() (cpp/fuzz/fuzz_input.hpp), including its
 *        Matrix Market path (a file whose first line starts with `%%MatrixMarket`).
 *
 * Options: 0 the weight columns K (0..3), 1 the ids (compact, as_is), 2 the index base (0, 1),
 * 3 symmetrize, 4 drop_self_loops, 5 the duplicates (merge, keep), 6 the parser threads (1..3),
 * 7 the result type (int32 ids with int32 weights, int64 ids with int32 weights, int32 ids
 * unweighted). One file. An accepted input must give a consistent, sorted edge list whose
 * id mapping and timestamps have the documented sizes, and write_edge_list() followed by
 * read_edge_list() (as_is, the same K, no self-loop dropping) must give the same edges.
 */
#include "fuzz_input.hpp"

#include <dyng/core/types.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <cstddef>
#include <cstdint>

namespace {

template <typename vertex_t, typename weight_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& path) {
  dyng::io::edge_list_options opt;
  opt.num_weights = in.option(0, 4);
  opt.ids = in.option(1, 2) == 0 ? dyng::io::vertex_ids::compact : dyng::io::vertex_ids::as_is;
  opt.index_base = in.option(2, 2);
  opt.symmetrize = in.option(3, 2) == 1;
  opt.drop_self_loops = in.option(4, 2) == 1;
  opt.duplicates =
      in.option(5, 2) == 0 ? dyng::io::duplicate_edges::merge : dyng::io::duplicate_edges::keep;
  opt.threads = 1 + in.option(6, 3);
  dyng::io::edge_list_info info;
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_edge_list<vertex_t, weight_t>(path, opt, &info); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& edges = *read;
  const std::size_t m = edges.num_edges();
  const bool keep = opt.duplicates == dyng::io::duplicate_edges::keep;
  DYNG_FUZZ_CHECK(edges.dst.size() == m);
  DYNG_FUZZ_CHECK(edges.weights.size() == m * static_cast<std::size_t>(edges.num_weights));
  DYNG_FUZZ_CHECK(info.timestamps.size() == (keep ? m : 0));
  if (opt.ids == dyng::io::vertex_ids::compact) {
    DYNG_FUZZ_CHECK(info.external_ids.size() == static_cast<std::size_t>(edges.num_vertices));
    for (std::size_t i = 1; i < info.external_ids.size(); ++i) {
      DYNG_FUZZ_CHECK(info.external_ids[i - 1] < info.external_ids[i]);
    }
  }
  for (std::size_t e = 0; e < m; ++e) {
    DYNG_FUZZ_CHECK(edges.src[e] >= 0 && edges.src[e] < edges.num_vertices);
    DYNG_FUZZ_CHECK(edges.dst[e] >= 0 && edges.dst[e] < edges.num_vertices);
    DYNG_FUZZ_CHECK(!opt.drop_self_loops || edges.src[e] != edges.dst[e]);
    if (e > 0) {
      const bool before = edges.src[e - 1] < edges.src[e] ||
                          (edges.src[e - 1] == edges.src[e] && edges.dst[e - 1] < edges.dst[e]);
      const bool same = edges.src[e - 1] == edges.src[e] && edges.dst[e - 1] == edges.dst[e];
      DYNG_FUZZ_CHECK(before || (keep && same));
    }
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string copy = dyng::fuzz::files().path("copy.txt");
    dyng::io::write_edge_list(copy, edges.view());
    dyng::fuzz::written(copy);
    dyng::io::edge_list_options back_opt;
    back_opt.num_weights = edges.num_weights;
    back_opt.ids = dyng::io::vertex_ids::as_is;
    back_opt.drop_self_loops = false;
    back_opt.duplicates = opt.duplicates;
    back_opt.threads = 1;
    const auto back = dyng::io::read_edge_list<vertex_t, weight_t>(copy, back_opt);
    DYNG_FUZZ_CHECK(back.src == edges.src);
    DYNG_FUZZ_CHECK(back.dst == edges.dst);
    DYNG_FUZZ_CHECK(back.weights == edges.weights);
    DYNG_FUZZ_CHECK(back.num_vertices <= edges.num_vertices);
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.txt", in.file(0));
  switch (in.option(7, 3)) {
    case 0:
      run<std::int32_t, std::int32_t>(in, path);
      break;
    case 1:
      run<std::int64_t, std::int32_t>(in, path);
      break;
    default:
      run<std::int32_t, dyng::unweighted>(in, path);
      break;
  }
  return 0;
}
