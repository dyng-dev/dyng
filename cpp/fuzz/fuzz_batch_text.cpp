// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_batch_text.cpp
 * @brief libFuzzer target of dyng::io::read_batches() and read_hyperedge_batches(), the `.dgt`
 *        batch text format (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the weight count K (-1 = from the first insertion, 0, 1, 2; graph batches), 1 the
 * vertex count (unchecked -1, 8, 1000), 2 the batch type (graph batches: int32 ids with int32
 * weights, int64 ids with int32 weights, int32 ids unweighted, int64 ids unweighted; hypergraph
 * batches: int32 ids unweighted, int32 ids with int32 weights). One file. An accepted input must
 * give consistent batches with ids in range, and the writer followed by the reader must give the
 * same batches.
 */
#include "fuzz_input.hpp"

#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/io/batch_io.hpp>

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

namespace {

template <typename vertex_t, typename weight_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& path) {
  constexpr std::int64_t counts[] = {-1, 8, 1000};
  dyng::io::batch_file_options opt;
  opt.num_weights = in.option(0, 4) - 1;
  opt.num_vertices = counts[in.option(1, 3)];
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_batches<vertex_t, weight_t>(path, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& batches = *read;
  const auto in_range = [&](vertex_t v) {
    return v >= 0 && (opt.num_vertices < 0 || static_cast<std::int64_t>(v) < opt.num_vertices);
  };
  std::vector<dyng::edge_batch_view<vertex_t, weight_t>> views;
  for (const auto& b : batches) {
    DYNG_FUZZ_CHECK(opt.num_weights < 0 || b.num_insertions() == 0 ||
                    b.num_weights() == opt.num_weights);
    DYNG_FUZZ_CHECK(b.insert_dst().size() == b.num_insertions());
    DYNG_FUZZ_CHECK(b.delete_dst().size() == b.num_deletions());
    DYNG_FUZZ_CHECK(b.insert_weights().size() ==
                    b.num_insertions() * static_cast<std::size_t>(b.num_weights()));
    for (std::size_t i = 0; i < b.num_insertions(); ++i) {
      DYNG_FUZZ_CHECK(in_range(b.insert_src()[i]) && in_range(b.insert_dst()[i]));
    }
    for (std::size_t i = 0; i < b.num_deletions(); ++i) {
      DYNG_FUZZ_CHECK(in_range(b.delete_src()[i]) && in_range(b.delete_dst()[i]));
    }
    views.push_back(b.view());
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string copy = dyng::fuzz::files().path("copy.dgt");
    dyng::io::write_batches(copy, views);
    dyng::fuzz::written(copy);
    const auto back = dyng::io::read_batches<vertex_t, weight_t>(copy, opt);
    DYNG_FUZZ_CHECK(back.size() == batches.size());
    for (std::size_t i = 0; i < back.size(); ++i) {
      DYNG_FUZZ_CHECK(back[i].insert_src() == batches[i].insert_src());
      DYNG_FUZZ_CHECK(back[i].insert_dst() == batches[i].insert_dst());
      DYNG_FUZZ_CHECK(back[i].insert_weights() == batches[i].insert_weights());
      DYNG_FUZZ_CHECK(back[i].delete_src() == batches[i].delete_src());
      DYNG_FUZZ_CHECK(back[i].delete_dst() == batches[i].delete_dst());
    }
  });
}

template <typename weight_t>
void run_hyperedges(const dyng::fuzz::fuzz_input& in, const std::string& path) {
  constexpr std::int64_t counts[] = {-1, 8, 1000};
  dyng::io::hyperedge_batch_file_options opt;
  opt.num_vertices = counts[in.option(1, 3)];
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return dyng::io::read_hyperedge_batches<std::int32_t, weight_t>(path, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& batches = *read;
  std::vector<dyng::hyperedge_batch_view<std::int32_t, weight_t>> views;
  for (const auto& b : batches) {
    const std::size_t n = b.num_insertions();
    DYNG_FUZZ_CHECK(n == 0 || (b.insert_offsets().front() == 0 &&
                               b.insert_offsets().back() ==
                                   static_cast<std::int64_t>(b.insert_members().size())));
    for (std::size_t i = 0; i < n; ++i) {
      DYNG_FUZZ_CHECK(b.insert_offsets()[i] < b.insert_offsets()[i + 1]);  // no empty insertion
    }
    for (const std::int32_t v : b.insert_members()) {
      DYNG_FUZZ_CHECK(v >= 0 && (opt.num_vertices < 0 || v < opt.num_vertices));
    }
    for (const std::int64_t id : b.delete_hyperedges()) DYNG_FUZZ_CHECK(id >= 0);
    DYNG_FUZZ_CHECK(dyng::is_unweighted_v<weight_t> ? b.insert_weights().empty()
                                                    : b.insert_weights().size() == n);
    views.push_back(b.view());
  }
  dyng::fuzz::round_trip([&] {
    // write -> read
    const std::string copy = dyng::fuzz::files().path("copy.dgt");
    dyng::io::write_hyperedge_batches(copy, views);
    dyng::fuzz::written(copy);
    const auto back = dyng::io::read_hyperedge_batches<std::int32_t, weight_t>(copy, opt);
    DYNG_FUZZ_CHECK(back.size() == batches.size());
    for (std::size_t i = 0; i < back.size(); ++i) {
      DYNG_FUZZ_CHECK(back[i].delete_hyperedges() == batches[i].delete_hyperedges());
      DYNG_FUZZ_CHECK(back[i].insert_offsets() == batches[i].insert_offsets());
      DYNG_FUZZ_CHECK(back[i].insert_members() == batches[i].insert_members());
      DYNG_FUZZ_CHECK(back[i].insert_weights() == batches[i].insert_weights());
    }
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.dgt", in.file(0));
  switch (in.option(2, 6)) {
    case 0:
      run<std::int32_t, std::int32_t>(in, path);
      break;
    case 1:
      run<std::int64_t, std::int32_t>(in, path);
      break;
    case 2:
      run<std::int32_t, dyng::unweighted>(in, path);
      break;
    case 3:
      run<std::int64_t, dyng::unweighted>(in, path);
      break;
    case 4:
      run_hyperedges<dyng::unweighted>(in, path);
      break;
    default:
      run_hyperedges<std::int32_t>(in, path);
      break;
  }
  return 0;
}
