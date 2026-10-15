// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_hg.cpp
 * @brief libFuzzer target of dyng::io::read_hypergraph(), the `.hg` hypergraph files
 *        (cpp/fuzz/fuzz_input.hpp).
 *
 * Options: 0 the relabelling (none, dense_sorted, first_appearance), 1 the index base (1, 0), 2
 * dedupe_hyperedges (off, on), 3 max_cardinality (-1, 2, 25), 4 the weights (unweighted, from the
 * file, random U[1, 100], H-SOSP's option set), 5 the vertex count (-1, 1000; relabelling none
 * only). One file. An accepted input must give a consistent hyperedge list (offsets from 0,
 * members in [0, n) and distinct within a hyperedge, sorted unless numbered by first appearance,
 * no hyperedge above max_cardinality, weights in range), and write_hypergraph() followed by
 * read_hypergraph() (relabelling none, the same vertex count) must give the same member sets and
 * weights.
 */
#include "fuzz_input.hpp"

#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace {

namespace io = dyng::io;

template <typename weight_t>
std::vector<std::vector<std::int32_t>> sorted_rows(
    const dyng::hyperedge_list<std::int32_t, weight_t>& l) {
  std::vector<std::vector<std::int32_t>> out;
  for (std::size_t i = 0; i + 1 < l.offsets.size(); ++i) {
    std::vector<std::int32_t> r(l.members.begin() + l.offsets[i],
                                l.members.begin() + l.offsets[i + 1]);
    std::sort(r.begin(), r.end());
    out.push_back(r);
  }
  return out;
}

template <typename weight_t>
void run(const dyng::fuzz::fuzz_input& in, const std::string& path,
         const io::hypergraph_read_options& opt) {
  const auto read = dyng::fuzz::read_or_reject(
      [&] { return io::read_hypergraph<std::int32_t, weight_t>(path, opt); });
  if (!read) {
    return;  // rejected with a documented exception
  }
  const auto& l = *read;
  (void)in;
  const std::int64_t n = l.num_vertices;
  DYNG_FUZZ_CHECK(n >= 0);
  DYNG_FUZZ_CHECK(!l.offsets.empty() && l.offsets.front() == 0);
  DYNG_FUZZ_CHECK(l.offsets.back() == static_cast<std::int64_t>(l.members.size()));
  for (std::size_t i = 0; i + 1 < l.offsets.size(); ++i) {
    const std::int64_t size = l.offsets[i + 1] - l.offsets[i];
    DYNG_FUZZ_CHECK(size >= 1);
    DYNG_FUZZ_CHECK(opt.max_cardinality < 0 || size <= opt.max_cardinality);
    std::vector<std::int32_t> r(l.members.begin() + l.offsets[i],
                                l.members.begin() + l.offsets[i + 1]);
    for (const std::int32_t v : r) DYNG_FUZZ_CHECK(v >= 0 && v < n);
    if (opt.relabel != io::vertex_relabel::first_appearance) {
      DYNG_FUZZ_CHECK(std::is_sorted(r.begin(), r.end()));
    }
    std::sort(r.begin(), r.end());
    DYNG_FUZZ_CHECK(std::adjacent_find(r.begin(), r.end()) == r.end());
  }
  if constexpr (!dyng::is_unweighted_v<weight_t>) {
    DYNG_FUZZ_CHECK(l.weights.size() + 1 == l.offsets.size());
    for (const weight_t w : l.weights) DYNG_FUZZ_CHECK(w >= 1);
    if (opt.random_weights) {
      for (const weight_t w : l.weights) {
        DYNG_FUZZ_CHECK(w >= opt.random_weights->lo && w <= opt.random_weights->hi);
      }
    }
  } else {
    DYNG_FUZZ_CHECK(l.weights.empty());
  }
  dyng::fuzz::round_trip([&] {
    // write -> read (relabelling none, the same vertex count, weights from the file)
    const std::string copy = dyng::fuzz::files().path("copy.hg");
    io::write_hypergraph(copy, l.view(), 1);
    dyng::fuzz::written(copy);
    io::hypergraph_read_options back_opt;
    back_opt.num_vertices = n;
    const auto back = io::read_hypergraph<std::int32_t, weight_t>(copy, back_opt);
    DYNG_FUZZ_CHECK(back.num_vertices == l.num_vertices);
    DYNG_FUZZ_CHECK(sorted_rows(back) == sorted_rows(l));
    DYNG_FUZZ_CHECK(back.weights == l.weights);
  });
}

}  // namespace

extern "C" int LLVMFuzzerTestOneInput(const std::uint8_t* data, std::size_t size) {
  const dyng::fuzz::fuzz_input in(data, size, 1);
  const std::string path = dyng::fuzz::files().write("input.hg", in.file(0));
  io::hypergraph_read_options opt;
  constexpr io::vertex_relabel relabels[] = {io::vertex_relabel::none,
                                             io::vertex_relabel::dense_sorted,
                                             io::vertex_relabel::first_appearance};
  opt.relabel = relabels[in.option(0, 3)];
  opt.index_base = in.option(1, 2) == 0 ? 1 : 0;
  opt.dedupe_hyperedges = in.option(2, 2) == 1;
  constexpr std::int64_t cards[] = {-1, 2, 25};
  opt.max_cardinality = cards[in.option(3, 3)];
  if (opt.relabel == io::vertex_relabel::none && in.option(5, 2) == 1) opt.num_vertices = 1000;
  switch (in.option(4, 4)) {
    case 0:
      run<dyng::unweighted>(in, path, opt);
      break;
    case 1:
      run<std::int32_t>(in, path, opt);
      break;
    case 2:
      opt.random_weights = io::weight_spec{1, 100, 7, false};
      run<std::int32_t>(in, path, opt);
      break;
    default:
      run<std::int32_t>(in, path, io::hypergraph_read_options::hsosp(1, opt.max_cardinality));
      break;
  }
  return 0;
}
