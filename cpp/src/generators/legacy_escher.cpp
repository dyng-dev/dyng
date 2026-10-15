// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:src/graphGeneration.cpp (hyperedge2vertex)
// Derived from ESCHER-GPU@abcf9b2:utils/batch.cpp (BatchGenerator::next)
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (DynamicHypergraph::update, the
// host model)
/**
 * @file legacy_escher.cpp
 * @brief generators::legacy for ESCHER-GPU: its synthetic hypergraph and its batch generator, one
 *        std::mt19937_64 consumed in A's draw order, over a hypergraph or over a host model
 *        (escher_host_model, A's host rows), and the run-time check of the standard library.
 *
 * Every std::uniform_int_distribution of the original (`<int>` for the sizes and vertices,
 * `<size_t>` for the deletions and the samples) draws from a 64-bit engine, which libstdc++
 * downscales with Lemire's method on 128-bit products: detail::legacy_uniform_int64 over the
 * range's width, plus the lower bound (for `<int>` the width is computed in 64-bit unsigned
 * arithmetic, as libstdc++ does).
 */
#include "hypergraph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/rng.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/generators/legacy.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::generators::legacy {

namespace {

/// std::uniform_int_distribution<int>(a, b)(rng) of libstdc++ with a 64-bit engine.
std::int64_t uniform_int(std::mt19937_64& rng, std::int64_t a, std::int64_t b) {
  const std::uint64_t width = static_cast<std::uint64_t>(b) - static_cast<std::uint64_t>(a);
  return a + static_cast<std::int64_t>(detail::legacy_uniform_int64(rng, 0, width));
}

void check_graph_options(const escher_hypergraph_options& opt, const char* who) {
  DYNG_EXPECTS(opt.num_hyperedges >= 0, who, ": num_hyperedges must be >= 0, got ",
               opt.num_hyperedges);
  DYNG_EXPECTS(opt.max_cardinality >= 1, who, ": max_cardinality must be >= 1, got ",
               opt.max_cardinality);
  DYNG_EXPECTS(opt.min_vertex >= 0 && opt.min_vertex <= opt.max_vertex, who, ": the vertex range [",
               opt.min_vertex, ", ", opt.max_vertex, "] is invalid");
}

/// hyperedge2vertex(n, m, r1, r2, rng) of ESCHER-GPU, 0-based: the rows, sorted.
std::vector<std::vector<std::int64_t>> hyperedge2vertex(std::int64_t n, std::int64_t m,
                                                        std::int64_t r1, std::int64_t r2,
                                                        std::mt19937_64& rng) {
  std::vector<std::vector<std::int64_t>> out(static_cast<std::size_t>(n));
  const std::int64_t range = r2 - r1 + 1;
  for (auto& row : out) {
    const std::int64_t size = std::min(uniform_int(rng, 1, m), range);
    std::set<std::int64_t> unique;
    while (static_cast<std::int64_t>(unique.size()) < size) unique.insert(uniform_int(rng, r1, r2));
    row.assign(unique.begin(), unique.end());
  }
  return out;
}

/// The weight of every generated hyperedge: 1 (nothing for unweighted).
template <typename weight_t>
constexpr weight_t unit_weight() noexcept {
  if constexpr (is_unweighted_v<weight_t>) {
    return weight_t{};
  } else {
    return weight_t{1};
  }
}

template <typename vertex_t, typename weight_t, typename rows_t>
void add_rows(hyperedge_list<vertex_t, weight_t>& list, const rows_t& rows) {
  std::vector<vertex_t> members;
  for (const auto& row : rows) {
    members.assign(row.begin(), row.end());
    list.add_hyperedge(host_view(members), unit_weight<weight_t>());
  }
}

}  // namespace

std::string_view to_string(escher_insert_mode mode) noexcept {
  switch (mode) {
    case escher_insert_mode::random:
      return "random";
    case escher_insert_mode::sample:
      return "sample";
  }
  return "unknown";
}

bool standard_library_matches_legacy_streams() try {
#if defined(__GLIBCXX__)
  bool same = true;
  for (const std::uint64_t seed : {1ULL, 5ULL, 20260725ULL}) {
    std::mt19937_64 ours(seed);
    std::mt19937_64 theirs(seed);
    for (const std::int64_t hi : {1LL, 10LL, 99LL, 333332LL, 2147483646LL}) {
      std::uniform_int_distribution<int> a(0, static_cast<int>(hi));
      std::uniform_int_distribution<std::size_t> b(3, static_cast<std::size_t>(hi) + 3);
      std::uniform_int_distribution<long long> c(-hi, hi);
      for (int i = 0; i < 64; ++i) {
        same = same && uniform_int(ours, 0, hi) == a(theirs);
        same = same && detail::legacy_uniform_int64(ours, 3, static_cast<std::uint64_t>(hi) + 3) ==
                           b(theirs);
        same = same && uniform_int(ours, -hi, hi) == c(theirs);
      }
    }
    std::mt19937 ours32(static_cast<std::uint32_t>(seed));
    std::mt19937 theirs32(static_cast<std::uint32_t>(seed));
    std::uniform_int_distribution<int> d(1, 100);
    for (int i = 0; i < 64; ++i)
      same = same && detail::legacy_uniform_int(ours32, 1, 100) == d(theirs32);
    std::vector<int> x(1001);
    std::iota(x.begin(), x.end(), 0);
    std::vector<int> y = x;
    detail::legacy_shuffle(x, ours);
    std::shuffle(y.begin(), y.end(), theirs);
    same = same && x == y && ours() == theirs();
  }
  return same;
#else
  return false;
#endif
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::standard_library_matches_legacy_streams")

template <typename vertex_t, typename weight_t>
hyperedge_list<vertex_t, weight_t> escher_hypergraph(std::mt19937_64& rng,
                                                     const escher_hypergraph_options& opt) try {
  check_graph_options(opt, "escher_hypergraph");
  const auto rows = hyperedge2vertex(opt.num_hyperedges, opt.max_cardinality, opt.min_vertex,
                                     opt.max_vertex, rng);
  hyperedge_list<vertex_t, weight_t> list;
  list.num_vertices = static_cast<vertex_t>(static_cast<std::int64_t>(opt.max_vertex) + 1);
  list.offsets.reserve(rows.size() + 1);
  add_rows(list, rows);
  return list;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::escher_hypergraph")

escher_batch_generator::escher_batch_generator(std::mt19937_64& rng,
                                               const escher_batch_options& opt)
    : rng_(&rng), opt_(opt) {
  const bool demo = opt.num_deletions == -1 && opt.num_insertions == -1;
  DYNG_EXPECTS(demo || (opt.num_deletions >= 0 && opt.num_insertions >= 0),
               "escher_batch_generator: the counts must both be >= 0, or both -1 (the demo "
               "batch), got ",
               opt.num_deletions, " and ", opt.num_insertions);
  if (opt.insert_mode == escher_insert_mode::random) {
    escher_hypergraph_options g = opt.graph;
    g.num_hyperedges = 0;
    check_graph_options(g, "escher_batch_generator");
  }
}

namespace {

/// BatchGenerator::next of ESCHER-GPU over a live state: `live` the live ids in increasing order,
/// `row(id)` the members of a live id, `num_vertices` the hypergraph's vertex count. Shared by
/// both next() overloads, so they draw the same batches from the same state.
template <typename vertex_t, typename weight_t, typename row_fn_t>
hyperedge_batch<vertex_t, weight_t> draw_batch(std::mt19937_64& rng,
                                               const escher_batch_options& opt,
                                               std::vector<std::int64_t> live, const row_fn_t& row,
                                               std::int64_t num_vertices) {
  const auto num_live = static_cast<std::int64_t>(live.size());
  std::int64_t num_del = opt.num_deletions;
  std::int64_t num_ins = opt.num_insertions;
  if (num_del == -1 && num_ins == -1) {
    num_del = std::min<std::int64_t>(2, num_live);  // the original demo batch
    num_ins = num_del + 1;
  }
  num_del = std::max<std::int64_t>(0, std::min(num_del, num_live));
  num_ins = std::max<std::int64_t>(0, num_ins);

  hyperedge_batch<vertex_t, weight_t> batch;
  // Deletions: distinct live keys, uniformly at random (partial Fisher-Yates), then sorted.
  for (std::int64_t i = 0; i < num_del; ++i) {
    const auto j = detail::legacy_uniform_int64(rng, static_cast<std::uint64_t>(i),
                                                static_cast<std::uint64_t>(num_live - 1));
    std::swap(live[static_cast<std::size_t>(i)], live[static_cast<std::size_t>(j)]);
  }
  std::vector<std::int64_t> deleted(live.begin(), live.begin() + num_del);
  std::sort(deleted.begin(), deleted.end());
  batch.reserve(static_cast<std::size_t>(num_ins), 0, deleted.size());
  for (const std::int64_t id : deleted) batch.delete_hyperedge(id);
  // Insertions: copies of sampled live hyperedges, or fresh random ones.
  if (opt.insert_mode == escher_insert_mode::sample && !live.empty()) {
    for (std::int64_t i = 0; i < num_ins; ++i) {
      const auto j = detail::legacy_uniform_int64(rng, 0, live.size() - 1);
      const std::vector<vertex_t> members = row(live[static_cast<std::size_t>(j)]);
      batch.insert_hyperedge(host_view(members), unit_weight<weight_t>());
    }
  } else {
    const std::int64_t r2 = std::max<std::int64_t>(opt.graph.max_vertex, num_vertices - 1);
    const auto rows =
        hyperedge2vertex(num_ins, opt.graph.max_cardinality, opt.graph.min_vertex, r2, rng);
    std::vector<vertex_t> members;
    for (const auto& r : rows) {
      members.assign(r.begin(), r.end());
      batch.insert_hyperedge(host_view(members), unit_weight<weight_t>());
    }
  }
  return batch;
}

template <typename value_t>
void expect_host(array_view<value_t> v, const char* who, const char* what) {
  DYNG_EXPECTS(v.empty() || is_host_accessible(v.space()), who, ": ", what,
               " must be host-accessible memory");
}

/// The members of row [b, e) of a host CSR, checked, sorted and deduplicated.
template <typename vertex_t>
std::vector<vertex_t> checked_row(array_view<const vertex_t> members, std::int64_t b,
                                  std::int64_t e, vertex_t num_vertices, const char* who) {
  std::vector<vertex_t> r(members.data() + b, members.data() + e);
  for (const vertex_t v : r) {
    DYNG_EXPECTS(v >= 0 && v < num_vertices, who, ": member ", v, " is outside [0, ", num_vertices,
                 ")");
  }
  std::sort(r.begin(), r.end());
  r.erase(std::unique(r.begin(), r.end()), r.end());
  return r;
}

void check_offsets(array_view<const std::int64_t> offsets, std::size_t members, const char* who) {
  DYNG_EXPECTS(offsets.empty() || (offsets[0] == 0 && offsets[offsets.size() - 1] ==
                                                          static_cast<std::int64_t>(members)),
               who, ": the offsets must run from 0 to the member count");
  for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
    DYNG_EXPECTS(offsets[i + 1] >= offsets[i], who, ": the offsets decrease at ", i);
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyperedge_batch<vertex_t, weight_t> escher_batch_generator::next(
    const resources& res, const hypergraph<vertex_t, weight_t>& h) try {
  DYNG_EXPECTS((res.get_backend() == backend::cuda) == (h.space() == memory_space::device),
               "escher_batch_generator::next: the resources (", to_string(res.get_backend()),
               ") are not of the hypergraph's backend");
  // The live state: A's host model (row key - 1, empty rows are absent keys).
  const auto view = h.h2v();
  const std::vector<std::int64_t> row_ptr = to_vector(res, view.row_ptr);
  const std::vector<vertex_t> col_ind = to_vector(res, view.col_ind);
  const auto row = [&](std::int64_t id) {
    return std::vector<vertex_t>(col_ind.begin() + row_ptr[static_cast<std::size_t>(id)],
                                 col_ind.begin() + row_ptr[static_cast<std::size_t>(id) + 1]);
  };
  std::vector<std::int64_t> live;
  live.reserve(static_cast<std::size_t>(h.num_hyperedges()));
  for (std::int64_t id = 0; id + 1 < static_cast<std::int64_t>(row_ptr.size()); ++id) {
    if (row_ptr[static_cast<std::size_t>(id) + 1] > row_ptr[static_cast<std::size_t>(id)]) {
      live.push_back(id);
    }
  }
  return draw_batch<vertex_t, weight_t>(*rng_, opt_, std::move(live), row,
                                        static_cast<std::int64_t>(h.num_vertices()));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::escher_batch_generator::next")

template <typename weight_t, typename vertex_t>
hyperedge_batch<vertex_t, weight_t> escher_batch_generator::next(
    const escher_host_model<vertex_t>& model) try {
  std::vector<std::int64_t> live;
  live.reserve(static_cast<std::size_t>(model.num_hyperedges()));
  for (std::int64_t id = 0; id < model.num_rows(); ++id) {
    if (!model.row(id).empty()) live.push_back(id);
  }
  const auto row = [&](std::int64_t id) { return model.row(id); };
  return draw_batch<vertex_t, weight_t>(*rng_, opt_, std::move(live), row,
                                        static_cast<std::int64_t>(model.num_vertices()));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::escher_batch_generator::next")

template <typename vertex_t>
template <typename weight_t>
escher_host_model<vertex_t>::escher_host_model(
    const hyperedge_list_view<vertex_t, weight_t>& list) try {
  constexpr const char* who = "escher_host_model";
  DYNG_EXPECTS(list.num_vertices >= 0, who, ": negative vertex count");
  expect_host(list.offsets, who, "the offsets");
  expect_host(list.members, who, "the members");
  check_offsets(list.offsets, list.members.size(), who);
  num_vertices_ = list.num_vertices;
  rows_.resize(static_cast<std::size_t>(list.num_hyperedges()));
  for (std::int64_t i = 0; i < list.num_hyperedges(); ++i) {
    rows_[static_cast<std::size_t>(i)] =
        checked_row(list.members, list.offsets[static_cast<std::size_t>(i)],
                    list.offsets[static_cast<std::size_t>(i) + 1], num_vertices_, who);
    num_live_ += rows_[static_cast<std::size_t>(i)].empty() ? 0 : 1;
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::escher_host_model")

template <typename vertex_t>
template <typename weight_t>
void escher_host_model<vertex_t>::apply(const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                        const hyper_apply_summary& summary) try {
  constexpr const char* who = "escher_host_model::apply";
  expect_host(batch.delete_hyperedges, who, "the deleted ids");
  expect_host(batch.insert_offsets, who, "the insertion offsets");
  expect_host(batch.insert_members, who, "the inserted members");
  check_offsets(batch.insert_offsets, batch.insert_members.size(), who);
  const std::size_t num_ins = batch.num_insertions();
  DYNG_EXPECTS(summary.inserted_ids.size() == num_ins, who, ": ", summary.inserted_ids.size(),
               " inserted ids for ", num_ins, " insertions");
  // Everything is checked and copied before the model changes (strong guarantee).
  std::vector<std::vector<vertex_t>> inserted(num_ins);
  std::int64_t top = static_cast<std::int64_t>(rows_.size());
  for (std::size_t i = 0; i < num_ins; ++i) {
    inserted[i] = checked_row(batch.insert_members, batch.insert_offsets[i],
                              batch.insert_offsets[i + 1], num_vertices_, who);
    DYNG_EXPECTS(!inserted[i].empty(), who, ": inserted hyperedge ", i, " is empty");
    DYNG_EXPECTS(summary.inserted_ids[i] >= 0, who, ": inserted id ", summary.inserted_ids[i],
                 " is negative");
    top = std::max(top, summary.inserted_ids[i] + 1);
  }
  rows_.reserve(static_cast<std::size_t>(top));
  // A's DynamicHypergraph::update: the deleted keys' rows are cleared, then the inserted rows set
  // at the keys the insert assigned (the vector grows, never shrinks).
  for (const std::int64_t id : batch.delete_hyperedges) {
    if (id >= 0 && id < static_cast<std::int64_t>(rows_.size()) &&
        !rows_[static_cast<std::size_t>(id)].empty()) {
      rows_[static_cast<std::size_t>(id)].clear();
      --num_live_;
    }
  }
  if (top > static_cast<std::int64_t>(rows_.size())) rows_.resize(static_cast<std::size_t>(top));
  for (std::size_t i = 0; i < num_ins; ++i) {
    auto& r = rows_[static_cast<std::size_t>(summary.inserted_ids[i])];
    num_live_ += r.empty() ? 1 : 0;
    r = std::move(inserted[i]);
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::escher_host_model::apply")

template <typename vertex_t>
const std::vector<vertex_t>& escher_host_model<vertex_t>::row(std::int64_t id) const {
  DYNG_EXPECTS(id >= 0 && id < static_cast<std::int64_t>(rows_.size()),
               "escher_host_model::row: id ", id, " is outside [0, ", rows_.size(), ")");
  return rows_[static_cast<std::size_t>(id)];
}

template class escher_host_model<std::int32_t>;

#define DYNG_INSTANTIATE_ESCHER_LEGACY(V, W)                                                      \
  template hyperedge_list<V, W> escher_hypergraph<V, W>(std::mt19937_64&,                         \
                                                        const escher_hypergraph_options&);        \
  template hyperedge_batch<V, W> escher_batch_generator::next<V, W>(const resources&,             \
                                                                    const hypergraph<V, W>&);     \
  template hyperedge_batch<V, W> escher_batch_generator::next<W, V>(const escher_host_model<V>&); \
  template escher_host_model<V>::escher_host_model(const hyperedge_list_view<V, W>&);             \
  template void escher_host_model<V>::apply<W>(const hyperedge_batch_view<V, W>&,                 \
                                               const hyper_apply_summary&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_ESCHER_LEGACY)
#undef DYNG_INSTANTIATE_ESCHER_LEGACY

}  // namespace dyng::generators::legacy
