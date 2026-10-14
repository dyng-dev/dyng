// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraphs.cpp
 * @brief Seeded hypergraph and hyperedge-batch generators (PLAN Section 5.8).
 *
 * The streams use std::mt19937_64 (fully specified by the C++ standard) and the library's own
 * algorithms on top of it: a uniform integer in [lo, hi] is Lemire's multiply-and-reject on
 * 128-bit products (detail::legacy_uniform_int64, the library's implementation), a shuffle is a
 * Fisher-Yates pass from the front with one such draw per position. Nothing depends on
 * the standard library's distributions, so the streams are bit-exact per (seed, major version)
 * everywhere; the golden hashes of cpp/tests/generators/hypergraph_generators_test.cpp pin them.
 * Changing a draw (its order, its range, the algorithm) changes the streams: a major-version
 * change (PLAN Section 5.9).
 */
#include "hypergraph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/parser.hpp"
#include "util/rng.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/generators/hypergraphs.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <set>
#include <string_view>
#include <utility>
#include <vector>

namespace dyng::generators {

std::string_view to_string(hyperedge_batch_mix mix) noexcept {
  switch (mix) {
    case hyperedge_batch_mix::insert_only:
      return "insert_only";
    case hyperedge_batch_mix::delete_only:
      return "delete_only";
    case hyperedge_batch_mix::mixed:
      return "mixed";
    case hyperedge_batch_mix::reuse:
      return "reuse";
    case hyperedge_batch_mix::heavy_delete:
      return "heavy_delete";
    case hyperedge_batch_mix::delete_all:
      return "delete_all";
    case hyperedge_batch_mix::reinsert:
      return "reinsert";
    case hyperedge_batch_mix::larger:
      return "larger";
    case hyperedge_batch_mix::duplicates:
      return "duplicates";
    case hyperedge_batch_mix::dead_ids:
      return "dead_ids";
    case hyperedge_batch_mix::skewed:
      return "skewed";
    case hyperedge_batch_mix::limits:
      return "limits";
    case hyperedge_batch_mix::empty:
      return "empty";
  }
  return "unknown";
}

namespace {

using engine = std::mt19937_64;

/// A uniform integer in [lo, hi] (lo <= hi).
std::int64_t draw(engine& rng, std::int64_t lo, std::int64_t hi) {
  const std::uint64_t width = static_cast<std::uint64_t>(hi) - static_cast<std::uint64_t>(lo);
  return static_cast<std::int64_t>(static_cast<std::uint64_t>(lo) +
                                   detail::legacy_uniform_int64(rng, 0, width));
}

/// Fisher-Yates from the front: position i swaps with a uniform position in [i, size).
template <typename value_t>
void shuffle_prefix(std::vector<value_t>& v, std::size_t count, engine& rng) {
  const auto n = static_cast<std::int64_t>(v.size());
  for (std::size_t i = 0; i < count && static_cast<std::int64_t>(i) + 1 < n; ++i) {
    const auto j = static_cast<std::size_t>(draw(rng, static_cast<std::int64_t>(i), n - 1));
    std::swap(v[i], v[j]);
  }
}

/// `card` distinct vertices of [0, n), sorted (card clamped to [1, n]).
template <typename vertex_t>
std::vector<vertex_t> draw_members(std::int64_t n, std::int64_t card, engine& rng) {
  card = std::max<std::int64_t>(1, std::min(card, n));
  std::vector<vertex_t> out;
  out.reserve(static_cast<std::size_t>(card));
  if (card * 2 > n) {
    std::vector<vertex_t> all(static_cast<std::size_t>(n));
    std::iota(all.begin(), all.end(), vertex_t{0});
    shuffle_prefix(all, static_cast<std::size_t>(card), rng);
    out.assign(all.begin(), all.begin() + card);
  } else {
    std::set<std::int64_t> chosen;
    while (static_cast<std::int64_t>(chosen.size()) < card) chosen.insert(draw(rng, 0, n - 1));
    for (const std::int64_t v : chosen) out.push_back(static_cast<vertex_t>(v));
  }
  std::sort(out.begin(), out.end());
  return out;
}

template <typename value_t>
constexpr std::int64_t largest() noexcept {
  if constexpr (is_unweighted_v<value_t>) {
    return 0;
  } else {
    return detail::max_as_int64<value_t>();
  }
}

template <typename vertex_t, typename weight_t>
void check_common(std::int64_t n, std::int64_t min_card, std::int64_t max_card,
                  std::int64_t min_weight, std::int64_t max_weight, const char* who) {
  DYNG_EXPECTS(n >= 0 && n <= detail::max_as_int64<vertex_t>(), who, ": the vertex count ", n,
               " must lie in [0, ", detail::max_as_int64<vertex_t>(), "]");
  DYNG_EXPECTS(min_card >= 1 && min_card <= max_card, who, ": the cardinality range [", min_card,
               ", ", max_card, "] must be non-empty and start at 1 or above");
  if constexpr (!is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(min_weight >= 1 && min_weight <= max_weight && max_weight <= largest<weight_t>(),
                 who, ": the weight range [", min_weight, ", ", max_weight, "] must lie in [1, ",
                 largest<weight_t>(), "]");
  } else {
    (void)min_weight;
    (void)max_weight;
  }
}

template <typename weight_t>
weight_t fixed_weight(std::int64_t w) noexcept {
  if constexpr (is_unweighted_v<weight_t>) {
    (void)w;
    return weight_t{};
  } else {
    return static_cast<weight_t>(w);
  }
}

template <typename weight_t>
weight_t draw_weight(engine& rng, std::int64_t lo, std::int64_t hi) {
  if constexpr (is_unweighted_v<weight_t>) {
    (void)rng;
    (void)lo;
    (void)hi;
    return weight_t{};
  } else {
    return static_cast<weight_t>(draw(rng, lo, hi));
  }
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyperedge_list<vertex_t, weight_t> uniform_hypergraph(const uniform_hypergraph_options& opt) try {
  check_common<vertex_t, weight_t>(opt.num_vertices, opt.min_cardinality, opt.max_cardinality,
                                   opt.min_weight, opt.max_weight, "uniform_hypergraph");
  DYNG_EXPECTS(opt.num_hyperedges >= 0, "uniform_hypergraph: num_hyperedges must be >= 0, got ",
               opt.num_hyperedges);
  DYNG_EXPECTS(opt.num_hyperedges == 0 || opt.num_vertices >= 1,
               "uniform_hypergraph: hyperedges need at least one vertex");
  engine rng(opt.seed);
  hyperedge_list<vertex_t, weight_t> out;
  out.num_vertices = static_cast<vertex_t>(opt.num_vertices);
  out.offsets.reserve(static_cast<std::size_t>(opt.num_hyperedges) + 1);
  for (std::int64_t i = 0; i < opt.num_hyperedges; ++i) {
    const std::int64_t card = draw(rng, opt.min_cardinality, opt.max_cardinality);
    const std::vector<vertex_t> members = draw_members<vertex_t>(opt.num_vertices, card, rng);
    out.add_hyperedge(host_view(members),
                      draw_weight<weight_t>(rng, opt.min_weight, opt.max_weight));
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::uniform_hypergraph")

template <typename vertex_t, typename weight_t>
hyperedge_batch<vertex_t, weight_t> random_hyperedge_batch(vertex_t num_vertices,
                                                           const incidence_view<vertex_t>& h2v,
                                                           array_view<const weight_t> weights,
                                                           const hyperedge_batch_options& opt) try {
  const auto n = static_cast<std::int64_t>(num_vertices);
  check_common<vertex_t, weight_t>(n, opt.min_cardinality, opt.max_cardinality, opt.min_weight,
                                   opt.max_weight, "random_hyperedge_batch");
  DYNG_EXPECTS(opt.size >= -1, "random_hyperedge_batch: size must be >= -1, got ", opt.size);
  DYNG_EXPECTS(opt.skewed_cardinality >= 1,
               "random_hyperedge_batch: skewed_cardinality must be >= 1");
  const auto host = [](const auto& v) { return v.empty() || is_host_accessible(v.space()); };
  DYNG_EXPECTS(host(h2v.row_ptr) && host(h2v.col_ind) && host(weights),
               "random_hyperedge_batch: the rows and weights must be in host-accessible memory");
  const std::int64_t bound = h2v.num_rows();
  DYNG_EXPECTS(bound == 0 || (h2v.row_ptr[0] == 0 &&
                              h2v.row_ptr[static_cast<std::size_t>(bound)] == h2v.num_entries()),
               "random_hyperedge_batch: the row offsets must run from 0 to the entry count");
  const auto row = [&](std::int64_t id) {
    const auto b = static_cast<std::size_t>(h2v.row_ptr[static_cast<std::size_t>(id)]);
    const auto e = static_cast<std::size_t>(h2v.row_ptr[static_cast<std::size_t>(id) + 1]);
    DYNG_EXPECTS(b <= e, "random_hyperedge_batch: the row offsets decrease at id ", id);
    return std::vector<vertex_t>(h2v.col_ind.begin() + b, h2v.col_ind.begin() + e);
  };
  const auto weight_of = [&](std::int64_t id) {
    if constexpr (is_unweighted_v<weight_t>) {
      (void)id;
      return weight_t{};
    } else {
      return static_cast<std::size_t>(id) < weights.size() ? weights[static_cast<std::size_t>(id)]
                                                           : fixed_weight<weight_t>(opt.min_weight);
    }
  };
  std::vector<std::int64_t> live;
  for (std::int64_t id = 0; id < bound; ++id) {
    if (h2v.row_ptr[static_cast<std::size_t>(id) + 1] > h2v.row_ptr[static_cast<std::size_t>(id)]) {
      live.push_back(id);
    }
  }
  engine rng(opt.seed);
  shuffle_prefix(live, live.size(), rng);  // the live ids in a random order
  const auto num_live = static_cast<std::int64_t>(live.size());
  const std::int64_t k = opt.size >= 0 ? opt.size : std::max<std::int64_t>(1, num_live / 10);
  std::size_t next_live = 0;
  const auto take = [&](std::int64_t count) {
    count = std::max<std::int64_t>(
        0, std::min<std::int64_t>(count, num_live - static_cast<std::int64_t>(next_live)));
    std::vector<std::int64_t> chosen(live.begin() + static_cast<std::ptrdiff_t>(next_live),
                                     live.begin() + static_cast<std::ptrdiff_t>(next_live) + count);
    next_live += static_cast<std::size_t>(count);
    return chosen;
  };
  hyperedge_batch<vertex_t, weight_t> out;
  const auto fresh = [&](std::int64_t card) {
    const std::vector<vertex_t> members = draw_members<vertex_t>(n, card, rng);
    out.insert_hyperedge(host_view(members),
                         draw_weight<weight_t>(rng, opt.min_weight, opt.max_weight));
  };
  const auto fresh_random = [&] { fresh(draw(rng, opt.min_cardinality, opt.max_cardinality)); };
  if (n == 0 || opt.mix == hyperedge_batch_mix::empty) {
    return out;  // no vertex: only the empty batch
  }
  switch (opt.mix) {
    case hyperedge_batch_mix::insert_only:
      for (std::int64_t i = 0; i < k; ++i) fresh_random();
      break;
    case hyperedge_batch_mix::delete_only:
      for (const std::int64_t id : take(k)) out.delete_hyperedge(id);
      break;
    case hyperedge_batch_mix::mixed:
      for (const std::int64_t id : take((k + 1) / 2)) out.delete_hyperedge(id);
      for (std::int64_t i = 0; i < (k + 1) / 2; ++i) fresh_random();
      break;
    case hyperedge_batch_mix::reuse: {
      // The largest live hyperedges go (ties: the smaller id); the new ones are no larger than
      // the smallest of them.
      std::vector<std::pair<std::int64_t, std::int64_t>> by_size;  // (-size, id)
      by_size.reserve(live.size());
      for (const std::int64_t id : live) {
        by_size.emplace_back(-(h2v.row_ptr[static_cast<std::size_t>(id) + 1] -
                               h2v.row_ptr[static_cast<std::size_t>(id)]),
                             id);
      }
      std::sort(by_size.begin(), by_size.end());
      const std::int64_t count = std::min<std::int64_t>(k, num_live);
      std::int64_t smallest = opt.max_cardinality;
      for (std::int64_t i = 0; i < count; ++i) {
        smallest = std::min(smallest, -by_size[static_cast<std::size_t>(i)].first);
        out.delete_hyperedge(by_size[static_cast<std::size_t>(i)].second);
      }
      for (std::int64_t i = 0; i < count; ++i) {
        fresh(draw(rng, 1, std::max<std::int64_t>(1, smallest)));
      }
      break;
    }
    case hyperedge_batch_mix::heavy_delete:
      for (const std::int64_t id : take(num_live - num_live / 10)) out.delete_hyperedge(id);
      for (std::int64_t i = 0; i < k; ++i) fresh_random();
      break;
    case hyperedge_batch_mix::delete_all:
      for (const std::int64_t id : take(num_live)) out.delete_hyperedge(id);
      break;
    case hyperedge_batch_mix::reinsert: {
      const std::vector<std::int64_t> gone = take(k);
      for (const std::int64_t id : gone) out.delete_hyperedge(id);
      for (const std::int64_t id : gone) {
        const std::vector<vertex_t> members = row(id);
        out.insert_hyperedge(host_view(members), weight_of(id));
      }
      break;
    }
    case hyperedge_batch_mix::larger: {
      for (const std::int64_t id : take(num_live / 2)) out.delete_hyperedge(id);
      for (std::int64_t i = 0; i < 2 * num_live + 3; ++i) fresh_random();
      break;
    }
    case hyperedge_batch_mix::duplicates: {
      for (std::int64_t i = 0; i < k; ++i) {
        std::vector<vertex_t> members =
            draw_members<vertex_t>(n, draw(rng, opt.min_cardinality, opt.max_cardinality), rng);
        const std::vector<vertex_t> copy = members;
        members.insert(members.end(), copy.begin(), copy.end());  // every member twice
        shuffle_prefix(members, members.size(), rng);
        out.insert_hyperedge(host_view(members),
                             draw_weight<weight_t>(rng, opt.min_weight, opt.max_weight));
      }
      if (!live.empty()) {  // a copy of the live hyperedge with the smallest id
        const std::int64_t first = *std::min_element(live.begin(), live.end());
        const std::vector<vertex_t> members = row(first);
        out.insert_hyperedge(host_view(members),
                             draw_weight<weight_t>(rng, opt.min_weight, opt.max_weight));
      }
      break;
    }
    case hyperedge_batch_mix::dead_ids: {
      const std::vector<std::int64_t> gone = take(std::min<std::int64_t>(k, 2));
      for (const std::int64_t id : gone) out.delete_hyperedge(id);
      for (const std::int64_t id : gone) out.delete_hyperedge(id);  // repeated deletions
      for (std::int64_t id = 0; id < bound; ++id) {                 // a dead id below the bound
        if (h2v.row_ptr[static_cast<std::size_t>(id) + 1] ==
            h2v.row_ptr[static_cast<std::size_t>(id)]) {
          out.delete_hyperedge(id);
          break;
        }
      }
      out.delete_hyperedge(bound);
      out.delete_hyperedge(-1);
      fresh_random();
      break;
    }
    case hyperedge_batch_mix::skewed:
      for (const std::int64_t id : take(std::min<std::int64_t>(k, 2))) out.delete_hyperedge(id);
      fresh(std::min(n, opt.skewed_cardinality));
      fresh(1);
      break;
    case hyperedge_batch_mix::limits: {
      if (!live.empty()) out.delete_hyperedge(*std::max_element(live.begin(), live.end()));
      out.delete_hyperedge(std::numeric_limits<std::int64_t>::min());
      out.delete_hyperedge(std::numeric_limits<std::int64_t>::max());
      out.delete_hyperedge(bound);
      const std::vector<vertex_t> ends = {0, static_cast<vertex_t>(n - 1)};
      out.insert_hyperedge(host_view(ends.data(), n > 1 ? 2 : 1),
                           draw_weight<weight_t>(rng, opt.min_weight, opt.max_weight));
      const std::vector<vertex_t> last = {static_cast<vertex_t>(n - 1)};
      const std::vector<vertex_t> zero = {0};
      out.insert_hyperedge(host_view(last), fixed_weight<weight_t>(opt.min_weight));
      out.insert_hyperedge(host_view(zero), fixed_weight<weight_t>(opt.max_weight));
      break;
    }
    case hyperedge_batch_mix::empty:
      break;
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::random_hyperedge_batch")

template <typename vertex_t, typename weight_t>
hyperedge_batch<vertex_t, weight_t> random_hyperedge_batch(const resources& res,
                                                           const hypergraph<vertex_t, weight_t>& h,
                                                           const hyperedge_batch_options& opt) try {
  DYNG_EXPECTS((res.get_backend() == backend::cuda) == (h.space() == memory_space::device),
               "random_hyperedge_batch: the resources (", to_string(res.get_backend()),
               ") are not of the hypergraph's backend");
  const auto view = h.h2v();
  const std::vector<std::int64_t> row_ptr = to_vector(res, view.row_ptr);
  const std::vector<vertex_t> col_ind = to_vector(res, view.col_ind);
  incidence_view<vertex_t> host_rows;
  host_rows.row_ptr = host_view(row_ptr);
  host_rows.col_ind = host_view(col_ind);
  host_rows.version = view.version;
  return random_hyperedge_batch<vertex_t, weight_t>(h.num_vertices(), host_rows, h.weights(), opt);
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::random_hyperedge_batch")

#define DYNG_INSTANTIATE_HYPERGRAPH_GENERATORS(V, W)                                         \
  template hyperedge_list<V, W> uniform_hypergraph<V, W>(const uniform_hypergraph_options&); \
  template hyperedge_batch<V, W> random_hyperedge_batch<V, W>(                               \
      V, const incidence_view<V>&, array_view<const W>, const hyperedge_batch_options&);     \
  template hyperedge_batch<V, W> random_hyperedge_batch<V, W>(                               \
      const resources&, const hypergraph<V, W>&, const hyperedge_batch_options&);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPERGRAPH_GENERATORS)
#undef DYNG_INSTANTIATE_HYPERGRAPH_GENERATORS

}  // namespace dyng::generators
