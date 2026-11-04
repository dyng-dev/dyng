// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_support.hpp
 * @brief Helpers of hyper_sssp's tests: slack_csr hypergraphs, the result on the host, the check
 *        against dyng::testing's oracle (hyper_sssp_reference, check_hyper_sssp_tree), and random
 *        batches with hyperedge and incidence changes and weights in [1, 2^28].
 */
#pragma once

#include "algorithms/hyper_sssp/virtual_endpoints.hpp"
#include "hypergraph/hypergraph_support.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hsssp {

using whgraph = hypergraph<std::int32_t, std::int32_t>;
using wbatch = hyperedge_batch<std::int32_t, std::int32_t>;
using result_t = hyper_sssp::result<std::int32_t>;

/// slack_csr storage and line graph (the properties hyper_sssp needs).
inline hypergraph_properties slack(id_reuse reuse = id_reuse::erase_first) {
  hypergraph_properties p;
  p.storage = hypergraph_storage::slack_csr;
  p.line_graph = line_graph_policy::slack_csr;
  p.reuse = reuse;
  return p;
}

/// A weighted hypergraph of rows and weights on n vertices.
inline whgraph make(const resources& res, std::int32_t n, const test::hyper::rows& r,
                    const std::vector<std::int32_t>& w,
                    const hypergraph_properties& props = slack()) {
  return whgraph::from_hyperedges(res, test::hyper::list_of<std::int32_t>(n, r, w).view(), props);
}

/// The distances and parents of a result on the host.
inline std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> host_of(const resources& res,
                                                                               const result_t& r) {
  return {to_vector(res, r.hyperedge_distances()), to_vector(res, r.hyperedge_parents())};
}

/// The hyperedges of a hypergraph (h2v by id) and its weights by id, on the host.
struct host_hypergraph {
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> members;
  std::vector<std::int64_t> weights;
  std::int32_t num_vertices = 0;
};

/// The host copy of `h`.
inline host_hypergraph host_copy(const resources& res, const whgraph& h) {
  host_hypergraph out;
  const auto v = h.h2v();
  out.offsets = to_vector(res, v.row_ptr);
  out.members = to_vector(res, v.col_ind);
  const auto w = h.weights();
  out.weights.assign(w.begin(), w.end());
  out.num_vertices = h.num_vertices();
  return out;
}

/**
 * @brief Whether `r` equals the oracle on `h` (distances and canonical parents by id).
 * @param zero The virtual mode's ids whose weight hyper_sssp reads as 0 (empty: public mode).
 */
inline ::testing::AssertionResult matches_oracle(const resources& res, const whgraph& h,
                                                 const result_t& r, std::int64_t source,
                                                 bool source_is_vertex,
                                                 const std::vector<std::int64_t>& zero = {}) {
  host_hypergraph g = host_copy(res, h);
  for (const std::int64_t id : zero) g.weights[static_cast<std::size_t>(id)] = 0;
  const auto got = host_of(res, r);
  const auto bad = testing::check_hyper_sssp_tree(g.offsets, g.members, g.num_vertices, g.weights,
                                                  source, source_is_vertex, got.first, got.second);
  if (bad.first == 0 && bad.second == 0) return ::testing::AssertionSuccess();
  return ::testing::AssertionFailure()
         << bad.first << " distance and " << bad.second << " parent differences from the oracle";
}

/// A random batch on `h`: hyperedge deletions (never `keep`), insertions with weights in
/// [1, max_weight], incidence removals and additions (never removing `keep_vertex` from `keep`).
inline wbatch random_batch(std::mt19937_64& rng, const whgraph& h, std::int64_t keep,
                           std::int32_t max_weight, int size = 6) {
  test::hyper::rows current;
  const std::vector<std::int64_t> live = test::hyper::live_ids(h, current);
  const std::int32_t n = h.num_vertices();
  std::uniform_int_distribution<std::int32_t> vertex(0, n - 1);
  std::uniform_int_distribution<std::int32_t> weight(1, max_weight);
  std::uniform_int_distribution<int> coin(0, 5);
  wbatch b;
  for (const std::int64_t id : live) {
    if (id == keep) continue;
    const int c = coin(rng);
    if (c == 0) b.delete_hyperedge(id);
    if (c == 1) b.insert_incidence(id, vertex(rng));
    if (c == 2) {
      const auto& row = current[static_cast<std::size_t>(id)];
      b.delete_incidence(id, row[static_cast<std::size_t>(vertex(rng)) % row.size()]);
    }
  }
  for (int i = 0; i < size; ++i) {
    std::vector<std::int32_t> m;
    const int k = 1 + coin(rng) % 3;
    for (int j = 0; j < k; ++j) m.push_back(vertex(rng));
    b.insert_hyperedge(host_view(m), weight(rng));
  }
  return b;
}

}  // namespace dyng::test::hsssp
