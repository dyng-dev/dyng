// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_operators_test.cpp
 * @brief label_propagation's operators (operators/; ADR 0102) on sequential_exec, against
 *        independent models: neighbor_reduce's averages against a double-precision sum over the
 *        live entries (every mapping, the hub degrees 8, 9, 32, 33, 1024, 1025, 3000 and 6000),
 *        its association order, binned_frontier's bins, halves, stamps and layout, expand's set,
 *        and connected_components' canonical ids against a breadth-first search. The CUDA test
 *        (label_propagation_operators_cuda_test.cu) holds cuda_exec to these host results.
 */
#include "algorithms/label_propagation/operators/binned_frontier.hpp"
#include "algorithms/label_propagation/operators/connected_components.hpp"
#include "algorithms/label_propagation/operators/neighbor_reduce.hpp"
#include "label_propagation_operators_cases.hpp"

#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace {

namespace ops = dyng::detail::operators;
namespace lpo = dyng::detail::lp_ops;
namespace cases = dyng::test::lp_operators;

using seq = ops::sequential_exec;

lpo::slot_rows<std::int32_t> view_of(const cases::host_rows& g) {
  lpo::slot_rows<std::int32_t> r;
  r.row_start = g.row_start.data();
  r.row_end = g.row_end.data();
  r.col_ind = g.col_ind.data();
  r.weights = g.weights.data();
  r.status = g.status.data();
  return r;
}

/// The guarded average in double precision over the live entries (the operator's definition).
double reference_average(const cases::host_rows& g, std::int32_t u) {
  double w = 0.0;
  double wx = 0.0;
  for (auto e = g.row_start[static_cast<std::size_t>(u)];
       e < g.row_end[static_cast<std::size_t>(u)]; ++e) {
    const std::int32_t v = g.col_ind[static_cast<std::size_t>(e)];
    if (v < 0 || g.status[static_cast<std::size_t>(v)] == 1) {
      continue;
    }
    w += g.weights[static_cast<std::size_t>(e)];
    wx += static_cast<double>(g.weights[static_cast<std::size_t>(e)]) *
          g.x[static_cast<std::size_t>(v)];
  }
  return w > 1e-10 ? wx / w : g.x[static_cast<std::size_t>(u)];
}

/// Records each apply call.
struct record_apply {
  std::vector<std::pair<std::int32_t, float>>* calls;
  void operator()(int i, std::int32_t u, float average) const {
    EXPECT_EQ(static_cast<std::size_t>(i), calls->size());
    calls->emplace_back(u, average);
  }
};

constexpr lpo::reduce_mapping mappings[] = {lpo::reduce_mapping::lanes_8, lpo::reduce_mapping::warp,
                                            lpo::reduce_mapping::block_256,
                                            lpo::reduce_mapping::block_1024};

TEST(LabelPropagationOperators, NeighborReduceIsTheGuardedAverageOnEveryMapping) {
  const dyng::resources res = dyng::resources::sequential();
  const cases::host_rows g = cases::make_rows(3000, 1);
  const std::vector<std::int32_t> list = cases::all_vertices(g);
  for (const lpo::reduce_mapping m : mappings) {
    SCOPED_TRACE(lpo::threads_of(m));
    std::vector<std::pair<std::int32_t, float>> calls;
    lpo::neighbor_reduce<seq>(res, m, view_of(g), g.x.data(), list.data(),
                              static_cast<int>(list.size()), record_apply{&calls});
    ASSERT_EQ(calls.size(), list.size());
    for (std::size_t i = 0; i < list.size(); ++i) {
      EXPECT_EQ(calls[i].first, list[i]);
      const double want = reference_average(g, list[i]);
      EXPECT_NEAR(calls[i].second, want, 2e-6 * std::max(1.0, std::fabs(want)))
          << "vertex " << list[i];
    }
  }
}

TEST(LabelPropagationOperators, NeighborReduceSumsInTheMappingsAssociationOrder) {
  // The sums of one row by hand for 8 lanes (lane l: entries l, l + 8, ...; then the tree
  // (p0 + p4) + (p2 + p6) + ((p1 + p5) + (p3 + p7))) and for a block (each warp's tree, then the
  // warps' sums in warp order): the model is exactly that order.
  const cases::host_rows g = cases::make_rows(100, 2);
  for (std::size_t h = 0; h < g.hubs.size(); ++h) {
    const std::int32_t u = g.hubs[h];
    SCOPED_TRACE(cases::hub_degrees[h]);
    for (const int lanes : {8, 32, 256, 1024}) {
      const int warp_lanes = std::min(lanes, 32);
      float block_w = 0.0F;
      for (int warp = 0; warp < std::max(1, lanes / 32); ++warp) {
        float p[32] = {};
        for (int l = 0; l < warp_lanes; ++l) {
          for (auto e = g.row_start[static_cast<std::size_t>(u)] + warp * 32 + l;
               e < g.row_end[static_cast<std::size_t>(u)]; e += lanes) {
            const std::int32_t v = g.col_ind[static_cast<std::size_t>(e)];
            if (v >= 0 && g.status[static_cast<std::size_t>(v)] != 1) {
              p[l] += g.weights[static_cast<std::size_t>(e)];
            }
          }
        }
        for (int offset = warp_lanes / 2; offset > 0; offset /= 2) {
          for (int l = 0; l < offset; ++l) {
            p[l] += p[l + offset];
          }
        }
        block_w = lanes <= 32 ? p[0] : block_w + p[0];
      }
      float w = 0.0F;
      float wx = 0.0F;
      lpo::model_sums(view_of(g), g.x.data(), u, static_cast<lpo::reduce_mapping>(lanes), w, wx);
      EXPECT_EQ(0, std::memcmp(&w, &block_w, sizeof(float))) << lanes << " threads";
    }
  }
}

TEST(LabelPropagationOperators, BinnedFrontierBinsHalvesStampsAndLayout) {
  constexpr int bins = 3;
  const int capacity[bins] = {64, 8, 4};
  std::vector<std::int32_t> q[2][bins];
  for (auto& set : q) {
    for (int b = 0; b < bins; ++b) {
      set[b].assign(static_cast<std::size_t>(capacity[b]), -1);
    }
  }
  std::vector<std::int32_t> counts(2 * 2 * bins, 0);
  std::vector<std::int32_t> host_counts(2 * 2 * bins, 0);
  std::vector<std::int32_t> stamp(100, 0);
  lpo::binned_frontier<std::int32_t, bins>::storage s{};
  for (int set = 0; set < 2; ++set) {
    for (int b = 0; b < bins; ++b) {
      s.queue[set][b] = q[set][b].data();
      s.capacity[b] = capacity[b];
    }
  }
  s.counts = counts.data();
  s.host_counts = host_counts.data();
  s.stamp = stamp.data();
  s.bins.max_degree[0] = 32;
  s.bins.max_degree[1] = 1024;
  const lpo::binned_frontier<std::int32_t, bins> f(s);
  EXPECT_EQ(s.bins.of(32), 0);
  EXPECT_EQ(s.bins.of(33), 1);
  EXPECT_EQ(s.bins.of(1024), 1);
  EXPECT_EQ(s.bins.of(1025), 2);
  const seq exec(dyng::resources::sequential());
  f.clear(exec, 1);
  const int tag = 7;
  const auto into = f.into(1, tag);
  const std::vector<std::pair<std::int32_t, int>> pushes = {
      {3, 8},    {4, 9},    {5, 32},    {6, 33}, {7, 1024},
      {8, 1025}, {9, 3000}, {10, 6000}, {3, 8},  {9, 3000}};
  for (const auto& [v, d] : pushes) {
    into.push(v, d);  // the repeats are dropped (the stamps)
  }
  EXPECT_EQ(f.read_size(exec, 1), 8);
  std::map<std::pair<int, int>, std::set<std::int32_t>> want;
  for (std::size_t k = 0; k < 8; ++k) {
    const auto [v, d] = pushes[k];
    EXPECT_EQ(stamp[static_cast<std::size_t>(v)], tag);
    want[{s.bins.of(d), lpo::half_sweep(v, tag)}].insert(v);
  }
  for (int b = 0; b < bins; ++b) {
    for (int half = 0; half < 2; ++half) {
      const int count = f.count(1, b, half);
      const std::int32_t* list = f.list(1, b, half);
      EXPECT_EQ(list, q[1][b].data() + (half == 0 ? 0 : capacity[b] - count));
      const std::set<std::int32_t> got(list, list + count);
      EXPECT_EQ(got, (want[{b, half}])) << "bin " << b << ", half " << half;
    }
  }
  // The next iteration's tag admits them again; the set's counters clear.
  const auto again = f.into(1, tag + 1);
  f.clear(exec, 1);
  again.push(3, 8);
  EXPECT_EQ(f.read_size(exec, 1), 1);
}

TEST(LabelPropagationOperators, ExpandPushesTheChangedAndTheirEligibleNeighbours) {
  const dyng::resources res = dyng::resources::sequential();
  const cases::host_rows g = cases::make_rows(400, 3);
  const std::vector<std::int32_t> list = cases::all_vertices(g);
  std::vector<std::uint8_t> changed(list.size());
  for (std::size_t i = 0; i < list.size(); ++i) {
    changed[i] = (i % 3 == 0 || i < g.hubs.size()) ? 1 : 0;
  }
  const std::vector<std::int32_t> gt = [&] {
    std::vector<std::int32_t> c(static_cast<std::size_t>(g.n), -1);
    for (std::size_t v = 0; v < c.size(); v += 11) {
      c[v] = static_cast<std::int32_t>(v % 2);
    }
    return c;
  }();
  struct eligible {
    const std::int32_t* status;
    const std::int32_t* gt;
    bool operator()(std::int32_t v) const {
      return status[v] == 0 && gt[v] < 0;
    }
  };
  constexpr int bins = 3;
  const int capacity[bins] = {g.n, g.n, g.n};
  std::vector<std::int32_t> q[bins];
  for (auto& b : q) {
    b.assign(static_cast<std::size_t>(g.n), -1);
  }
  std::vector<std::int32_t> counts(2 * bins, 0);
  std::vector<std::int32_t> stamp(static_cast<std::size_t>(g.n), 0);
  lpo::binned_push<std::int32_t, bins> out{};
  for (int b = 0; b < bins; ++b) {
    out.queue[b] = q[b].data();
    out.capacity[b] = capacity[b];
  }
  out.counts = counts.data();
  out.stamp = stamp.data();
  out.tag = 1;
  out.bins.max_degree[0] = 32;
  out.bins.max_degree[1] = 1024;
  lpo::expand<seq>(res, 8, view_of(g), list.data(), changed.data(), static_cast<int>(list.size()),
                   eligible{g.status.data(), gt.data()}, out);
  std::set<std::int32_t> want;
  for (std::size_t i = 0; i < list.size(); ++i) {
    if (changed[i] == 0) {
      continue;
    }
    const std::int32_t u = list[i];
    want.insert(u);
    for (auto e = g.row_start[static_cast<std::size_t>(u)];
         e < g.row_end[static_cast<std::size_t>(u)]; ++e) {
      const std::int32_t v = g.col_ind[static_cast<std::size_t>(e)];
      if (v >= 0 && g.status[static_cast<std::size_t>(v)] == 0 &&
          gt[static_cast<std::size_t>(v)] < 0) {
        want.insert(v);
      }
    }
  }
  std::set<std::int32_t> got;
  for (int b = 0; b < bins; ++b) {
    for (int half = 0; half < 2; ++half) {
      const int count = counts[static_cast<std::size_t>(b * 2 + half)];
      const std::int32_t* list_bh = q[b].data() + lpo::half_offset(half, capacity[b], count);
      for (int k = 0; k < count; ++k) {
        EXPECT_TRUE(got.insert(list_bh[k]).second) << "vertex " << list_bh[k] << " twice";
        EXPECT_EQ(out.bins.of(view_of(g).degree(list_bh[k])), b);
      }
    }
  }
  EXPECT_EQ(got, want);
}

/// The canonical ids by breadth-first search: a component's root is its smallest vertex, the
/// roots numbered in list order.
std::vector<std::int32_t> canonical_ids(const cases::edge_case& c) {
  std::vector<std::vector<std::int32_t>> adj(static_cast<std::size_t>(c.n));
  for (std::size_t e = 0; e < c.src.size(); ++e) {
    adj[static_cast<std::size_t>(c.src[e])].push_back(c.dst[e]);
    adj[static_cast<std::size_t>(c.dst[e])].push_back(c.src[e]);
  }
  std::vector<std::int32_t> root(static_cast<std::size_t>(c.n), -1);
  for (const std::int32_t s : c.vertices) {
    if (root[static_cast<std::size_t>(s)] >= 0) {
      continue;
    }
    std::vector<std::int32_t> members = {s};
    root[static_cast<std::size_t>(s)] = s;
    for (std::size_t k = 0; k < members.size(); ++k) {
      for (const std::int32_t w : adj[static_cast<std::size_t>(members[k])]) {
        if (root[static_cast<std::size_t>(w)] < 0) {
          root[static_cast<std::size_t>(w)] = s;
          members.push_back(w);
        }
      }
    }
    const std::int32_t smallest = *std::min_element(members.begin(), members.end());
    for (const std::int32_t m : members) {
      root[static_cast<std::size_t>(m)] = smallest;
    }
  }
  std::vector<std::int32_t> id_of_root(static_cast<std::size_t>(c.n), -1);
  std::int32_t next = 0;
  for (const std::int32_t v : c.vertices) {
    if (root[static_cast<std::size_t>(v)] == v) {
      id_of_root[static_cast<std::size_t>(v)] = next++;
    }
  }
  std::vector<std::int32_t> id(static_cast<std::size_t>(c.n), -1);
  for (const std::int32_t v : c.vertices) {
    id[static_cast<std::size_t>(v)] =
        id_of_root[static_cast<std::size_t>(root[static_cast<std::size_t>(v)])];
  }
  return id;
}

TEST(LabelPropagationOperators, ConnectedComponentsGiveTheCanonicalIds) {
  const dyng::resources res = dyng::resources::sequential();
  for (const auto& [n, listed, edges] :
       {std::tuple{50, 30, 0}, std::tuple{2000, 1500, 600}, std::tuple{20000, 12000, 11000}}) {
    SCOPED_TRACE(edges);
    const cases::edge_case c = cases::make_edges(n, listed, edges, 77 + edges);
    std::vector<std::int32_t> parent(static_cast<std::size_t>(n));
    std::vector<std::int32_t> root_id(static_cast<std::size_t>(n));
    std::vector<std::int32_t> component(static_cast<std::size_t>(n), -1);
    lpo::components_scratch s;
    s.parent = parent.data();
    s.root_id = root_id.data();
    const std::int32_t num_edges = edges;
    const lpo::components_result r =
        lpo::connected_components<seq>(res, c.src.data(), c.dst.data(), &num_edges, edges,
                                       c.vertices.data(), listed, s, component.data());
    const std::vector<std::int32_t> want = canonical_ids(c);
    EXPECT_EQ(component, want);
    EXPECT_EQ(r.count, *std::max_element(want.begin(), want.end()) + 1);
    EXPECT_EQ(r.rounds, 0);
  }
}

}  // namespace
