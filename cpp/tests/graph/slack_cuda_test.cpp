// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slack_cuda_test.cpp
 * @brief row_layout::slack on the CUDA backend (ADR 0103): the device apply with the host mirror
 *        of the layout gives the host apply's row starts, limits, ends, slots, states and counts
 *        after every batch (moves to the tail with the arena growing on the device, compactions
 *        with the kept entries read back, vertex growth; random streams and vertex-arrival streams
 *        with deliberately small capacities; 32- and 64-bit offsets), the model's rows, and
 *        clones between the backends. Every test skips without a visible device.
 */
#include "core/cuda_runtime.hpp"
#include "graph/graph_impl.hpp"
#include "support/slack_model.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace {

using dyng::batch_semantics;
using dyng::graph_properties;
using dyng::host_view;
using dyng::row_layout;
using dyng::vertex_status;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using lp_graph64 = dyng::graph<std::int32_t, std::int64_t, float>;
using lp_batch = dyng::edge_batch<std::int32_t, float>;

bool have_device() {
  return dyng::backend_available(dyng::backend::cuda) && dyng::detail::cuda_device_count() > 0;
}

#define DYNG_REQUIRE_DEVICE()                      \
  do {                                             \
    if (!have_device()) {                          \
      GTEST_SKIP() << "no CUDA device is visible"; \
    }                                              \
  } while (0)

graph_properties slack_props(bool growth = false) {
  graph_properties p;
  p.directed = false;
  p.layout = row_layout::slack;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = batch_semantics::net_effect();
  p.semantics.allow_vertex_growth = growth;
  return p;
}

/// The device graph's layout (downloaded) equals the host graph's: starts, limits, ends, slots,
/// states, counts, the dead slots.
template <typename graph_t>
void expect_same_layout(const graph_t& h, const graph_t& d, const std::string& where) {
  const dyng::resources seq = dyng::resources::sequential();
  const auto& hi = dyng::detail::graph_access::impl(h);
  const auto& di = dyng::detail::graph_access::impl(d);
  const auto& hs = hi.host_edges();
  const auto& ds = di.host_edges();  // the download
  ASSERT_EQ(hs.row_ptr, ds.row_ptr) << where;
  ASSERT_EQ(hi.slotted.row_limit, di.slotted.row_limit) << where;
  ASSERT_EQ(hi.slotted.row_end, di.slotted.row_end) << where;
  ASSERT_EQ(hs.col_ind, ds.col_ind) << where;
  ASSERT_EQ(hs.weights.size(), ds.weights.size()) << where;
  EXPECT_EQ(std::memcmp(hs.weights.data(), ds.weights.data(), hs.weights.size() * sizeof(float)), 0)
      << where;
  ASSERT_EQ(hi.slotted.status, di.slotted.status) << where;
  EXPECT_EQ(hi.slotted.dead_slots, di.slotted.dead_slots) << where;
  EXPECT_EQ(h.num_edges(), d.num_edges()) << where;
  EXPECT_EQ(h.num_active_vertices(), d.num_active_vertices()) << where;
  EXPECT_EQ(h.num_vertices(), d.num_vertices()) << where;
  d.check_integrity(dyng::resources::cuda(0));
  (void)seq;
}

template <typename graph_t>
void random_stream(std::int64_t n, int batches, int edges, std::uint64_t seed, int hub_share,
                   bool growth, std::int64_t& moves, std::int64_t& compactions) {
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  using edge_t = typename graph_t::edge_type;
  const std::vector<edge_t> none;
  const std::int64_t start_n = growth ? n / 4 : n;
  graph_t h = graph_t::with_capacity(seq, static_cast<std::int32_t>(start_n), host_view(none),
                                     slack_props(growth));
  graph_t d = graph_t::with_capacity(cuda, static_cast<std::int32_t>(start_n), host_view(none),
                                     slack_props(growth));
  dyng::test::slack_model model(start_n);
  std::mt19937_64 rng(seed);
  std::vector<std::int32_t> present;
  std::int32_t next = 0;
  moves = 0;
  compactions = 0;
  for (int b = 0; b < batches; ++b) {
    lp_batch batch(1);
    const int arrivals = std::min<std::int32_t>(static_cast<std::int32_t>(n) - next, 25);
    for (int i = 0; i < arrivals; ++i) {
      batch.insert_vertex(next, static_cast<std::int8_t>((next % 7) == 0 ? next % 2 : -1));
      present.push_back(next++);
    }
    for (int e = 0; e < edges && present.size() > 1; ++e) {
      const auto u = (static_cast<int>(rng() % 100) < hub_share) ? present[0]
                                                                 : present[rng() % present.size()];
      const auto v = present[rng() % present.size()];
      batch.insert_edge(u, v, {static_cast<float>(rng() % 1000 + 1) / 1000.0F});
    }
    if (b % 3 == 2 && present.size() > 4) {
      for (int k = 0; k < 2; ++k) {
        const std::size_t i = 1 + rng() % (present.size() - 1);
        if ((present[i] % 7) == 0) continue;
        batch.delete_vertex(present[i]);
        present.erase(present.begin() + static_cast<std::ptrdiff_t>(i));
      }
    }
    const dyng::apply_summary hs = h.apply(seq, batch.view());
    const dyng::apply_summary ds = d.apply(cuda, batch.view());
    EXPECT_EQ(hs.relocated_rows, ds.relocated_rows);
    EXPECT_EQ(hs.compactions, ds.compactions);
    EXPECT_EQ(hs.inserted_edges, ds.inserted_edges);
    EXPECT_EQ(hs.dropped_insertions, ds.dropped_insertions);
    EXPECT_EQ(hs.num_vertices_after, ds.num_vertices_after);
    model.apply(batch.view(), hs.compactions > 0);
    moves += hs.relocated_rows;
    compactions += hs.compactions;
    expect_same_layout(h, d, "batch " + std::to_string(b));
    model.expect_equal(d, "batch " + std::to_string(b));
    if (::testing::Test::HasFatalFailure()) return;
  }
}

TEST(SlackCuda, RandomStreamsEqualTheHostApply) {
  DYNG_REQUIRE_DEVICE();
  for (const int hub : {0, 40, 90}) {
    SCOPED_TRACE(hub);
    std::int64_t moves = 0;
    std::int64_t compactions = 0;
    random_stream<lp_graph>(400, 30, 60, 11 + hub, hub, false, moves, compactions);
    EXPECT_GT(moves, 0);
    EXPECT_GT(compactions, 0);
  }
  std::int64_t moves = 0;
  std::int64_t compactions = 0;
  random_stream<lp_graph64>(300, 20, 50, 5, 50, false, moves, compactions);
}

TEST(SlackCuda, VertexGrowthOnTheDevice) {
  DYNG_REQUIRE_DEVICE();
  std::int64_t moves = 0;
  std::int64_t compactions = 0;
  random_stream<lp_graph>(400, 20, 40, 3, 30, true, moves, compactions);
}

TEST(SlackCuda, VertexArrivalStreamsWithSmallCapacities) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = 5000;
  o.num_batches = 5;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(seq, o);
  for (const int percent : {0, 50, 100}) {
    SCOPED_TRACE(percent);
    std::vector<std::int32_t> hints;
    if (percent > 0) {
      for (const auto c : stream.row_capacity) hints.push_back(c * percent / 100);
    }
    lp_graph h = lp_graph::with_capacity(seq, 5000, host_view(hints), slack_props());
    lp_graph d = lp_graph::with_capacity(cuda, 5000, host_view(hints), slack_props());
    for (std::size_t b = 0; b < stream.batches.size(); ++b) {
      (void)h.apply(seq, stream.batches[b].view());
      (void)d.apply(cuda, stream.batches[b].view());
      expect_same_layout(h, d, "batch " + std::to_string(b));
      if (::testing::Test::HasFatalFailure()) return;
    }
  }
}

TEST(SlackCuda, ClonesBetweenTheBackends) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = 3000;
  o.num_batches = 4;
  const auto stream =
      dyng::generators::generate_vertex_arrival_stream<std::int32_t, std::int32_t>(seq, o);
  lp_graph h =
      lp_graph::with_capacity(seq, 3000, dyng::array_view<const std::int32_t>(), slack_props());
  for (std::size_t b = 0; b + 1 < stream.batches.size(); ++b) {
    (void)h.apply(seq, stream.batches[b].view());
  }
  lp_graph d = h.clone(cuda);  // the host rows uploaded with their limits
  const auto& last = stream.batches.back();
  (void)h.apply(seq, last.view());
  (void)d.apply(cuda, last.view());
  expect_same_layout(h, d, "after the clone");
  const lp_graph back = d.clone(seq);
  back.check_integrity(seq);
  EXPECT_EQ(back.num_edges(), h.num_edges());
}

}  // namespace
