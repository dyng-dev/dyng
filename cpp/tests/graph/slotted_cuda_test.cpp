// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slotted_cuda_test.cpp
 * @brief row_layout::slotted on the CUDA backend (ADR 0074): the resident device copy and the
 *        device apply of vertex batches give the host apply's rows, row ends, slots, states and
 *        counts after every batch (hand batches and a DynLP stream, pinned and pageable arrays,
 *        32- and 64-bit offsets, integer weights and unweighted graphs); the errors Step 0 finds
 *        keep the graph unchanged; an overflow found on the device raises capacity_error and
 *        poisons the graph; LabelPropagation-CUDA@a276a3a's nine input_contract cases; clones
 *        between the backends; graph::reserve() of the device scratch; the lazy download.
 *        Every test skips without a visible device.
 */
#include "core/cuda_runtime.hpp"
#include "graph/graph_impl.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <climits>
#include <cstring>
#include <optional>
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

graph_properties slotted_props() {
  graph_properties p;
  p.directed = false;
  p.layout = row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = batch_semantics::net_effect();
  return p;
}

template <typename graph_t>
graph_t make(const dyng::resources& res, const std::vector<typename graph_t::edge_type>& cap,
             graph_properties p = slotted_props()) {
  return graph_t::with_capacity(res, static_cast<std::int32_t>(cap.size()), host_view(cap), p);
}

/// Everything observable of a slotted graph, as bytes.
template <typename graph_t>
std::string snap(const graph_t& g) {
  const auto v = g.view();
  std::string s;
  const auto put = [&s](const void* p, std::size_t bytes) {
    s.append(static_cast<const char*>(p), bytes);
    s.push_back('|');
  };
  put(v.out.row_ptr.data(), v.out.row_ptr.size_bytes());
  put(v.out.col_ind.data(), v.out.col_ind.size_bytes());
  put(v.out.weights.data(), v.out.weights.size_bytes());
  put(v.row_end.data(), v.row_end.size_bytes());
  put(v.status.data(), v.status.size_bytes());
  const auto counts = std::to_string(g.num_edges()) + "/" + std::to_string(g.num_active_vertices());
  s += counts;
  return s;
}

/// Apply every batch to a host graph and a CUDA graph; after each batch both are equal and
/// consistent, and the summaries agree.
template <typename graph_t, typename batch_t>
void expect_same_as_host(const std::vector<typename graph_t::edge_type>& cap,
                         const std::vector<batch_t>& batches,
                         graph_properties p = slotted_props()) {
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  graph_t h = make<graph_t>(seq, cap, p);
  graph_t d = make<graph_t>(cuda, cap, p);
  EXPECT_EQ(snap(h), snap(d));
  for (std::size_t b = 0; b < batches.size(); ++b) {
    const dyng::apply_summary sh = h.apply(seq, batches[b].view());
    const dyng::apply_summary sd = d.apply(cuda, batches[b].view());
    EXPECT_EQ(sh.inserted_edges, sd.inserted_edges) << "batch " << b;
    EXPECT_EQ(sh.dropped_insertions, sd.dropped_insertions) << "batch " << b;
    EXPECT_EQ(sh.inserted_vertices, sd.inserted_vertices) << "batch " << b;
    EXPECT_EQ(sh.deleted_vertices, sd.deleted_vertices) << "batch " << b;
    // The counts without a download, then everything after it.
    EXPECT_EQ(h.num_edges(), d.num_edges()) << "batch " << b;
    EXPECT_EQ(h.num_active_vertices(), d.num_active_vertices()) << "batch " << b;
    EXPECT_FALSE(dyng::detail::graph_access::impl(d).host_current()) << "batch " << b;
    ASSERT_EQ(snap(h), snap(d)) << "batch " << b;
    EXPECT_NO_THROW(d.check_integrity(cuda)) << "batch " << b;
  }
}

std::vector<lp_batch> hand_batches() {
  std::vector<lp_batch> out(4);
  out[0].insert_vertex(0, 1);
  out[0].insert_vertex(1);
  out[0].insert_vertex(2, 0);
  out[0].insert_edge(0, 1, {0.25F});
  out[0].insert_edge(1, 2, {0.5F});
  out[0].insert_edge(2, 3, {0.75F});  // 3 is not active: dropped
  out[1].insert_vertex(3);
  out[1].insert_vertex(4);
  out[1].delete_vertex(4);  // Ins ∩ Del: never enters
  out[1].insert_edge(3, 0, {0.125F});
  out[1].insert_edge(3, 1, {0.375F});
  out[1].insert_edge(4, 2, {0.625F});  // dropped (4 is deleted)
  out[1].insert_edge(3, 3, {1.0F});    // a self-loop (both half-edges in row 3)
  out[2].delete_vertex(1);
  out[2].delete_vertex(3);  // adjacent deletions: the live entries counted once
  out[2].delete_vertex(5);  // never inserted: retired
  out[3].insert_vertex(6);
  out[3].insert_edge(6, 0, {0.5F});
  out[3].insert_edge(6, 2, {0.5F});
  out[3].insert_edge(0, 2, {0.9F});  // an edge between two vertices already present
  return out;
}

TEST(SlottedCuda, HandBatchesEqualTheHostApply) {
  DYNG_REQUIRE_DEVICE();
  expect_same_as_host<lp_graph>({4, 3, 3, 4, 1, 1, 2}, hand_batches());
  expect_same_as_host<lp_graph64>({4, 3, 3, 4, 1, 1, 2}, hand_batches());
}

TEST(SlottedCuda, IntegerWeightsAndUnweightedGraphs) {
  DYNG_REQUIRE_DEVICE();
  std::vector<dyng::edge_batch<std::int32_t, std::int32_t>> ib(1);
  ib[0].insert_vertex(0);
  ib[0].insert_vertex(1);
  ib[0].insert_edge(0, 1, {7});
  expect_same_as_host<dyng::graph<std::int32_t, std::int32_t, std::int32_t>>({1, 1}, ib);
  graph_properties p = slotted_props();
  p.num_weights = 0;
  std::vector<dyng::edge_batch<std::int32_t, dyng::unweighted>> ub(2);
  ub[0].insert_vertex(0);
  ub[0].insert_vertex(1);
  ub[0].insert_edge(0, 1);
  ub[1].delete_vertex(0);
  expect_same_as_host<dyng::graph<std::int32_t, std::int64_t, dyng::unweighted>>({1, 1}, ub, p);
}

/// A DynLP stream (the legacy generator), with its arrays in pinned host memory as
/// dyng-compat-dynlp passes them, and once more pageable.
TEST(SlottedCuda, DynlpStreamEqualsTheHostApply) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = 5000;
  opt.num_batches = 5;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(seq, opt);
  for (const bool pinned : {true, false}) {
    lp_graph h =
        lp_graph::with_capacity(seq, 5000, host_view(stream.row_capacity), slotted_props());
    lp_graph d =
        lp_graph::with_capacity(cuda, 5000, host_view(stream.row_capacity), slotted_props());
    d.reserve(cuda, static_cast<std::int32_t>(stream.max_batch_edges));
    auto& pinned_mr = dyng::default_pinned_host_memory_resource();
    for (std::size_t b = 0; b < stream.batches.size(); ++b) {
      const auto& batch = stream.batches[b];
      dyng::edge_batch_view<std::int32_t, float> view = batch.view();
      std::vector<dyng::buffer<std::int32_t>> ids;
      std::vector<dyng::buffer<float>> ws;
      std::vector<dyng::buffer<std::int8_t>> ls;
      if (pinned) {
        const auto pin = [&](dyng::array_view<const std::int32_t> v) {
          ids.emplace_back(v.size(), dyng::stream_ref(), dyng::memory_resource_ref(pinned_mr));
          std::memcpy(ids.back().data(), v.data(), v.size_bytes());
          return dyng::array_view<const std::int32_t>(ids.back().data(), v.size(),
                                                      dyng::memory_space::pinned_host);
        };
        view.insert_vertices = pin(view.insert_vertices);
        view.delete_vertices = pin(view.delete_vertices);
        view.insert_src = pin(view.insert_src);
        view.insert_dst = pin(view.insert_dst);
        ws.emplace_back(view.insert_weights.size(), dyng::stream_ref(),
                        dyng::memory_resource_ref(pinned_mr));
        std::memcpy(ws.back().data(), view.insert_weights.data(), view.insert_weights.size_bytes());
        view.insert_weights = dyng::array_view<const float>(
            ws.back().data(), view.insert_weights.size(), dyng::memory_space::pinned_host);
        ls.emplace_back(view.insert_vertex_labels.size(), dyng::stream_ref(),
                        dyng::memory_resource_ref(pinned_mr));
        std::memcpy(ls.back().data(), view.insert_vertex_labels.data(),
                    view.insert_vertex_labels.size_bytes());
        view.insert_vertex_labels = dyng::array_view<const std::int8_t>(
            ls.back().data(), view.insert_vertex_labels.size(), dyng::memory_space::pinned_host);
      }
      const auto sh = h.apply(seq, batch.view());
      const auto sd = d.apply(cuda, view);
      EXPECT_EQ(sh.inserted_edges, sd.inserted_edges);
      EXPECT_EQ(h.num_edges(), d.num_edges()) << "batch " << b;
      ASSERT_EQ(snap(h), snap(d)) << "batch " << b << (pinned ? " (pinned)" : " (pageable)");
      EXPECT_NO_THROW(d.check_integrity(cuda));
    }
  }
}

TEST(SlottedCuda, StepZeroErrorsKeepTheGraphUnchanged) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources res = dyng::resources::cuda(0);
  lp_graph g = make<lp_graph>(res, {2, 2, 2, 1});
  lp_batch b0;
  b0.insert_vertex(0, 1);
  b0.insert_vertex(1);
  b0.insert_edge(0, 1, {0.5F});
  (void)g.apply(res, b0.view());
  const std::string before = snap(g);
  const auto unchanged = [&](const lp_batch& b, const char* what, auto error) {
    using error_t = decltype(error);
    EXPECT_THROW((void)g.apply(res, b.view()), error_t) << what;
    EXPECT_EQ(snap(g), before) << what;
    EXPECT_NO_THROW(g.check_integrity(res)) << what;
  };
  lp_batch b;
  b.insert_vertex(4);
  unchanged(b, "an inserted id out of range", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2);
  b.insert_edge(2, 9, {0.5F});
  unchanged(b, "an edge endpoint out of range", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(1);
  unchanged(b, "a re-insertion", dyng::invalid_argument_error{""});
  b.clear();
  b.insert_vertex(2, 1);
  b.delete_vertex(2);
  unchanged(b, "a seed label on a vertex the batch deletes", dyng::invalid_argument_error{""});
  b.clear();
  b.delete_edge(0, 1);
  unchanged(b, "an edge deletion", dyng::not_supported_error{""});
  b.clear();
  b.insert_vertex(2);
  b.insert_edge(1, 2, {0.3F});
  EXPECT_NO_THROW((void)g.apply(res, b.view()));
  EXPECT_EQ(g.num_edges(), 4);
}

/// The overflow is found on the device (ADR 0074): capacity_error after the batch was applied,
/// the appends past the end of the row dropped, and the graph poisoned (the basic guarantee).
TEST(SlottedCuda, DeviceOverflowPoisonsTheGraph) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources res = dyng::resources::cuda(0);
  lp_graph g = make<lp_graph>(res, {2, 2, 2, 1});
  lp_batch b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  b.insert_vertex(3);
  b.insert_edge(3, 0, {0.1F});
  b.insert_edge(3, 1, {0.1F});  // row 3 has one slot
  EXPECT_THROW((void)g.apply(res, b.view()), dyng::capacity_error);
  EXPECT_EQ(g.view().row_end[3], 7);  // one entry appended (row 3 is slot 6), the second dropped
  EXPECT_THROW(g.check_integrity(res), dyng::internal_error);
  lp_batch c;
  c.insert_vertex(2);
  EXPECT_THROW((void)g.apply(res, c.view()), dyng::capacity_error);
}

/// The poison is the graph's host state (M10a review): after a device overflow, an apply on any
/// resources raises capacity_error (a host apply too, which would otherwise drop the device copy
/// and the poison with it), and a clone to either backend is poisoned as well.
TEST(SlottedCuda, ThePoisonSurvivesHostAppliesAndClones) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const dyng::resources seq = dyng::resources::sequential();
  lp_graph g = make<lp_graph>(cuda, {2, 2, 2, 1});
  lp_batch b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  b.insert_vertex(3);
  b.insert_edge(3, 0, {0.1F});
  b.insert_edge(3, 1, {0.1F});  // row 3 has one slot
  EXPECT_THROW((void)g.apply(cuda, b.view()), dyng::capacity_error);
  lp_batch c;
  c.insert_vertex(2);
  const lp_batch empty;
  const std::uint64_t version = g.version();
  EXPECT_THROW((void)g.apply(seq, c.view()), dyng::capacity_error);  // a host apply
  EXPECT_THROW((void)g.apply(cuda, empty.view()), dyng::capacity_error);
  EXPECT_EQ(g.version(), version);
  EXPECT_THROW(g.check_integrity(seq), dyng::internal_error);
  for (const dyng::resources* to : {&cuda, &seq}) {
    SCOPED_TRACE(to == &cuda ? "clone to cuda" : "clone to sequential");
    lp_graph copy = g.clone(*to);
    EXPECT_THROW((void)copy.apply(*to, c.view()), dyng::capacity_error);
    EXPECT_THROW((void)copy.apply(cuda, empty.view()), dyng::capacity_error);
    try {
      copy.check_integrity(*to);
      ADD_FAILURE() << "a clone of a poisoned graph passed check_integrity()";
    } catch (const dyng::internal_error& e) {
      EXPECT_NE(std::string(e.what()).find("poisoned"), std::string::npos) << e.what();
    }
  }
}

/// A row at the end of a slot array of 2^31 - 1 slots (int32 offsets): the device append's slot
/// and row-end arithmetic must not wrap past the row's end check (M10a review: it wrote about
/// 8 GB before the slot array). Unweighted, so the slots take 8 GiB on each side; skipped when the
/// device or the host cannot hold them.
TEST(SlottedCuda, AnOverflowNearTheInt32LimitStaysInsideTheRow) {
  DYNG_REQUIRE_DEVICE();
  using u_graph = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>;
  const dyng::resources res = dyng::resources::cuda(0);
  graph_properties p = slotted_props();
  p.num_weights = 0;
  const std::vector<std::int32_t> cap = {INT32_MAX - 1, 1};
  std::optional<u_graph> g;
  try {
    g.emplace(u_graph::with_capacity(res, 2, host_view(cap), p));
    (void)dyng::detail::graph_access::slotted_device(res, *g);  // the 8 GiB device copy
  } catch (const dyng::out_of_memory_error& e) {
    GTEST_SKIP() << "no room for 2^31 - 1 slots: " << e.what();
  }
  dyng::edge_batch<std::int32_t, dyng::unweighted> b;
  b.insert_vertex(0);
  b.insert_vertex(1);
  for (int k = 0; k < 64; ++k) {
    b.insert_edge(1, 0);  // 64 deliveries of one edge: row 1 has one slot
  }
  EXPECT_THROW((void)g->apply(res, b.view()), dyng::capacity_error);
  const auto v = g->view();
  EXPECT_EQ(v.row_end[1], INT32_MAX);  // its one slot, never past the end of the row
  EXPECT_EQ(v.out.col_ind[static_cast<std::size_t>(INT32_MAX) - 1], 0);
}

/// LabelPropagation-CUDA@a276a3a:tests/input_contract_test.cu's nine scenarios on CUDA: the same
/// verdicts as on the host (ADR 0069); dupedge's overflow is found on the device (poisoned).
TEST(SlottedCuda, OriginalInputContract) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources res = dyng::resources::cuda(0);
  const auto first = [] {
    lp_batch b;
    b.insert_vertex(0, 0);
    b.insert_vertex(1, 1);
    b.insert_vertex(2);
    b.insert_edge(0, 2, {0.9F});
    b.insert_edge(1, 2, {0.8F});
    return b;
  };
  const auto scenario = [&](const std::string& name) {
    lp_batch b0 = first();
    lp_batch b1;
    lp_batch b2;
    if (name == "dupedge") {
      b0.insert_edge(0, 2, {0.9F});
      return std::vector<lp_batch>{b0};
    }
    if (name == "dupid") {
      b1.insert_vertex(2);
      return std::vector<lp_batch>{b0, b1};
    }
    if (name == "dupsame") {
      b0.insert_vertex(2);
      return std::vector<lp_batch>{b0};
    }
    if (name == "insdel" || name == "insdel2") {
      lp_batch c;
      c.insert_vertex(0, 0);
      c.insert_vertex(1, 1);
      c.insert_vertex(2);
      if (name == "insdel2") {
        c.insert_vertex(2);
      }
      c.delete_vertex(2);
      return std::vector<lp_batch>{c};
    }
    if (name == "insdelre") {
      b1.insert_vertex(2);
      b1.delete_vertex(2);
      return std::vector<lp_batch>{b0, b1};
    }
    if (name == "range") {
      b1.delete_vertex(3);
      return std::vector<lp_batch>{b0, b1};
    }
    b1.delete_vertex(2);
    if (name == "valid") {
      return std::vector<lp_batch>{b0, b1};
    }
    b2.insert_vertex(2);
    b2.insert_edge(0, 2, {0.9F});
    b2.insert_edge(1, 2, {0.8F});
    return std::vector<lp_batch>{b0, b1, b2};
  };
  for (const std::string name : {"valid", "insdel", "reinsert", "dupid", "dupsame", "range",
                                 "dupedge", "insdelre", "insdel2"}) {
    lp_graph g = make<lp_graph>(res, {1, 1, 2});
    std::string error;
    std::string before;
    for (const auto& b : scenario(name)) {
      before = snap(g);
      try {
        (void)g.apply(res, b.view());
      } catch (const dyng::capacity_error&) {
        error = "capacity";
        break;
      } catch (const dyng::invalid_argument_error&) {
        error = "invalid";
        break;
      }
    }
    if (error == "invalid") {
      EXPECT_EQ(snap(g), before) << name << ": the refused batch changed the graph";
      EXPECT_NO_THROW(g.check_integrity(res)) << name;
    }
    if (name == "valid" || name == "insdel") {
      EXPECT_EQ(error, "") << name;
      EXPECT_NO_THROW(g.check_integrity(res)) << name;
    } else if (name == "dupedge") {
      EXPECT_EQ(error, "capacity") << name;
    } else {
      EXPECT_EQ(error, "invalid") << name;
    }
  }
}

TEST(SlottedCuda, ClonesBetweenTheBackends) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources seq = dyng::resources::sequential();
  const dyng::resources cuda = dyng::resources::cuda(0);
  const std::vector<lp_batch> batches = hand_batches();
  lp_graph h = make<lp_graph>(seq, {4, 3, 3, 4, 1, 1, 2});
  lp_graph d = make<lp_graph>(cuda, {4, 3, 3, 4, 1, 1, 2});
  (void)h.apply(seq, batches[0].view());
  (void)d.apply(cuda, batches[0].view());
  lp_graph h_to_d = h.clone(cuda);  // uploaded on first device use
  lp_graph d_to_h = d.clone(seq);   // downloaded by the copy
  for (std::size_t b = 1; b < batches.size(); ++b) {
    (void)h.apply(seq, batches[b].view());
    (void)h_to_d.apply(cuda, batches[b].view());
    (void)d_to_h.apply(seq, batches[b].view());
    ASSERT_EQ(snap(h), snap(h_to_d)) << "batch " << b;
    ASSERT_EQ(snap(h), snap(d_to_h)) << "batch " << b;
  }
  EXPECT_NO_THROW(h_to_d.check_integrity(cuda));
}

TEST(SlottedCuda, ReserveSizesTheDeviceScratch) {
  DYNG_REQUIRE_DEVICE();
  const dyng::resources res = dyng::resources::cuda(0);
  lp_graph g = make<lp_graph>(res, {4, 3, 3, 4, 1, 1, 2});
  g.reserve(res, 100);
  const auto& dev = dyng::detail::graph_access::slotted_device(res, g);
  EXPECT_GE(dev.max_edges, 100);
  EXPECT_THROW(g.reserve(res, -1), dyng::invalid_argument_error);
  for (const auto& b : hand_batches()) {
    (void)g.apply(res, b.view());
  }
  EXPECT_NO_THROW(g.check_integrity(res));
}

}  // namespace
