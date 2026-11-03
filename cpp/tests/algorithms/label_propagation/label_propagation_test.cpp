// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_test.cpp
 * @brief Hand cases of label_propagation on the sequential backend (M10a step 3; ADRs 0071,
 *        0072), compiled a second time with DYNG_TEST_CUDA=1 for the cuda backend (M10a step 4;
 *        ADR 0075; the labels are copied to the host, run-to-run bits required for damped only):
 *        the published flow on a DynLP stream (the deterministic stages against
 *        dyng::testing::lp_host_model, the harmonic oracle of AD7 at delta 1e-6, seeds exact,
 *        labels in [0, 1], no cap), the original's three-vertex graph by hand, an edge-only batch,
 *        the run-to-run bits of both schedules, compute() (IrLP), on_limit, the options and the
 *        errors (AD19's backend message), dyng::update(), clone(), and no host allocation in the
 *        algorithm phase after reserve().
 */
#include "conformance/allocation_counter.hpp"
#include "core/budget_counters.hpp"
#include "framework/budgets.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/testing/harmonic_solution.hpp>
#include <dyng/testing/lp_host_model.hpp>
#include <dyng/update.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace lp = dyng::label_propagation;

#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
constexpr bool on_cuda = true;
#else
constexpr bool on_cuda = false;
#endif

/// The backend under test: sequential, or cuda (DYNG_TEST_CUDA=1).
dyng::resources test_resources() {
  return on_cuda ? dyng::resources::cuda(0) : dyng::resources::sequential();
}

/// The labels of a result on the host (copied from the device for a CUDA result).
std::vector<float> host_labels(const dyng::resources& res, const lp::result<std::int32_t>& r) {
  const auto v = r.labels();
  if (dyng::is_host_accessible(v.space())) {
    return std::vector<float>(v.begin(), v.end());
  }
  return dyng::to_vector(res, v);
}
using dyng::host_view;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using lp_graph64 = dyng::graph<std::int32_t, std::int64_t, float>;
using stream_t = dyng::generators::vertex_arrival_stream<std::int32_t, std::int32_t>;

dyng::graph_properties slotted_props() {
  dyng::graph_properties p;
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  return p;
}

stream_t make_stream(std::int64_t n, std::int32_t batches, std::uint32_t seed = 42) {
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = n;
  opt.num_batches = batches;
  opt.seed = seed;
  return dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(
      dyng::resources::sequential(), opt);
}

template <typename graph_t = lp_graph>
graph_t capacity_graph(const dyng::resources& res, const stream_t& s) {
  std::vector<typename graph_t::edge_type> cap(s.row_capacity.begin(), s.row_capacity.end());
  const std::vector<typename graph_t::edge_type>& c = cap;
  return graph_t::with_capacity(res, static_cast<std::int32_t>(c.size()), host_view(c),
                                slotted_props());
}

lp::options published(const stream_t& s, float delta, lp::schedule rule) {
  lp::options opt;
  opt.similarity_threshold = s.similarity_threshold;
  opt.tolerance = delta;
  opt.update_schedule = rule;
  return opt;
}

lp::result<std::int32_t> empty_compute(const dyng::resources& res, const lp_graph& g,
                                       const lp::options& opt) {
  return lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                     dyng::array_view<const std::int8_t>(), opt);
}

std::vector<std::uint32_t> bits(const std::vector<float>& labels) {
  std::vector<std::uint32_t> out(labels.size());
  std::memcpy(out.data(), labels.data(), labels.size() * sizeof(float));
  return out;
}

dyng::testing::lp_model_batch model_batch(
    const dyng::generators::vertex_arrival_batch<std::int32_t>& b) {
  dyng::testing::lp_model_batch m;
  m.insert_vertices = b.insert_vertices;
  m.insert_labels = b.insert_vertex_labels;
  m.delete_vertices = b.delete_vertices;
  m.edge_sources = b.edge_sources;
  m.edge_destinations = b.edge_destinations;
  m.edge_weights = b.edge_weights;
  return m;
}

/// The published flow on a stream, every batch checked against lp_host_model (the deterministic
/// counts), the harmonic oracle (AD7) and the label invariants.
void run_stream(const stream_t& s, float delta, lp::schedule rule) {
  const dyng::resources res = test_resources();
  lp_graph g = capacity_graph(res, s);
  auto r = empty_compute(res, g, published(s, delta, rule));
  r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
  std::vector<std::int64_t> cap(s.row_capacity.begin(), s.row_capacity.end());
  dyng::testing::lp_host_model model(cap);
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    const lp::stats st = lp::update(res, g, s.batches[b].view(), r);
    const auto report = model.apply(model_batch(s.batches[b]), s.similarity_threshold);
    EXPECT_EQ(st.initial_frontier, static_cast<std::int64_t>(report.affected.size()));
    EXPECT_EQ(st.components, report.num_components);
    EXPECT_EQ(st.sparsified_edges, static_cast<std::int64_t>(report.sparsified.size() / 2));
    EXPECT_EQ(st.batch.inserted_vertices, static_cast<std::int64_t>(report.inserted.size()));
    EXPECT_EQ(st.batch.deleted_vertices, static_cast<std::int64_t>(report.deleted.size()));
    EXPECT_TRUE(st.converged);  // no batch reaches the cap
    EXPECT_FALSE(st.fallback_used);
    EXPECT_EQ(r.graph_version(), g.version());
    const std::vector<float> labels = host_labels(res, r);
    const auto seeds = r.seed_labels();
    std::vector<std::int8_t> seed_vec(seeds.begin(), seeds.end());
    EXPECT_EQ(seed_vec, model.seed_labels());
    for (std::size_t v = 0; v < labels.size(); ++v) {
      ASSERT_TRUE(labels[v] >= 0.0F && labels[v] <= 1.0F) << v;
      if (seeds[v] == 0) {
        ASSERT_EQ(labels[v], 0.0F) << v;  // GT labels exact
      } else if (seeds[v] == 1) {
        ASSERT_EQ(labels[v], 1.0F) << v;
      } else if (model.states()[v] == dyng::testing::lp_host_model::vertex_state::inactive) {
        ASSERT_EQ(labels[v], 0.5F) << v;  // never inserted
      }
    }
    // AD7: mean |F - F*| <= 2000 delta, the PCG converged.
    const std::vector<float>& f = labels;
    const auto h = dyng::testing::harmonic_solution(
        s.graph.row_ptr, s.graph.col_ind, s.graph.weights, model.present(), model.seed_labels());
    const auto err = dyng::testing::label_error_against(f, h, model.present(), model.seed_labels());
    EXPECT_TRUE(h.converged);
    EXPECT_LE(err.mean, 2000.0 * static_cast<double>(delta));
  }
  EXPECT_NO_THROW(g.check_integrity(res));
}

TEST(LabelPropagation, DampedStreamMeetsTheOracle) {
  run_stream(make_stream(5000, 5), 1e-6F, lp::schedule::damped);
}

TEST(LabelPropagation, InPlaceStreamMeetsTheOracle) {
  run_stream(make_stream(5000, 5), 1e-6F, lp::schedule::in_place);
}

TEST(LabelPropagation, OtherSeedsAndSixtyFourBitOffsets) {
  run_stream(make_stream(2000, 4, 7), 1e-5F, lp::schedule::damped);
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(2000, 4, 7);
  lp_graph g = capacity_graph(res, s);
  lp_graph64 g64 = capacity_graph<lp_graph64>(res, s);
  const lp::options opt = published(s, 1e-5F, lp::schedule::damped);
  auto r = empty_compute(res, g, opt);
  auto r64 = lp::compute(res, g64, dyng::array_view<const std::int32_t>(),
                         dyng::array_view<const std::int8_t>(), opt);
  for (const auto& b : s.batches) {
    (void)lp::update(res, g, b.view(), r);
    (void)lp::update(res, g64, b.view(), r64);
  }
  // the offset type does not change the bits
  EXPECT_EQ(bits(host_labels(res, r)), bits(host_labels(res, r64)));
}

/// The original's input_contract_test graph: 0 (GT class 0), 1 (GT class 1), 2 unlabelled;
/// edges 0-2 (0.9) and 1-2 (0.8). Vertex 2's label is 0.8 / 1.7 after the initialization and
/// stays there (its neighbours are seeds).
TEST(LabelPropagation, ThreeVertexGraphByHand) {
  const dyng::resources res = test_resources();
  const std::vector<std::int32_t> cap = {1, 1, 2};
  lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  opt.update_schedule = lp::schedule::damped;
  auto r = empty_compute(res, g, opt);
  for (const float x : host_labels(res, r)) {
    EXPECT_EQ(x, 0.5F);
  }
  dyng::edge_batch<std::int32_t, float> b;
  b.insert_vertex(0, 0);
  b.insert_vertex(1, 1);
  b.insert_vertex(2);
  b.insert_edge(0, 2, {0.9F});
  b.insert_edge(1, 2, {0.8F});
  const lp::stats st = lp::update(res, g, b.view(), r);
  EXPECT_EQ(st.components, 1);        // G' = {0-2, 1-2}
  EXPECT_EQ(st.sparsified_edges, 2);  // both > tau = 0.5
  EXPECT_EQ(st.initial_frontier, 1);  // vertex 2 (0 and 1 are seeds)
  EXPECT_EQ(st.affected, 3);          // 0 -> 0, 1 -> 1, 2 -> 0.47...
  EXPECT_EQ(st.iterations, 1);        // the initial label is the fixed point
  EXPECT_EQ(st.frontier_visits, 1);
  const std::vector<float> labels = host_labels(res, r);
  EXPECT_EQ(labels[0], 0.0F);
  EXPECT_EQ(labels[1], 1.0F);
  const double w0 = std::nearbyint(0.9F * 4294967296.0) / 4294967296.0;
  const double w1 = std::nearbyint(0.8F * 4294967296.0) / 4294967296.0;
  if (on_cuda) {
    // The damped iteration stores 0.5 (F + average) with the original's --use_fast_math
    // division: within an ulp or two of the initial label.
    EXPECT_NEAR(labels[2], static_cast<float>(w1 / (w0 + w1)), 1e-6);
  } else {
    EXPECT_EQ(labels[2], static_cast<float>(w1 / (w0 + w1)));
  }
  // The vertex 2 leaves: it keeps its last label.
  dyng::edge_batch<std::int32_t, float> d;
  d.delete_vertex(2);
  const lp::stats sd = lp::update(res, g, d.view(), r);
  EXPECT_EQ(sd.initial_frontier, 0);  // its neighbours are seeds
  EXPECT_EQ(sd.components, 0);
  EXPECT_EQ(host_labels(res, r)[2], labels[2]);
}

/// A batch of edges only, between present vertices: both endpoints are affected (ADR 0072: the
/// original skips its marks when nothing is inserted, unreachable from its generator).
TEST(LabelPropagation, EdgeOnlyBatchMarksItsEndpoints) {
  const dyng::resources res = test_resources();
  const std::vector<std::int32_t> cap = {1, 1, 2, 2};
  lp_graph g = lp_graph::with_capacity(res, 4, host_view(cap), slotted_props());
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  auto r = empty_compute(res, g, opt);
  dyng::edge_batch<std::int32_t, float> b0;
  b0.insert_vertex(0, 0);
  b0.insert_vertex(1, 1);
  b0.insert_vertex(2);
  b0.insert_vertex(3);
  b0.insert_edge(0, 2, {0.9F});
  b0.insert_edge(1, 3, {0.9F});
  (void)lp::update(res, g, b0.view(), r);
  EXPECT_EQ(host_labels(res, r)[2], 0.0F);
  EXPECT_EQ(host_labels(res, r)[3], 1.0F);
  dyng::edge_batch<std::int32_t, float> b1;
  b1.insert_edge(2, 3, {0.4F});
  const lp::stats st = lp::update(res, g, b1.view(), r);
  EXPECT_EQ(st.batch.inserted_edges, 2);
  EXPECT_EQ(st.initial_frontier, 2);  // 2 and 3
  EXPECT_EQ(st.components, 0);        // nothing inserted: no G'
  EXPECT_GT(host_labels(res, r)[2], 0.0F);
  EXPECT_LT(host_labels(res, r)[3], 1.0F);
}

TEST(LabelPropagation, RunToRunBitsOfBothSchedules) {
  const stream_t s = make_stream(3000, 4);
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    std::vector<std::vector<std::uint32_t>> runs;
    for (int run = 0; run < 2; ++run) {
      const dyng::resources res = test_resources();
      lp_graph g = capacity_graph(res, s);
      auto r = empty_compute(res, g, published(s, 1e-5F, rule));
      std::vector<std::uint32_t> all;
      for (const auto& b : s.batches) {
        (void)lp::update(res, g, b.view(), r);
        const auto x = bits(host_labels(res, r));
        all.insert(all.end(), x.begin(), x.end());
      }
      runs.push_back(std::move(all));
    }
    if (rule == lp::schedule::damped || !on_cuda) {
      EXPECT_EQ(runs[0], runs[1]);  // bitwise per backend (in_place on CUDA is not reproducible)
    }
  }
}

/// A seed-free pair of adjacent vertices with different labels (the structure the original's
/// D-d found in large numbers: plain Jacobi has the eigenvalue -1 there and swaps the two labels
/// forever): the vertex 1 inherits its seed's class, the seed leaves, and the vertex 2 arrives
/// next to it. The damped rule and the half-sweeps of the in-place rule converge at once; plain
/// Jacobi reaches any cap (the mutations inplace_plain_jacobi and damped_no_damping, AD9).
TEST(LabelPropagation, ASeedFreePairConverges) {
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    SCOPED_TRACE(rule == lp::schedule::damped ? "damped" : "in_place");
    const dyng::resources res = test_resources();
    const std::vector<std::int32_t> cap = {1, 2, 1};
    lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
    lp::options opt;
    opt.similarity_threshold = 0.5F;
    opt.tolerance = 1e-6F;
    opt.max_iterations = 1000;
    opt.update_schedule = rule;
    auto r = empty_compute(res, g, opt);
    dyng::edge_batch<std::int32_t, float> b0;
    b0.insert_vertex(0, 1);
    b0.insert_vertex(1);
    b0.insert_edge(0, 1, {0.9F});
    (void)lp::update(res, g, b0.view(), r);
    EXPECT_EQ(host_labels(res, r)[1], 1.0F);
    dyng::edge_batch<std::int32_t, float> b1;
    b1.delete_vertex(0);
    b1.insert_vertex(2);
    b1.insert_edge(1, 2, {0.9F});
    const lp::stats st = lp::update(res, g, b1.view(), r);
    EXPECT_EQ(st.initial_frontier, 2);
    EXPECT_TRUE(st.converged) << st.iterations << " iterations";
    EXPECT_LT(st.iterations, 100);
    const std::vector<float> labels = host_labels(res, r);
    EXPECT_NEAR(labels[1], labels[2], 1e-5);  // one seed-free component: one value
  }
}

/// An id inserted and deleted in one batch ends deleted, exactly as in the original (D-e: its
/// insertion applied after the deletion re-activated it); its edges are dropped.
TEST(LabelPropagation, InsertedAndDeletedInOneBatchEndsDeleted) {
  const dyng::resources res = test_resources();
  const std::vector<std::int32_t> cap = {2, 2, 2};
  lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  auto r = empty_compute(res, g, opt);
  dyng::edge_batch<std::int32_t, float> b;
  b.insert_vertex(0, 0);
  b.insert_vertex(1);
  b.insert_vertex(2);
  b.insert_edge(0, 1, {0.9F});
  b.insert_edge(1, 2, {0.9F});
  b.delete_vertex(2);
  const lp::stats st = lp::update(res, g, b.view(), r);
  EXPECT_EQ(st.batch.inserted_vertices, 2);
  EXPECT_EQ(st.batch.inserted_edges, 2);      // 0-1 in both rows
  EXPECT_EQ(st.batch.dropped_insertions, 2);  // 1-2: the vertex 2 ends deleted
  EXPECT_EQ(g.num_active_vertices(), 2);
  EXPECT_EQ(g.view().status[2], dyng::vertex_status::deleted);
  EXPECT_EQ(host_labels(res, r)[2], 0.5F);  // it never entered the graph
}

/// The damped schedule's iteration counts are the original's on its 5k5 stream (delta 1e-4,
/// seed 42; LabelPropagation-CUDA@a276a3a's results.csv, the golden label_propagation_damped
/// case 5k5_d1e-4_s42): 0, 1217, 548, 327 and 65 (the mutations damped_no_damping and
/// lp_d_c_stale_buffer change them).
TEST(LabelPropagation, DampedIterationsAreTheOriginals) {
  const stream_t s = make_stream(5000, 5);
  const dyng::resources res = test_resources();
  lp_graph g = capacity_graph(res, s);
  auto r = empty_compute(res, g, published(s, 1e-4F, lp::schedule::damped));
  r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
  const std::vector<std::int64_t> original = {0, 1217, 548, 327, 65};
  std::vector<std::int64_t> iterations;
  for (const auto& b : s.batches) {
    iterations.push_back(lp::update(res, g, b.view(), r).iterations);
  }
  EXPECT_EQ(iterations, original);
}

TEST(LabelPropagation, ComputeIsIrLP) {
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(3000, 4);
  lp_graph g = capacity_graph(res, s);
  lp::options opt = published(s, 1e-6F, lp::schedule::damped);
  auto r = empty_compute(res, g, opt);
  for (const auto& b : s.batches) {
    (void)lp::update(res, g, b.view(), r);
  }
  // IrLP on the current graph with its present seeds meets the oracle too (AD7, "for compute").
  const auto view = g.view();
  std::vector<std::int32_t> seeds;
  std::vector<std::int8_t> classes;
  std::vector<std::uint8_t> present(view.status.size());
  std::vector<std::int8_t> seed_of(view.status.size(), -1);
  for (std::size_t v = 0; v < view.status.size(); ++v) {
    present[v] = view.status[v] == dyng::vertex_status::active ? 1 : 0;
    if (present[v] != 0 && r.seed_labels()[v] >= 0) {
      seeds.push_back(static_cast<std::int32_t>(v));
      classes.push_back(r.seed_labels()[v]);
      seed_of[v] = r.seed_labels()[v];
    }
  }
  const std::vector<std::int32_t>& sv = seeds;
  const std::vector<std::int8_t>& sc = classes;
  const auto irlp = lp::compute(res, g, host_view(sv), host_view(sc), opt);
  const std::vector<float> f = host_labels(res, irlp);
  for (std::size_t v = 0; v < f.size(); ++v) {
    if (present[v] == 0) {
      EXPECT_EQ(f[v], 0.5F) << v;  // every vertex that is not active is 0.5
    }
  }
  const auto h = dyng::testing::harmonic_solution(s.graph.row_ptr, s.graph.col_ind, s.graph.weights,
                                                  present, seed_of);
  const auto err = dyng::testing::label_error_against(f, h, present, seed_of);
  EXPECT_LE(err.mean, 2000.0 * 1e-6);
  // compute() also reads a compact float graph (the same edges: the same labels).
  const auto c = g.to_csr(res);
  dyng::graph_properties compact;
  compact.directed = false;
  compact.order = dyng::row_order::append;
  compact.parallel_edges = dyng::multi_edges::allow;
  const lp_graph cg = lp_graph::from_csr(res, c.view(), compact);
  const auto irlp_c = lp::compute(res, cg, host_view(sv), host_view(sc), opt);
  EXPECT_EQ(bits(host_labels(res, irlp)), bits(host_labels(res, irlp_c)));
}

TEST(LabelPropagation, OnLimit) {
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(2000, 3);
  for (const lp::on_limit rule :
       {lp::on_limit::report, lp::on_limit::error, lp::on_limit::fallback_recompute}) {
    lp_graph g = capacity_graph(res, s);
    lp::options opt = published(s, 1e-6F, lp::schedule::damped);
    auto r = empty_compute(res, g, opt);
    (void)lp::update(res, g, s.batches[0].view(), r);
    opt.max_iterations = 1;
    opt.on_iteration_limit = rule;
    r.set_options(opt);
    if (rule == lp::on_limit::error) {
      EXPECT_THROW((void)lp::update(res, g, s.batches[1].view(), r), dyng::convergence_error);
      EXPECT_THROW((void)r.labels(), dyng::stale_result_error);  // poisoned
      continue;
    }
    const lp::stats st = lp::update(res, g, s.batches[1].view(), r);
    if (rule == lp::on_limit::report) {
      EXPECT_FALSE(st.converged);
      EXPECT_FALSE(st.fallback_used);
      EXPECT_EQ(st.iterations, 1);
    } else {
      EXPECT_TRUE(st.fallback_used);
      // IrLP under the same cap of 1 iteration reaches it too: not converged (M10a review).
      EXPECT_FALSE(st.converged);
    }
  }
}

/// on_limit::fallback_recompute when IrLP reaches the cap too (M10a review): the update keeps
/// IrLP's labels and reports stats::converged = false (compute() raises convergence_error there).
TEST(LabelPropagation, AFallbackThatReachesTheCapIsNotConverged) {
  const dyng::resources res = test_resources();
  constexpr std::int32_t n = 200;
  std::vector<std::int32_t> cap(n, 2);
  cap.front() = 1;
  cap.back() = 1;
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    SCOPED_TRACE(rule == lp::schedule::damped ? "damped" : "in_place");
    lp_graph g = lp_graph::with_capacity(res, n, host_view(cap), slotted_props());
    lp::options opt;
    opt.similarity_threshold = 0.95F;  // no edge joins G': every vertex starts from its supernode
    opt.tolerance = 1e-6F;
    opt.max_iterations = 1;
    opt.update_schedule = rule;
    opt.on_iteration_limit = lp::on_limit::fallback_recompute;
    auto r = empty_compute(res, g, opt);
    dyng::edge_batch<std::int32_t, float> b;  // a path with a seed of each class at its ends
    b.insert_vertex(0, 0);
    for (std::int32_t v = 1; v + 1 < n; ++v) {
      b.insert_vertex(v);
    }
    b.insert_vertex(n - 1, 1);
    for (std::int32_t v = 0; v + 1 < n; ++v) {
      b.insert_edge(v, v + 1, {0.9F});
    }
    const lp::stats st = lp::update(res, g, b.view(), r);
    EXPECT_TRUE(st.fallback_used);
    EXPECT_FALSE(st.converged) << "the fallback's IrLP reached the cap too";
    EXPECT_NO_THROW((void)r.labels());  // IrLP's labels are kept
    // compute() with the same options raises convergence_error at the cap.
    std::vector<std::int32_t> sv = {0, n - 1};
    std::vector<std::int8_t> sc = {0, 1};
    const std::vector<std::int32_t>& seeds = sv;
    const std::vector<std::int8_t>& classes = sc;
    EXPECT_THROW((void)lp::compute(res, g, host_view(seeds), host_view(classes), opt),
                 dyng::convergence_error);
  }
}

/// The similarity threshold is strict (the original's buildSubgraphKernel: w > tau): an edge of
/// weight exactly tau between two inserted vertices is not in G', the next float up is.
TEST(LabelPropagation, AnEdgeOfWeightTauIsNotInGPrime) {
  const dyng::resources res = test_resources();
  for (const float w : {0.5F, std::nextafter(0.5F, 1.0F)}) {
    SCOPED_TRACE(w);
    const std::vector<std::int32_t> cap = {1, 1};
    lp_graph g = lp_graph::with_capacity(res, 2, host_view(cap), slotted_props());
    lp::options opt;
    opt.similarity_threshold = 0.5F;
    auto r = empty_compute(res, g, opt);
    dyng::edge_batch<std::int32_t, float> b;
    b.insert_vertex(0, 0);
    b.insert_vertex(1);
    b.insert_edge(0, 1, {w});
    const lp::stats st = lp::update(res, g, b.view(), r);
    EXPECT_EQ(st.sparsified_edges, w > 0.5F ? 1 : 0);
    EXPECT_EQ(st.components, w > 0.5F ? 1 : 2);  // one supernode, or two singletons
    EXPECT_NEAR(host_labels(res, r)[1], 0.0F, 1e-3);  // its one neighbour is the class-0 seed
  }
}

/// The average live weight of slotted rows as options::similarity_threshold = std::nullopt
/// documents it: a double sum in vertex order and slot order over the entries whose two endpoints
/// are active, rounded to float (0.5 for a graph without edges).
float live_mean(const lp_graph& g) {
  const auto v = g.view();
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::size_t u = 0; u < v.status.size(); ++u) {
    if (v.status[u] != dyng::vertex_status::active) {
      continue;
    }
    for (auto e = static_cast<std::size_t>(v.out.row_ptr[u]);
         e < static_cast<std::size_t>(v.row_end[u]); ++e) {
      const std::int32_t w = v.out.col_ind[e];
      if (w >= 0 && v.status[static_cast<std::size_t>(w)] == dyng::vertex_status::active) {
        sum += v.out.weights[e];
        ++count;
      }
    }
  }
  return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : 0.5F;
}

/// options::similarity_threshold unset (the default): every update uses the current graph's
/// average live weight, bitwise the host formula on every backend (an update with that value
/// passed explicitly gives the same bits and counts).
TEST(LabelPropagation, TheDefaultThresholdIsTheAverageLiveWeight) {
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(3000, 4);
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    SCOPED_TRACE(rule == lp::schedule::damped ? "damped" : "in_place");
    lp_graph g = capacity_graph(res, s);
    lp_graph h = capacity_graph(res, s);
    lp::options opt = published(s, 1e-5F, rule);
    opt.similarity_threshold.reset();
    auto r = empty_compute(res, g, opt);
    auto q = empty_compute(res, h, opt);
    for (std::size_t b = 0; b < s.batches.size(); ++b) {
      SCOPED_TRACE("batch " + std::to_string(b));
      const lp::stats st = lp::update(res, g, s.batches[b].view(), r);
      lp::options explicit_tau = opt;
      explicit_tau.similarity_threshold = live_mean(g);  // G_{t+1}, after the update
      q.set_options(explicit_tau);
      const lp::stats sq = lp::update(res, h, s.batches[b].view(), q);
      EXPECT_EQ(st.sparsified_edges, sq.sparsified_edges);
      EXPECT_EQ(st.components, sq.components);
      EXPECT_EQ(st.initial_frontier, sq.initial_frontier);
      if (rule == lp::schedule::damped || !on_cuda) {
        EXPECT_EQ(bits(host_labels(res, r)), bits(host_labels(res, q)));
      }
    }
    EXPECT_NE(live_mean(g), s.similarity_threshold);  // the default is not the published tau
  }
  // One edge: the mean of its two entries is its weight, which is not > tau.
  const std::vector<std::int32_t> cap = {1, 1};
  lp_graph g = lp_graph::with_capacity(res, 2, host_view(cap), slotted_props());
  lp::options opt;
  auto r = empty_compute(res, g, opt);
  dyng::edge_batch<std::int32_t, float> b;
  b.insert_vertex(0, 0);
  b.insert_vertex(1);
  b.insert_edge(0, 1, {0.5F});  // the only entries' mean is 0.5 itself: not > tau
  EXPECT_EQ(lp::update(res, g, b.view(), r).sparsified_edges, 0);
}

/// FNV-1a of the label bits after every batch of the 5k5 stream (damped, delta 1e-4): a change of
/// the label initialization's precision (double sums, M10a review) or of the damped rule changes
/// them. The cuda value is the original's (damped byte parity, AD6); the sequential one is the
/// host engine's (bitwise per backend; -ffp-contract=off).
TEST(LabelPropagation, TheLabelBitsOfTheFiveThousandVertexStream) {
  const stream_t s = make_stream(5000, 5);
  const dyng::resources res = test_resources();
  lp_graph g = capacity_graph(res, s);
  auto r = empty_compute(res, g, published(s, 1e-4F, lp::schedule::damped));
  std::uint64_t h = 1469598103934665603ULL;
  for (const auto& b : s.batches) {
    (void)lp::update(res, g, b.view(), r);
    for (const std::uint32_t x : bits(host_labels(res, r))) {
      for (int k = 0; k < 4; ++k) {
        h = (h ^ ((x >> (8 * k)) & 0xFFU)) * 1099511628211ULL;
      }
    }
  }
  const std::uint64_t expected = on_cuda ? 0x1d034ffaed566004ULL : 0xfe32f56e533f87feULL;
  EXPECT_EQ(h, expected) << std::hex << "0x" << h;
}

/// The entry checks come before the batch is read (ADR 0070, M10a review): a stale result with an
/// invalid batch raises stale_result_error, and resources of another backend raise
/// not_supported_error, not Step 0's invalid_argument_error.
TEST(LabelPropagation, EntryChecksComeBeforeTheBatch) {
  const dyng::resources res = test_resources();
  const std::vector<std::int32_t> cap = {1, 1, 2};
  lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  auto r = empty_compute(res, g, opt);
  dyng::edge_batch<std::int32_t, float> v;
  v.insert_vertex(0);
  (void)g.apply(res, v.view());  // r is stale now
  dyng::edge_batch<std::int32_t, float> invalid;
  invalid.insert_vertex(0);  // a re-insertion: Step 0 would refuse it
  EXPECT_THROW((void)lp::update(res, g, invalid.view(), r), dyng::stale_result_error);
  EXPECT_THROW((void)dyng::update(res, g, invalid.view(), r), dyng::stale_result_error);
  if (dyng::backend_available(dyng::backend::openmp)) {
    auto fresh = empty_compute(res, g, opt);
    EXPECT_THROW((void)lp::update(dyng::resources::openmp(), g, invalid.view(), fresh),
                 dyng::not_supported_error);
  }
}

TEST(LabelPropagation, OptionsAndErrors) {
  const dyng::resources res = test_resources();
  const std::vector<std::int32_t> cap = {1, 1, 2};
  lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
  const auto none_v = dyng::array_view<const std::int32_t>();
  const auto none_l = dyng::array_view<const std::int8_t>();
  lp::options bad;
  bad.tolerance = 0.0F;
  EXPECT_THROW((void)lp::compute(res, g, none_v, none_l, bad), dyng::invalid_argument_error);
  bad = {};
  bad.tolerance = std::nanf("");
  EXPECT_THROW((void)lp::compute(res, g, none_v, none_l, bad), dyng::invalid_argument_error);
  bad = {};
  bad.max_iterations = 0;
  EXPECT_THROW((void)lp::compute(res, g, none_v, none_l, bad), dyng::invalid_argument_error);
  bad = {};
  bad.similarity_threshold = std::nanf("");
  EXPECT_THROW((void)lp::compute(res, g, none_v, none_l, bad), dyng::invalid_argument_error);
  // Seeds: sizes, range, label, not active (nothing is inserted yet), duplicates.
  const std::vector<std::int32_t> one = {0};
  const std::vector<std::int8_t> none;
  const std::vector<std::int8_t> two = {1, 1};
  EXPECT_THROW((void)lp::compute(res, g, host_view(one), host_view(none)),
               dyng::invalid_argument_error);
  const std::vector<std::int8_t> cls = {1};
  EXPECT_THROW((void)lp::compute(res, g, host_view(one), host_view(cls)),
               dyng::invalid_argument_error);  // vertex 0 is not active
  // OpenMP resources: not a backend of label_propagation in 0.4 (AD19).
  if (dyng::backend_available(dyng::backend::openmp)) {
    try {
      (void)lp::compute(dyng::resources::openmp(), g, none_v, none_l);
      ADD_FAILURE() << "openmp accepted";
    } catch (const dyng::not_supported_error& e) {
      const std::string m = e.what();
      EXPECT_NE(m.find("sequential"), std::string::npos) << m;
      EXPECT_NE(m.find("cuda"), std::string::npos) << m;
      EXPECT_NE(m.find("dyng.Resources(\"sequential\")"), std::string::npos) << m;
    }
  }
  // A directed graph is refused; update() needs slotted rows.
  const std::vector<std::int32_t> src = {0};
  const std::vector<std::int32_t> dst = {1};
  const std::vector<float> w = {0.5F};
  const lp_graph directed =
      lp_graph::from_edges(res, {2, host_view(src), host_view(dst), host_view(w), 1});
  EXPECT_THROW((void)lp::compute(res, directed, none_v, none_l), dyng::invalid_argument_error);
  dyng::graph_properties undirected;
  undirected.directed = false;
  lp_graph compact =
      lp_graph::from_edges(res, {2, host_view(src), host_view(dst), host_view(w), 1}, undirected);
  auto rc = lp::compute(res, compact, none_v, none_l);
  dyng::edge_batch<std::int32_t, float> b;
  b.insert_edge(0, 1, {0.3F});
  EXPECT_THROW((void)lp::update(res, compact, b.view(), rc), dyng::not_supported_error);
  // A stale result.
  auto r = lp::compute(res, g, none_v, none_l);
  dyng::edge_batch<std::int32_t, float> v;
  v.insert_vertex(0);
  (void)g.apply(res, v.view());
  dyng::edge_batch<std::int32_t, float> next;
  next.insert_vertex(1);
  EXPECT_THROW((void)lp::update(res, g, next.view(), r), dyng::stale_result_error);
  // set_options validates.
  lp::options o = r.get_options();
  o.max_iterations = -1;
  EXPECT_THROW(r.set_options(o), dyng::invalid_argument_error);
}

TEST(LabelPropagation, CompositionAndClone) {
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(2000, 3);
  lp_graph g = capacity_graph(res, s);
  lp_graph h = capacity_graph(res, s);
  const lp::options opt = published(s, 1e-5F, lp::schedule::damped);
  auto a = empty_compute(res, g, opt);
  auto b = empty_compute(res, h, opt);
  for (const auto& batch : s.batches) {
    const auto [st] = dyng::update(res, g, batch.view(), a);
    const lp::stats sb = lp::update(res, h, batch.view(), b);
    EXPECT_EQ(st.iterations, sb.iterations);
    EXPECT_EQ(st.components, sb.components);
  }
  EXPECT_EQ(bits(host_labels(res, a)), bits(host_labels(res, b)));  // dyng::update() is update()
  auto c = a.clone(res);
  EXPECT_EQ(bits(host_labels(res, c)), bits(host_labels(res, a)));
  EXPECT_EQ(c.graph_version(), a.graph_version());
}

TEST(LabelPropagation, NoHostAllocationInTheAlgorithmPhaseAfterReserve) {
  namespace fw = dyng::detail::framework;
  if (!dyng::conformance::host_allocation_counting_available()) {
    GTEST_SKIP() << "the counting operator new is not compiled in (sanitizer build)";
  }
  if (!dyng::detail::budgets_enabled()) {
    GTEST_SKIP() << "not a DYNG_DEBUG_BUDGETS build";
  }
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(3000, 5);
  lp_graph g = capacity_graph(res, s);
  auto r = empty_compute(res, g, published(s, 1e-5F, lp::schedule::damped));
  r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    dyng::conformance::host_allocation_counter armed;
    (void)lp::update(res, g, s.batches[b].view(), r);
    armed.stop();
    const fw::budget_report report = fw::last_budget_report();
    ASSERT_TRUE(report.measured);
    EXPECT_FALSE(report.reserving());
    EXPECT_EQ(report.used.own_allocations(), 0);
    if (!on_cuda) {
      EXPECT_EQ(report.used.own_host_syncs(), 0);
    } else {
      // The fused engine's run-dependent synchronizations, within its declared budget.
      EXPECT_TRUE(report.limit.allows(report.used)) << report.used.own_host_syncs();
      EXPECT_TRUE(report.limit.bounded());
    }
  }
}

}  // namespace
