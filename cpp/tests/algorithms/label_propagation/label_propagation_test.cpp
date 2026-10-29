// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_test.cpp
 * @brief Hand cases of label_propagation on the sequential backend (M10a step 3; ADRs 0071,
 *        0072): the published flow on a DynLP stream (the deterministic stages against
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

std::vector<std::uint32_t> bits(dyng::array_view<const float> labels) {
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
  const dyng::resources res = dyng::resources::sequential();
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
    const auto labels = r.labels();
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
    const std::vector<float> f(labels.begin(), labels.end());
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
  const dyng::resources res = dyng::resources::sequential();
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
  EXPECT_EQ(bits(r.labels()), bits(r64.labels()));  // the offset type does not change the bits
}

/// The original's input_contract_test graph: 0 (GT class 0), 1 (GT class 1), 2 unlabelled;
/// edges 0-2 (0.9) and 1-2 (0.8). Vertex 2's label is 0.8 / 1.7 after the initialization and
/// stays there (its neighbours are seeds).
TEST(LabelPropagation, ThreeVertexGraphByHand) {
  const dyng::resources res = dyng::resources::sequential();
  const std::vector<std::int32_t> cap = {1, 1, 2};
  lp_graph g = lp_graph::with_capacity(res, 3, host_view(cap), slotted_props());
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  opt.update_schedule = lp::schedule::damped;
  auto r = empty_compute(res, g, opt);
  for (const float x : r.labels()) {
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
  const auto labels = r.labels();
  EXPECT_EQ(labels[0], 0.0F);
  EXPECT_EQ(labels[1], 1.0F);
  const double w0 = std::nearbyint(0.9F * 4294967296.0) / 4294967296.0;
  const double w1 = std::nearbyint(0.8F * 4294967296.0) / 4294967296.0;
  EXPECT_EQ(labels[2], static_cast<float>(w1 / (w0 + w1)));
  // The vertex 2 leaves: it keeps its last label.
  dyng::edge_batch<std::int32_t, float> d;
  d.delete_vertex(2);
  const lp::stats sd = lp::update(res, g, d.view(), r);
  EXPECT_EQ(sd.initial_frontier, 0);  // its neighbours are seeds
  EXPECT_EQ(sd.components, 0);
  EXPECT_EQ(r.labels()[2], labels[2]);
}

/// A batch of edges only, between present vertices: both endpoints are affected (ADR 0072: the
/// original skips its marks when nothing is inserted, unreachable from its generator).
TEST(LabelPropagation, EdgeOnlyBatchMarksItsEndpoints) {
  const dyng::resources res = dyng::resources::sequential();
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
  EXPECT_EQ(r.labels()[2], 0.0F);
  EXPECT_EQ(r.labels()[3], 1.0F);
  dyng::edge_batch<std::int32_t, float> b1;
  b1.insert_edge(2, 3, {0.4F});
  const lp::stats st = lp::update(res, g, b1.view(), r);
  EXPECT_EQ(st.batch.inserted_edges, 2);
  EXPECT_EQ(st.initial_frontier, 2);  // 2 and 3
  EXPECT_EQ(st.components, 0);        // nothing inserted: no G'
  EXPECT_GT(r.labels()[2], 0.0F);
  EXPECT_LT(r.labels()[3], 1.0F);
}

TEST(LabelPropagation, RunToRunBitsOfBothSchedules) {
  const stream_t s = make_stream(3000, 4);
  for (const lp::schedule rule : {lp::schedule::damped, lp::schedule::in_place}) {
    std::vector<std::vector<std::uint32_t>> runs;
    for (int run = 0; run < 2; ++run) {
      const dyng::resources res = dyng::resources::sequential();
      lp_graph g = capacity_graph(res, s);
      auto r = empty_compute(res, g, published(s, 1e-5F, rule));
      std::vector<std::uint32_t> all;
      for (const auto& b : s.batches) {
        (void)lp::update(res, g, b.view(), r);
        const auto x = bits(r.labels());
        all.insert(all.end(), x.begin(), x.end());
      }
      runs.push_back(std::move(all));
    }
    EXPECT_EQ(runs[0], runs[1]);  // bitwise on the sequential backend
  }
}

TEST(LabelPropagation, ComputeIsIrLP) {
  const dyng::resources res = dyng::resources::sequential();
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
  const std::vector<float> f(irlp.labels().begin(), irlp.labels().end());
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
  EXPECT_EQ(bits(irlp.labels()), bits(irlp_c.labels()));
}

TEST(LabelPropagation, OnLimit) {
  const dyng::resources res = dyng::resources::sequential();
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
      EXPECT_TRUE(st.converged);
    }
  }
}

TEST(LabelPropagation, OptionsAndErrors) {
  const dyng::resources res = dyng::resources::sequential();
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
  const dyng::resources res = dyng::resources::sequential();
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
  EXPECT_EQ(bits(a.labels()), bits(b.labels()));  // dyng::update() is update()
  auto c = a.clone(res);
  EXPECT_EQ(bits(c.labels()), bits(a.labels()));
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
  const dyng::resources res = dyng::resources::sequential();
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
    EXPECT_EQ(report.used.own_host_syncs(), 0);
  }
}

}  // namespace
