// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_checker_test.cpp
 * @brief LabelPropagation-CUDA@a276a3a's checker oracles as GoogleTests (M10a criterion 14; PLAN
 *        6.4.7; AD7), on the sequential backend and, compiled with DYNG_TEST_CUDA=1, on cuda: the
 *        published flow on the DynLP streams 5k5 and 50k at delta 1e-6 (the corpus C), both
 *        schedules, every batch checked against the independent oracles:
 *
 * - the components of G' = union_find over the sparsified edges on Ins_eff (stats::components);
 * - the initial frontier = V_aff (stats::initial_frontier, lp_host_model);
 * - the statuses (by name) and the rows = lp_host_model, in exact slot order (row ends, slot
 *   columns and weight bits);
 * - no batch at the iteration cap; the seeds' labels exact; every label in [0, 1];
 * - the harmonic oracle: mean |F - F*| <= 2000 delta, the PCG converged (0 skips), no class flip
 *   where |F* - 0.5| > 3m (m = 2.9217e-3); and IrLP (compute()) on every snapshot at 5k5;
 * - IrLP's class agreement with DynLP after the last batch within 0.5 pp (reported).
 *
 * The 50k cases are labelled slow (ctest -L slow; the gpu_local label_propagation step runs them
 * on cuda); a backend's run of the original's flow at 5k5 takes about a second.
 */
#include <dyng/core/array_view.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/testing/harmonic_solution.hpp>
#include <dyng/testing/lp_host_model.hpp>
#include <dyng/testing/union_find.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

namespace lp = dyng::label_propagation;
namespace dt = dyng::testing;
using dyng::host_view;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using stream_t = dyng::generators::vertex_arrival_stream<std::int32_t, std::int32_t>;

#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
constexpr bool on_cuda = true;
#else
constexpr bool on_cuda = false;
#endif

constexpr double class_margin = 2.9217e-3;  // m (parity/results/M10a-class-margin.json)

dyng::resources test_resources() {
  return on_cuda ? dyng::resources::cuda(0) : dyng::resources::sequential();
}

std::vector<float> host_labels(const dyng::resources& res, const lp::result<std::int32_t>& r) {
  const auto v = r.labels();
  if (dyng::is_host_accessible(v.space())) {
    return std::vector<float>(v.begin(), v.end());
  }
  return dyng::to_vector(res, v);
}

dyng::graph_properties slotted_props() {
  dyng::graph_properties p;
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  return p;
}

stream_t make_stream(std::int64_t n, std::int32_t batches) {
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = n;
  opt.num_batches = batches;
  return dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(
      dyng::resources::sequential(), opt);
}

dt::lp_model_batch model_batch(const dyng::generators::vertex_arrival_batch<std::int32_t>& b) {
  dt::lp_model_batch m;
  m.insert_vertices = b.insert_vertices;
  m.insert_labels = b.insert_vertex_labels;
  m.delete_vertices = b.delete_vertices;
  m.edge_sources = b.edge_sources;
  m.edge_destinations = b.edge_destinations;
  m.edge_weights = b.edge_weights;
  return m;
}

/// The model's state of a vertex, by name, against the graph's vertex_status (ADR 0069, C8).
bool same_state(dt::lp_host_model::vertex_state m, dyng::vertex_status g) {
  switch (m) {
    case dt::lp_host_model::vertex_state::inactive:
      return g == dyng::vertex_status::inactive;
    case dt::lp_host_model::vertex_state::active:
      return g == dyng::vertex_status::active;
    case dt::lp_host_model::vertex_state::deleted:
      return g == dyng::vertex_status::deleted;
  }
  return false;
}

/// The harmonic oracle of one set of labels (AD7): mean <= 2000 delta, converged, the 3m rule.
::testing::AssertionResult near_harmonic(const stream_t& s, const dt::lp_host_model& model,
                                         const std::vector<float>& labels, float delta,
                                         const char* what) {
  const auto h = dt::harmonic_solution(s.graph.row_ptr, s.graph.col_ind, s.graph.weights,
                                       model.present(), model.seed_labels());
  if (!h.converged) {
    return ::testing::AssertionFailure() << what << ": the PCG did not converge (a skip)";
  }
  const auto err = dt::label_error_against(labels, h, model.present(), model.seed_labels());
  if (err.mean > 2000.0 * static_cast<double>(delta)) {
    return ::testing::AssertionFailure()
           << what << ": mean |F - F*| = " << err.mean << " above 2000 delta = " << 2000.0 * delta;
  }
  const auto ca =
      dt::class_agreement_with(labels, h, model.present(), model.seed_labels(), 3.0 * class_margin);
  if (ca.flips_beyond_margin > 0) {
    return ::testing::AssertionFailure()
           << what << ": " << ca.flips_beyond_margin << " class flips where |F* - 0.5| > 3m";
  }
  return ::testing::AssertionSuccess() << what << ": margin " << 2000.0 * delta / err.mean;
}

/// IrLP (compute()) on the graph's current state, seeded by the model's present seeds.
lp::result<std::int32_t> irlp(const dyng::resources& res, const lp_graph& g,
                              const dt::lp_host_model& model, const lp::options& opt) {
  std::vector<std::int32_t> seeds;
  std::vector<std::int8_t> classes;
  for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
    if (model.present()[static_cast<std::size_t>(v)] != 0 &&
        model.seed_labels()[static_cast<std::size_t>(v)] >= 0) {
      seeds.push_back(v);
      classes.push_back(model.seed_labels()[static_cast<std::size_t>(v)]);
    }
  }
  const std::vector<std::int32_t>& sv = seeds;
  const std::vector<std::int8_t>& sl = classes;
  return lp::compute(res, g, host_view(sv), host_view(sl), opt);
}

/// The original's checker over the published flow of one stream.
void check_stream(std::int64_t n, std::int32_t batches, lp::schedule rule, bool irlp_snapshots) {
  const float delta = 1e-6F;
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(n, batches);
  std::vector<std::int32_t> cap(s.row_capacity.begin(), s.row_capacity.end());
  const std::vector<std::int32_t>& c = cap;
  lp_graph g = lp_graph::with_capacity(res, static_cast<std::int32_t>(c.size()), host_view(c),
                                       slotted_props());
  lp::options opt;
  opt.similarity_threshold = s.similarity_threshold;
  opt.tolerance = delta;
  opt.update_schedule = rule;
  auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  r.reserve(res, s.max_batch_vertices, s.max_batch_edges);
  std::vector<std::int64_t> cap64(s.row_capacity.begin(), s.row_capacity.end());
  dt::lp_host_model model(cap64);
  std::int64_t ins_and_del = 0;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    const auto& batch = s.batches[b];
    const lp::stats st = lp::update(res, g, batch.view(), r);
    const dt::lp_model_report rep = model.apply(model_batch(batch), s.similarity_threshold);
    // The components of G' = union_find over the sparsified edges, on Ins_eff.
    dt::union_find uf(n);
    for (std::size_t i = 0; i + 1 < rep.sparsified.size(); i += 2) {
      uf.unite(rep.sparsified[i], rep.sparsified[i + 1]);
    }
    std::set<std::int64_t> roots;
    for (const std::int32_t v : rep.inserted) {
      roots.insert(uf.find(v));
    }
    EXPECT_EQ(st.components, static_cast<std::int64_t>(roots.size()))
        << "the components of G' differ from union_find";
    EXPECT_EQ(st.components, rep.num_components);
    EXPECT_EQ(st.sparsified_edges, static_cast<std::int64_t>(rep.sparsified.size() / 2));
    // The initial frontier = V_aff.
    EXPECT_EQ(st.initial_frontier, static_cast<std::int64_t>(rep.affected.size()))
        << "the initial frontier differs from V_aff";
    EXPECT_EQ(st.batch.inserted_vertices, static_cast<std::int64_t>(rep.inserted.size()));
    EXPECT_EQ(st.batch.deleted_vertices, static_cast<std::int64_t>(rep.deleted.size()));
    EXPECT_EQ(st.batch.inserted_edges, rep.appended);
    ins_and_del += static_cast<std::int64_t>(batch.insert_vertices.size() - rep.inserted.size());
    // No batch at the cap.
    EXPECT_TRUE(st.converged) << "the batch reached the iteration cap";
    EXPECT_FALSE(st.fallback_used);
    // The statuses (by name) and the rows, in exact slot order.
    {
      const auto view = g.view();
      std::int64_t status_diff = 0;
      std::int64_t row_diff = 0;
      std::int64_t slot_diff = 0;
      for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
        const auto vv = static_cast<std::size_t>(v);
        status_diff += same_state(model.states()[vv], view.status[vv]) ? 0 : 1;
        row_diff += static_cast<std::int64_t>(view.row_end[vv]) == model.row_end()[vv] ? 0 : 1;
      }
      const std::size_t slots = model.slot_columns().size();
      for (std::size_t e = 0; e < slots; ++e) {
        std::uint32_t a = 0;
        std::uint32_t w = 0;
        std::memcpy(&a, &view.out.weights[e], sizeof(float));
        std::memcpy(&w, &model.slot_weights()[e], sizeof(float));
        slot_diff += (view.out.col_ind[e] == model.slot_columns()[e] && a == w) ? 0 : 1;
      }
      EXPECT_EQ(status_diff, 0) << "statuses differ from lp_host_model";
      EXPECT_EQ(row_diff, 0) << "row ends differ from lp_host_model";
      EXPECT_EQ(slot_diff, 0) << "slots differ from lp_host_model (columns or weight bits)";
    }
    // The labels: seeds exact, every label in [0, 1], the harmonic oracle.
    const std::vector<float> labels = host_labels(res, r);
    std::int64_t outside = 0;
    std::int64_t seeds_wrong = 0;
    for (std::size_t v = 0; v < labels.size(); ++v) {
      outside += (labels[v] >= 0.0F && labels[v] <= 1.0F) ? 0 : 1;
      const std::int8_t seed = model.seed_labels()[v];
      if (seed >= 0 && labels[v] != static_cast<float>(seed)) {
        ++seeds_wrong;
      }
    }
    EXPECT_EQ(outside, 0) << "labels outside [0, 1]";
    EXPECT_EQ(seeds_wrong, 0) << "seed labels not exact";
    const ::testing::AssertionResult update_near = near_harmonic(s, model, labels, delta, "update");
    EXPECT_TRUE(update_near);
    ::testing::Test::RecordProperty("batch" + std::to_string(b) + "_margin", update_near.message());
    if (irlp_snapshots) {
      const auto snapshot = irlp(res, g, model, opt);
      EXPECT_TRUE(near_harmonic(s, model, host_labels(res, snapshot), delta, "IrLP"));
    }
  }
  ::testing::Test::RecordProperty("inserted_and_deleted", static_cast<int>(ins_and_del));
  // IrLP agreement with DynLP after the last batch (reported; within 0.5 pp).
  const std::vector<float> dynlp = host_labels(res, r);
  const std::vector<float> irlp_labels = host_labels(res, irlp(res, g, model, opt));
  std::int64_t compared = 0;
  std::int64_t agree = 0;
  for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
    const auto vv = static_cast<std::size_t>(v);
    if (model.present()[vv] != 0 && model.seed_labels()[vv] < 0) {
      ++compared;
      agree += (dynlp[vv] >= 0.5F) == (irlp_labels[vv] >= 0.5F) ? 1 : 0;
    }
  }
  const double agreement =
      compared > 0 ? 100.0 * static_cast<double>(agree) / static_cast<double>(compared) : 100.0;
  ::testing::Test::RecordProperty("irlp_agreement_percent", std::to_string(agreement));
  std::printf("[ checker ] n=%lld %s: IrLP agrees with DynLP on %.3f %% of %lld vertices\n",
              static_cast<long long>(n), rule == lp::schedule::damped ? "damped" : "in_place",
              agreement, static_cast<long long>(compared));
  EXPECT_GE(agreement, 99.5) << "IrLP and DynLP disagree on more than 0.5 pp of the vertices";
  EXPECT_NO_THROW(g.check_integrity(res));
}

TEST(LabelPropagationChecker, Damped5k5) {
  check_stream(5000, 5, lp::schedule::damped, true);
}

TEST(LabelPropagationChecker, InPlace5k5) {
  check_stream(5000, 5, lp::schedule::in_place, true);
}

TEST(LabelPropagationChecker, Damped50k) {
  check_stream(50000, 10, lp::schedule::damped, false);
}

TEST(LabelPropagationChecker, InPlace50k) {
  check_stream(50000, 10, lp::schedule::in_place, false);
}

}  // namespace
