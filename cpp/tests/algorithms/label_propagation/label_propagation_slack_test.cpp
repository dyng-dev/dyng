// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_slack_test.cpp
 * @brief label_propagation on row_layout::slack (ADRs 0103, 0104), on the sequential backend and,
 *        compiled with DYNG_TEST_CUDA=1, on cuda:
 *
 * - damped on slack rows with the published threshold is bitwise slotted's on the same stream
 *   while no compaction has run (the published capacities: nothing moves; half of them: rows
 *   move to the tail, their entries in order);
 * - the streaming mode (unknown capacities, options::similarity_threshold = std::nullopt) on
 *   corpus C (5k5 and, with DYNG_TEST_SLACK_LARGE, 50k at delta 1e-6, both schedules): every
 *   batch against the harmonic oracle (AD7's bound, no PCG skip, the 3m class rule), the seeds
 *   exact and every label in [0, 1], the components of G' against union_find with the threshold
 *   recomputed independently by the documented accumulation order, the initial frontier against
 *   lp_host_model's V_aff, the states against lp_host_model and the rows against the kit's slack
 *   model; with half the capacities, a headroom of 0 and a compaction forced after every second
 *   batch (graph_access::compact_slack: the rows of a 5k5 stream never outgrow their dead slots
 *   in 5 batches), the oracles still pass;
 * - compactions inside update() (no hints, a headroom of 0, 24 batches of a 5k stream): the
 *   states, the rows, the seeds and the label range every batch, and damped's labels against
 *   slotted rows (bitwise until the first compaction, then within 100 delta on average);
 * - the vertex growth of slack rows is refused by update() before anything changes.
 */
#include "graph/graph_impl.hpp"
#include "support/slack_model.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
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

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <optional>
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

dyng::graph_properties props(dyng::row_layout layout, double headroom = 0.125) {
  dyng::graph_properties p;
  p.directed = false;
  p.layout = layout;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  p.headroom = headroom;
  return p;
}

stream_t make_stream(std::int64_t n, std::int32_t batches) {
  dyng::generators::vertex_arrival_options opt;
  opt.total_vertices = n;
  opt.num_batches = batches;
  return dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(
      dyng::resources::sequential(), opt);
}

lp_graph make_graph(const dyng::resources& res, const stream_t& s, dyng::row_layout layout,
                    int percent, double headroom = 0.125) {
  std::vector<std::int32_t> hints;
  if (layout == dyng::row_layout::slotted || percent > 0) {
    for (const auto c : s.row_capacity) {
      hints.push_back(layout == dyng::row_layout::slotted ? c : c * percent / 100);
    }
  }
  const std::vector<std::int32_t>& h = hints;
  return lp_graph::with_capacity(res, static_cast<std::int32_t>(s.true_labels.size()), host_view(h),
                                 props(layout, headroom));
}

/// The labels after every batch of the published flow, and the compactions seen before each.
struct run_record {
  std::vector<std::vector<std::uint32_t>> bits;
  std::vector<std::int64_t> compactions_so_far;
  std::int64_t relocated = 0;
};

run_record run_published(const stream_t& s, dyng::row_layout layout, int percent,
                         lp::schedule rule) {
  const dyng::resources res = test_resources();
  lp_graph g = make_graph(res, s, layout, percent);
  lp::options opt;
  opt.similarity_threshold = s.similarity_threshold;
  opt.update_schedule = rule;
  auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  run_record out;
  std::int64_t compactions = 0;
  for (const auto& batch : s.batches) {
    const lp::stats st = lp::update(res, g, batch.view(), r);
    compactions += st.batch.compactions;
    out.relocated += st.batch.relocated_rows;
    const std::vector<float> labels = host_labels(res, r);
    std::vector<std::uint32_t> bits(labels.size());
    std::memcpy(bits.data(), labels.data(), labels.size() * sizeof(float));
    out.bits.push_back(std::move(bits));
    out.compactions_so_far.push_back(compactions);
  }
  return out;
}

void expect_bitwise_slotted(std::int64_t n, std::int32_t batches) {
  const stream_t s = make_stream(n, batches);
  const run_record slotted = run_published(s, dyng::row_layout::slotted, 100, lp::schedule::damped);
  for (const int percent : {100, 50}) {
    SCOPED_TRACE(percent);
    const run_record slack =
        run_published(s, dyng::row_layout::slack, percent, lp::schedule::damped);
    std::size_t compared = 0;
    for (std::size_t b = 0; b < s.batches.size(); ++b) {
      if (slack.compactions_so_far[b] > 0) {
        break;  // a compaction changes the order of a row's entries (ADR 0103)
      }
      EXPECT_EQ(slack.bits[b], slotted.bits[b]) << "batch " << b;
      ++compared;
    }
    if (percent == 100) {
      EXPECT_EQ(slack.relocated, 0);
      EXPECT_EQ(compared, s.batches.size());
    } else {
      EXPECT_GT(slack.relocated, 0);
      EXPECT_GE(compared, 2U);
    }
    ::testing::Test::RecordProperty("batches_compared_" + std::to_string(percent),
                                    static_cast<int>(compared));
  }
}

/// The harmonic oracle of one set of labels (AD7): mean <= 2000 delta, converged, the 3m rule.
::testing::AssertionResult near_harmonic(const stream_t& s, const dt::lp_host_model& model,
                                         const std::vector<float>& labels, float delta) {
  const auto h = dt::harmonic_solution(s.graph.row_ptr, s.graph.col_ind, s.graph.weights,
                                       model.present(), model.seed_labels());
  if (!h.converged) {
    return ::testing::AssertionFailure() << "the PCG did not converge (a skip)";
  }
  const auto err = dt::label_error_against(labels, h, model.present(), model.seed_labels());
  if (err.mean > 2000.0 * static_cast<double>(delta)) {
    return ::testing::AssertionFailure()
           << "mean |F - F*| = " << err.mean << " above 2000 delta = " << 2000.0 * delta;
  }
  const auto ca =
      dt::class_agreement_with(labels, h, model.present(), model.seed_labels(), 3.0 * class_margin);
  if (ca.flips_beyond_margin > 0) {
    return ::testing::AssertionFailure()
           << ca.flips_beyond_margin << " class flips where |F* - 0.5| > 3m";
  }
  return ::testing::AssertionSuccess() << "margin " << 2000.0 * delta / err.mean;
}

/// The streaming mode's threshold, recomputed by the documented accumulation order (ADR 0104)
/// from lp_host_model's rows of G_t and the batch: the deleted vertices' live entries out (in the
/// order of Del ∩ V_t, each vertex's partial sum in slot order), the appended entries in.
class streaming_tau {
 public:
  float next(const dt::lp_host_model& m,
             const dyng::generators::vertex_arrival_batch<std::int32_t>& b) {
    using st = dt::lp_host_model::vertex_state;
    const auto& state = m.states();
    std::vector<std::int32_t> del_eff;
    for (const std::int32_t v : b.delete_vertices) {
      if (state[static_cast<std::size_t>(v)] == st::active) {
        del_eff.push_back(v);
      }
    }
    std::vector<std::int32_t> sorted = del_eff;
    std::sort(sorted.begin(), sorted.end());
    for (const std::int32_t u : del_eff) {
      double partial = 0.0;
      std::int64_t count = 0;
      const auto uu = static_cast<std::size_t>(u);
      for (auto e = static_cast<std::size_t>(m.row_ptr()[uu]);
           e < static_cast<std::size_t>(m.row_end()[uu]); ++e) {
        const std::int32_t c = m.slot_columns()[e];
        if (c < 0 || state[static_cast<std::size_t>(c)] != st::active) {
          continue;
        }
        const double w = static_cast<double>(m.slot_weights()[e]);
        partial += w;
        ++count;
        if (c != u && !std::binary_search(sorted.begin(), sorted.end(), c)) {
          partial += w;
          ++count;
        }
      }
      sum_ -= partial;
      count_ -= count;
    }
    std::vector<char> active(state.size(), 0);
    for (std::size_t v = 0; v < state.size(); ++v) {
      active[v] = state[v] == st::active ? 1 : 0;
    }
    for (const std::int32_t v : b.delete_vertices) {
      active[static_cast<std::size_t>(v)] = 0;
    }
    std::vector<char> deleted(state.size(), 0);
    for (const std::int32_t v : b.delete_vertices) {
      deleted[static_cast<std::size_t>(v)] = 1;
    }
    for (const std::int32_t v : b.insert_vertices) {
      if (deleted[static_cast<std::size_t>(v)] == 0) {
        active[static_cast<std::size_t>(v)] = 1;
      }
    }
    for (std::size_t e = 0; e < b.edge_sources.size(); ++e) {
      if (active[static_cast<std::size_t>(b.edge_sources[e])] != 0 &&
          active[static_cast<std::size_t>(b.edge_destinations[e])] != 0) {
        const double w = static_cast<double>(b.edge_weights[e]);
        sum_ += w;
        sum_ += w;
        count_ += 2;
      }
    }
    return count_ > 0 ? static_cast<float>(sum_ / static_cast<double>(count_)) : 0.5F;
  }

 private:
  double sum_ = 0.0;
  std::int64_t count_ = 0;
};

/// The streaming mode on one stream: unknown capacities (`percent` = 0) or a fraction of them,
/// the threshold std::nullopt, every batch checked.
void check_streaming(std::int64_t n, std::int32_t batches, lp::schedule rule, int percent,
                     double headroom, bool force_compactions, bool irlp_snapshots,
                     dyng::row_layout layout = dyng::row_layout::slack) {
  const float delta = 1e-6F;
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(n, batches);
  lp_graph g = make_graph(res, s, layout, percent, headroom);
  lp::options opt;
  opt.similarity_threshold = std::nullopt;  // the streaming mode (AD16's advanced mode)
  opt.tolerance = delta;
  opt.update_schedule = rule;
  auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  std::vector<std::int64_t> cap64(s.row_capacity.begin(), s.row_capacity.end());
  dt::lp_host_model model(cap64);
  dyng::test::slack_model rows(n);
  streaming_tau tau;
  std::int64_t compactions = 0;
  std::int64_t relocated = 0;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    const auto& batch = s.batches[b];
    const float t = tau.next(model, batch);
    const lp::stats st = lp::update(res, g, batch.view(), r);
    dt::lp_model_batch mb;
    mb.insert_vertices = batch.insert_vertices;
    mb.insert_labels = batch.insert_vertex_labels;
    mb.delete_vertices = batch.delete_vertices;
    mb.edge_sources = batch.edge_sources;
    mb.edge_destinations = batch.edge_destinations;
    mb.edge_weights = batch.edge_weights;
    const dt::lp_model_report rep = model.apply(mb, t);
    rows.apply(batch.view(), st.batch.compactions > 0);
    compactions += st.batch.compactions;
    relocated += st.batch.relocated_rows;
    // G' with the streaming threshold: the components against union_find, V_aff.
    dt::union_find uf(n);
    for (std::size_t i = 0; i + 1 < rep.sparsified.size(); i += 2) {
      uf.unite(rep.sparsified[i], rep.sparsified[i + 1]);
    }
    std::set<std::int64_t> roots;
    for (const std::int32_t v : rep.inserted) {
      roots.insert(uf.find(v));
    }
    EXPECT_EQ(st.components, static_cast<std::int64_t>(roots.size()));
    EXPECT_EQ(st.sparsified_edges, static_cast<std::int64_t>(rep.sparsified.size() / 2))
        << "the streaming threshold differs from the documented accumulation";
    EXPECT_EQ(st.initial_frontier, static_cast<std::int64_t>(rep.affected.size()));
    EXPECT_EQ(st.batch.inserted_edges, rep.appended);
    EXPECT_TRUE(st.converged);
    // The states against lp_host_model (by name), the rows against the slack model.
    const auto view = g.view();
    std::int64_t status_diff = 0;
    for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
      const auto vv = static_cast<std::size_t>(v);
      const auto ms = model.states()[vv];
      const auto gs = view.status[vv];
      const bool same =
          (ms == dt::lp_host_model::vertex_state::inactive &&
           gs == dyng::vertex_status::inactive) ||
          (ms == dt::lp_host_model::vertex_state::active && gs == dyng::vertex_status::active) ||
          (ms == dt::lp_host_model::vertex_state::deleted && gs == dyng::vertex_status::deleted);
      status_diff += same ? 0 : 1;
    }
    EXPECT_EQ(status_diff, 0);
    rows.expect_equal(g, "batch " + std::to_string(b));
    // The labels: seeds exact, in [0, 1], the harmonic oracle.
    const std::vector<float> labels = host_labels(res, r);
    std::int64_t outside = 0;
    std::int64_t seeds_wrong = 0;
    for (std::size_t v = 0; v < labels.size(); ++v) {
      outside += (labels[v] >= 0.0F && labels[v] <= 1.0F) ? 0 : 1;
      const std::int8_t seed = model.seed_labels()[v];
      seeds_wrong += (seed >= 0 && labels[v] != static_cast<float>(seed)) ? 1 : 0;
    }
    EXPECT_EQ(outside, 0);
    EXPECT_EQ(seeds_wrong, 0);
    const ::testing::AssertionResult near = near_harmonic(s, model, labels, delta);
    EXPECT_TRUE(near);
    ::testing::Test::RecordProperty("batch" + std::to_string(b) + "_margin", near.message());
    if (irlp_snapshots) {
      std::vector<std::int32_t> sv;
      std::vector<std::int8_t> sl;
      for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
        const auto vv = static_cast<std::size_t>(v);
        if (model.present()[vv] != 0 && model.seed_labels()[vv] >= 0) {
          sv.push_back(v);
          sl.push_back(model.seed_labels()[vv]);
        }
      }
      const std::vector<std::int32_t>& svr = sv;
      const std::vector<std::int8_t>& slr = sl;
      const auto snapshot = lp::compute(res, g, host_view(svr), host_view(slr), opt);
      EXPECT_TRUE(near_harmonic(s, model, host_labels(res, snapshot), delta)) << "IrLP";
    }
    if (::testing::Test::HasFatalFailure()) return;
    if (force_compactions && b % 2 == 1) {
      // A forced compaction between updates (the graph's rows rewritten without the entries of
      // deleted vertices; the result continues on the same graph state).
      dyng::detail::graph_access::compact_slack(res, g);
      rows.compact();
      rows.expect_equal(g, "compacted after batch " + std::to_string(b));
      ++compactions;
    }
  }
  ::testing::Test::RecordProperty("compactions", static_cast<int>(compactions));
  ::testing::Test::RecordProperty("relocated_rows", static_cast<int>(relocated));
  EXPECT_GT(relocated, 0);
  if (force_compactions) {
    EXPECT_GT(compactions, 0);
  }
  EXPECT_NO_THROW(g.check_integrity(res));
}

#if !defined(DYNG_TEST_SLACK_LARGE)

/// Compactions inside update() (no capacity hints, a headroom of 0, a long stream: the dead slots
/// of the moved rows outgrow the owned ones, and apply() compacts within the batch, after the
/// engines' workspaces were sized for the batch). The published threshold; every batch: the
/// states against lp_host_model, the rows against the slack model, the seeds exact, every label in
/// [0, 1], converged; under damped also the labels against the same stream on slotted rows (dynG
/// against dynG: bitwise until the first compaction, then within 100 delta on average, since a
/// compaction changes the order in which the device sums a row's entries; in_place is not
/// repeatable on cuda, so its labels have no slotted reference here). The harmonic
/// oracle's bound (AD7) is calibrated on corpus C's short streams (5 and 10 batches); on a stream
/// this long O's incremental rule drifts past it on slotted rows as well (ADR 0104), so it is not
/// the reference here.
void check_natural_compactions(std::int64_t n, std::int32_t batches, lp::schedule rule) {
  const float delta = 1e-6F;
  const dyng::resources res = test_resources();
  const stream_t s = make_stream(n, batches);
  const bool damped = rule == lp::schedule::damped;
  const run_record slotted =
      damped ? run_published(s, dyng::row_layout::slotted, 100, rule) : run_record{};
  lp_graph g = make_graph(res, s, dyng::row_layout::slack, 0, 0.0);
  lp::options opt;
  opt.similarity_threshold = s.similarity_threshold;
  opt.update_schedule = rule;
  auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  std::vector<std::int64_t> cap64(s.row_capacity.begin(), s.row_capacity.end());
  dt::lp_host_model model(cap64);
  dyng::test::slack_model rows(n);
  std::int64_t compactions = 0;
  double worst_mean = 0.0;
  float worst_max = 0.0F;
  for (std::size_t b = 0; b < s.batches.size(); ++b) {
    SCOPED_TRACE("batch " + std::to_string(b));
    const auto& batch = s.batches[b];
    const lp::stats st = lp::update(res, g, batch.view(), r);
    dt::lp_model_batch mb;
    mb.insert_vertices = batch.insert_vertices;
    mb.insert_labels = batch.insert_vertex_labels;
    mb.delete_vertices = batch.delete_vertices;
    mb.edge_sources = batch.edge_sources;
    mb.edge_destinations = batch.edge_destinations;
    mb.edge_weights = batch.edge_weights;
    (void)model.apply(mb, s.similarity_threshold);
    rows.apply(batch.view(), st.batch.compactions > 0);
    compactions += st.batch.compactions;
    EXPECT_TRUE(st.converged);
    const auto view = g.view();
    std::int64_t status_diff = 0;
    for (std::int32_t v = 0; v < model.num_vertices(); ++v) {
      const auto vv = static_cast<std::size_t>(v);
      const auto ms = model.states()[vv];
      const auto gs = view.status[vv];
      const bool same =
          (ms == dt::lp_host_model::vertex_state::inactive &&
           gs == dyng::vertex_status::inactive) ||
          (ms == dt::lp_host_model::vertex_state::active && gs == dyng::vertex_status::active) ||
          (ms == dt::lp_host_model::vertex_state::deleted && gs == dyng::vertex_status::deleted);
      status_diff += same ? 0 : 1;
    }
    EXPECT_EQ(status_diff, 0);
    rows.expect_equal(g, "batch " + std::to_string(b));
    const std::vector<float> labels = host_labels(res, r);
    std::vector<float> reference = labels;
    if (damped) {
      std::memcpy(reference.data(), slotted.bits[b].data(), reference.size() * sizeof(float));
    }
    std::int64_t outside = 0;
    std::int64_t seeds_wrong = 0;
    double sum = 0.0;
    float largest = 0.0F;
    std::int64_t present = 0;
    for (std::size_t v = 0; v < labels.size(); ++v) {
      outside += (labels[v] >= 0.0F && labels[v] <= 1.0F) ? 0 : 1;
      const std::int8_t seed = model.seed_labels()[v];
      seeds_wrong += (seed >= 0 && labels[v] != static_cast<float>(seed)) ? 1 : 0;
      if (model.present()[v] != 0) {
        const float d =
            labels[v] > reference[v] ? labels[v] - reference[v] : reference[v] - labels[v];
        sum += static_cast<double>(d);
        largest = std::max(largest, d);
        ++present;
      }
    }
    EXPECT_EQ(outside, 0);
    EXPECT_EQ(seeds_wrong, 0);
    const double mean = present > 0 ? sum / static_cast<double>(present) : 0.0;
    worst_mean = std::max(worst_mean, mean);
    worst_max = std::max(worst_max, largest);
    if (compactions == 0) {
      EXPECT_EQ(labels, reference) << "no compaction yet: bitwise slotted";
    }
    // The tolerance's scale: the two runs stop within delta of each other's iterates.
    EXPECT_LE(mean, 100.0 * static_cast<double>(delta)) << "mean |F_slack - F_slotted|";
    if (::testing::Test::HasFatalFailure()) return;
  }
  ::testing::Test::RecordProperty("compactions", static_cast<int>(compactions));
  ::testing::Test::RecordProperty("worst_mean", std::to_string(worst_mean));
  ::testing::Test::RecordProperty("worst_max", std::to_string(worst_max));
  EXPECT_GT(compactions, 0) << "the stream ran no compaction inside update()";
  EXPECT_NO_THROW(g.check_integrity(res));
}

#endif  // !DYNG_TEST_SLACK_LARGE

#if !defined(DYNG_TEST_SLACK_LARGE)

TEST(LabelPropagationSlack, DampedIsBitwiseSlottedWhileNoCompactionRan5k5) {
  expect_bitwise_slotted(5000, 5);
}

TEST(LabelPropagationSlack, StreamingDamped5k5) {
  check_streaming(5000, 5, lp::schedule::damped, 0, 0.125, false, true);
}

TEST(LabelPropagationSlack, StreamingInPlace5k5) {
  check_streaming(5000, 5, lp::schedule::in_place, 0, 0.125, false, false);
}

TEST(LabelPropagationSlack, ForcedCompactionsAndTheOracles5k5) {
  check_streaming(5000, 5, lp::schedule::damped, 50, 0.0, true, true);
  check_streaming(5000, 5, lp::schedule::in_place, 50, 0.0, true, false);
}

TEST(LabelPropagationSlack, NaturalCompactionsInsideUpdate5k24) {
  check_natural_compactions(5000, 24, lp::schedule::damped);
  check_natural_compactions(5000, 24, lp::schedule::in_place);
}

TEST(LabelPropagationSlack, VertexGrowthIsRefusedBeforeAnythingChanges) {
  const dyng::resources res = test_resources();
  dyng::graph_properties p = props(dyng::row_layout::slack);
  p.semantics.allow_vertex_growth = true;
  lp_graph g = lp_graph::with_capacity(res, 4, dyng::array_view<const std::int32_t>(), p);
  lp::options opt;
  opt.similarity_threshold = 0.5F;
  auto r = lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                       dyng::array_view<const std::int8_t>(), opt);
  dyng::edge_batch<std::int32_t, float> b(1);
  b.insert_vertex(0, 1);
  b.insert_vertex(6);
  b.insert_edge(0, 6, {0.75F});
  const std::uint64_t version = g.version();
  EXPECT_THROW((void)lp::update(res, g, b.view(), r), dyng::not_supported_error);
  EXPECT_EQ(g.num_vertices(), 4);
  EXPECT_EQ(g.version(), version);
  dyng::edge_batch<std::int32_t, float> ok(1);
  ok.insert_vertex(0, 1);
  ok.insert_vertex(1);
  ok.insert_edge(0, 1, {0.75F});
  EXPECT_NO_THROW((void)lp::update(res, g, ok.view(), r));
}

#else

TEST(LabelPropagationSlack, DampedIsBitwiseSlottedWhileNoCompactionRan50k) {
  expect_bitwise_slotted(50000, 10);
}

TEST(LabelPropagationSlack, StreamingDamped50k) {
  check_streaming(50000, 10, lp::schedule::damped, 0, 0.125, false, false);
}

TEST(LabelPropagationSlack, StreamingInPlace50k) {
  check_streaming(50000, 10, lp::schedule::in_place, 0, 0.125, false, false);
}

#endif

}  // namespace
