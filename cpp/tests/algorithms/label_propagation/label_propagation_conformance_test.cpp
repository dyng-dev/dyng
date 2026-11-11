// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_conformance_test.cpp
 * @brief The conformance kit for label_propagation (PLAN Section 8.2; ADR 0076): C0-C12 on the
 *        float slotted axis with vertex batches, for schedule::in_place (`label_propagation`) and
 *        schedule::damped (the test-only tag `label_propagation_damped`), on every backend of the
 *        executable (host: sequential; DYNG_TEST_CUDA=1: cuda, compared with sequential), and the
 *        axis's own generator check (a kit tests only what its generator makes).
 */
#include "conformance/conformance.hpp"
#include "conformance/registry.hpp"
#include "conformance/slotted_generators.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <random>

DYNG_CONFORMANCE_SUITE(label_propagation);
DYNG_CONFORMANCE_SUITE(label_propagation_damped);

namespace {

namespace dc = ::dyng::conformance;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
using ct = dc::container_traits<dc::slotted_axis<lp_graph>>;
using traits = dc::test_traits<dc::tags::label_propagation>;

/// One batch of `mix` from a fresh small model, applied to a sequential graph of the model.
struct applied_mix {
  ct::generated_type gen;    ///< the batch
  dyng::apply_summary sum;   ///< what the commit did
  std::int64_t pending = 0;  ///< pending edges before the batch
};

applied_mix apply_mix(dc::vertex_mix mix) {
  std::mt19937_64 rng(91);
  ct::model_type model = ct::random_model<traits>(dc::size_class::small, rng);
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = ct::build(model, res, ct::presets<traits>().front().props);
  EXPECT_TRUE(ct::matches(g, model));
  applied_mix out;
  out.pending = dc::slotted_detail::pending_edges(model);
  out.gen = ct::random_batch(model, mix, rng);
  out.sum = g.apply(res, out.gen.batch.view());
  EXPECT_TRUE(ct::matches(g, model));
  EXPECT_NO_THROW(g.check_integrity(res));
  return out;
}

TEST(LabelPropagationKitGenerators, EveryMixMakesWhatItsNamePromises) {
  using dc::vertex_mix;
  {
    const applied_mix a = apply_mix(vertex_mix::arrival_with_labels);
    EXPECT_GT(a.sum.inserted_vertices, 0);
    EXPECT_GT(a.gen.labelled_arrivals, 0);
    EXPECT_GT(a.sum.inserted_edges, 0);
    EXPECT_EQ(a.sum.deleted_vertices, 0);
  }
  {
    const applied_mix a = apply_mix(vertex_mix::departure);
    EXPECT_GT(a.sum.deleted_vertices, 0);
    EXPECT_EQ(a.sum.inserted_vertices, 0);
    EXPECT_EQ(a.gen.batch.num_insertions(), 0U);
  }
  {
    const applied_mix a = apply_mix(vertex_mix::mixed);
    EXPECT_GT(a.sum.inserted_vertices, 0);
    EXPECT_GT(a.sum.deleted_vertices, 0);
  }
  {
    // Ins and Del: the transient ids end deleted (as in the original), and their edges are
    // dropped.
    const applied_mix a = apply_mix(vertex_mix::insert_and_delete_same_batch);
    EXPECT_GT(a.gen.inserted_and_deleted, 0);
    EXPECT_EQ(a.sum.inserted_vertices, a.gen.arrivals - a.gen.inserted_and_deleted);
    EXPECT_GT(a.sum.dropped_insertions, 0);
  }
  {
    const applied_mix a = apply_mix(vertex_mix::edges_between_present);
    EXPECT_GT(a.pending, 0);
    EXPECT_GT(a.gen.delivered_pending, 0);
    EXPECT_EQ(a.sum.inserted_edges, 2 * a.gen.delivered_pending);
    EXPECT_EQ(a.sum.inserted_vertices + a.sum.deleted_vertices, 0);
  }
}

}  // namespace
