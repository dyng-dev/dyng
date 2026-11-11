// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_allocation_failure_test.cpp
 * @brief A host allocation failure inside label_propagation::compute(), update(),
 *        result::reserve() or result::clone() is an out_of_memory_error with the guarantee the
 *        header documents (M10a review; the ADR 0060 precedent): no std::bad_alloc leaves the
 *        library.
 *
 * This executable replaces the global operator new (support/failing_allocator.hpp) and fails
 * every allocation of each call once, in turn, on the sequential backend. Not under the
 * sanitizers, which replace operator new themselves (the tests skip).
 */
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/label_propagation.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <optional>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;
namespace lp = dyng::label_propagation;
using dyng::host_view;
using lp_graph = dyng::graph<std::int32_t, std::int32_t, float>;
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

const stream_t& stream() {
  static const stream_t s = [] {
    dyng::generators::vertex_arrival_options opt;
    opt.total_vertices = 300;
    opt.num_batches = 3;
    return dyng::generators::legacy::dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(
        dyng::resources::sequential(), opt);
  }();
  return s;
}

lp_graph capacity_graph(const dyng::resources& res) {
  const stream_t& s = stream();
  return lp_graph::with_capacity(res, static_cast<std::int32_t>(s.row_capacity.size()),
                                 host_view(s.row_capacity), slotted_props());
}

lp::options published() {
  lp::options opt;
  opt.similarity_threshold = stream().similarity_threshold;
  opt.update_schedule = lp::schedule::damped;
  return opt;
}

lp::result<std::int32_t> empty_compute(const dyng::resources& res, const lp_graph& g) {
  return lp::compute(res, g, dyng::array_view<const std::int32_t>(),
                     dyng::array_view<const std::int8_t>(), published());
}

std::vector<std::uint32_t> bits(const lp::result<std::int32_t>& r) {
  const auto v = r.labels();
  std::vector<std::uint32_t> out(v.size());
  std::memcpy(out.data(), v.data(), v.size() * sizeof(float));
  return out;
}

TEST(LabelPropagationAllocationFailure, ComputeReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  std::optional<lp_graph> g;
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread,
                [&] {
                  g.emplace(capacity_graph(res));
                  (void)g->apply(res, stream().batches[0].view());
                },
                [&] { (void)empty_compute(res, *g); }),
            0);
}

TEST(LabelPropagationAllocationFailure, ReserveReportsOutOfMemoryAndKeepsTheResult) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  std::optional<lp_graph> g;
  std::optional<lp::result<std::int32_t>> r;
  std::vector<std::uint32_t> before;
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread,
                [&] {
                  g.emplace(capacity_graph(res));
                  r.emplace(empty_compute(res, *g));
                  before = bits(*r);
                },
                [&] { r->reserve(res, 100000, 400000); },
                [&](bool threw) {
                  if (threw) {
                    EXPECT_EQ(bits(*r), before);  // the result is still usable
                    EXPECT_NO_THROW((void)lp::update(res, *g, stream().batches[0].view(), *r));
                  }
                }),
            0);
}

TEST(LabelPropagationAllocationFailure, CloneReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  lp_graph g = capacity_graph(res);
  auto r = empty_compute(res, g);
  (void)lp::update(res, g, stream().batches[0].view(), r);
  const std::vector<std::uint32_t> want = bits(r);
  EXPECT_GT(alloc::fail_each_allocation<dyng::out_of_memory_error>(
                alloc::scope::calling_thread, [] {}, [&] { (void)r.clone(res); },
                [&](bool) { EXPECT_EQ(bits(r), want); }),
            0);
}

// update(): before the commit `g` and `r` are unchanged and usable; after it `g` is at the new
// version and `r` is poisoned (stale_result_error), as the header documents.
TEST(LabelPropagationAllocationFailure, UpdateReportsOutOfMemoryWithTheDocumentedGuarantee) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const auto& batches = stream().batches;
  std::vector<std::uint32_t> want;
  {
    lp_graph g = capacity_graph(res);
    auto r = empty_compute(res, g);
    (void)lp::update(res, g, batches[0].view(), r);
    (void)lp::update(res, g, batches[1].view(), r);
    want = bits(r);
  }
  std::optional<lp_graph> g;
  std::optional<lp::result<std::int32_t>> r;
  std::vector<std::uint32_t> before;
  std::uint64_t version = 0;
  int unchanged = 0;
  int after_commit = 0;
  const int failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
      alloc::scope::calling_thread,
      [&] {
        g.emplace(capacity_graph(res));
        r.emplace(empty_compute(res, *g));
        (void)lp::update(res, *g, batches[0].view(), *r);
        before = bits(*r);
        version = g->version();
      },
      [&] { (void)lp::update(res, *g, batches[1].view(), *r); },
      [&](bool threw) {
        if (!threw) {
          EXPECT_EQ(bits(*r), want);
          return;
        }
        if (g->version() == version) {  // before the commit: strong
          ++unchanged;
          EXPECT_EQ(bits(*r), before);
          (void)lp::update(res, *g, batches[1].view(), *r);  // a retry succeeds
          EXPECT_EQ(bits(*r), want);
        } else {  // after the commit: the result is poisoned
          ++after_commit;
          EXPECT_EQ(g->version(), version + 1);
          EXPECT_THROW((void)r->labels(), dyng::stale_result_error);
        }
      });
  EXPECT_GT(failures, 0);
  EXPECT_GT(unchanged, 0);
  (void)after_commit;  // reached where the update allocates after the commit
}

}  // namespace
