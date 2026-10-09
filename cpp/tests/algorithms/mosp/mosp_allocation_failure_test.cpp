// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file mosp_allocation_failure_test.cpp
 * @brief mosp::result::set_options() keeps the strong guarantee the header states (the mosp API
 *        review, ADR 0035): a rejected change leaves the options as they were, and an accepted
 *        one allocates nothing after the checks (it succeeds while every allocation fails).
 *
 * This executable replaces the global operator new with one that can be told to fail the n-th
 * allocation on the calling thread (not compiled under the sanitizers, which replace operator new
 * themselves; the tests then skip).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/mosp.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <vector>

namespace {

/// Allocations on this thread until one fails (0: none fails).
thread_local int fail_countdown = 0;

}  // namespace

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  if (fail_countdown > 0 && --fail_countdown == 0) {
    throw std::bad_alloc();
  }
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
// NOLINTEND(cppcoreguidelines-no-malloc)
#endif

namespace {

/// Fail the n-th allocation of the calling thread from now on.
struct failing_allocation {
  explicit failing_allocation(int n) noexcept {
    fail_countdown = n;
  }
  failing_allocation(const failing_allocation&) = delete;
  failing_allocation& operator=(const failing_allocation&) = delete;
  failing_allocation(failing_allocation&&) = delete;
  failing_allocation& operator=(failing_allocation&&) = delete;
  ~failing_allocation() {
    fail_countdown = 0;
  }
};

using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;

/// A 4-vertex graph with three weight columns.
graph_t small_graph(const dyng::resources& res) {
  dyng::edge_list<std::int32_t, std::int32_t> edges;
  edges.num_vertices = 4;
  edges.num_weights = 3;
  edges.add_edge(0, 1, {1, 2, 3});
  edges.add_edge(1, 2, {2, 1, 1});
  edges.add_edge(0, 2, {4, 4, 1});
  edges.add_edge(2, 3, {1, 1, 1});
  dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
  props.num_weights = 3;
  return graph_t::from_edges(res, edges.view(), props);
}

TEST(MospAllocationFailure, SetOptionsIsStrong) {
  if (DYNG_TEST_SANITIZED) {
    GTEST_SKIP() << "operator new is the sanitizer's";
  }
  const dyng::resources res = dyng::resources::sequential();
  const graph_t g = small_graph(res);
  dyng::mosp::options opt;
  opt.preferences = {4, 1, 4};
  opt.delta = 3;
  auto r = dyng::mosp::compute(res, g, 0, opt);
  // A rejected change leaves every option as it was.
  dyng::mosp::options rejected = opt;
  rejected.delta = 9;
  rejected.preferences = {4, 1, 2};
  EXPECT_THROW(r.set_options(rejected), dyng::invalid_argument_error);
  rejected = opt;
  rejected.compute_path_costs = false;
  rejected.delta = -1;
  EXPECT_THROW(r.set_options(rejected), dyng::invalid_argument_error);
  EXPECT_EQ(r.get_options().delta, 3);
  EXPECT_TRUE(r.get_options().compute_path_costs);
  // An accepted change allocates nothing: it succeeds although every allocation would fail.
  dyng::mosp::options changed = opt;
  changed.delta = 9;
  changed.compute_path_costs = false;
  {
    const failing_allocation fail(1);
    EXPECT_NO_THROW(r.set_options(changed));
  }
  EXPECT_EQ(r.get_options().delta, 9);
  EXPECT_FALSE(r.get_options().compute_path_costs);
  EXPECT_EQ(r.get_options().preferences, (std::vector<std::int32_t>{4, 1, 4}));
}

}  // namespace
