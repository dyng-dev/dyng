// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_cuda_test.cpp
 * @brief The CUDA backend of sssp beyond the shared suites: engine selection (the forced
 *        "no cooperative launch" path, engine::operators), placement of graphs and results, device
 *        arrays, the packing boundary n = 2^17 - 1 (47 distance bits), the distance-only fallback
 *        (320 x 320 grid, weights 2 * 10^9), poisoned results, steady-state allocations, profiler
 *        device times and kernel warm-up.
 *
 * Every test skips when no CUDA device is visible. The shared suites (sssp_test.cpp,
 * sssp_random_test.cpp, sssp_fixture_test.cpp) are compiled into the same executable with
 * DYNG_TEST_CUDA=1 and run their cases on the cuda backend, the randomized ones against the
 * sequential and OpenMP backends.
 */
#include "algorithms/sssp/problem.hpp"
#include "core/resources_access.hpp"
#include "framework/workspace.hpp"
#include "graph/graph_impl.hpp"
#include "support/gtest_helpers.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>
#include <dyng/testing/check_sssp.hpp>
#include <dyng/testing/dijkstra.hpp>
#include <dyng/update.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <random>
#include <string>
#include <vector>

namespace {

using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
using result_t = dyng::sssp::result<std::int32_t>;
using batch_t = dyng::edge_batch<std::int32_t, std::int32_t>;
using dyng::test::host_copy;
constexpr std::int64_t inf = dyng::infinite_distance<std::int64_t>();

struct edge {
  std::int32_t u;
  std::int32_t v;
  std::int32_t w;
};

graph_t make_graph(const dyng::resources& res, std::int32_t n, const std::vector<edge>& edges) {
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  for (const edge& e : edges) {
    list.add_edge(e.u, e.v, {e.w});
  }
  return graph_t::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
}

/// The 5-vertex "ties" graph of MOSP_ESCHER's test_mosp_update.
graph_t ties_graph(const dyng::resources& res) {
  return make_graph(res, 5, {{0, 1, 5}, {0, 2, 1}, {2, 3, 1}, {1, 4, 2}, {3, 4, 2}});
}

bool contains(const std::string& text, const std::string& part) {
  return text.find(part) != std::string::npos;
}

class SsspCuda : public ::testing::Test {
 protected:
  void SetUp() override {
    DYNG_SKIP_IF_NO_CUDA();
  }
};

TEST_F(SsspCuda, ResultArraysLiveOnTheDevice) {
  const auto res = dyng::resources::cuda();
  auto g = ties_graph(res);
  EXPECT_EQ(g.space(), dyng::memory_space::device);
  result_t r = dyng::sssp::compute(res, g, 0);
  EXPECT_EQ(r.space(), dyng::memory_space::device);
  EXPECT_EQ(r.distances().space(), dyng::memory_space::device);
  EXPECT_EQ(r.parents().device(), res.device());
  EXPECT_EQ(dyng::to_vector(res, r.distances()), (std::vector<std::int64_t>{0, 5, 1, 2, 4}));
  EXPECT_EQ(dyng::to_vector(res, r.parents()), (std::vector<std::int32_t>{-1, 0, 0, 2, 3}));
}

TEST_F(SsspCuda, WithoutCooperativeLaunchAutomaticAndFusedThrowNotSupported) {
  const auto res = dyng::resources::cuda();
  auto g = ties_graph(res);
  result_t r = dyng::sssp::compute(res, g, 0);
  // A device without cooperative launch (forced; the recorded flag is shared by every copy).
  dyng::detail::resources_access::force_cooperative_launch(res, false);
  for (const dyng::engine e : {dyng::engine::automatic, dyng::engine::fused}) {
    dyng::sssp::options opt;
    opt.cuda_engine = e;
    try {
      (void)dyng::sssp::compute(res, g, 0, opt);
      FAIL() << "compute ran without cooperative launch";
    } catch (const dyng::not_supported_error& error) {
      EXPECT_TRUE(contains(error.what(), "cooperative launch")) << error.what();
      EXPECT_TRUE(contains(error.what(), "resources::openmp()")) << error.what();
    }
    r.set_options(opt);
    batch_t b;
    b.delete_edge(0, 2);
    EXPECT_THROW((void)dyng::sssp::update(res, g, b.view(), r), dyng::not_supported_error);
    EXPECT_EQ(g.version(), 0u);  // rejected before the batch was applied
  }
  // Nothing was changed: the result is still usable once the capability is back.
  EXPECT_EQ(host_copy(r.distances()), (std::vector<std::int64_t>{0, 5, 1, 2, 4}));
  dyng::detail::resources_access::force_cooperative_launch(res, true);
  batch_t b;
  b.delete_edge(0, 2);
  const auto st = dyng::sssp::update(res, g, b.view(), r);
  EXPECT_EQ(st.engine_used, dyng::engine::fused);
  EXPECT_EQ(host_copy(r.distances()), (std::vector<std::int64_t>{0, 5, inf, inf, 7}));
}

TEST_F(SsspCuda, OperatorsEngineArrivesIn02) {
  const auto res = dyng::resources::cuda();
  auto g = ties_graph(res);
  dyng::sssp::options opt;
  opt.cuda_engine = dyng::engine::operators;
  try {
    (void)dyng::sssp::compute(res, g, 0, opt);
    FAIL() << "engine::operators ran";
  } catch (const dyng::not_supported_error& error) {
    EXPECT_TRUE(contains(error.what(), "engine::operators")) << error.what();
  }
  // The host backends ignore the CUDA engine option.
  const auto host = dyng::resources::sequential();
  auto hg = ties_graph(host);
  EXPECT_NO_THROW((void)dyng::sssp::compute(host, hg, 0, opt));
}

TEST_F(SsspCuda, GraphsAndResultsStayWithTheirBackendUntilCloned) {
  const auto cuda = dyng::resources::cuda();
  const auto host = dyng::resources::sequential();
  auto hg = ties_graph(host);
  auto dg = ties_graph(cuda);
  // An algorithm never copies a whole graph silently (PLAN Section 4.6 rule 5).
  try {
    (void)dyng::sssp::compute(cuda, hg, 0);
    FAIL() << "a host graph was used with cuda resources";
  } catch (const dyng::invalid_argument_error& error) {
    EXPECT_TRUE(contains(error.what(), "g.clone(res)")) << error.what();
  }
  EXPECT_THROW((void)dyng::sssp::compute(host, dg, 0), dyng::invalid_argument_error);
  // clone(res) moves a graph to the backend of res; the content and the state are kept.
  auto moved = hg.clone(cuda);
  EXPECT_EQ(moved.space(), dyng::memory_space::device);
  result_t hr = dyng::sssp::compute(host, hg, 0);
  result_t dr = dyng::sssp::compute(cuda, moved, 0);
  EXPECT_EQ(host_copy(dr.distances()), host_copy(hr.distances()));
  // A host result cannot be updated with cuda resources (and the reverse) until cloned.
  batch_t b;
  b.insert_edge(2, 1, {1});
  EXPECT_THROW((void)dyng::sssp::update(cuda, moved, b.view(), hr), dyng::invalid_argument_error);
  EXPECT_THROW((void)dyng::sssp::update(host, hg, b.view(), dr), dyng::invalid_argument_error);
  EXPECT_EQ(moved.version(), 0u);
  result_t to_device = hr.clone(cuda);
  result_t to_host = dr.clone(host);
  EXPECT_EQ(to_device.space(), dyng::memory_space::device);
  EXPECT_EQ(to_host.space(), dyng::memory_space::host);
  (void)dyng::sssp::update(cuda, moved, b.view(), to_device);
  (void)dyng::sssp::update(host, hg, b.view(), to_host);
  EXPECT_EQ(host_copy(to_device.distances()), host_copy(to_host.distances()));
  EXPECT_EQ(host_copy(to_device.parents()), host_copy(to_host.parents()));
}

// A result computed on one stream and updated with vertex growth through a handle on another
// stream (the caller orders the streams, PLAN 4.7.4): the old arrays are released on the updating
// stream, after the copies into the grown arrays read them (M1b review).
TEST_F(SsspCuda, VertexGrowthThroughAHandleOnAnotherStream) {
  cudaStream_t a = nullptr;
  cudaStream_t b = nullptr;
  ASSERT_EQ(cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking), cudaSuccess);
  ASSERT_EQ(cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking), cudaSuccess);
  {
    const auto on_a = dyng::resources::cuda(0, dyng::stream_ref(a));
    const auto on_b = dyng::resources::cuda(0, dyng::stream_ref(b));
    const auto host = dyng::resources::sequential();
    dyng::graph_properties props = dyng::graph_properties::mosp_compatible();
    props.semantics.allow_vertex_growth = true;
    const auto build = [&](const dyng::resources& res) {
      dyng::edge_list<std::int32_t, std::int32_t> list;
      list.num_vertices = 5;
      list.num_weights = 1;
      for (const edge& e : std::vector<edge>{{0, 1, 5}, {0, 2, 1}, {2, 3, 1}, {1, 4, 2}}) {
        list.add_edge(e.u, e.v, {e.w});
      }
      return graph_t::from_edges(res, list.view(), props);
    };
    auto g = build(on_a);
    auto hg = build(host);
    result_t r = dyng::sssp::compute(on_a, g, 0);
    result_t hr = dyng::sssp::compute(host, hg, 0);
    batch_t batch;
    batch.insert_edge(4, 6, {3});  // grows the graph to 7 vertices
    batch.insert_edge(3, 4, {1});
    batch.delete_edge(0, 1);
    (void)dyng::sssp::update(on_b, g, batch.view(), r);
    (void)dyng::sssp::update(host, hg, batch.view(), hr);
    EXPECT_EQ(g.num_vertices(), 7);
    EXPECT_EQ(dyng::to_vector(on_b, r.distances()), host_copy(hr.distances()));
    EXPECT_EQ(dyng::to_vector(on_b, r.parents()), host_copy(hr.parents()));
    // The grown arrays, and the release of the old ones, are ordered on the updating stream.
    const auto& state = dyng::detail::sssp_access::state(r);
    EXPECT_EQ(state.device_distances.stream(), on_b.stream());
    EXPECT_EQ(state.device_parents.stream(), on_b.stream());
    on_b.synchronize();
  }
  ASSERT_EQ(cudaStreamSynchronize(a), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(a), cudaSuccess);
  ASSERT_EQ(cudaStreamDestroy(b), cudaSuccess);
}

TEST_F(SsspCuda, TheDeviceCopyIsUploadedOncePerGraphState) {
  const auto res = dyng::resources::cuda();
  auto g = ties_graph(res);
  const auto& impl = dyng::detail::graph_access::impl(g);
  EXPECT_FALSE(impl.has_device_edges());  // built on first use
  result_t r = dyng::sssp::compute(res, g, 0);
  EXPECT_TRUE(impl.has_device_edges());
  EXPECT_FALSE(impl.has_in_edges());  // the host in-edges of a CUDA graph are never needed
  batch_t b;
  b.insert_edge(4, 0, {3});
  (void)g.apply(res, b.view());
  EXPECT_FALSE(impl.has_device_edges());  // dropped with the state
  r = dyng::sssp::compute(res, g, 0);
  EXPECT_TRUE(impl.has_device_edges());
  (void)dyng::sssp::update(res, g, b.view(), r);  // the commit uploads the new state
  EXPECT_TRUE(impl.has_device_edges());
  EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
}

TEST_F(SsspCuda, FromArraysTakesHostOrDeviceArrays) {
  const auto res = dyng::resources::cuda();
  auto g = ties_graph(res);
  const result_t computed = dyng::sssp::compute(res, g, 0);
  result_t from_device =
      result_t::from_arrays(res, g, 0, computed.distances(), computed.parents(), false);
  const auto d = host_copy(computed.distances());
  const auto p = host_copy(computed.parents());
  result_t from_host = result_t::from_arrays(res, g, 0, dyng::host_view(d), dyng::host_view(p));
  EXPECT_EQ(host_copy(from_device.distances()), d);
  EXPECT_EQ(host_copy(from_host.parents()), p);
  // The host backends take host arrays only.
  const auto host = dyng::resources::sequential();
  auto hg = ties_graph(host);
  EXPECT_THROW(
      (void)result_t::from_arrays(host, hg, 0, computed.distances(), computed.parents(), false),
      dyng::invalid_argument_error);
}

TEST_F(SsspCuda, AnOverflowingImportedTreePoisonsTheResult) {
  // A tree imported without validation whose distance does not fit (n - 1) * max weight: the
  // kernel reports it (MOSP-CUDA's "the initial tree does not belong to this graph").
  const auto res = dyng::resources::cuda();
  auto g = make_graph(res, 3, {{0, 1, 1}, {1, 2, 1}});
  const std::vector<std::int64_t> d{0, 1, 1000};
  const std::vector<std::int32_t> p{-1, 0, 1};
  dyng::sssp::options opt;
  opt.validate_inputs = false;
  result_t r = result_t::from_arrays(res, g, 0, dyng::host_view(d), dyng::host_view(p), false, opt);
  batch_t b;
  b.insert_edge(0, 2, {5});
  try {
    (void)dyng::sssp::update(res, g, b.view(), r);
    FAIL() << "the overflowing distance was not reported";
  } catch (const dyng::stale_result_error&) {
    FAIL() << "expected invalid_argument_error";
  } catch (const dyng::invalid_argument_error& error) {
    EXPECT_TRUE(contains(error.what(), "does not belong to this graph")) << error.what();
  }
  EXPECT_EQ(g.version(), 1u);  // the batch was applied before the engine ran
  EXPECT_THROW((void)r.distances(), dyng::stale_result_error);
  EXPECT_THROW((void)r.clone(res), dyng::stale_result_error);
  const batch_t empty;
  EXPECT_THROW((void)dyng::sssp::update(res, g, empty.view(), r), dyng::stale_result_error);
  r = dyng::sssp::compute(res, g, 0);
  EXPECT_EQ(host_copy(r.distances()), (std::vector<std::int64_t>{0, 1, 2}));
  // The handle's workspace of the failed run was discarded; the next runs are correct.
  (void)dyng::sssp::update(res, g, empty.view(), r);
  EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
}

/// mospTest runPackingBoundary: n = 2^17 - 1 needs 17 parent bits and leaves 47 for distances.
/// On the path 0 -> 1 -> ... -> n-1 with every weight W = 2^30 + 2^14, (n - 1) * W = 2^47 - 2^15
/// still fits (packed words), but an edge n-1 -> 1 forms the candidate n * W > 2^47 - 1, which
/// must not be packed. Three entry points: the pull pass (the edge is inserted), the push loop
/// (the edge exists and a cheaper last path edge makes n-1 push it) and compute().
TEST_F(SsspCuda, PackingBoundaryOf47DistanceBits) {
  constexpr std::int32_t n = (1 << 17) - 1;
  constexpr std::int32_t w = (1 << 30) + (1 << 14);
  const auto res = dyng::resources::cuda();
  const auto host = dyng::resources::sequential();
  auto path = [&](const dyng::resources& r, bool back_edge) {
    dyng::edge_list<std::int32_t, std::int32_t> list;
    list.num_vertices = n;
    list.num_weights = 1;
    for (std::int32_t u = 0; u + 1 < n; ++u) {
      list.add_edge(u, u + 1, {w});
    }
    if (back_edge) {
      list.add_edge(n - 1, 1, {w});
    }
    return graph_t::from_edges(r, list.view(), dyng::graph_properties::mosp_compatible());
  };
  struct scenario {
    const char* name;
    bool back_edge;
    std::int32_t from;
    std::int32_t to;
    std::int32_t weight;
  };
  for (const scenario& c :
       {scenario{"pull", false, n - 1, 1, w}, scenario{"push", true, n - 2, n - 1, w - 1},
        scenario{"from-scratch", true, -1, -1, 0}}) {
    SCOPED_TRACE(c.name);
    auto g = path(res, c.back_edge);
    auto hg = path(host, c.back_edge);
    result_t r = dyng::sssp::compute(res, g, 0);
    result_t hr = dyng::sssp::compute(host, hg, 0);
    if (c.from >= 0) {
      batch_t b;
      b.insert_edge(c.from, c.to, {c.weight});
      const auto st = dyng::sssp::update(res, g, b.view(), r);
      (void)dyng::sssp::update(host, hg, b.view(), hr);
      // MOSP-CUDA's rule: the bound (n - 1) * W fits in 47 bits, so the words are packed. (The
      // OpenMP engine also counts one more edge, so it keeps distances only here; the trees are
      // equal either way.)
      EXPECT_TRUE(st.packed_parents);
    }
    const auto check = dyng::testing::check_sssp_tree(g, r);
    EXPECT_TRUE(check.ok()) << check.summary();
    EXPECT_EQ(host_copy(r.distances()), host_copy(hr.distances()));
    EXPECT_EQ(host_copy(r.parents()), host_copy(hr.parents()));
  }
}

/// mospTest runLargeWeights / runLargeWeightTies: on a 320 x 320 grid with weights 2 * 10^9 (every
/// vertex has two tight in-neighbours; the recovery must pick the lower id) and with random
/// weights up to 2^31 - 1, (n - 1) * max weight does not fit next to 17 parent bits, so the kernel
/// keeps distances only and recovers the parents after the search.
TEST_F(SsspCuda, LargeWeightsUseTheDistanceOnlyFallback) {
  constexpr std::int32_t side = 320;
  constexpr std::int32_t n = side * side;
  const auto res = dyng::resources::cuda();
  const auto omp = dyng::test::make_resources(dyng::test::host_backends().back(), 4);
  std::mt19937 rng(20260927);
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 2;
  auto random_weight = [&] {
    return static_cast<std::int32_t>(
        std::uniform_int_distribution<std::int64_t>(1, 2147483647)(rng));
  };
  for (std::int32_t y = 0; y < side; ++y) {
    for (std::int32_t x = 0; x < side; ++x) {
      const std::int32_t v = y * side + x;
      if (x + 1 < side) {
        list.add_edge(v, v + 1, {2000000000, random_weight()});
        list.add_edge(v + 1, v, {2000000000, random_weight()});
      }
      if (y + 1 < side) {
        list.add_edge(v, v + side, {2000000000, random_weight()});
        list.add_edge(v + side, v, {2000000000, random_weight()});
      }
    }
  }
  const auto props = dyng::graph_properties::mosp_compatible();
  auto g = graph_t::from_edges(res, list.view(), props);
  auto hg = graph_t::from_edges(omp, list.view(), props);
  for (int k = 0; k < 2; ++k) {
    SCOPED_TRACE("objective " + std::to_string(k));
    dyng::sssp::options opt;
    opt.objective = k;
    result_t r = dyng::sssp::compute(res, g, 0, opt);
    result_t hr = dyng::sssp::compute(omp, hg, 0, opt);
    EXPECT_EQ(host_copy(r.parents()), host_copy(hr.parents()));
    for (int round = 0; round < 3; ++round) {
      // 1000 deletions of tree edges and existing edges, 1000 insertions (weights 2 * 10^9 or
      // random), like the 2000-change batches of mospTest.
      batch_t b(2);
      const auto parents = host_copy(r.parents());
      for (int i = 0; i < 1000; ++i) {
        const auto v =
            static_cast<std::int32_t>(std::uniform_int_distribution<std::int32_t>(1, n - 1)(rng));
        if (parents[static_cast<std::size_t>(v)] >= 0) {
          b.delete_edge(parents[static_cast<std::size_t>(v)], v);
        }
        const auto a = std::uniform_int_distribution<std::int32_t>(0, n - 1)(rng);
        const auto c = std::uniform_int_distribution<std::int32_t>(0, n - 1)(rng);
        if (a != c) {
          b.insert_edge(a, c, {2000000000, random_weight()});
        }
      }
      const auto st = dyng::sssp::update(res, g, b.view(), r);
      const auto hst = dyng::sssp::update(omp, hg, b.view(), hr);
      EXPECT_FALSE(st.packed_parents);
      EXPECT_FALSE(hst.packed_parents);
      EXPECT_EQ(st.invalidated, hst.invalidated);
      EXPECT_EQ(st.affected, hst.affected);
      EXPECT_EQ(host_copy(r.distances()), host_copy(hr.distances()));
      EXPECT_EQ(host_copy(r.parents()), host_copy(hr.parents()));
      const auto check = dyng::testing::check_sssp_tree(g, r);
      EXPECT_TRUE(check.ok()) << check.summary();
    }
  }
}

/// Counts the calls into an upstream device resource.
class counting_resource {
 public:
  explicit counting_resource(dyng::memory_resource_ref upstream) : upstream_(upstream) {}
  void* allocate(dyng::stream_ref s, std::size_t b, std::size_t a) {
    ++allocations;
    return upstream_.allocate(s, b, a);
  }
  void deallocate(dyng::stream_ref s, void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate(s, p, b, a);
  }
  void* allocate_sync(std::size_t b, std::size_t a) {
    ++allocations;
    return upstream_.allocate_sync(b, a);
  }
  void deallocate_sync(void* p, std::size_t b, std::size_t a) noexcept {
    upstream_.deallocate_sync(p, b, a);
  }
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return upstream_.space();
  }
  std::atomic<int> allocations{0};

 private:
  dyng::memory_resource_ref upstream_;
};

/// Invariant I9 on the CUDA backend: after the first batch, an update of a stable workload
/// allocates exactly what uploading the new graph state allocates (the straight port of MOSP's host
/// apply re-uploads the graph per batch; that is the graph's budget, not the algorithm's), and the
/// algorithm phase (workspace, change lists, the fused engine) allocates nothing.
TEST_F(SsspCuda, SteadyStateUpdatesAllocateOnlyTheGraphUpload) {
  auto res = dyng::resources::cuda();
  counting_resource counter(res.memory());
  res.set_memory_resource(counter);
  std::mt19937 rng(7);
  const std::int32_t n = 2000;
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 3;
  for (std::int32_t i = 0; i < 8 * n; ++i) {
    const auto u = std::uniform_int_distribution<std::int32_t>(0, n - 1)(rng);
    const auto v = std::uniform_int_distribution<std::int32_t>(0, n - 1)(rng);
    list.add_edge(u, v,
                  {std::uniform_int_distribution<std::int32_t>(1, 100)(rng),
                   std::uniform_int_distribution<std::int32_t>(1, 100)(rng),
                   std::uniform_int_distribution<std::int32_t>(1, 100)(rng)});
  }
  auto g = graph_t::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
  std::vector<result_t> results;
  for (int k = 0; k < 3; ++k) {
    dyng::sssp::options opt;
    opt.objective = k;
    results.push_back(dyng::sssp::compute(res, g, 0, opt));
  }
  std::vector<result_t*> pointers{&results[0], &results[1], &results[2]};
  auto& pool = dyng::detail::resources_access::workspaces(res);
  for (int round = 0; round < 5; ++round) {
    SCOPED_TRACE("batch " + std::to_string(round));
    // Deletions of existing edges and insertions of new (u, v) pairs only: the change lists have
    // the same length in every batch.
    const auto csr = g.to_csr(res);
    batch_t b(3);
    for (int i = 0; i < 100; ++i) {
      const auto u = std::uniform_int_distribution<std::int32_t>(0, n - 1)(rng);
      const auto e = csr.row_ptr[static_cast<std::size_t>(u)];
      if (e < csr.row_ptr[static_cast<std::size_t>(u) + 1]) {
        b.delete_edge(u, csr.col_ind[static_cast<std::size_t>(e)]);
      } else {
        b.delete_edge(u, (u + 1) % n);  // a missing edge still counts as a deletion
      }
      // Weight 1 never raises an existing edge's weight, so no insertion joins the change list.
      b.insert_edge(static_cast<std::int32_t>(n - 1 - (round * 100 + i) % n),
                    static_cast<std::int32_t>((round * 100 + i) % n), {1, 1, 1});
    }
    const std::uint64_t created = pool.statistics().created;
    const int before = counter.allocations.load();
    (void)dyng::update_each(res, g, b.view(),
                            dyng::array_view<result_t* const>(pointers.data(), pointers.size()));
    const int during = counter.allocations.load() - before;
    // What uploading this graph state alone allocates.
    const auto copy = g.clone(res);
    const int before_upload = counter.allocations.load();
    (void)dyng::detail::graph_access::device(res, copy);
    const int upload = counter.allocations.load() - before_upload;
    if (round >= 1) {
      EXPECT_EQ(during, upload);
      EXPECT_EQ(pool.statistics().created, created);
    }
    for (const result_t& r : results) {
      EXPECT_TRUE(dyng::testing::check_sssp_tree(g, r).ok());
    }
  }
  res.synchronize();
  results.clear();
  g = graph_t();
  res.release_workspaces();
}

TEST_F(SsspCuda, ProfilerRecordsTheDeviceTimeOfTheFusedEngine) {
  auto res = dyng::resources::cuda();
  dyng::profiler_options options;
  options.cuda_events = true;
  dyng::profiler prof(options);
  auto g = ties_graph(res);
  result_t r = dyng::sssp::compute(res, g, 0);
  res.attach_profiler(&prof);
  batch_t b;
  b.delete_edge(0, 2);
  (void)dyng::sssp::update(res, g, b.view(), r);
  res.attach_profiler(nullptr);
  bool seen = false;
  for (const auto& s : prof.samples()) {
    if (s.name == "sssp.enact_fused") {
      seen = true;
      EXPECT_GT(s.device_ms, 0.0);
      EXPECT_GE(s.host_ms, 0.0);
    }
  }
  EXPECT_TRUE(seen);
  std::vector<std::string> names;
  for (const auto& s : prof.stages()) {
    names.push_back(s.name);
  }
  for (const char* expected : {"sssp.update", "sssp.prepare", "sssp.commit", "graph.apply",
                               "graph.upload", "sssp.workspace", "sssp.changes"}) {
    EXPECT_NE(std::find(names.begin(), names.end(), expected), names.end()) << expected;
  }
}

TEST_F(SsspCuda, WarmUpLoadsTheSsspKernels) {
  const auto res = dyng::resources::cuda();
  int persistent = 0;
  int transpose = 0;
  for (const auto& kernel : dyng::detail::registered_kernels()) {
    const std::string name = kernel.name;
    persistent += contains(name, "sssp_persistent_kernel") ? 1 : 0;
    transpose += contains(name, "fill_reverse_kernel") ? 1 : 0;
  }
  EXPECT_EQ(persistent, 3);  // one per instantiated graph type
  EXPECT_EQ(transpose, 3);
  EXPECT_NO_THROW(res.warm_up());
}

}  // namespace
