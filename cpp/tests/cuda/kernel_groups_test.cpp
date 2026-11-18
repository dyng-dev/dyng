// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file kernel_groups_test.cpp
 * @brief Kernel groups (core/kernel_groups.hpp, ADR 0109 part B): each algorithm's CUDA kernels
 *        are registered with the runtime when the process first uses them, not when libdyng is
 *        loaded.
 *
 * CTest runs every test of this executable in a process of its own (gtest_discover_tests), so a
 * test that needs a fresh process checks that no group is loaded yet and skips otherwise (as when
 * the executable runs all of them at once).
 */
#include "core/kernel_groups.hpp"

#include "core/cuda_runtime.hpp"
#include "support/gtest_helpers.hpp"
#include "util/kernel_registry.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/generators/vertex_arrival.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/sssp.hpp>

#include <cuda.h>  // CUfunctionLoadingState (the driver function is fetched through the runtime)
#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <vector>

namespace {

using dyng::detail::kernel_group_info;
namespace kernel_groups = dyng::detail::kernel_groups;

const kernel_group_info* info(const std::vector<kernel_group_info>& groups, const char* name) {
  for (const kernel_group_info& g : groups) {
    if (g.name == name) {
      return &g;
    }
  }
  return nullptr;
}

bool loaded(const char* name) {
  const auto groups = kernel_groups::list();
  const kernel_group_info* g = info(groups, name);
  return g != nullptr && g->loaded;
}

/// Skips unless no group is loaded yet (this test runs in a process of its own).
#define DYNG_SKIP_UNLESS_FRESH_PROCESS()                                   \
  do {                                                                     \
    for (const kernel_group_info& g : kernel_groups::list()) {             \
      if (g.loaded) {                                                      \
        GTEST_SKIP() << "needs a fresh process (group " << g.name          \
                     << " is loaded already); CTest runs each test alone"; \
      }                                                                    \
    }                                                                      \
  } while (false)

#if DYNG_TEST_KERNEL_GROUPS
#define DYNG_SKIP_WITHOUT_KERNEL_GROUPS() \
  do {                                    \
  } while (false)
#else
#define DYNG_SKIP_WITHOUT_KERNEL_GROUPS() GTEST_SKIP() << "built with DYNG_DEFER_KERNEL_GROUPS=OFF"
#endif

}  // namespace

TEST(KernelGroups, EveryAlgorithmWithCudaKernelsAndTheHypergraphsHaveAGroup) {
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  const auto groups = kernel_groups::list();
  std::set<std::string> names;
  for (const kernel_group_info& g : groups) {
    EXPECT_TRUE(names.insert(g.name).second) << "two groups named " << g.name;
    EXPECT_GE(g.fatbins, 1u) << g.name;
    EXPECT_GE(g.kernels, 1u) << g.name;
  }
  // The algorithms built in every CUDA preset (DYNG_ALGORITHMS=all) and the hypergraph containers.
  for (const char* name : {"sssp", "cycle_count", "mosp", "dynamic_bfs", "triangle_delta",
                           "triad_count", "hyper_sssp", "label_propagation", "hypergraph"}) {
    EXPECT_EQ(names.count(name), 1u) << "no kernel group " << name;
  }
}

TEST(KernelGroups, TheContextLoadsNoGroup) {
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
  const auto res = dyng::resources::cuda();
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  for (const kernel_group_info& g : kernel_groups::list()) {
    EXPECT_FALSE(g.loaded) << g.name;
  }
}

TEST(KernelGroups, LoadLoadsTheNamedGroupsOnly) {
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
  const auto res = dyng::resources::cuda();
  kernel_groups::load({"cycle_count", "no_such_group"});
  EXPECT_TRUE(loaded("cycle_count"));
  EXPECT_FALSE(loaded("label_propagation"));
  EXPECT_FALSE(loaded("hypergraph"));
  kernel_groups::load({"cycle_count"});  // idempotent
  EXPECT_TRUE(loaded("cycle_count"));
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(KernelGroups, LoadingAGroupLoadsEveryKernelOfIt) {
  // A group's load is the eager loading of its modules (M10b step 4): the runtime does not load
  // the kernels of a fat binary registered after the context exists until their first use, which
  // put label_propagation's module loading into the first timed update. Every kernel of a loaded
  // group must be loaded (cuFuncIsLoaded, CUDA 12.4 and later) before any of them runs.
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
#if CUDART_VERSION >= 12050
  using is_loaded_fn = CUresult (*)(CUfunctionLoadingState*, CUfunction);
  void* entry = nullptr;
  cudaDriverEntryPointQueryResult found{};
  if (cudaGetDriverEntryPointByVersion("cuFuncIsLoaded", &entry, 12040, cudaEnableDefault,
                                       &found) != cudaSuccess ||
      found != cudaDriverEntryPointSuccess || entry == nullptr) {
    (void)cudaGetLastError();
    GTEST_SKIP() << "the driver has no cuFuncIsLoaded";
  }
  const auto is_loaded = reinterpret_cast<is_loaded_fn>(entry);
  const auto state_of = [&](const void* host_function) {
    cudaFunction_t f = nullptr;
    EXPECT_EQ(cudaGetFuncBySymbol(&f, host_function), cudaSuccess);
    CUfunctionLoadingState state = CU_FUNCTION_LOADING_STATE_MAX;
    EXPECT_EQ(is_loaded(&state, reinterpret_cast<CUfunction>(f)), CUDA_SUCCESS);
    return state;
  };
  const auto res = dyng::resources::cuda();
  ASSERT_EQ(cudaFree(nullptr), cudaSuccess);
  kernel_groups::load({"cycle_count"});
  ASSERT_TRUE(loaded("cycle_count"));
  std::size_t checked = 0;
  for (const auto& kernel : dyng::detail::registered_kernels()) {
    if (kernel.group != nullptr && std::strcmp(kernel.group, "cycle_count") == 0) {
      EXPECT_EQ(state_of(kernel.function), CU_FUNCTION_LOADING_STATE_LOADED) << kernel.name;
      ++checked;
    }
  }
  EXPECT_GT(checked, 0u);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
#else
  GTEST_SKIP() << "cuFuncIsLoaded needs CUDA 12.5's cudaGetDriverEntryPointByVersion";
#endif
}

TEST(KernelGroups, TheFirstUseOfAnAlgorithmLoadsItsGroupOnly) {
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
  const auto res = dyng::resources::cuda();
  using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = 5;
  list.num_weights = 1;
  for (const auto& [u, v, w] : std::vector<std::array<std::int32_t, 3>>{
           {0, 1, 5}, {0, 2, 1}, {2, 3, 1}, {1, 4, 2}, {3, 4, 2}}) {
    list.add_edge(u, v, {w});
  }
  const graph_t g =
      graph_t::from_edges(res, list.view(), dyng::graph_properties::mosp_compatible());
  EXPECT_FALSE(loaded("sssp"));
  // No explicit load: the first launch (or occupancy query) of sssp's code loads its group.
  const auto r = dyng::sssp::compute(res, g, std::int32_t{0});
  const std::vector<std::int64_t> d = dyng::to_vector(res, r.distances());
  EXPECT_EQ(d, (std::vector<std::int64_t>{0, 5, 1, 2, 4}));
  EXPECT_TRUE(loaded("sssp"));
  EXPECT_FALSE(loaded("label_propagation"));
  EXPECT_FALSE(loaded("cycle_count"));
  EXPECT_FALSE(loaded("hypergraph"));
}

// The published flow of label_propagation (compute() on the capacity graph, then updates) loads
// its own group and no other: its CUB calls (cub::DeviceScan::ExclusiveSum, ...) may run another
// group's copy of the same template, which the linker kept (M10b step 4's final re-read found the
// first update loading the hypergraph group through it, about 8 ms inside the timed update at
// 50k); such a call goes to the runtime without loading that group (kernel_group::ensure_for()).
#if DYNG_TEST_KG_LABEL_PROPAGATION
TEST(KernelGroups, LabelPropagationLoadsItsGroupOnly) {
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
  namespace lp = dyng::label_propagation;
  using graph_t = dyng::graph<std::int32_t, std::int32_t, float>;
  const auto res = dyng::resources::cuda();
  dyng::generators::vertex_arrival_options o;
  o.total_vertices = 2000;
  o.num_batches = 3;
  const auto stream =
      dyng::generators::legacy::dynlp_vertex_arrival_stream(dyng::resources::sequential(), o);
  dyng::graph_properties p;
  p.directed = false;
  p.layout = dyng::row_layout::slotted;
  p.order = dyng::row_order::append;
  p.parallel_edges = dyng::multi_edges::allow;
  p.semantics = dyng::batch_semantics::net_effect();
  graph_t g = graph_t::with_capacity(res, static_cast<std::int32_t>(o.total_vertices),
                                     dyng::host_view(stream.row_capacity), p);
  lp::options opt;
  opt.update_schedule = lp::schedule::damped;
  opt.similarity_threshold = stream.similarity_threshold;
  lp::result<std::int32_t> r = lp::compute(res, g, {}, {}, opt);
  EXPECT_TRUE(loaded("label_propagation"));
  for (const auto& batch : stream.batches) {
    (void)lp::update(res, g, batch.view(), r);
  }
  EXPECT_GT(g.num_active_vertices(), 0);
  for (const kernel_group_info& group : kernel_groups::list()) {
    EXPECT_EQ(group.loaded, group.name == "label_propagation") << group.name;
  }
}
#endif

TEST(KernelGroups, WarmUpLoadsEveryGroupAndKernel) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  res.warm_up();
  for (const kernel_group_info& g : kernel_groups::list()) {
    EXPECT_TRUE(g.loaded) << g.name;
  }
  EXPECT_EQ(dyng::detail::cuda_warm_up(res.device(), res.stream(), res.memory()),
            dyng::detail::registered_kernels().size());
}

TEST(KernelGroups, TheNarrowWarmUpLeavesTheOtherGroupsUnloaded) {
  DYNG_SKIP_IF_NO_CUDA();
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  DYNG_SKIP_UNLESS_FRESH_PROCESS();
  const auto res = dyng::resources::cuda();
  const std::size_t warmed = dyng::detail::warm_up_kernel_groups(res, {"mosp"});
  EXPECT_TRUE(loaded("mosp"));
  EXPECT_FALSE(loaded("label_propagation"));
  EXPECT_FALSE(loaded("sssp"));
  // The kernels outside every group, and mosp's.
  std::size_t expected = 0;
  for (const auto& kernel : dyng::detail::registered_kernels()) {
    expected += kernel.group == nullptr || std::strcmp(kernel.group, "mosp") == 0 ? 1 : 0;
  }
  EXPECT_EQ(warmed, expected);
  EXPECT_LT(warmed, dyng::detail::registered_kernels().size());
  EXPECT_EQ(dyng::detail::warm_up_kernel_groups(dyng::resources::sequential(), {"mosp"}), 0u);
}

TEST(KernelGroups, RegisteredKernelsCarryTheirGroup) {
  DYNG_SKIP_WITHOUT_KERNEL_GROUPS();
  std::size_t outside_fill = 0;
  std::size_t sssp = 0;
  for (const auto& kernel : dyng::detail::registered_kernels()) {
    if (kernel.group == nullptr) {
      outside_fill += std::strstr(kernel.name, "fill_kernel") != nullptr ? 1 : 0;
      continue;
    }
    EXPECT_NE(kernel_groups::find(kernel.group), nullptr) << kernel.name << " in " << kernel.group;
    const std::string name = kernel.name;
    if (name.find("sssp_persistent_kernel") != std::string::npos) {
      EXPECT_STREQ(kernel.group, "sssp");
      ++sssp;
    }
  }
  EXPECT_GE(outside_fill, 7u);  // the core's fill kernels (util/device_fill.cu) are in no group
  EXPECT_EQ(sssp, 3u);          // one per graph type
}
