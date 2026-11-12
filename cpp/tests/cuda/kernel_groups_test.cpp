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
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>

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
