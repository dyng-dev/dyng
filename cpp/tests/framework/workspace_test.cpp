// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file workspace_test.cpp
 * @brief The workspace pool of a resources handle (ADR 0015): reuse, distinct leases, discard on
 *        failure, release, statistics, sharing between copies of a handle, thread safety.
 */
#include "framework/workspace.hpp"

#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace {

using dyng::detail::pooled_workspace;
using dyng::detail::resources_access;
using dyng::detail::workspace_pool;

struct scratch_a final : pooled_workspace {
  std::vector<int> data;
  int runs = 0;
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return data.capacity() * sizeof(int);
  }
};

struct scratch_b final : pooled_workspace {
  [[nodiscard]] std::size_t bytes() const noexcept override {
    return 1;
  }
};

TEST(WorkspacePool, ReusesTheReturnedWorkspace) {
  workspace_pool pool;
  const scratch_a* first = nullptr;
  {
    auto ws = pool.acquire<scratch_a>();
    ws->data.resize(100);
    ws->runs = 1;
    first = &ws.get();
    EXPECT_EQ(pool.statistics().leased, 1u);
    EXPECT_EQ(pool.statistics().idle, 0u);
  }
  EXPECT_EQ(pool.statistics().idle, 1u);
  EXPECT_EQ(pool.statistics().idle_bytes, 100 * sizeof(int));
  {
    auto ws = pool.acquire<scratch_a>();
    EXPECT_EQ(&ws.get(), first);  // the same object, sized and with its state
    EXPECT_EQ(ws->runs, 1);
    EXPECT_EQ(ws->data.size(), 100u);
  }
  const auto s = pool.statistics();
  EXPECT_EQ(s.created, 1u);
  EXPECT_EQ(s.leases, 2u);
  EXPECT_EQ(s.leased, 0u);
  EXPECT_EQ(s.discarded, 0u);
}

TEST(WorkspacePool, TypesAreKeptApart) {
  workspace_pool pool;
  {
    auto a = pool.acquire<scratch_a>();
  }
  {
    auto b = pool.acquire<scratch_b>();
  }
  {
    auto a = pool.acquire<scratch_a>();
  }
  EXPECT_EQ(pool.statistics().created, 2u);
  EXPECT_EQ(pool.statistics().idle, 2u);
}

TEST(WorkspacePool, OverlappingLeasesGetDistinctWorkspaces) {
  workspace_pool pool;
  {
    auto a = pool.acquire<scratch_a>();
    auto b = pool.acquire<scratch_a>();
    EXPECT_NE(&a.get(), &b.get());
    auto moved = std::move(b);  // a moved lease returns its workspace once
    EXPECT_EQ(pool.statistics().leased, 2u);
  }
  const auto s = pool.statistics();
  EXPECT_EQ(s.created, 2u);
  EXPECT_EQ(s.idle, 2u);
  EXPECT_EQ(s.leased, 0u);
}

TEST(WorkspacePool, AFailedRunDiscardsItsWorkspace) {
  workspace_pool pool;
  {
    auto ws = pool.acquire<scratch_a>();
  }
  try {
    auto ws = pool.acquire<scratch_a>();
    ws->runs = 99;  // a flag the next run must not inherit
    throw std::runtime_error("engine failed");
  } catch (const std::runtime_error&) {
  }
  auto s = pool.statistics();
  EXPECT_EQ(s.discarded, 1u);
  EXPECT_EQ(s.idle, 0u);
  EXPECT_EQ(s.leased, 0u);
  auto ws = pool.acquire<scratch_a>();
  EXPECT_EQ(ws->runs, 0);  // a fresh workspace
  EXPECT_EQ(pool.statistics().created, 2u);
}

TEST(WorkspacePool, LeasesThatEndDuringUnwindingOfAnOuterExceptionAreKept) {
  // A lease taken and returned inside a destructor that runs during unwinding saw the same
  // number of uncaught exceptions at both ends: its run completed, so the workspace is kept.
  workspace_pool pool;
  struct returns_on_unwind {
    workspace_pool& pool;
    ~returns_on_unwind() {
      auto ws = pool.acquire<scratch_a>();
      ws->runs = 7;
    }
  };
  try {
    returns_on_unwind guard{pool};
    throw std::runtime_error("outer");
  } catch (const std::runtime_error&) {
  }
  EXPECT_EQ(pool.statistics().discarded, 0u);
  EXPECT_EQ(pool.statistics().idle, 1u);
}

TEST(WorkspacePool, ReleaseIdleFreesOnlyIdleWorkspaces) {
  workspace_pool pool;
  {
    auto ws = pool.acquire<scratch_a>();
  }
  auto held = pool.acquire<scratch_b>();
  pool.release_idle();
  EXPECT_EQ(pool.statistics().idle, 0u);
  EXPECT_EQ(pool.statistics().leased, 1u);
}

TEST(WorkspacePool, ThreadsNeverShareALease) {
  workspace_pool pool;
  constexpr int threads = 8;
  constexpr int rounds = 2000;
  std::vector<std::thread> workers;
  workers.reserve(threads);
  for (int t = 0; t < threads; ++t) {
    workers.emplace_back([&pool] {
      for (int i = 0; i < rounds; ++i) {
        auto ws = pool.acquire<scratch_a>();
        // Exclusive use: a data race here would be reported by the tsan preset.
        ws->runs += 1;
        ws->data.assign(8, ws->runs);
      }
    });
  }
  for (std::thread& w : workers) {
    w.join();
  }
  const auto s = pool.statistics();
  EXPECT_LE(s.created, static_cast<std::uint64_t>(threads));
  EXPECT_EQ(s.leases, static_cast<std::uint64_t>(threads) * rounds);
  EXPECT_EQ(s.leased, 0u);
  EXPECT_EQ(s.idle, s.created);
}

TEST(ResourcesWorkspaces, CopiesShareOnePoolAndReleaseFreesIt) {
  const auto res = dyng::resources::sequential();
  const dyng::resources copy = res;  // NOLINT(performance-unnecessary-copy-initialization)
  {
    auto ws = resources_access::workspaces(res).acquire<scratch_a>();
    ws->data.resize(10);
  }
  EXPECT_EQ(&resources_access::workspaces(res), &resources_access::workspaces(copy));
  EXPECT_EQ(copy.workspace_bytes(), 10 * sizeof(int));
  copy.release_workspaces();
  EXPECT_EQ(res.workspace_bytes(), 0u);
  EXPECT_EQ(resources_access::workspaces(res).statistics().idle, 0u);

  const auto other = dyng::resources::sequential();  // independent handles, independent pools
  EXPECT_NE(&resources_access::workspaces(res), &resources_access::workspaces(other));
}

}  // namespace
