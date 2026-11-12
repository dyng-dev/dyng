// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_openmp_allocation_failure_test.cpp
 * @brief A host allocation failure inside an OpenMP region of sssp's OpenMP engine is a
 *        std::bad_alloc of the hook, after the region, and out_of_memory_error of sssp's entry
 *        points with the guarantee sssp.hpp documents (M11b, ADR 0094; the D13 class of ADR
 *        0061, the carried item (a) of 0.4).
 *
 * Before ADR 0094 a per-thread list that grew (thread_list::push_back) or a gather output that
 * was resized (list_gather, inside `omp single`) threw out of the region and ended the process.
 * This executable replaces the global operator new, the aligned form included
 * (support/failing_allocator.hpp, DYNG_TEST_FAIL_ALIGNED_NEW), and:
 *   - drives the engine's hooks directly (sssp_openmp_engine: identify_affected, seed, loop) on
 *     inputs shaped so that one allocation site of one region allocates first: the lists and
 *     the gathers of identify_affected (the invalidated vertices, the insertion heads), of seed
 *     (the pull pass's frontier) and of loop (the first near/far split, a push iteration, a far
 *     re-split), 18 sites; at one thread the first allocation of the hook is at the site (its
 *     kind, aligned or plain, and the near-far counters at the failure say so), and every
 *     allocation of the hook is failed in turn at 1, 8 and 56 threads, of the calling thread or
 *     of any thread, once or from then on (every thread's allocations fail), each sweep under a
 *     deadlock watchdog: every failure comes out of the hook as std::bad_alloc, none is lost,
 *     and the hook's output without a failure is the reference;
 *   - sweeps sssp::compute and sssp::update on openmp with 1, 8 and 56 threads (the calling
 *     thread's allocations, and any thread's): out_of_memory_error, with compute's strong
 *     guarantee and update's two outcomes (before the commit: `g` and `r` unchanged; after it:
 *     `g` updated and `r` poisoned), both reached.
 * mosp's OpenMP update, which runs this engine per objective, has its own executable
 * (algorithms/mosp/mosp_openmp_allocation_failure_test.cpp). Not under the sanitizers, which
 * replace operator new themselves (the tests skip).
 */
#define DYNG_TEST_FAIL_ALIGNED_NEW 1
#include "algorithms/sssp/problem.hpp"
#include "support/failing_allocator.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/edge_list.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/sssp.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <queue>
#include <random>
#include <string>
#include <utility>
#include <vector>

namespace {

namespace alloc = dyng::test::alloc;
namespace dd = dyng::detail;

using vid = std::int32_t;
using eid = std::int32_t;
using wgt = std::int32_t;
using engine_t = dd::sssp_openmp_engine<vid, eid, wgt>;
using run_t = dd::sssp_run<vid, eid, wgt>;
using ws_t = dd::sssp_workspace<vid>;

/// The thread counts of the sweeps (56: every hardware thread of the development machine).
constexpr int thread_counts[] = {1, 8, 56};
/// A sweep that has not finished after this long is a deadlock (a thread missed a barrier).
constexpr int watchdog_seconds = 600;

bool openmp_built() {
  return dyng::backend_available(dyng::backend::openmp);
}

// ---- the engine's inputs ------------------------------------------------------------------------

struct edge_rec {
  vid from;
  vid to;
  wgt w;
};

/// One objective's CSR pair (out- and in-edges), as sssp_problem builds it.
struct host_graph {
  vid n = 0;
  std::vector<eid> out_row, in_row;
  std::vector<vid> out_col, in_col;
  std::vector<wgt> out_w, in_w;
};

host_graph make_graph(vid n, std::vector<edge_rec> edges) {
  host_graph g;
  g.n = n;
  const auto fill = [n](std::vector<edge_rec> list, bool by_head, std::vector<eid>& row,
                        std::vector<vid>& col, std::vector<wgt>& w) {
    std::stable_sort(list.begin(), list.end(), [by_head](const edge_rec& a, const edge_rec& b) {
      return by_head ? a.to < b.to : a.from < b.from;
    });
    row.assign(static_cast<std::size_t>(n) + 1, 0);
    for (const edge_rec& e : list) {
      ++row[static_cast<std::size_t>(by_head ? e.to : e.from) + 1];
    }
    for (std::size_t v = 0; v < static_cast<std::size_t>(n); ++v) {
      row[v + 1] += row[v];
    }
    for (const edge_rec& e : list) {
      col.push_back(by_head ? e.from : e.to);
      w.push_back(e.w);
    }
  };
  fill(edges, false, g.out_row, g.out_col, g.out_w);
  fill(std::move(edges), true, g.in_row, g.in_col, g.in_w);
  return g;
}

/// Shortest distances from 0 and the lowest-id parent among the tight in-edges.
void shortest_tree(const host_graph& g, std::vector<std::int64_t>& dist, std::vector<vid>& parent) {
  const auto n = static_cast<std::size_t>(g.n);
  dist.assign(n, dd::sssp_infinity);
  parent.assign(n, -1);
  using item = std::pair<std::int64_t, vid>;
  std::priority_queue<item, std::vector<item>, std::greater<>> queue;
  dist[0] = 0;
  queue.emplace(0, 0);
  while (!queue.empty()) {
    const auto [d, u] = queue.top();
    queue.pop();
    if (d != dist[static_cast<std::size_t>(u)]) continue;
    for (eid e = g.out_row[static_cast<std::size_t>(u)];
         e < g.out_row[static_cast<std::size_t>(u) + 1]; ++e) {
      const vid v = g.out_col[static_cast<std::size_t>(e)];
      const std::int64_t nd = d + g.out_w[static_cast<std::size_t>(e)];
      if (nd < dist[static_cast<std::size_t>(v)]) {
        dist[static_cast<std::size_t>(v)] = nd;
        queue.emplace(nd, v);
      }
    }
  }
  for (std::size_t v = 1; v < n; ++v) {
    for (eid e = g.in_row[v]; e < g.in_row[v + 1]; ++e) {
      const vid u = g.in_col[static_cast<std::size_t>(e)];
      if (dist[static_cast<std::size_t>(u)] < dd::sssp_infinity &&
          dist[static_cast<std::size_t>(u)] + g.in_w[static_cast<std::size_t>(e)] == dist[v] &&
          (parent[v] < 0 || u < parent[v])) {
        parent[v] = u;
      }
    }
  }
}

/// The hook a site test arms (the earlier hooks of the run are called unarmed in prepare()).
enum class hook { identify_affected, seed, loop };

/// The engine's inputs of one site: the graph after the batch, the old tree, the change lists.
struct engine_input {
  host_graph g;
  std::vector<std::int64_t> dist;
  std::vector<vid> parent;
  std::vector<vid> changed_from, changed_to, insert_heads;
  bool update = true;  ///< false: compute() (reset, seed_static)
  std::int64_t delta = 1;
  std::int64_t max_weight = 1;
};

constexpr vid site_vertices = 3000;

/// A chain 0 -> 1 -> ... (weight 1), skips i -> i + 2 (weight 3) and direct edges 0 -> v
/// (weight 2v + 5); the old tree is the chain; `batch` says what the update changed.
enum class batch_kind {
  delete_first_chain_edge,  ///< (0, 1) deleted: every vertex but 0 is invalidated
  insert_heads_only,        ///< no deletion; every vertex but 0 is an insertion head
};

engine_input chain_input(batch_kind batch, std::int64_t delta) {
  const vid n = site_vertices;
  std::vector<edge_rec> after;
  for (vid v = 0; v + 1 < n; ++v) {
    if (!(batch == batch_kind::delete_first_chain_edge && v == 0)) {
      after.push_back({v, v + 1, 1});
    }
    if (v + 2 < n) after.push_back({v, v + 2, 3});
  }
  for (vid v = 1; v < n; ++v) after.push_back({0, v, 2 * v + 5});
  std::vector<edge_rec> before = after;
  if (batch == batch_kind::delete_first_chain_edge) before.push_back({0, 1, 1});
  engine_input in;
  in.g = make_graph(n, after);
  shortest_tree(make_graph(n, before), in.dist, in.parent);
  if (batch == batch_kind::delete_first_chain_edge) {
    in.changed_from = {0};
    in.changed_to = {1};
  } else {
    for (vid v = 1; v < n; ++v) in.insert_heads.push_back(v);
  }
  in.delta = delta;
  in.max_weight = 2 * n + 5;
  return in;
}

/// compute(): a star 0 -> v (weight `w`) over every other vertex.
engine_input star_input(wgt w, std::int64_t delta) {
  std::vector<edge_rec> edges;
  for (vid v = 1; v < site_vertices; ++v) edges.push_back({0, v, w});
  engine_input in;
  in.g = make_graph(site_vertices, edges);
  in.update = false;
  in.delta = delta;
  in.max_weight = w;
  in.dist.assign(static_cast<std::size_t>(site_vertices), dd::sssp_infinity);
  in.parent.assign(static_cast<std::size_t>(site_vertices), -1);
  return in;
}

/// compute(): the path 0 -> 1 -> 2 -> 3 (weight 1) and far edges from 0, 1 and 2 (weights 100,
/// 200 and 300) to a third of the other vertices each: three push iterations put `far_per_step`
/// vertices each on the far pile, and the first re-split keeps two thirds of it.
constexpr vid far_per_step = (site_vertices - 4) / 3;
engine_input path_input() {
  std::vector<edge_rec> edges = {{0, 1, 1}, {1, 2, 1}, {2, 3, 1}};
  for (vid i = 0; i < 3 * far_per_step; ++i) {
    edges.push_back({i / far_per_step, 4 + i, 100 * (i / far_per_step + 1)});
  }
  engine_input in;
  in.g = make_graph(site_vertices, edges);
  in.update = false;
  in.delta = 10;
  in.max_weight = 300;
  in.dist.assign(static_cast<std::size_t>(site_vertices), dd::sssp_infinity);
  in.parent.assign(static_cast<std::size_t>(site_vertices), -1);
  return in;
}

/// One run of the engine: its inputs, workspace and the engine bound to them.
struct engine_run {
  engine_input in;
  dd::sssp_changes<vid> changes;
  ws_t ws;
  run_t run;
  std::optional<engine_t> engine;

  engine_run(const dyng::resources& res, engine_input input) : in(std::move(input)) {
    changes.changed_from = in.changed_from.data();
    changes.changed_to = in.changed_to.data();
    changes.num_changed = in.changed_from.size();
    changes.insert_heads = in.insert_heads.data();
    changes.num_insert_heads = in.insert_heads.size();
    run.graph = {in.g.n,
                 in.g.out_row.data(),
                 in.g.out_col.data(),
                 in.g.out_w.data(),
                 in.g.in_row.data(),
                 in.g.in_col.data(),
                 in.g.in_w.data()};
    run.changes = in.update ? &changes : nullptr;
    run.source = 0;
    run.delta = in.delta;
    run.max_weight = in.max_weight;
    run.distances = in.dist.data();
    run.parents = in.parent.data();
    run.ws = &ws;
    engine.emplace(res, run);
  }
  engine_run(const engine_run&) = delete;
  engine_run& operator=(const engine_run&) = delete;
  engine_run(engine_run&&) = delete;
  engine_run& operator=(engine_run&&) = delete;
  ~engine_run() = default;

  /// Call the hooks before `target`, unarmed.
  void run_before(hook target) {
    if (!in.update) {
      engine->reset();
      engine->seed_static();
      return;
    }
    if (target != hook::identify_affected) engine->identify_affected();
    if (target == hook::loop) engine->seed();
  }

  void call(hook target) {
    switch (target) {
      case hook::identify_affected:
        engine->identify_affected();
        break;
      case hook::seed:
        engine->seed();
        break;
      case hook::loop:
        engine->loop();
        break;
    }
  }

  /// Every thread's list `slot` with capacity `capacity` (0: empty, no block).
  void lists(int slot, std::size_t capacity) {
    for (std::size_t i = static_cast<std::size_t>(slot); i < ws.thread_lists.size(); i += 2) {
      dd::thread_list<vid>().swap(ws.thread_lists[i].items);
      ws.thread_lists[i].items.reserve(capacity);
    }
  }
};

/// An output list of the workspace without capacity, so that the gather into it allocates.
void without_capacity(std::vector<vid>& list) {
  std::vector<vid>().swap(list);
}

/// The hook's output, compared with the reference run's.
struct hook_output {
  std::vector<vid> list;             ///< identify_affected: candidates; seed: frontier (sorted)
  std::vector<std::uint64_t> words;  ///< loop: the packed words
  std::int64_t invalidated = 0;

  bool operator==(const hook_output& o) const {
    return list == o.list && words == o.words && invalidated == o.invalidated;
  }
};

hook_output output_of(engine_run& r, hook target) {
  hook_output out;
  if (target == hook::identify_affected) {
    out.list = r.ws.candidates;
  } else if (target == hook::seed) {
    out.list = r.ws.frontier;
  } else {
    out.words = r.ws.packed;
  }
  std::sort(out.list.begin(), out.list.end());
  out.invalidated = r.run.counters.invalidated;
  return out;
}

/// One allocation site: the input, the hook, the shaping of the workspace that makes the site
/// allocate first, the kind of its allocation and, for loop, the near-far counters at it.
struct site {
  const char* name;
  std::function<engine_input()> input;
  hook target;
  std::function<void(engine_run&)> shape;
  alloc::allocation_kind kind;
  std::int64_t iterations = 0;  ///< loop: run.counters.iterations when the site fails
  std::int64_t epochs = 0;      ///< loop: run.counters.epochs when the site fails
};

constexpr std::size_t big = 4 * site_vertices;

std::vector<site> sites() {
  using kind = alloc::allocation_kind;
  const auto lists_empty = [](engine_run& r) {
    r.lists(0, 0);
    r.lists(1, 0);
  };
  const auto lists_big = [](engine_run& r) {
    r.lists(0, big);
    r.lists(1, big);
  };
  const auto chain_deleted = [] { return chain_input(batch_kind::delete_first_chain_edge, 10); };
  const auto chain_heads = [] { return chain_input(batch_kind::insert_heads_only, 10); };
  const auto chain_all_near = [] {
    return chain_input(batch_kind::delete_first_chain_edge, 1'000'000);
  };
  const auto star_near = [] { return star_input(1, 1'000'000); };
  const auto star_far = [] { return star_input(100, 10); };
  return {
      // identify_affected: the invalidated vertices and the insertion heads, and their gathers.
      {"identify_affected.invalid_list", chain_deleted, hook::identify_affected, lists_empty,
       kind::aligned},
      {"identify_affected.heads_list", chain_heads, hook::identify_affected, lists_empty,
       kind::aligned},
      {"identify_affected.invalid_gather", chain_deleted, hook::identify_affected,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.candidates);
       },
       kind::plain},
      {"identify_affected.heads_gather", chain_heads, hook::identify_affected,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.candidates);
       },
       kind::plain},
      // seed: the pull pass's frontier and its gather.
      {"seed.frontier_list", chain_deleted, hook::seed, lists_empty, kind::aligned},
      {"seed.frontier_gather", chain_deleted, hook::seed,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.frontier);
       },
       kind::plain},
      // loop: the first split of the frontier (update), near and far, and its gathers.
      {"loop.split.near_list", chain_all_near, hook::loop,
       [](engine_run& r) {
         r.lists(0, 0);
         r.lists(1, big);
       },
       kind::aligned, 0, 0},
      {"loop.split.far_list", chain_deleted, hook::loop,
       [](engine_run& r) {
         r.lists(0, big);
         r.lists(1, 0);
       },
       kind::aligned, 0, 0},
      {"loop.split.near_gather", chain_all_near, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.near_a);
       },
       kind::plain, 0, 0},
      {"loop.split.far_gather", chain_deleted, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.far);
       },
       kind::plain, 0, 0},
      // loop: a push iteration (compute from the source), near and far, and its gathers.
      {"loop.push.near_list", star_near, hook::loop,
       [](engine_run& r) {
         r.lists(0, 1);
         r.lists(1, big);
       },
       kind::aligned, 1, 0},
      {"loop.push.far_list", star_far, hook::loop,
       [](engine_run& r) {
         r.lists(0, 1);
         r.lists(1, 0);
       },
       kind::aligned, 1, 0},
      {"loop.push.near_gather", star_near, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.near_b);
       },
       kind::plain, 1, 0},
      {"loop.push.far_gather", star_far, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.far);
       },
       kind::plain, 1, 0},
      // loop: a far re-split, its near and kept lists, and their gathers.
      {"loop.resplit.near_list", star_far, hook::loop,
       [](engine_run& r) {
         r.lists(0, 1);
         r.lists(1, big);
       },
       kind::aligned, 1, 1},
      {"loop.resplit.keep_list", path_input, hook::loop,
       [](engine_run& r) {
         r.lists(0, big);
         r.lists(1, static_cast<std::size_t>(far_per_step));
       },
       kind::aligned, 4, 1},
      {"loop.resplit.near_gather", star_far, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.near_b);
       },
       kind::plain, 1, 1},
      {"loop.resplit.keep_gather", path_input, hook::loop,
       [=](engine_run& r) {
         lists_big(r);
         without_capacity(r.ws.far2);
       },
       kind::plain, 4, 1},
  };
}

/// A fresh run of `s` on `threads` threads, the hooks before its own called, the workspace shaped.
std::unique_ptr<engine_run> prepared(const site& s, int threads) {
  auto r = std::make_unique<engine_run>(dyng::resources::openmp(threads), s.input());
  r->run_before(s.target);
  s.shape(*r);
  return r;
}

class SsspOpenmpSites : public ::testing::Test {
 protected:
  void SetUp() override {
    if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
    if (!openmp_built()) GTEST_SKIP() << "the openmp backend is not built";
  }
};

// At one thread the first allocation of the armed hook is at the site: of the site's kind, and
// (loop) in the region the near-far counters name; it comes out of the hook as std::bad_alloc.
TEST_F(SsspOpenmpSites, TheFirstAllocationOfTheHookIsAtTheSite) {
  for (const site& s : sites()) {
    SCOPED_TRACE(s.name);
    const auto before = prepared(s, 1);
    const long long plain0 = alloc::plain_allocations.load();
    const long long aligned0 = alloc::aligned_allocations.load();
    before->call(s.target);
    const long long plain = alloc::plain_allocations.load() - plain0;
    const long long aligned = alloc::aligned_allocations.load() - aligned0;
    // Only the site's kind allocates in the hook (the other lists and outputs have room).
    if (s.kind == alloc::allocation_kind::aligned) {
      EXPECT_GT(aligned, 0);
      EXPECT_EQ(plain, 0);
    } else {
      EXPECT_GT(plain, 0);
      EXPECT_EQ(aligned, 0);
    }
    const auto r = prepared(s, 1);
    bool threw = false;
    bool reached = false;
    alloc::allocation_kind failed = alloc::allocation_kind::none;
    alloc::within_seconds(watchdog_seconds, s.name, [&] {
      alloc::arm(alloc::scope::calling_thread, 1);
      try {
        r->call(s.target);
      } catch (const std::bad_alloc&) {
        threw = true;
      }
      failed = alloc::failed_kind();
      reached = alloc::disarm();
    });
    EXPECT_TRUE(reached);
    EXPECT_TRUE(threw);
    EXPECT_EQ(static_cast<int>(failed), static_cast<int>(s.kind));
    if (s.target == hook::loop) {
      EXPECT_EQ(r->run.counters.iterations, s.iterations);
      EXPECT_EQ(r->run.counters.epochs, s.epochs);
    }
  }
}

// Every allocation of the hook failed in turn, at 1, 8 and 56 threads, of the calling thread or
// of any thread, once or from then on: each reached failure is a std::bad_alloc of the hook (no
// thread ends the process, none waits forever at a barrier, no failure is lost), and the hook's
// output when nothing fails is the reference.
TEST_F(SsspOpenmpSites, EveryAllocationFailedInTurnIsABadAllocOfTheHook) {
  for (const site& s : sites()) {
    hook_output reference;
    {
      const auto r = prepared(s, 1);
      r->call(s.target);
      reference = output_of(*r, s.target);
    }
    for (const int threads : thread_counts) {
      for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
        for (const alloc::failure mode : {alloc::failure::once, alloc::failure::from_then_on}) {
          const std::string what =
              std::string(s.name) + ", " + std::to_string(threads) + " threads, " +
              (scope == alloc::scope::calling_thread ? "calling thread" : "any thread") + ", " +
              (mode == alloc::failure::once ? "once" : "from then on");
          SCOPED_TRACE(what);
          int failures = 0;
          alloc::within_seconds(watchdog_seconds, what.c_str(), [&] {
            for (long long k = 1;; ++k) {
              const auto r = prepared(s, threads);
              bool threw = false;
              std::string other;
              alloc::arm(scope, k, mode);
              try {
                r->call(s.target);
              } catch (const std::bad_alloc&) {
                threw = true;
              } catch (const std::exception& e) {
                other = e.what();
              }
              const bool reached = alloc::disarm();
              ASSERT_TRUE(other.empty()) << "allocation " << k << ": " << other;
              if (!reached) {
                // The hook made fewer than k allocations: it ran to the end, unharmed.
                EXPECT_FALSE(threw) << k;
                EXPECT_TRUE(output_of(*r, s.target) == reference) << k;
                break;
              }
              EXPECT_TRUE(threw) << "allocation " << k << " failed, but the hook returned";
              failures += threw ? 1 : 0;
            }
          });
          // Every allocation of the hook is inside a region, and the site allocates: at one
          // thread always, at more threads in the any-thread sweeps. (The calling thread may make
          // none at 8 or 56 threads: the `single` of a gather runs on whichever thread arrives
          // first. And the re-split's kept list does not grow there: the static schedule gives
          // each thread a share of the far pile below the capacity the shaping gives its list.)
          const bool site_allocates =
              threads == 1 || (scope == alloc::scope::any_thread &&
                               std::string(s.name) != "loop.resplit.keep_list");
          if (site_allocates) {
            EXPECT_GT(failures, 0);
          }
        }
      }
    }
  }
}

// ---- sssp's entry points ----------------------------------------------------------------------

using graph_t = dyng::graph<std::int32_t, std::int64_t, std::int32_t>;
using result_t = dyng::sssp::result<std::int32_t>;
using batch_t = dyng::edge_batch<std::int32_t, std::int32_t>;

/// A random graph with a spanning path (most vertices reachable), as sssp_workspace_test's.
dyng::edge_list<std::int32_t, std::int32_t> random_edges(std::int32_t n, int m,
                                                         std::uint32_t seed) {
  std::mt19937 rng(seed);
  dyng::edge_list<std::int32_t, std::int32_t> list;
  list.num_vertices = n;
  list.num_weights = 1;
  for (std::int32_t v = 1; v < n; ++v) {
    list.add_edge(v - 1, v, {1 + static_cast<std::int32_t>(rng() % 20)});
  }
  for (int i = 0; i < m; ++i) {
    list.add_edge(static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
                  static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
                  {1 + static_cast<std::int32_t>(rng() % 20)});
  }
  return list;
}

/// Deletes the first out-edge of `count` random vertices and inserts as many random edges.
batch_t random_batch(const graph_t& g, int count, std::uint32_t seed) {
  std::mt19937 rng(seed);
  const auto out = g.view().out;
  const std::int32_t n = g.num_vertices();
  batch_t b(1);
  for (int i = 0; i < count; ++i) {
    const auto u = static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n));
    if (out.row_ptr[static_cast<std::size_t>(u) + 1] > out.row_ptr[static_cast<std::size_t>(u)]) {
      b.delete_edge(u, out.col_ind[static_cast<std::size_t>(out.row_ptr[u])]);
    }
    b.insert_edge(static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
                  static_cast<std::int32_t>(rng() % static_cast<std::uint32_t>(n)),
                  {1 + static_cast<std::int32_t>(rng() % 20)});
  }
  return b;
}

constexpr std::int32_t api_vertices = 1000;
constexpr int api_edges = 4000;

graph_t api_graph(const dyng::resources& res) {
  return graph_t::from_edges(res, random_edges(api_vertices, api_edges, 7).view(),
                             dyng::graph_properties::mosp_compatible());
}

struct tree {
  std::vector<std::int64_t> distances;
  std::vector<std::int32_t> parents;
  bool operator==(const tree& o) const {
    return distances == o.distances && parents == o.parents;
  }
};

tree tree_of(const result_t& r) {
  const auto d = r.distances();
  const auto p = r.parents();
  return {std::vector<std::int64_t>(d.begin(), d.end()),
          std::vector<std::int32_t>(p.begin(), p.end())};
}

class SsspOpenmpEntryPoints : public ::testing::Test {
 protected:
  void SetUp() override {
    if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
    if (!openmp_built()) GTEST_SKIP() << "the openmp backend is not built";
  }
};

// compute(): out_of_memory_error, `g` unchanged (strong), and the result of the call without a
// failure is the reference.
TEST_F(SsspOpenmpEntryPoints, ComputeIsStrongUnderEveryAllocationFailure) {
  const tree reference = tree_of(dyng::sssp::compute(dyng::resources::sequential(),
                                                     api_graph(dyng::resources::sequential()), 0));
  for (const int threads : thread_counts) {
    for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
      const std::string what = "compute, " + std::to_string(threads) + " threads, " +
                               (scope == alloc::scope::calling_thread ? "calling" : "any");
      SCOPED_TRACE(what);
      std::optional<dyng::resources> res;
      std::optional<graph_t> g;
      std::optional<result_t> r;
      std::uint64_t version = 0;
      int failures = 0;
      alloc::within_seconds(watchdog_seconds, what.c_str(), [&] {
        failures = alloc::fail_each_allocation<dyng::out_of_memory_error>(
            scope,
            [&] {
              r.reset();
              g.reset();
              res.emplace(dyng::resources::openmp(threads));
              g.emplace(api_graph(*res));
              version = g->version();
            },
            [&] { r.emplace(dyng::sssp::compute(*res, *g, 0)); },
            [&](bool threw) {
              EXPECT_EQ(g->version(), version);
              if (!threw) {
                ASSERT_TRUE(r.has_value());
                EXPECT_TRUE(tree_of(*r) == reference);
              }
            });
      });
      EXPECT_GT(failures, 0);
    }
  }
}

// update(): out_of_memory_error with one of sssp.hpp's two outcomes: before the commit `g` and
// `r` are unchanged (and the update then succeeds); after it `g` holds the new version and `r` is
// poisoned (stale_result_error). Both are reached; without a failure the result is the
// reference.
TEST_F(SsspOpenmpEntryPoints, UpdateKeepsItsGuaranteeUnderEveryAllocationFailure) {
  const dyng::resources seq = dyng::resources::sequential();
  graph_t g0 = api_graph(seq);
  result_t r0 = dyng::sssp::compute(seq, g0, 0);
  const tree old_tree = tree_of(r0);
  const batch_t batch = random_batch(g0, 300, 11);
  (void)dyng::sssp::update(seq, g0, batch.view(), r0);
  const tree reference = tree_of(r0);
  for (const int threads : thread_counts) {
    for (const alloc::scope scope : {alloc::scope::calling_thread, alloc::scope::any_thread}) {
      const std::string what = "update, " + std::to_string(threads) + " threads, " +
                               (scope == alloc::scope::calling_thread ? "calling" : "any");
      SCOPED_TRACE(what);
      std::optional<dyng::resources> res;
      std::optional<graph_t> g;
      std::optional<result_t> r;
      std::uint64_t version = 0;
      int before_commit = 0;
      int after_commit = 0;
      alloc::within_seconds(watchdog_seconds, what.c_str(), [&] {
        (void)alloc::fail_each_allocation<dyng::out_of_memory_error>(
            scope,
            [&] {
              r.reset();
              g.reset();
              // The update runs through a handle of its own: its workspace pool is empty, so
              // the per-thread lists grow inside the update's regions.
              const dyng::resources setup = dyng::resources::openmp(threads);
              res.emplace(dyng::resources::openmp(threads));
              g.emplace(api_graph(setup));
              r.emplace(dyng::sssp::compute(setup, *g, 0));
              version = g->version();
            },
            [&] { (void)dyng::sssp::update(*res, *g, batch.view(), *r); },
            [&](bool threw) {
              if (!threw) {
                EXPECT_EQ(g->version(), version + 1);
                EXPECT_TRUE(tree_of(*r) == reference);
              } else if (g->version() == version) {
                ++before_commit;
                EXPECT_TRUE(tree_of(*r) == old_tree);
                (void)dyng::sssp::update(*res, *g, batch.view(), *r);
                EXPECT_TRUE(tree_of(*r) == reference);
              } else {
                ++after_commit;
                EXPECT_EQ(g->version(), version + 1);
                EXPECT_THROW((void)r->distances(), dyng::stale_result_error);
                EXPECT_THROW((void)dyng::sssp::update(*res, *g, batch.view(), *r),
                             dyng::stale_result_error);
              }
            });
      });
      EXPECT_GT(before_commit, 0);
      EXPECT_GT(after_commit, 0);
    }
  }
}

}  // namespace
