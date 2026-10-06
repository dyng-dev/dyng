// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file engine.hpp
 * @brief dynamic_bfs_engine_impl<exec_t>: every pass of dynamic_bfs, written once with the
 *        framework operators and run by the executor `exec_t` of a backend (TEACHING MATERIAL).
 *
 * Each pass is a small functor (a struct of raw pointers with a `DYNG_HD` call operator) that does
 * the work of ONE element: one deleted edge, one frontier vertex, one invalidated vertex. The
 * executor runs it for every element: a loop (sequential.cpp), an OpenMP parallel loop
 * (openmp.cpp) or a CUDA kernel (cuda.cu). Words that several elements may write in one pass are
 * changed with the atomics of operators/execution.hpp, so the same functor is correct on all three.
 *
 * A level is stored as std::int64_t with -1 for "unreached". Read as an unsigned 64-bit word, -1
 * is the largest value, so "offer a smaller level" is one atomic minimum on that word.
 *
 * The frontier push (dedup by round stamps) and the out-edge loop of `advance` are operators with
 * one user, so they live here and not in cpp/src/operators (the rule of two, PLAN 4.5.3).
 */
#pragma once

#include "algorithms/dynamic_bfs/problem.hpp"
#include "operators/execution.hpp"

#include <cstddef>
#include <cstdint>
#include <limits>

namespace dyng::detail {

namespace dynamic_bfs_ops {

using operators::atomic_add;
using operators::atomic_exchange;
using operators::atomic_load;
using operators::atomic_min;
using operators::atomic_store;

/// A level as the unsigned word the atomic minimum compares (-1, unreached, is the largest).
DYNG_HD inline std::uint64_t* word(std::int64_t* p) {
  return reinterpret_cast<std::uint64_t*>(p);
}

/// Read a level or a parent that other elements of the pass may change.
DYNG_HD inline std::int64_t load(const std::int64_t* p) {
  return static_cast<std::int64_t>(atomic_load(reinterpret_cast<const std::uint64_t*>(p)));
}

/// Write a level or a parent that other elements of the pass may read.
DYNG_HD inline void store(std::int64_t* p, std::int64_t value) {
  atomic_store(word(p), static_cast<std::uint64_t>(value));
}

/**
 * @brief Append to a vertex list, at most once per stamp (the frontier push, dedup by stamps:
 *        a stamp array is never cleared, each round or run draws a new stamp).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct push_once {
  vertex_t* items;      ///< the list
  std::uint64_t* size;  ///< its length (a counter on the backend)
  std::uint32_t* mark;  ///< the stamp of every vertex
  std::uint32_t stamp;  ///< this round's (or run's) stamp

  /// Append v unless it was appended with this stamp already.
  DYNG_HD void operator()(vertex_t v) const {
    if (atomic_exchange(&mark[v], stamp) != stamp) {
      items[atomic_add(size, 1)] = v;
    }
  }
};

/**
 * @brief Invalidate one vertex (once per update: the `invalid` flag is claimed atomically): its
 *        level and parent are reset, it joins the invalidated and touched lists and the next
 *        frontier of the subtree walk.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct invalidator {
  std::int64_t* levels;              ///< the levels
  std::int64_t* parents;             ///< the parents
  std::uint32_t* invalid;            ///< the flags
  vertex_t* next;                    ///< the next frontier of the walk
  std::uint64_t* next_size;          ///< its length
  vertex_t* invalidated;             ///< every invalidated vertex
  std::uint64_t* invalidated_count;  ///< its length
  push_once<vertex_t> touch;         ///< the touched list

  /// Invalidate v unless it is invalid already.
  DYNG_HD void operator()(vertex_t v) const {
    if (atomic_exchange(&invalid[v], 1U) != 0U) {
      return;
    }
    store(&levels[v], -1);
    store(&parents[v], -1);
    next[atomic_add(next_size, 1)] = v;
    invalidated[atomic_add(invalidated_count, 1)] = v;
    touch(v);
  }
};

/**
 * @brief classify: the head v of a deleted edge u -> v is a root if u -> v was its tree edge and
 *        no parallel u -> v is left in G_{t+1}.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct find_roots {
  dynamic_bfs_graph<vertex_t, edge_t> g;  ///< G_{t+1}
  const vertex_t* tails;                  ///< the deleted edges' tails
  const vertex_t* heads;                  ///< the deleted edges' heads
  invalidator<vertex_t> invalidate;       ///< the invalidation

  /// Deletion i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t u = tails[i];
    const vertex_t v = heads[i];
    if (u < 0 || v < 0 || u >= g.num_vertices || v >= g.num_vertices ||
        load(&invalidate.parents[v]) != static_cast<std::int64_t>(u)) {
      return;  // not v's tree edge
    }
    for (edge_t e = g.out_offsets[u]; e < g.out_offsets[u + 1]; ++e) {
      if (g.out_targets[e] == v) {
        return;  // a parallel u -> v is left: v keeps its level
      }
    }
    invalidate(v);
  }
};

/**
 * @brief invalidate_subtree: the tree children of an invalidated vertex are invalidated too
 *        (one pass per tree level, from the roots down).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct invalidate_children {
  dynamic_bfs_graph<vertex_t, edge_t> g;  ///< G_{t+1}
  const vertex_t* current;                ///< the vertices invalidated in the previous pass
  invalidator<vertex_t> invalidate;       ///< the invalidation

  /// Frontier vertex i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t x = current[i];
    for (edge_t e = g.out_offsets[x]; e < g.out_offsets[x + 1]; ++e) {
      const vertex_t y = g.out_targets[e];
      if (load(&invalidate.parents[y]) == static_cast<std::int64_t>(x)) {
        invalidate(y);
      }
    }
  }
};

/**
 * @brief seed, part 1: an invalidated vertex pulls the best level from its valid in-neighbours
 *        (their levels do not change in this pass) and joins the frontier if it found one.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct pull_level {
  dynamic_bfs_graph<vertex_t, edge_t> g;  ///< G_{t+1}
  const vertex_t* invalidated;            ///< the invalidated vertices
  std::int64_t* levels;                   ///< the levels
  const std::uint32_t* invalid;           ///< the flags
  push_once<vertex_t> frontier;           ///< the first frontier

  /// Invalidated vertex i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t x = invalidated[i];
    std::int64_t best = -1;
    for (edge_t e = g.in_offsets[x]; e < g.in_offsets[x + 1]; ++e) {
      const vertex_t u = g.in_sources[e];
      if (invalid[u] != 0U) {
        continue;  // invalidated: its level is being rebuilt (it pushes later if it improves)
      }
      const std::int64_t lu = levels[u];
      if (lu >= 0 && (best < 0 || lu + 1 < best)) {
        best = lu + 1;
      }
    }
    if (best >= 0) {
      store(&levels[x], best);
      frontier(x);
    }
  }
};

/**
 * @brief Offer the level `level` to v (an atomic minimum); if v's level dropped, v joins the
 *        frontier and the touched list. Used by the insertions (seed) and by advance (loop).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct offer {
  std::int64_t* levels;          ///< the levels
  push_once<vertex_t> frontier;  ///< the next frontier
  push_once<vertex_t> touch;     ///< the touched list

  /// Offer `level` to v.
  DYNG_HD void operator()(vertex_t v, std::int64_t level) const {
    const auto candidate = static_cast<std::uint64_t>(level);
    if (candidate < atomic_min(word(&levels[v]), candidate)) {
      frontier(v);
      touch(v);
    }
  }
};

/**
 * @brief seed, part 2: an inserted edge u -> v offers level[u] + 1 to v.
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct offer_insertion {
  const vertex_t* tails;     ///< the inserted edges' tails
  const vertex_t* heads;     ///< the inserted edges' heads
  offer<vertex_t> offer_to;  ///< the offer

  /// Insertion i.
  DYNG_HD void operator()(std::int64_t i) const {
    const std::int64_t lu = load(&offer_to.levels[tails[i]]);
    if (lu >= 0) {
      offer_to(heads[i], lu + 1);
    }
  }
};

/**
 * @brief advance (push): a frontier vertex u offers level[u] + 1 to every out-neighbour.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct advance_one {
  dynamic_bfs_graph<vertex_t, edge_t> g;  ///< G_{t+1}
  const vertex_t* frontier;               ///< the frontier
  offer<vertex_t> offer_to;               ///< the offer

  /// Frontier vertex i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t u = frontier[i];
    const std::int64_t next = load(&offer_to.levels[u]) + 1;
    for (edge_t e = g.out_offsets[u]; e < g.out_offsets[u + 1]; ++e) {
      offer_to(g.out_targets[e], next);
    }
  }
};

/**
 * @brief finalize: the parent of a touched vertex is its lowest-id in-neighbour one level up
 *        (the levels are final), -1 for the source and the unreached; the vertex counts as
 *        affected if its level differs from the level before the update.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename vertex_t, typename edge_t>
struct repair_parent {
  dynamic_bfs_graph<vertex_t, edge_t> g;  ///< G_{t+1}
  const vertex_t* touched;                ///< the touched vertices
  const std::int64_t* levels;             ///< the levels
  std::int64_t* parents;                  ///< the parents
  const std::int64_t* before;             ///< the levels before the update, or nullptr
  std::uint64_t* changed;                 ///< the count of changed levels

  /// Touched vertex i.
  DYNG_HD void operator()(std::int64_t i) const {
    const vertex_t x = touched[i];
    const std::int64_t lx = levels[x];
    std::int64_t parent = -1;
    if (lx > 0) {
      for (edge_t e = g.in_offsets[x]; e < g.in_offsets[x + 1]; ++e) {
        const vertex_t u = g.in_sources[e];
        if (levels[u] == lx - 1 && (parent < 0 || u < parent)) {
          parent = u;
        }
      }
    }
    parents[x] = parent;
    if (before != nullptr && before[x] != lx) {
      atomic_add(changed, 1);
    }
  }
};

/**
 * @brief finalize: clear the flag of an invalidated vertex (the next update starts clean).
 * @tparam vertex_t Vertex id type.
 */
template <typename vertex_t>
struct clear_flag {
  const vertex_t* invalidated;  ///< the invalidated vertices
  std::uint32_t* invalid;       ///< the flags

  /// Invalidated vertex i.
  DYNG_HD void operator()(std::int64_t i) const {
    invalid[invalidated[i]] = 0U;
  }
};

}  // namespace dynamic_bfs_ops

/**
 * @brief The passes of dynamic_bfs on the backend of `exec_t` (see the file comment and
 *        dynamic_bfs_engine in problem.hpp).
 * @tparam exec_t   The executor (operators::sequential_exec, openmp_exec or cuda_exec).
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 */
template <typename exec_t, typename vertex_t, typename edge_t>
class dynamic_bfs_engine_impl final : public dynamic_bfs_engine<vertex_t, edge_t> {
 public:
  using graph_type = dynamic_bfs_graph<vertex_t, edge_t>;  ///< the graph
  using run_type = dynamic_bfs_run<vertex_t>;              ///< the run
  using workspace_type = dynamic_bfs_workspace<vertex_t>;  ///< the workspace

  void begin(const resources& res, const graph_type& g, run_type& run,
             bool snapshot) const override {
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    const auto n = static_cast<std::size_t>(g.num_vertices);
    // Size the arrays (a no-op once they are large enough: invariant I9). The flag and stamp
    // arrays are cleared only when they are (re)allocated; afterwards every run leaves the flags
    // clear and draws new stamps.
    const bool fresh = ws.invalid.capacity() < n;
    ws.before.reserve(res, n);
    ws.lists[0].reserve(res, n);
    ws.lists[1].reserve(res, n);
    ws.invalidated.reserve(res, n);
    ws.touched.reserve(res, n);
    ws.frontier_mark.reserve(res, n);
    ws.touched_mark.reserve(res, n);
    ws.invalid.reserve(res, n);
    ws.counters.reserve(res, workspace_type::num_counters);
    // A run draws one run stamp and at most n + 2 round stamps (the seed and one per round).
    constexpr std::uint64_t last_stamp = std::numeric_limits<std::uint32_t>::max();
    if (fresh || ws.run == last_stamp ||
        static_cast<std::uint64_t>(ws.round) + static_cast<std::uint64_t>(n) + 2 >= last_stamp) {
      const auto capacity = static_cast<std::int64_t>(ws.invalid.capacity());
      exec.fill(ws.invalid.data(), capacity, 0U);
      exec.fill(ws.frontier_mark.data(), static_cast<std::int64_t>(ws.frontier_mark.capacity()),
                0U);
      exec.fill(ws.touched_mark.data(), static_cast<std::int64_t>(ws.touched_mark.capacity()), 0U);
      ws.round = 0;
      ws.run = 0;
    }
    ++ws.run;
    exec.fill(ws.counters.data(), workspace_type::num_counters, std::uint64_t{0});
    if (snapshot) {
      exec.copy(ws.before.data(), run.levels, g.num_vertices);
    }
  }

  void invalidate(const resources& res, const graph_type& g, run_type& run, const vertex_t* tails,
                  const vertex_t* heads, std::int64_t count) const override {
    if (count == 0) {
      return;  // nothing deleted: nothing to invalidate (and no host synchronization)
    }
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    vertex_t* changes = upload_changes(exec, res, ws, tails, heads, count);
    // Pass 0 (classify): the roots go to list 0. Pass k: the children of list k % 2 go to the
    // other list. Each pass reads the length of its output (on cuda a host synchronization).
    int current = 0;
    exec.write(ws.counters.data() + 0, std::uint64_t{0});
    exec.for_each(count, dynamic_bfs_ops::find_roots<vertex_t, edge_t>{
                             g, changes, changes + count, make_invalidator(ws, run, 0)});
    std::int64_t size = static_cast<std::int64_t>(exec.read(ws.counters.data() + 0));
    run.invalidation_rounds = 1;
    run.invalidated = size;
    while (size > 0) {
      const int next = 1 - current;
      exec.write(ws.counters.data() + next, std::uint64_t{0});
      exec.for_each(size, dynamic_bfs_ops::invalidate_children<vertex_t, edge_t>{
                              g, ws.lists[current].data(), make_invalidator(ws, run, next)});
      size = static_cast<std::int64_t>(exec.read(ws.counters.data() + next));
      run.invalidated += size;
      ++run.invalidation_rounds;
      current = next;
    }
  }

  void seed(const resources& res, const graph_type& g, run_type& run, const vertex_t* tails,
            const vertex_t* heads, std::int64_t count, dynamic_bfs_frontier& f) const override {
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    f.list = 0;
    const dynamic_bfs_ops::push_once<vertex_t> frontier = frontier_push(ws, 0);
    exec.write(ws.counters.data() + 0, std::uint64_t{0});
    // The pull reads the levels of valid vertices only, which no element of the pass writes.
    exec.for_each(run.invalidated,
                  dynamic_bfs_ops::pull_level<vertex_t, edge_t>{
                      g, ws.invalidated.data(), run.levels, ws.invalid.data(), frontier});
    // Then the insertions, in a pass of their own (they may lower any level).
    vertex_t* changes = upload_changes(exec, res, ws, tails, heads, count);
    exec.for_each(count, dynamic_bfs_ops::offer_insertion<vertex_t>{
                             changes, changes + count,
                             dynamic_bfs_ops::offer<vertex_t>{run.levels, frontier, touch(ws)}});
    f.size = static_cast<std::int64_t>(exec.read(ws.counters.data() + 0));
  }

  void seed_source(const resources& res, const graph_type& g, run_type& run, std::int64_t source,
                   dynamic_bfs_frontier& f) const override {
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    exec.fill(run.levels, g.num_vertices, std::int64_t{-1});
    exec.fill(run.parents, g.num_vertices, std::int64_t{-1});
    exec.write(run.levels + source, std::int64_t{0});
    // The source is the first frontier and the first touched vertex (its stamps written too, so
    // a later offer cannot push it twice).
    const auto v = static_cast<vertex_t>(source);
    exec.write(ws.lists[0].data(), v);
    exec.write(ws.touched.data(), v);
    exec.write(ws.frontier_mark.data() + source, ++ws.round);
    exec.write(ws.touched_mark.data() + source, ws.run);
    exec.write(ws.counters.data() + workspace_type::touched_count, std::uint64_t{1});
    f.list = 0;
    f.size = 1;
  }

  void advance(const resources& res, const graph_type& g, run_type& run,
               const dynamic_bfs_frontier& in, dynamic_bfs_frontier& out) const override {
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    out.list = 1 - in.list;
    exec.write(ws.counters.data() + out.list, std::uint64_t{0});
    const dynamic_bfs_ops::offer<vertex_t> offer_to{run.levels, frontier_push(ws, out.list),
                                                    touch(ws)};
    exec.for_each(in.size, dynamic_bfs_ops::advance_one<vertex_t, edge_t>{
                               g, ws.lists[in.list].data(), offer_to});
    out.size = static_cast<std::int64_t>(exec.read(ws.counters.data() + out.list));
    ++run.iterations;
    run.frontier_visits += in.size;
  }

  std::int64_t finish(const resources& res, const graph_type& g, run_type& run,
                      bool count_affected) const override {
    const exec_t exec(res);
    workspace_type& ws = *run.ws;
    std::uint64_t* changed = ws.counters.data() + workspace_type::changed;
    const auto touched =
        static_cast<std::int64_t>(exec.read(ws.counters.data() + workspace_type::touched_count));
    exec.for_each(touched, dynamic_bfs_ops::repair_parent<vertex_t, edge_t>{
                               g, ws.touched.data(), run.levels, run.parents,
                               count_affected ? ws.before.data() : nullptr, changed});
    exec.for_each(run.invalidated,
                  dynamic_bfs_ops::clear_flag<vertex_t>{ws.invalidated.data(), ws.invalid.data()});
    return count_affected ? static_cast<std::int64_t>(exec.read(changed)) : 0;
  }

 private:
  /// The push into frontier list `list`, with a new round stamp.
  static dynamic_bfs_ops::push_once<vertex_t> frontier_push(workspace_type& ws, int list) {
    return {ws.lists[list].data(), ws.counters.data() + list, ws.frontier_mark.data(), ++ws.round};
  }

  /// The push into the touched list (the run's stamp).
  static dynamic_bfs_ops::push_once<vertex_t> touch(workspace_type& ws) {
    return {ws.touched.data(), ws.counters.data() + workspace_type::touched_count,
            ws.touched_mark.data(), ws.run};
  }

  /// The invalidation into frontier list `list`.
  static dynamic_bfs_ops::invalidator<vertex_t> make_invalidator(workspace_type& ws, run_type& run,
                                                                 int list) {
    return {run.levels,
            run.parents,
            ws.invalid.data(),
            ws.lists[list].data(),
            ws.counters.data() + list,
            ws.invalidated.data(),
            ws.counters.data() + workspace_type::invalidated_count,
            touch(ws)};
  }

  /// The batch's changes in the backend's memory: tails, then heads.
  static vertex_t* upload_changes(const exec_t& exec, const resources& res, workspace_type& ws,
                                  const vertex_t* tails, const vertex_t* heads,
                                  std::int64_t count) {
    vertex_t* changes = ws.changes.reserve(res, 2 * static_cast<std::size_t>(count));
    exec.upload(changes, tails, count);
    exec.upload(changes + count, heads, count);
    return changes;
  }
};

}  // namespace dyng::detail
