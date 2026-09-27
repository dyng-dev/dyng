// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/sequentialSOSPUpdate.cpp (sequentialSOSPUpdate, adapted)
/**
 * @file sequential.cpp
 * @brief The sequential backend of sssp: the legacy re-scan SOSP update, adapted into the hooks
 *        of problem.hpp. It is the readable reference backend (PLAN Section 6.4.2).
 *
 * sequentialSOSPUpdate() is a file-path API: it reads the CSR, the trees and the batch, applies
 * the batch to adjacency lists, updates and writes files. Only its algorithm is kept here; the
 * graph container has already applied the batch (commit) and classified it (apply_delta), so the
 * hooks receive the updated out- and in-edges and the per-objective change list:
 *
 *   - identify_affected: roots are the heads v of deleted or weight-increased edges (u,v) with
 *     parent[v] == u; the subtree of every root is invalidated (distance infinite, parent -1) by a
 *     traversal of the children lists of the old tree.
 *   - seed: every invalidated vertex, then every insertion head (not the source), re-evaluates
 *     its best (distance, lowest parent id) over its in-neighbours; those whose distance decreased
 *     are affected.
 *   - loop: the candidates of a round are the out-neighbours (not the source) of the affected
 *     vertices; each candidate re-evaluates its in-neighbours and keeps a strictly better pair;
 *     those whose distance decreased are affected in the next round. Distances only decrease, so
 *     the loop terminates without an iteration cap.
 *   - finalize: count the vertices whose distance or parent changed.
 *
 * compute() runs the same loop from the source (reset -> seed_static -> loop -> finalize).
 *
 * Changes from the original: the hook structure; the per-round `vector<bool>` candidate flags
 * (O(n) per round) are replaced by generation stamps in the workspace; the children lists and all
 * other scratch arrays live in the result's workspace (reserved once); the classification of
 * weight increases is the graph's (MOSP applyChangeBatch semantics, identical for both backends)
 * instead of the adjacency-list scan of the original.
 */
#include "algorithms/sssp/problem.hpp"
#include "graph/instantiate.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::detail {

namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
class sequential_problem {
 public:
  explicit sequential_problem(sssp_run<vertex_t, edge_t, weight_t>& run)
      : run_(run), ws_(*run.ws), n_(static_cast<std::int64_t>(run.graph.num_vertices)) {
    ws_.reserve(n_);
    // The sequential engine's own arrays, sized on its first run for this result (the OpenMP
    // engine does not need them).
    if (ws_.child_start.size() < static_cast<std::size_t>(n_) + 1) {
      ws_.child_start.assign(static_cast<std::size_t>(n_) + 1, vertex_t{0});
    }
    for (std::vector<vertex_t>* list :
         {&ws_.children, &ws_.invalid, &ws_.affected, &ws_.candidate_list}) {
      list->reserve(static_cast<std::size_t>(n_));
    }
    run_.counters = sssp_counters{};
    ws_.touched.clear();
    ws_.old_distance.clear();
    ws_.old_parent.clear();
    // `state` marks the vertices saved in the touch list (the OpenMP backend leaves its walk
    // states there, so clear it first).
    std::fill(ws_.state.begin(), ws_.state.begin() + n_, char{0});
  }

  // ---- compute() -------------------------------------------------------------------------------

  void reset() {
    for (std::int64_t v = 0; v < n_; ++v) {
      run_.distances[v] = sssp_infinity;
      run_.parents[v] = -1;
    }
    run_.distances[run_.source] = 0;
    count_changes_ = false;
  }

  void seed_static() {
    ws_.affected.assign(1, run_.source);
  }

  // ---- update() --------------------------------------------------------------------------------

  /// Roots and subtree invalidation (a traversal of the old tree's children lists).
  void identify_affected() {
    const sssp_changes<vertex_t>& changes = *run_.changes;
    std::int64_t* distances = run_.distances;
    vertex_t* parent = run_.parents;
    const vertex_t source = run_.source;
    int* stamp = ws_.stamp.data();
    const int invalid_generation = ws_.next_generation();
    auto is_invalid = [&](vertex_t v) { return stamp[v] == invalid_generation; };

    // Roots: the head v of every deleted or weight-increased edge (u,v) that is the tree edge of
    // v (parent[v] == u).
    std::vector<vertex_t>& invalidated = ws_.invalid;
    invalidated.clear();
    for (std::size_t i = 0; i < changes.changed_to.size(); ++i) {
      const vertex_t u = changes.changed_from[i];
      const vertex_t v = changes.changed_to[i];
      if (parent[v] == u && !is_invalid(v)) {
        stamp[v] = invalid_generation;
        invalidated.push_back(v);
      }
    }
    if (!invalidated.empty()) {
      // Children lists of the tree, then a traversal from the roots.
      std::vector<vertex_t>& child_start = ws_.child_start;
      std::vector<vertex_t>& children = ws_.children;
      std::fill(child_start.begin(), child_start.begin() + n_ + 1, vertex_t{0});
      for (std::int64_t v = 0; v < n_; ++v) {
        if (v != source && parent[v] >= 0) {
          ++child_start[static_cast<std::size_t>(parent[v]) + 1];
        }
      }
      for (std::int64_t v = 0; v < n_; ++v) {
        child_start[static_cast<std::size_t>(v) + 1] += child_start[static_cast<std::size_t>(v)];
      }
      children.resize(static_cast<std::size_t>(child_start[static_cast<std::size_t>(n_)]));
      // The cursor of row p is child_start[p] minus the children placed so far; placing children
      // in increasing id order keeps the traversal order of the original.
      std::vector<vertex_t>& cursor = ws_.candidate_list;
      cursor.assign(child_start.begin(), child_start.begin() + n_);
      for (std::int64_t v = 0; v < n_; ++v) {
        if (v != source && parent[v] >= 0) {
          children[static_cast<std::size_t>(cursor[static_cast<std::size_t>(parent[v])]++)] =
              static_cast<vertex_t>(v);
        }
      }
      for (std::size_t i = 0; i < invalidated.size(); ++i) {
        const vertex_t x = invalidated[i];
        for (vertex_t c = child_start[static_cast<std::size_t>(x)];
             c < child_start[static_cast<std::size_t>(x) + 1]; ++c) {
          const vertex_t child = children[static_cast<std::size_t>(c)];
          if (!is_invalid(child)) {
            stamp[child] = invalid_generation;
            invalidated.push_back(child);
          }
        }
      }
      for (const vertex_t v : invalidated) {
        save(v);
        distances[v] = sssp_infinity;
        parent[v] = -1;
      }
    }
    run_.counters.invalidated = static_cast<std::int64_t>(invalidated.size());
  }

  /// The invalidated vertices and the insertion heads pull their best in-neighbour.
  void seed() {
    const sssp_changes<vertex_t>& changes = *run_.changes;
    const vertex_t source = run_.source;
    ws_.affected.clear();
    affected_generation_ = ws_.next_generation();
    for (const vertex_t v : ws_.invalid) {
      if (relax(v)) {
        mark_affected(v);
      }
    }
    for (std::size_t i = 0; i < changes.num_insert_heads; ++i) {
      const vertex_t v = changes.insert_heads[i];
      if (v != source && relax(v)) {
        mark_affected(v);
      }
    }
  }

  // ---- both ------------------------------------------------------------------------------------

  /// Propagate: re-evaluate the out-neighbours of the affected vertices until nothing decreases.
  void loop() {
    const auto& g = run_.graph;
    const vertex_t source = run_.source;
    int* stamp = ws_.stamp.data();
    std::vector<vertex_t>& candidates = ws_.candidate_list;
    while (!ws_.affected.empty()) {
      ++run_.counters.iterations;
      if (run_.counters.iterations > n_) {
        // Every round settles at least one more hop of every shortest path (as in the original).
        DYNG_FAIL("sssp: the sequential update did not converge");
      }
      run_.counters.pushes += static_cast<std::int64_t>(ws_.affected.size());
      // Candidates: the out-neighbours of the affected vertices (never the source).
      const int candidate_generation = ws_.next_generation();
      candidates.clear();
      for (const vertex_t a : ws_.affected) {
        for (edge_t e = g.out_row_ptr[a]; e < g.out_row_ptr[a + 1]; ++e) {
          const vertex_t w = g.out_col_ind[e];
          if (w == source || stamp[w] == candidate_generation) {
            continue;
          }
          stamp[w] = candidate_generation;
          candidates.push_back(w);
        }
      }
      ws_.affected.clear();
      affected_generation_ = ws_.next_generation();
      for (const vertex_t c : candidates) {
        if (relax(c)) {
          mark_affected(c);
        }
      }
    }
  }

  /// Count the vertices whose distance or parent changed.
  void finalize() {
    if (!count_changes_) {
      std::int64_t reachable = 0;
      for (std::int64_t v = 0; v < n_; ++v) {
        reachable += run_.distances[v] < sssp_infinity / 2 ? 1 : 0;
      }
      run_.counters.affected = reachable;
      return;
    }
    std::int64_t affected = 0;
    for (std::size_t i = 0; i < ws_.touched.size(); ++i) {
      const vertex_t v = ws_.touched[i];
      affected += (run_.distances[v] != ws_.old_distance[i] || run_.parents[v] != ws_.old_parent[i])
                      ? 1
                      : 0;
    }
    run_.counters.affected = affected;
  }

 private:
  /// Remember the values of `v` before its first change (for `affected`).
  void save(vertex_t v) {
    if (!count_changes_) {
      return;
    }
    // Only the first change of a vertex is saved: it holds the values before the update.
    char& seen = ws_.state[static_cast<std::size_t>(v)];
    if (seen != 0) {
      return;
    }
    seen = 1;
    ws_.touched.push_back(v);
    ws_.old_distance.push_back(run_.distances[v]);
    ws_.old_parent.push_back(run_.parents[v]);
  }

  void mark_affected(vertex_t v) {
    int* stamp = ws_.stamp.data();
    if (stamp[v] != affected_generation_) {
      stamp[v] = affected_generation_;
      ws_.affected.push_back(v);
    }
  }

  /// Re-evaluate one vertex over its in-neighbours; returns true if its distance decreased.
  bool relax(vertex_t v) {
    const auto& g = run_.graph;
    const std::int64_t* distances = run_.distances;
    vertex_t best_parent = -1;
    std::int64_t best_distance = sssp_infinity;
    for (edge_t e = g.in_row_ptr[v]; e < g.in_row_ptr[v + 1]; ++e) {
      const vertex_t candidate_parent = g.in_col_ind[e];
      // Skip unreachable in-neighbours to avoid overflow.
      if (distances[candidate_parent] >= sssp_infinity / 2) {
        continue;
      }
      const std::int64_t candidate_distance =
          distances[candidate_parent] + static_cast<std::int64_t>(g.in_weights[e]);
      // Ties go to the lowest parent id (canonical tree).
      if (candidate_distance < best_distance ||
          (candidate_distance == best_distance && candidate_parent < best_parent)) {
        best_distance = candidate_distance;
        best_parent = candidate_parent;
      }
    }
    const bool better =
        best_distance < run_.distances[v] ||
        (best_distance == run_.distances[v] && best_parent >= 0 && best_parent < run_.parents[v]);
    if (!better) {
      return false;
    }
    const bool decreased = best_distance < run_.distances[v];
    save(v);
    run_.distances[v] = best_distance;
    run_.parents[v] = best_parent;
    return decreased;
  }

  sssp_run<vertex_t, edge_t, weight_t>& run_;
  sssp_workspace<vertex_t>& ws_;
  std::int64_t n_;
  int affected_generation_ = 0;
  bool count_changes_ = true;
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_sequential_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  sequential_problem<vertex_t, edge_t, weight_t> problem(run);
  sssp_enact_update(res, problem);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_sequential_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  DYNG_EXPECTS(run.source >= 0 && run.source < run.graph.num_vertices, "sssp: source ", run.source,
               " is out of range [0, ", run.graph.num_vertices, ")");
  sequential_problem<vertex_t, edge_t, weight_t> problem(run);
  sssp_enact_compute(res, problem);
}

#define DYNG_INSTANTIATE_SSSP_SEQUENTIAL(V, E, W)                                      \
  template void sssp_sequential_update<V, E, W>(const resources&, sssp_run<V, E, W>&); \
  template void sssp_sequential_compute<V, E, W>(const resources&, sssp_run<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_SEQUENTIAL)
#undef DYNG_INSTANTIATE_SSSP_SEQUENTIAL

}  // namespace dyng::detail
