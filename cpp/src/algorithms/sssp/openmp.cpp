// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/sospUpdateCpu.cpp (sospUpdateCpu, sospFromScratchCpu)
/**
 * @file openmp.cpp
 * @brief The OpenMP backend of sssp: MOSP-OpenMP's work-efficient, disconnection-safe SOSP update
 *        ported straight, its phases placed in the hooks of problem.hpp.
 *
 * Algorithm (unchanged; the same as MOSP-CUDA's sospUpdateGpu):
 *
 * Step 1 (from the change list):
 *   - Roots: the head v of every deleted or weight-increased edge (u,v) with parent[v] == u.
 *   - Subtree invalidation: every vertex walks up its parent chain to the first vertex whose state
 *     is known (a root: invalid; the tree root: valid) and writes that state along the path, so
 *     each vertex is resolved about once. The invalid vertices lose their distance and parent.
 *   - Pull pass: every invalidated vertex and every head of an inserted edge takes the best
 *     (distance, parent id) pair over its in-neighbours.
 *
 * Step 2 (propagation): a push-based near-far worklist (Delta-stepping variant). Improved vertices
 * relax their out-edges with an atomic minimum (compare-and-swap) on the packed 64-bit word
 * (distance << b | parent); a head below the threshold joins the next near frontier (deduplicated
 * by generation stamps), otherwise the far pile; when the near frontier is empty the threshold
 * moves to the smallest far distance plus Delta. Distances only decrease, so there is no iteration
 * cap, no counting to infinity and no reachability post-pass; ties go to the lowest parent id
 * (canonical trees); only improved vertices are expanded. If n * maxWeight does not fit next to the
 * parent ids, the words hold the distance alone and parents are recovered in one pass over the
 * out-edges after the search.
 *
 * Mechanical changes only: names (snake_case), templates on the index types, namespace
 * dyng::detail, exceptions (invalid_argument_error) instead of `bool` + `cerr`, no globals, the
 * thread count of `resources` on every parallel region (`num_threads`) instead of the global
 * OpenMP setting, the workspace owned by the result and reserved once, the phases split into the
 * hooks identify_affected / seed / loop / finalize, and one added counter in the unpack pass
 * (`affected`: vertices whose distance or parent changed).
 */
#include "algorithms/sssp/problem.hpp"
#include "graph/instantiate.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#if DYNG_HAS_OPENMP

#include "util/list_gather.hpp"

#include <omp.h>

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

using u64 = std::uint64_t;
constexpr u64 packed_inf = ~0ULL;
constexpr u64 output_max_distance = static_cast<u64>(sssp_infinity / 2 - 1);

/// Packed (distance, parent) words; parent_bits == 0 means distance only.
template <typename vertex_t>
struct packing {
  int parent_bits;
  u64 no_parent;

  [[nodiscard]] bool has_parents() const {
    return parent_bits > 0;
  }
  [[nodiscard]] u64 pack(u64 distance, vertex_t parent) const {
    if (parent_bits == 0) {
      return distance;
    }
    return (distance << parent_bits) | (parent < 0 ? no_parent : static_cast<u64>(parent));
  }
  [[nodiscard]] u64 distance(u64 word) const {
    return word >> parent_bits;
  }
  [[nodiscard]] vertex_t parent(u64 word) const {
    const u64 p = word & no_parent;
    return p == no_parent ? vertex_t{-1} : static_cast<vertex_t>(p);
  }
  [[nodiscard]] u64 max_distance() const {
    return (packed_inf >> parent_bits) - 1;
  }
};

/// @p largest_candidate: the largest distance a word may have to hold, including candidates
/// formed from a tree distance plus one edge.
template <typename vertex_t>
packing<vertex_t> make_packing(std::int64_t num_vertices, u64 largest_candidate) {
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<u64>(num_vertices)) {
    ++bits;
  }
  packing<vertex_t> packed{bits, (1ULL << bits) - 1};
  return largest_candidate <= packed.max_distance() ? packed : packing<vertex_t>{0, 0};
}

inline u64 load(const u64* p) {
  return __atomic_load_n(p, __ATOMIC_RELAXED);
}

/// Atomic minimum; returns the value before the operation.
inline u64 atomic_min(u64* p, u64 value) {
  u64 old = load(p);
  while (value < old &&
         !__atomic_compare_exchange_n(p, &old, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
  return old;
}

template <typename vertex_t>
inline void atomic_min_id(vertex_t* p, vertex_t value) {
  vertex_t old = __atomic_load_n(p, __ATOMIC_RELAXED);
  while (value < old &&
         !__atomic_compare_exchange_n(p, &old, value, true, __ATOMIC_RELAXED, __ATOMIC_RELAXED)) {
  }
}

template <typename vertex_t>
inline bool claim(int* stamp, vertex_t v, int generation) {
  return __atomic_exchange_n(&stamp[v], generation, __ATOMIC_RELAXED) != generation;
}

/// Smallest distance over a vertex list.
template <typename vertex_t>
u64 smallest_distance(const std::vector<vertex_t>& list, const std::vector<u64>& packed_words,
                      const packing<vertex_t>& packed, int threads) {
  const auto count = static_cast<std::int64_t>(list.size());
  u64 smallest = packed_inf;
#pragma omp parallel for num_threads(threads) reduction(min : smallest) schedule(static)
  for (std::int64_t i = 0; i < count; ++i) {
    const u64 word =
        load(&packed_words[static_cast<std::size_t>(list[static_cast<std::size_t>(i)])]);
    if (word != packed_inf) {
      smallest = std::min(smallest, packed.distance(word));
    }
  }
  return smallest;
}

/// Choose the packing; fails only if distances could overflow 62 bits.
template <typename vertex_t>
void choose_packing(std::int64_t n, std::int64_t max_weight, packing<vertex_t>& packed,
                    u64& bound) {
  const u64 weight = static_cast<u64>(std::max<std::int64_t>(max_weight, 1));
  const u64 hops = static_cast<u64>(std::max<std::int64_t>(n - 1, 1));
  DYNG_EXPECTS(weight <= output_max_distance / hops, "sssp: distances up to ", weight, " * ", hops,
               " do not fit in 62 bits");
  bound = weight * hops;
  // Tree distances are at most bound, but a relaxation or the pull pass forms bound + weight
  // before comparing (e.g. from the farthest vertex back into the tree): that value must fit next
  // to the parent bits too, or the shift drops its top bits and a wrong small distance wins.
  packed = make_packing<vertex_t>(n, bound + weight);
}

/// The OpenMP problem: MOSP-OpenMP's sospUpdateCpu() / sospFromScratchCpu(), one hook per phase.
template <typename vertex_t, typename edge_t, typename weight_t>
class openmp_problem {
 public:
  openmp_problem(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run)
      : threads_(std::max(1, res.num_threads())), run_(run), ws_(*run.ws) {
    const std::int64_t n = run_.graph.num_vertices;
    DYNG_EXPECTS(run_.delta > 0, "sssp: the near-far width delta must be > 0, got ", run_.delta);
    ws_.reserve(n);
    choose_packing(n, run_.max_weight, packing_, bound_);
    run_.counters = sssp_counters{};
    run_.counters.packed_parents = packing_.has_parents();
  }

  // ---- compute(): sospFromScratchCpu -----------------------------------------------------------

  void reset() {
    const auto n = static_cast<std::int64_t>(run_.graph.num_vertices);
    const vertex_t source = run_.source;
    u64* packed_words = ws_.packed.data();
    const packing<vertex_t> packed = packing_;
#pragma omp parallel for num_threads(threads_) schedule(static)
    for (std::int64_t v = 0; v < n; ++v) {
      packed_words[v] = v == source ? packed.pack(0, -1) : packed_inf;
    }
  }

  void seed_static() {
    ws_.frontier.assign(1, run_.source);
  }

  // ---- update(): sospUpdateCpu ------------------------------------------------------------------

  /// Pack the old tree, then Step 1: roots and subtree invalidation, and the insertion heads.
  void identify_affected() {
    const auto n = static_cast<std::int64_t>(run_.graph.num_vertices);
    const sssp_changes<vertex_t>& changes = *run_.changes;
    u64* packed_words = ws_.packed.data();
    char* state = ws_.state.data();
    int* stamp = ws_.stamp.data();
    const std::int64_t* distances = run_.distances;
    const vertex_t* parent = run_.parents;
    const packing<vertex_t> packed = packing_;
    const u64 bound = bound_;
    const vertex_t source = run_.source;

    // ---- Pack the old tree. ----
    bool overflow = false;
#pragma omp parallel for num_threads(threads_) schedule(static) reduction(|| : overflow)
    for (std::int64_t v = 0; v < n; ++v) {
      const std::int64_t d = distances[v];
      state[v] = 0;
      if (d >= sssp_infinity / 2) {
        packed_words[v] = packed_inf;
      } else if (d < 0 || static_cast<u64>(d) > bound) {
        overflow = true;
        packed_words[v] = packed_inf;
      } else {
        packed_words[v] = packed.pack(static_cast<u64>(d), parent[v]);
      }
    }
    DYNG_EXPECTS(!overflow, "sssp: an input distance exceeds (n - 1) * max weight");

    // ---- Step 1: roots and subtree invalidation. ----
    ws_.candidates.clear();
    const int generation = ws_.next_generation();
    const auto num_changed = static_cast<std::int64_t>(changes.changed_to.size());
    if (num_changed > 0) {
      const vertex_t* changed_from = changes.changed_from.data();
      const vertex_t* changed_to = changes.changed_to.data();
#pragma omp parallel for num_threads(threads_) schedule(static)
      for (std::int64_t i = 0; i < num_changed; ++i) {
        const vertex_t v = changed_to[i];
        if (parent[v] == changed_from[i]) {
          __atomic_store_n(&state[v], 2, __ATOMIC_RELAXED);
        }
      }
      auto load_state = [&](vertex_t v) { return __atomic_load_n(&state[v], __ATOMIC_RELAXED); };
      list_gather<vertex_t> invalid_gather(ws_.candidates, threads_);
      // A walk longer than n steps means the input tree has a parent cycle (e.g. a corrupt
      // imported tree): report it instead of looping forever.
      int cyclic = 0;
#pragma omp parallel num_threads(threads_)
      {
        // 0 unknown, 1 valid, 2 invalid (a root among the vertex and its ancestors). Concurrent
        // walks over a shared path write the same state, so relaxed atomics suffice.
#pragma omp for schedule(dynamic, 1024)
        for (std::int64_t v = 0; v < n; ++v) {
          if (load_state(static_cast<vertex_t>(v)) != 0 ||
              __atomic_load_n(&cyclic, __ATOMIC_RELAXED)) {
            continue;
          }
          auto u = static_cast<vertex_t>(v);
          std::int64_t steps = 0;
          while (load_state(u) == 0 && parent[u] >= 0 && steps <= n) {
            u = parent[u];
            ++steps;
          }
          if (steps > n) {
            __atomic_store_n(&cyclic, 1, __ATOMIC_RELAXED);
            continue;
          }
          const char result = load_state(u) == 0 ? char{1} : load_state(u);
          for (u = static_cast<vertex_t>(v); u >= 0 && load_state(u) == 0; u = parent[u]) {
            __atomic_store_n(&state[u], result, __ATOMIC_RELAXED);
          }
        }
        std::vector<vertex_t> local_invalid;
#pragma omp for schedule(static)
        for (std::int64_t v = 0; v < n; ++v) {
          if (state[v] == 2) {
            packed_words[v] = packed_inf;
            stamp[v] = generation;
            local_invalid.push_back(static_cast<vertex_t>(v));
          }
        }
        invalid_gather.gather(local_invalid);
      }
      DYNG_EXPECTS(cyclic == 0, "sssp: the input shortest-path tree has a parent cycle");
    }
    run_.counters.invalidated = static_cast<std::int64_t>(ws_.candidates.size());
    const auto num_heads = static_cast<std::int64_t>(changes.num_insert_heads);
    if (num_heads > 0) {
      const vertex_t* insert_heads = changes.insert_heads;
      list_gather<vertex_t> head_gather(ws_.candidates, threads_);
#pragma omp parallel num_threads(threads_)
      {
        std::vector<vertex_t> local_heads;
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < num_heads; ++i) {
          const vertex_t v = insert_heads[i];
          if (v != source && claim(stamp, v, generation)) {
            local_heads.push_back(v);
          }
        }
        head_gather.gather(local_heads);
      }
    }
  }

  /// Step 1: the pull pass.
  void seed() {
    ws_.frontier.clear();
    const int pull_generation = ws_.next_generation();
    const auto count = static_cast<std::int64_t>(ws_.candidates.size());
    const vertex_t* candidates = ws_.candidates.data();
    const auto& g = run_.graph;
    u64* packed_words = ws_.packed.data();
    int* stamp = ws_.stamp.data();
    const packing<vertex_t> packed = packing_;
    list_gather<vertex_t> frontier_gather(ws_.frontier, threads_);
#pragma omp parallel num_threads(threads_)
    {
      std::vector<vertex_t> local_frontier;
#pragma omp for schedule(dynamic, 64)
      for (std::int64_t i = 0; i < count; ++i) {
        const vertex_t v = candidates[i];
        const u64 current = load(&packed_words[v]);
        u64 best = current;
        for (edge_t e = g.in_row_ptr[v]; e < g.in_row_ptr[v + 1]; ++e) {
          const vertex_t u = g.in_col_ind[e];
          const u64 word = load(&packed_words[u]);
          if (word != packed_inf) {
            best = std::min(
                best, packed.pack(packed.distance(word) + static_cast<u64>(g.in_weights[e]), u));
          }
        }
        if (best < current) {
          const u64 old = atomic_min(&packed_words[v], best);
          if (packed.distance(best) < packed.distance(old) && claim(stamp, v, pull_generation)) {
            local_frontier.push_back(v);
          }
        }
      }
      frontier_gather.gather(local_frontier);
    }
  }

  /// Step 2: near-far propagation from ws.frontier.
  void loop() {
    if (ws_.frontier.empty()) {
      return;
    }
    const auto& g = run_.graph;
    const vertex_t source = run_.source;
    const u64 delta = static_cast<u64>(run_.delta);
    const packing<vertex_t> packed = packing_;
    sssp_counters& stats = run_.counters;
    u64* packed_words = ws_.packed.data();
    int* stamp = ws_.stamp.data();
    char* in_far = ws_.in_far.data();
    const u64 smallest = smallest_distance(ws_.frontier, ws_.packed, packed, threads_);
    u64 threshold = (smallest == packed_inf ? 0 : smallest) + delta;

    std::vector<vertex_t>* current = &ws_.near_a;
    std::vector<vertex_t>* next = &ws_.near_b;
    current->clear();
    ws_.far.clear();
    {
      const int generation = ws_.next_generation();
      const auto count = static_cast<std::int64_t>(ws_.frontier.size());
      const vertex_t* frontier = ws_.frontier.data();
      list_gather<vertex_t> near_gather(*current, threads_);
      list_gather<vertex_t> far_gather(ws_.far, threads_);
#pragma omp parallel num_threads(threads_)
      {
        std::vector<vertex_t> local_near;
        std::vector<vertex_t> local_far;
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < count; ++i) {
          const vertex_t v = frontier[i];
          const u64 word = load(&packed_words[v]);
          const u64 d = word == packed_inf ? packed_inf : packed.distance(word);
          if (d < threshold) {
            if (claim(stamp, v, generation)) {
              local_near.push_back(v);
            }
          } else if (__atomic_exchange_n(&in_far[v], 1, __ATOMIC_RELAXED) == 0) {
            local_far.push_back(v);
          }
        }
        near_gather.gather(local_near);
        far_gather.gather(local_far);
      }
    }

    while (true) {
      while (!current->empty()) {
        ++stats.iterations;
        stats.pushes += static_cast<std::int64_t>(current->size());
        const int generation = ws_.next_generation();
        const auto count = static_cast<std::int64_t>(current->size());
        const vertex_t* near = current->data();
        next->clear();
        list_gather<vertex_t> near_gather(*next, threads_);
        list_gather<vertex_t> far_gather(ws_.far, threads_);
#pragma omp parallel num_threads(threads_)
        {
          std::vector<vertex_t> local_near;
          std::vector<vertex_t> local_far;
#pragma omp for schedule(dynamic, 64)
          for (std::int64_t i = 0; i < count; ++i) {
            const vertex_t u = near[i];
            const u64 word = load(&packed_words[u]);
            if (word == packed_inf) {
              continue;
            }
            const u64 du = packed.distance(word);
            for (edge_t e = g.out_row_ptr[u]; e < g.out_row_ptr[u + 1]; ++e) {
              const vertex_t w = g.out_col_ind[e];
              if (w == source) {
                continue;
              }
              const u64 nd = du + static_cast<u64>(g.out_weights[e]);
              const u64 candidate = packed.pack(nd, u);
              if (candidate >= load(&packed_words[w])) {
                continue;
              }
              const u64 old = atomic_min(&packed_words[w], candidate);
              if (nd < packed.distance(old)) {
                if (nd < threshold) {
                  if (claim(stamp, w, generation)) {
                    local_near.push_back(w);
                  }
                } else if (__atomic_exchange_n(&in_far[w], 1, __ATOMIC_RELAXED) == 0) {
                  local_far.push_back(w);
                }
              }
            }
          }
          near_gather.gather(local_near);
          far_gather.gather(local_far);
        }
        std::swap(current, next);
      }
      if (ws_.far.empty()) {
        break;
      }
      // Raise the threshold past the far pile and re-split it.
      ++stats.epochs;
      threshold =
          std::max(threshold, smallest_distance(ws_.far, ws_.packed, packed, threads_)) + delta;
      const int generation = ws_.next_generation();
      const auto count = static_cast<std::int64_t>(ws_.far.size());
      const vertex_t* far = ws_.far.data();
      current->clear();
      ws_.far2.clear();
      list_gather<vertex_t> near_gather(*current, threads_);
      list_gather<vertex_t> keep_gather(ws_.far2, threads_);
#pragma omp parallel num_threads(threads_)
      {
        std::vector<vertex_t> local_near;
        std::vector<vertex_t> local_keep;
#pragma omp for schedule(static)
        for (std::int64_t i = 0; i < count; ++i) {
          const vertex_t v = far[i];
          const u64 word = load(&packed_words[v]);
          const u64 d = word == packed_inf ? packed_inf : packed.distance(word);
          if (d < threshold) {
            in_far[v] = 0;
            if (claim(stamp, v, generation)) {
              local_near.push_back(v);
            }
          } else {
            local_keep.push_back(v);
          }
        }
        near_gather.gather(local_near);
        keep_gather.gather(local_keep);
      }
      ws_.far.swap(ws_.far2);
    }
  }

  /// Write distances and parents from the packed words (and count the affected vertices).
  void finalize() {
    const auto n = static_cast<std::int64_t>(run_.graph.num_vertices);
    const auto& g = run_.graph;
    const vertex_t source = run_.source;
    const packing<vertex_t> packed = packing_;
    const u64* packed_words = ws_.packed.data();
    std::int64_t* distances = run_.distances;
    vertex_t* parent = run_.parents;
    std::int64_t affected = 0;
    if (packed.has_parents()) {
#pragma omp parallel for num_threads(threads_) schedule(static) reduction(+ : affected)
      for (std::int64_t v = 0; v < n; ++v) {
        const u64 word = packed_words[v];
        const std::int64_t d =
            word == packed_inf ? sssp_infinity : static_cast<std::int64_t>(packed.distance(word));
        const vertex_t p = word == packed_inf ? vertex_t{-1} : packed.parent(word);
        affected += (d != distances[v] || p != parent[v]) ? 1 : 0;
        distances[v] = d;
        parent[v] = p;
      }
      run_.counters.affected = affected;
      return;
    }
    // Distance-only words: recover the lowest-id parent over tight edges.
    ws_.saved_parents.resize(static_cast<std::size_t>(n));
    vertex_t* saved = ws_.saved_parents.data();
    std::vector<char>& distance_changed = ws_.state;
#pragma omp parallel for num_threads(threads_) schedule(static)
    for (std::int64_t v = 0; v < n; ++v) {
      const u64 word = packed_words[v];
      const std::int64_t d = word == packed_inf ? sssp_infinity : static_cast<std::int64_t>(word);
      distance_changed[static_cast<std::size_t>(v)] = d != distances[v] ? 1 : 0;
      saved[v] = parent[v];
      distances[v] = d;
      parent[v] =
          word == packed_inf || v == source ? vertex_t{-1} : std::numeric_limits<vertex_t>::max();
    }
#pragma omp parallel for num_threads(threads_) schedule(dynamic, 256)
    for (std::int64_t u = 0; u < n; ++u) {
      const u64 du = packed_words[u];
      if (du == packed_inf) {
        continue;
      }
      for (edge_t e = g.out_row_ptr[u]; e < g.out_row_ptr[u + 1]; ++e) {
        const vertex_t w = g.out_col_ind[e];
        if (w != source && du + static_cast<u64>(g.out_weights[e]) == packed_words[w]) {
          atomic_min_id(&parent[w], static_cast<vertex_t>(u));
        }
      }
    }
#pragma omp parallel for num_threads(threads_) schedule(static) reduction(+ : affected)
    for (std::int64_t v = 0; v < n; ++v) {
      affected +=
          (distance_changed[static_cast<std::size_t>(v)] != 0 || saved[v] != parent[v]) ? 1 : 0;
    }
    run_.counters.affected = affected;
  }

 private:
  int threads_;
  sssp_run<vertex_t, edge_t, weight_t>& run_;
  sssp_workspace<vertex_t>& ws_;
  packing<vertex_t> packing_{0, 0};
  u64 bound_ = 0;
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_update(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  if (run.graph.num_vertices == 0) {
    run.counters = sssp_counters{};
    return;
  }
  openmp_problem<vertex_t, edge_t, weight_t> problem(res, run);
  sssp_enact_update(res, problem);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_compute(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  DYNG_EXPECTS(run.source >= 0 && run.source < run.graph.num_vertices, "sssp: source ", run.source,
               " is out of range [0, ", run.graph.num_vertices, ")");
  openmp_problem<vertex_t, edge_t, weight_t> problem(res, run);
  sssp_enact_compute(res, problem);
}

#define DYNG_INSTANTIATE_SSSP_OPENMP(V, E, W)                                      \
  template void sssp_openmp_update<V, E, W>(const resources&, sssp_run<V, E, W>&); \
  template void sssp_openmp_compute<V, E, W>(const resources&, sssp_run<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_OPENMP)
#undef DYNG_INSTANTIATE_SSSP_OPENMP

}  // namespace dyng::detail

#else  // !DYNG_HAS_OPENMP

namespace dyng::detail {

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_update(const resources&, sssp_run<vertex_t, edge_t, weight_t>&) {
  throw not_supported_error("dyng: sssp: the openmp backend is not built; available: sequential");
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_openmp_compute(const resources&, sssp_run<vertex_t, edge_t, weight_t>&) {
  throw not_supported_error("dyng: sssp: the openmp backend is not built; available: sequential");
}

#define DYNG_INSTANTIATE_SSSP_OPENMP(V, E, W)                                      \
  template void sssp_openmp_update<V, E, W>(const resources&, sssp_run<V, E, W>&); \
  template void sssp_openmp_compute<V, E, W>(const resources&, sssp_run<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_OPENMP)
#undef DYNG_INSTANTIATE_SSSP_OPENMP

}  // namespace dyng::detail

#endif  // DYNG_HAS_OPENMP
