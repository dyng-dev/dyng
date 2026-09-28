// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/mospUpdate.cpp (canonicalizeTree, the per-objective
// preparation of mospUpdate) and src/sospUpdateCpu.cpp (SospWorkspace::reserve / nextGeneration,
// defaultDelta, the choosePacking precondition)
/**
 * @file sssp.cpp
 * @brief sssp: argument validation, version checks, backend dispatch and result bookkeeping
 *        (compute, update, result, from_arrays) and the explicit instantiations.
 *
 * The per-objective preparation follows MOSP-OpenMP@c352151 mospUpdate(): the largest weight
 * (original graph and batch insertions, at least 1) and the weight sum of the original graph give
 * the packing bound and the default near-far width; the change list of the objective is every
 * deletion plus every insertion that raised the objective's weight (apply_delta), and the
 * insertion heads are the heads of all insertions.
 */
#include "algorithms/sssp/problem.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/device_fill.hpp"

#include <dyng/config.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <climits>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail {

// ------------------------------------------------------------------------------------------------
// Workspace (MOSP-OpenMP@c352151:src/sospUpdateCpu.cpp SospWorkspace::reserve / nextGeneration)
// ------------------------------------------------------------------------------------------------

template <typename vertex_t>
void sssp_workspace<vertex_t>::reserve(std::int64_t requested) {
  if (requested <= capacity) {
    return;
  }
  const auto n = static_cast<std::size_t>(requested);
  packed.assign(n, ~0ULL);
  stamp.assign(n, 0);
  in_far.assign(n, 0);
  state.assign(n, 0);
  for (std::vector<vertex_t>* list : {&near_a, &near_b, &far, &far2, &candidates, &frontier}) {
    list->clear();
    list->reserve(n);
  }
  capacity = requested;
  generation = 0;
}

template <typename vertex_t>
std::size_t sssp_workspace<vertex_t>::bytes() const noexcept {
  const auto bytes_of = [](const auto& v) { return v.capacity() * sizeof(v[0]); };
  std::size_t total = bytes_of(packed) + bytes_of(stamp) + bytes_of(in_far) + bytes_of(state) +
                      bytes_of(old_distance);
  for (const std::vector<vertex_t>* list :
       {&near_a, &near_b, &far, &far2, &candidates, &frontier, &saved_parents, &child_start,
        &children, &invalid, &affected, &candidate_list, &touched, &old_parent, &changed_from,
        &changed_to}) {
    total += bytes_of(*list);
  }
  return total + thread_list_bytes();
}

template <typename vertex_t>
std::size_t sssp_workspace<vertex_t>::thread_list_bytes() const noexcept {
  std::size_t total = thread_lists.capacity() * sizeof(padded_thread_list<vertex_t>);
  for (const padded_thread_list<vertex_t>& list : thread_lists) {
    total += (list.items.capacity() * sizeof(vertex_t) + cache_line_bytes - 1) / cache_line_bytes *
             cache_line_bytes;
  }
  return total;
}

/// Lease the workspace of the handle's pool and size it for `n` vertices (profiler stage
/// sssp.workspace; a no-op once the pooled workspace is large enough, ADR 0015). MOSP reserves its
/// one SospWorkspace in mospUpdate()'s "prepare"; the frontier lists are only reserved, so their
/// pages are first touched by the first run that uses them, as in the original.
template <typename vertex_t>
workspace_pool::lease<sssp_workspace<vertex_t>> lease_workspace(const resources& res,
                                                                std::int64_t n) {
  scoped_stage stage(res, "sssp.workspace");
  auto ws = resources_access::workspaces(res).acquire<sssp_workspace<vertex_t>>(res);
  ws->reserve(n);
  return ws;
}

/// Lease the device workspace of a CUDA handle's pool and size it for `n` vertices (profiler stage
/// sssp.workspace; MOSP-CUDA reserves its one SospWorkspace in mospUpdate()'s "upload").
template <typename vertex_t>
workspace_pool::lease<sssp_cuda_workspace<vertex_t>> lease_cuda_workspace(const resources& res,
                                                                          std::int64_t n) {
  scoped_stage stage(res, "sssp.workspace");
  auto ws = resources_access::workspaces(res).acquire<sssp_cuda_workspace<vertex_t>>(res);
  ws->reserve(res, n);
  return ws;
}

template <typename vertex_t>
int sssp_workspace<vertex_t>::next_generation() {
  if (generation == INT_MAX) {
    std::fill(stamp.begin(), stamp.end(), 0);
    generation = 0;
  }
  return ++generation;
}

template struct sssp_workspace<std::int32_t>;
template struct sssp_workspace<std::int64_t>;

std::int64_t sssp_default_delta(std::int64_t num_edges, std::int64_t num_vertices,
                                std::int64_t weight_sum) {
  if (num_edges <= 0 || num_vertices <= 0) {
    return 1;
  }
  const double average_weight = static_cast<double>(weight_sum) / static_cast<double>(num_edges);
  const double average_degree = static_cast<double>(num_edges) / static_cast<double>(num_vertices);
  return std::max<std::int64_t>(1,
                                static_cast<std::int64_t>(32.0 * average_weight / average_degree));
}

bool sssp_distances_fit(std::int64_t num_vertices, std::int64_t max_weight) {
  const auto weight = static_cast<std::uint64_t>(std::max<std::int64_t>(max_weight, 1));
  const auto hops = static_cast<std::uint64_t>(std::max<std::int64_t>(num_vertices - 1, 1));
  const auto output_max = static_cast<std::uint64_t>(sssp_infinity / 2 - 1);
  return weight <= output_max / hops;
}

bool sssp_packs_parents(std::int64_t num_vertices, std::int64_t max_weight) {
  // make_packing(n, bound + weight) of openmp.cpp (MOSP's choosePacking), without the engine.
  const auto weight = static_cast<std::uint64_t>(std::max<std::int64_t>(max_weight, 1));
  const auto hops = static_cast<std::uint64_t>(std::max<std::int64_t>(num_vertices - 1, 1));
  int bits = 1;
  while ((1ULL << bits) - 1 < static_cast<std::uint64_t>(num_vertices)) {
    ++bits;
  }
  const std::uint64_t max_distance = (~0ULL >> bits) - 1;
  const std::uint64_t largest_candidate = weight * hops + weight;
  return weight <= max_distance / hops && largest_candidate <= max_distance;
}

template <typename vertex_t, typename distance_t>
sssp_state<vertex_t, distance_t>& sssp_access::state(sssp::result<vertex_t, distance_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "sssp::result: use of a moved-from result");
  return *r.impl_;
}

template <typename vertex_t, typename distance_t>
sssp::result<vertex_t, distance_t> sssp_access::make(
    std::unique_ptr<sssp_state<vertex_t, distance_t>> state) {
  return sssp::result<vertex_t, distance_t>(std::move(state));
}

namespace {

bool is_cuda(const resources& res) noexcept {
  return res.get_backend() == backend::cuda;
}

/// The backends sssp runs on in this build.
void expect_supported_backend(const resources& res, const char* what) {
  const backend b = res.get_backend();
  if (b == backend::cuda && !DYNG_HAS_CUDA) {
    throw not_supported_error(std::string("dyng: ") + what +
                              ": the cuda backend is not built; available: sequential, openmp");
  }
}

/// The engine of the CUDA backend (PLAN Section 4.5.4): engine::automatic and engine::fused run
/// the fused persistent cooperative kernel; without cooperative launch there is no engine in this
/// release (the operators engine arrives in 0.2, decision O24), so both throw, and so does
/// engine::operators. Checked before anything is changed.
void select_cuda_engine(const resources& res, engine requested, const char* what) {
  if (!is_cuda(res)) {
    return;
  }
  if (requested == engine::operators) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": engine::operators of sssp on the cuda backend arrives in 0.2 (the multi-kernel "
        "engine); use engine::fused or engine::automatic"));
  }
  const cuda_device_properties& props = resources_access::device_properties(res);
  if (!props.cooperative_launch) {
    throw not_supported_error(concat_message(
        "dyng: ", what, ": the fused sssp engine needs cooperative launch, which CUDA device ",
        props.ordinal, " (", props.name, ") does not support",
        requested == engine::automatic
            ? "; engine::automatic has no other CUDA engine in this release (the operators engine "
              "arrives in 0.2)"
            : "",
        "; run sssp with resources::openmp() or resources::sequential() instead"));
  }
}

/// Host threads of the host-side work of a call (column summaries, tree import and checks).
int host_threads(const resources& res) noexcept {
  return resources_access::host_threads(res);
}

}  // namespace

/// A failed update leaves the arrays inconsistent: every later use throws until the result is
/// rebuilt (PLAN Section 4.7.3).
template <typename vertex_t, typename distance_t>
void expect_not_poisoned(const sssp_state<vertex_t, distance_t>& state, const char* what) {
  if (state.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", what,
        ": the result was left unusable by a failed update; recompute it with sssp::compute() "
        "(or import a valid tree with sssp::result::from_arrays())"));
  }
}

namespace {

/// A result lives in the memory of the backend class that built it (host arrays for sequential
/// and openmp, device arrays of one device for cuda); clone(res) moves it.
template <typename vertex_t, typename distance_t>
void expect_result_placement(const resources& res, const sssp_state<vertex_t, distance_t>& st,
                             const char* what) {
  if (is_cuda(res)) {
    DYNG_EXPECTS(
        st.space == memory_space::device && st.device == res.device(), what, ": the result is in ",
        st.space == memory_space::host ? std::string("host memory")
                                       : concat_message("CUDA device ", st.device),
        " but the resources are cuda on device ", res.device(), "; copy it with r.clone(res)");
  } else {
    DYNG_EXPECTS(st.space == memory_space::host, what, ": the result is on CUDA device ", st.device,
                 " but the resources are ", to_string(res.get_backend()),
                 "; copy it with r.clone(res)");
  }
}

void expect_options(const sssp::options& opt) {
  DYNG_EXPECTS(opt.delta >= 0, "sssp: options.delta must be >= 0 (0 = automatic), got ", opt.delta);
  DYNG_EXPECTS(opt.objective >= 0, "sssp: options.objective must be >= 0, got ", opt.objective);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void expect_graph(const graph<vertex_t, edge_t, weight_t>& g, const sssp::options& opt,
                  const char* what) {
  DYNG_EXPECTS(g.has_transposed(), what,
               ": sssp needs the in-edges; build the graph with "
               "graph_properties::store_transposed = true");
  DYNG_EXPECTS(opt.objective < g.num_weights(), what, ": options.objective ", opt.objective,
               " is out of range for a graph with ", g.num_weights(), " weight column(s)");
}

/// Largest weight (at least 1) and weight sum of one objective column; every weight must be >= 1.
struct column_summary {
  std::int64_t max_weight = 1;
  std::int64_t weight_sum = 0;
  std::int64_t first_bad = -1;
};

template <typename weight_t>
column_summary summarize_column(const resources& res, array_view<const weight_t> column) {
  const auto m = static_cast<std::int64_t>(column.size());
  const weight_t* w = column.data();
  std::int64_t largest = 1;
  std::int64_t sum = 0;
  std::int64_t bad = m;
#if DYNG_HAS_OPENMP
  if (host_threads(res) > 1) {
    const int threads = host_threads(res);
#pragma omp parallel for num_threads(threads) schedule(static) reduction(max : largest) \
    reduction(+ : sum) reduction(min : bad)
    for (std::int64_t e = 0; e < m; ++e) {
      const auto value = static_cast<std::int64_t>(w[e]);
      largest = std::max(largest, value);
      sum += value;
      if (value < 1) {
        bad = std::min(bad, e);
      }
    }
    return column_summary{largest, sum, bad == m ? -1 : bad};
  }
#endif
  (void)res;
  for (std::int64_t e = 0; e < m; ++e) {
    const auto value = static_cast<std::int64_t>(w[e]);
    largest = std::max(largest, value);
    sum += value;
    if (value < 1 && bad == m) {
      bad = e;
    }
  }
  return column_summary{largest, sum, bad == m ? -1 : bad};
}

/// One objective's view of the graph. The in-edges are built (graph_access::view) only when
/// `with_in_edges`; otherwise the in-edge pointers stay null.
template <typename vertex_t, typename edge_t, typename weight_t>
sssp_graph<vertex_t, edge_t, weight_t> objective_graph(const resources& res,
                                                       const graph<vertex_t, edge_t, weight_t>& g,
                                                       int objective, bool with_in_edges = true) {
  sssp_graph<vertex_t, edge_t, weight_t> out;
  const auto out_edges = graph_access::out_view(g);
  out.num_vertices = out_edges.num_vertices();
  out.out_row_ptr = out_edges.row_ptr.data();
  out.out_col_ind = out_edges.col_ind.data();
  out.out_weights = out_edges.weight_column(objective).data();
  if (with_in_edges) {
    const auto view = graph_access::view(res, g);
    out.in_row_ptr = view.in.row_ptr.data();
    out.in_col_ind = view.in.col_ind.data();
    out.in_weights = view.in.weight_column(objective).data();
  }
  return out;
}

/// One objective's view of the device copy of a CUDA graph (MOSP's DeviceGraph::out(k) and in(k));
/// uploads the graph state first if it is not resident yet.
template <typename vertex_t, typename edge_t, typename weight_t>
sssp_graph<vertex_t, edge_t, weight_t> device_objective_graph(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g, int objective) {
  const device_graph<vertex_t, edge_t, weight_t>& d = graph_access::device(res, g);
  sssp_graph<vertex_t, edge_t, weight_t> out;
  out.num_vertices = d.num_vertices;
  out.out_row_ptr = d.out_row_ptr.data();
  out.out_col_ind = d.out_col_ind.data();
  out.out_weights = d.out_column(objective);
  out.in_row_ptr = d.in_row_ptr.data();
  out.in_col_ind = d.in_col_ind.data();
  out.in_weights = d.in_column(objective);
  return out;
}

/// Copy `n` elements of a host array into a device array, ordered on the stream of `res` (from
/// pageable memory: the host array may be reused as soon as the call returns).
template <typename value_t>
void upload(const resources& res, value_t* device, const value_t* host, std::size_t n) {
  if (n > 0) {
    copy_bytes(device, memory_space::device, host, memory_space::host, n * sizeof(value_t),
               res.stream(), res.device());
  }
}

/// Copy an array of any memory space into a host vector (from_arrays of device inputs).
template <typename value_t>
void assign_host(const resources& res, std::vector<value_t>& out, array_view<const value_t> in) {
  if (is_host_accessible(in.space())) {
    out.assign(in.begin(), in.end());
    return;
  }
  out.resize(in.size());
  if (!in.empty()) {
    const int device = in.device() >= 0 ? in.device() : res.device();
    const stream_ref stream = is_cuda(res) ? res.stream() : stream_ref{};
    copy_bytes(out.data(), memory_space::host, in.data(), in.space(), in.size_bytes(), stream,
               device);
    cuda_synchronize(device, stream);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void run_update_engine(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  if (res.get_backend() == backend::openmp) {
    sssp_openmp_update(res, run);
  } else {
    sssp_sequential_update(res, run);
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void run_compute_engine(const resources& res, sssp_run<vertex_t, edge_t, weight_t>& run) {
  if (res.get_backend() == backend::openmp) {
    sssp_openmp_compute(res, run);
  } else {
    sssp_sequential_compute(res, run);
  }
}

/// MOSP's canonicalizeTree(): for every edge (u,v) with dist[u] + w(u,v) == dist[v] and
/// u < parent[v], parent[v] becomes u.
template <typename vertex_t, typename edge_t, typename weight_t>
void canonicalize_tree(const sssp_graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                       const std::vector<std::int64_t>& distances, std::vector<vertex_t>& parent) {
  const std::int64_t n = g.num_vertices;
  for (std::int64_t u = 0; u < n; ++u) {
    const std::int64_t du = distances[static_cast<std::size_t>(u)];
    if (du >= sssp_infinity / 2) {
      continue;
    }
    for (edge_t e = g.out_row_ptr[u]; e < g.out_row_ptr[u + 1]; ++e) {
      const vertex_t v = g.out_col_ind[e];
      auto& pv = parent[static_cast<std::size_t>(v)];
      if (v != source && u < pv &&
          du + static_cast<std::int64_t>(g.out_weights[e]) ==
              distances[static_cast<std::size_t>(v)]) {
        pv = static_cast<vertex_t>(u);
      }
    }
  }
}

/// The O(n) checks of options::validate_inputs (rooted at the source, no parent cycle), one vertex
/// after the other; throws the message of the first violation in vertex order.
template <typename vertex_t>
void validate_tree_sequential(vertex_t source, const std::vector<std::int64_t>& distances,
                              const std::vector<vertex_t>& parents, std::int64_t bound) {
  const auto n = static_cast<std::int64_t>(distances.size());
  const auto s = static_cast<std::size_t>(source);
  DYNG_EXPECTS(distances[s] == 0, "sssp: the source ", source, " must have distance 0, got ",
               distances[s]);
  DYNG_EXPECTS(parents[s] == -1, "sssp: the source ", source, " must have parent -1, got ",
               parents[s]);
  for (std::int64_t v = 0; v < n; ++v) {
    const auto i = static_cast<std::size_t>(v);
    const std::int64_t d = distances[i];
    const vertex_t p = parents[i];
    if (d >= sssp_infinity / 2) {
      DYNG_EXPECTS(p == -1, "sssp: unreachable vertex ", v, " has parent ", p, " (expected -1)");
      continue;
    }
    DYNG_EXPECTS(d >= 0 && d <= bound, "sssp: the distance ", d, " of vertex ", v,
                 " is outside [0, (n - 1) * max weight = ", bound, "]");
    if (v == static_cast<std::int64_t>(source)) {
      continue;
    }
    DYNG_EXPECTS(p >= 0, "sssp: reachable vertex ", v, " (distance ", d, ") has no parent");
    DYNG_EXPECTS(distances[static_cast<std::size_t>(p)] < sssp_infinity / 2, "sssp: vertex ", v,
                 " has the unreachable parent ", p);
  }
  // Every parent chain must end at the source: 0 unknown, 1 on the current walk, 2 rooted.
  std::vector<char> state(static_cast<std::size_t>(n), 0);
  state[s] = 2;
  std::vector<vertex_t> path;
  for (std::int64_t v = 0; v < n; ++v) {
    if (state[static_cast<std::size_t>(v)] != 0 ||
        distances[static_cast<std::size_t>(v)] >= sssp_infinity / 2) {
      continue;
    }
    path.clear();
    auto u = static_cast<vertex_t>(v);
    while (state[static_cast<std::size_t>(u)] == 0) {
      state[static_cast<std::size_t>(u)] = 1;
      path.push_back(u);
      u = parents[static_cast<std::size_t>(u)];
    }
    DYNG_EXPECTS(state[static_cast<std::size_t>(u)] == 2,
                 "sssp: the input shortest-path tree has a parent cycle through vertex ", u);
    for (const vertex_t x : path) {
      state[static_cast<std::size_t>(x)] = 2;
    }
  }
}

/// validate_tree_sequential() with OpenMP: the same checks in parallel (per-vertex conditions,
/// then a chain walk that marks every vertex whose parent chain reaches the source, as the OpenMP
/// engine's invalidation does). A valid tree passes without the sequential pass; on any violation
/// the sequential checks run and throw, so the message does not depend on the thread count.
template <typename vertex_t>
void validate_tree(const resources& res, vertex_t source,
                   const std::vector<std::int64_t>& distances, const std::vector<vertex_t>& parents,
                   std::int64_t bound, std::vector<char>& scratch) {
#if DYNG_HAS_OPENMP
  const auto n = static_cast<std::int64_t>(distances.size());
  if (host_threads(res) > 1 && n >= 4096) {
    const int threads = host_threads(res);
    const std::int64_t* d = distances.data();
    const vertex_t* p = parents.data();
    bool bad = d[source] != 0 || p[source] != -1;
#pragma omp parallel for num_threads(threads) schedule(static) reduction(|| : bad)
    for (std::int64_t v = 0; v < n; ++v) {
      const std::int64_t dv = d[v];
      const vertex_t pv = p[v];
      if (dv >= sssp_infinity / 2) {
        bad = bad || pv != -1;
      } else if (dv < 0 || dv > bound) {
        bad = true;
      } else if (v != static_cast<std::int64_t>(source)) {
        bad = bad || pv < 0 || d[pv] >= sssp_infinity / 2;
      }
    }
    if (!bad) {
      // 0 unknown, 1 rooted. Every reachable vertex has a reachable parent now, so a walk from a
      // reachable vertex either meets a rooted vertex (the source first of all) or cycles; a walk
      // longer than n steps means a cycle. Concurrent walks over a shared path write the same
      // state, so relaxed atomics suffice.
      scratch.assign(static_cast<std::size_t>(n), 0);
      char* state = scratch.data();
      state[source] = 1;
      int cyclic = 0;
      auto load_state = [state](vertex_t v) {
        return __atomic_load_n(&state[v], __ATOMIC_RELAXED);
      };
#pragma omp parallel for num_threads(threads) schedule(dynamic, 1024)
      for (std::int64_t v = 0; v < n; ++v) {
        if (d[v] >= sssp_infinity / 2 || load_state(static_cast<vertex_t>(v)) != 0 ||
            __atomic_load_n(&cyclic, __ATOMIC_RELAXED) != 0) {
          continue;
        }
        auto u = static_cast<vertex_t>(v);
        std::int64_t steps = 0;
        while (load_state(u) == 0 && steps <= n) {
          u = p[u];
          ++steps;
        }
        if (steps > n) {
          __atomic_store_n(&cyclic, 1, __ATOMIC_RELAXED);
          continue;
        }
        for (u = static_cast<vertex_t>(v); load_state(u) == 0; u = p[u]) {
          __atomic_store_n(&state[u], char{1}, __ATOMIC_RELAXED);
        }
      }
      if (cyclic == 0) {
        return;
      }
    }
  }
#else
  (void)res;
  (void)scratch;
#endif
  validate_tree_sequential(source, distances, parents, bound);
}

/// The always-on part of importing a tree (from_arrays): distances at or above half of MOSP's
/// DISTANCE_INF become sssp_infinity (MOSP writes and reads them as "INF"), and every parent id
/// must lie in [-1, n) (memory safety of the engines, with or without validate_inputs). In
/// parallel on the OpenMP backend; the error names the first bad vertex either way.
template <typename vertex_t>
void normalize_imported_tree(const resources& res, std::vector<std::int64_t>& distances,
                             const std::vector<vertex_t>& parents) {
  const auto n = static_cast<std::int64_t>(distances.size());
  std::int64_t* d = distances.data();
  const vertex_t* p = parents.data();
  std::int64_t first_bad = n;
  const int threads = host_threads(res);
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (threads > 1 && n >= 65536) schedule(static) \
    reduction(min : first_bad)
#else
  (void)threads;
#endif
  for (std::int64_t v = 0; v < n; ++v) {
    if (d[v] >= sssp_infinity / 2) {
      d[v] = sssp_infinity;
    }
    if ((p[v] < -1 || p[v] >= n) && v < first_bad) {
      first_bad = v;
    }
  }
  DYNG_EXPECTS(first_bad == n, "sssp::result::from_arrays: vertex ", first_bad, " has parent ",
               first_bad < n ? static_cast<std::int64_t>(p[first_bad]) : 0, ", outside [-1, ", n,
               ")");
}

/// The participant of one sssp result in run_update() (and dyng::update()).
template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
class sssp_participant final : public update_participant<vertex_t, edge_t, weight_t> {
  static_assert(std::is_same_v<distance_t, std::int64_t>, "sssp: distance_t must be int64_t");

 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;
  using batch_type = edge_batch_view<vertex_t, weight_t>;

  sssp_participant(sssp::result<vertex_t, distance_t>& r, sssp::stats& out)
      : result_(r), out_(out) {}

  [[nodiscard]] const void* target() const noexcept override {
    return &result_;
  }

  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    expect_supported_backend(res, "sssp::update");
    state_ = &sssp_access::state(result_);
    expect_not_poisoned(*state_, "sssp::update");
    if (state_->version != g.version()) {
      throw stale_result_error(detail::concat_message(
          "dyng: sssp::update: the result matches graph version ", state_->version,
          " but the graph is at version ", g.version(),
          " (the graph was changed without updating this result; recompute it, or update all "
          "results together with dyng::update(res, g, batch, results...))"));
    }
    if (state_->graph_state != graph_access::impl(g).state_id) {
      throw stale_result_error(detail::concat_message(
          "dyng: sssp::update: the result was computed on another graph (or on an earlier state "
          "of a graph variable that was reassigned since), although both are at version ",
          g.version(), "; recompute it on this graph"));
    }
    const sssp::options& opt = state_->opt;
    expect_graph(g, opt, "sssp::update");
    graph_access::expect_placement(res, g, "sssp::update");
    expect_result_placement(res, *state_, "sssp::update");
    select_cuda_engine(res, opt.cuda_engine, "sssp::update");
    DYNG_EXPECTS(static_cast<std::int64_t>(state_->num_vertices()) ==
                     static_cast<std::int64_t>(g.num_vertices()),
                 "sssp::update: the result has ", state_->num_vertices(),
                 " vertices but the graph has ", g.num_vertices(),
                 " (is it a result of this graph?)");

    // ---- prepare (on G_t, as MOSP's mospUpdate() does before applying the batch) ----
    scoped_stage stage(res, "sssp.prepare");
    const int k = opt.objective;
    const int num_weights = g.num_weights();
    DYNG_EXPECTS(batch.num_weights == num_weights, "sssp::update: the batch has ",
                 batch.num_weights, " weight(s) per insertion but the graph has ", num_weights);
    const std::size_t num_inserts = batch.num_insertions();
    DYNG_EXPECTS(batch.insert_weights.size() == num_inserts * static_cast<std::size_t>(num_weights),
                 "sssp::update: the batch has ", batch.insert_weights.size(),
                 " insertion weights for ", num_inserts, " insertions of ", num_weights,
                 " weight(s)");
    DYNG_EXPECTS(batch.insert_weights.empty() || is_host_accessible(batch.insert_weights.space()),
                 "sssp::update: the batch must be in host memory");
    DYNG_EXPECTS(batch.insert_dst.empty() || is_host_accessible(batch.insert_dst.space()),
                 "sssp::update: the batch must be in host memory");
    std::int64_t largest = 1;
    std::int64_t max_id = -1;
    for (std::size_t i = 0; i < num_inserts; ++i) {
      const auto w =
          static_cast<std::int64_t>(batch.insert_weights[i * static_cast<std::size_t>(num_weights) +
                                                         static_cast<std::size_t>(k)]);
      DYNG_EXPECTS(w >= 1, "sssp::update: insertion ", i, " has weight ", w, " in objective ", k,
                   "; sssp needs weights in [1, 2^31 - 1]");
      largest = std::max(largest, w);
      if (i < batch.insert_src.size() && i < batch.insert_dst.size()) {
        max_id = std::max<std::int64_t>(
            max_id, std::max<std::int64_t>(batch.insert_src[i], batch.insert_dst[i]));
      }
    }
    const column_summary column = summarize_column(res, graph_access::out_view(g).weight_column(k));
    max_weight_ = std::max(largest, column.max_weight);
    const std::int64_t n = g.num_vertices();
    const std::int64_t m = g.num_edges();
    delta_ = opt.delta > 0 ? opt.delta
                           : sssp_default_delta(std::max<std::int64_t>(m, 1), n, column.weight_sum);
    std::int64_t n_after = n;
    if (g.properties().semantics.allow_vertex_growth) {
      n_after = std::max(n_after, max_id + 1);
    }
    DYNG_EXPECTS(sssp_distances_fit(n_after, max_weight_), "sssp::update: distances up to ",
                 max_weight_, " * ", n_after - 1, " do not fit in 62 bits");
  }

  void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                   const apply_delta<vertex_t>& delta) override {
    sssp_state<vertex_t, distance_t>& st = *state_;
    const auto n = static_cast<std::size_t>(g.num_vertices());
    // New vertices (vertex growth) start unreachable.
    if (n > st.num_vertices()) {
      grow(res, st, n);
    }
    DYNG_EXPECTS(sssp_distances_fit(static_cast<std::int64_t>(n), max_weight_),
                 "sssp::update: distances up to ", max_weight_, " * ", n - 1,
                 " do not fit in 62 bits");
    sssp_counters counters =
        is_cuda(res) ? update_cuda(res, g, delta, st) : update_host(res, g, delta, st);

    sssp::stats s;
    s.affected = counters.affected;
    s.iterations = counters.iterations;
    s.frontier_visits = counters.pushes;
    s.fallback_used = false;
    s.converged = true;
    s.engine_used = res.get_backend() == backend::sequential ? engine::operators : engine::fused;
    s.batch = summary;
    s.invalidated = counters.invalidated;
    s.epochs = counters.epochs;
    s.pushes = counters.pushes;
    s.packed_parents = counters.packed_parents;
    out_ = s;
    st.version = g.version();
    st.graph_state = graph_access::impl(g).state_id;
  }

  void poison() noexcept override {
    if (state_ != nullptr) {
      state_->poisoned = true;
    }
  }

 private:
  /// The change list of objective k: every deletion, then every insertion that raised the
  /// objective's weight (MOSP's weightIncreaseMask, one byte per insertion and objective here);
  /// the insertion heads are the heads of all insertions.
  static void build_changes(const apply_delta<vertex_t>& delta, int k,
                            std::vector<vertex_t>& changed_from,
                            std::vector<vertex_t>& changed_to) {
    changed_from.assign(delta.delete_src.begin(), delta.delete_src.end());
    changed_to.assign(delta.delete_dst.begin(), delta.delete_dst.end());
    const std::size_t num_ins = delta.insert_src.size();
    const auto num_objectives = static_cast<std::size_t>(delta.num_weights);
    for (std::size_t i = 0; i < num_ins; ++i) {
      if (delta.weight_increased[i * num_objectives + static_cast<std::size_t>(k)] != 0) {
        changed_from.push_back(delta.insert_src[i]);
        changed_to.push_back(delta.insert_dst[i]);
      }
    }
  }

  /// Vertex growth: the new vertices are unreachable (INF, parent -1).
  static void grow(const resources& res, sssp_state<vertex_t, distance_t>& st, std::size_t n) {
    if (st.space == memory_space::host) {
      st.distances.resize(n, sssp_infinity);
      st.parents.resize(n, vertex_t{-1});
      return;
    }
#if DYNG_HAS_CUDA
    const std::size_t old_n = st.num_vertices();
    buffer<distance_t> distances(res, n);
    buffer<vertex_t> parents(res, n);
    copy_bytes(distances.data(), memory_space::device, st.device_distances.data(),
               memory_space::device, old_n * sizeof(distance_t), res.stream(), res.device());
    copy_bytes(parents.data(), memory_space::device, st.device_parents.data(), memory_space::device,
               old_n * sizeof(vertex_t), res.stream(), res.device());
    {
      const scoped_device guard(res.device());
      fill_async(res.stream(), distances.data() + old_n, n - old_n, distance_t{sssp_infinity});
      fill_async(res.stream(), parents.data() + old_n, n - old_n, vertex_t{-1});
    }
    // The old arrays are released on res.stream(), after the copies above that read them: the
    // result may have been computed (and its arrays allocated) on another stream, whose earlier
    // work the caller has ordered before this call (PLAN 4.7.4). Released on their own stream they
    // could be reused by another allocation before the copies ran.
    st.device_distances.set_stream(res.stream());
    st.device_parents.set_stream(res.stream());
    st.device_distances = std::move(distances);
    st.device_parents = std::move(parents);
#else
    (void)res;
    DYNG_FAIL("sssp: a device result without the cuda backend");
#endif
  }

  /// Host backends: the objective's change list in the pooled workspace, then the engine.
  sssp_counters update_host(const resources& res, const graph_type& g,
                            const apply_delta<vertex_t>& delta,
                            sssp_state<vertex_t, distance_t>& st) const {
    const int k = st.opt.objective;
    // The scratch memory of the engines, shared with the other results run through `res` (the K
    // objectives of dyng::update_each() use one workspace one after the other, as MOSP's
    // mospUpdate() shares its SospWorkspace; ADR 0015).
    auto ws = lease_workspace<vertex_t>(res, static_cast<std::int64_t>(g.num_vertices()));
    // Kept in the workspace, whose capacity is reused from batch to batch.
    build_changes(delta, k, ws->changed_from, ws->changed_to);
    sssp_changes<vertex_t> changes;
    changes.changed_from = ws->changed_from.data();
    changes.changed_to = ws->changed_to.data();
    changes.num_changed = ws->changed_to.size();
    changes.insert_heads = delta.insert_dst.data();
    changes.num_insert_heads = delta.insert_dst.size();

    sssp_run<vertex_t, edge_t, weight_t> run;
    run.graph = objective_graph(res, g, k);
    run.changes = &changes;
    run.source = st.source;
    run.delta = delta_;
    run.max_weight = max_weight_;
    run.distances = st.distances.data();
    run.parents = st.parents.data();
    run.ws = &ws.get();
    run_update_engine(res, run);
    return run.counters;
  }

  /// CUDA backend: the change lists are built on the host (as mospUpdate() builds them) and
  /// uploaded into the pooled device workspace (stage sssp.changes), then the fused engine runs
  /// (stage sssp.enact_fused, MOSP's per-objective "sosp_update_gpu" region).
  sssp_counters update_cuda(const resources& res, const graph_type& g,
                            const apply_delta<vertex_t>& delta,
                            sssp_state<vertex_t, distance_t>& st) const {
    const int k = st.opt.objective;
    auto ws = lease_cuda_workspace<vertex_t>(res, static_cast<std::int64_t>(g.num_vertices()));
    sssp_changes<vertex_t> changes;
    {
      scoped_stage stage(res, "sssp.changes");
      build_changes(delta, k, ws->changed_from, ws->changed_to);
      const std::size_t num_changed = ws->changed_to.size();
      const std::size_t num_heads = delta.insert_dst.size();
      vertex_t* from = ws->device_changed_from.reserve(res, num_changed);
      vertex_t* to = ws->device_changed_to.reserve(res, num_changed);
      vertex_t* heads = ws->device_insert_heads.reserve(res, num_heads);
      upload(res, from, ws->changed_from.data(), num_changed);
      upload(res, to, ws->changed_to.data(), num_changed);
      upload(res, heads, delta.insert_dst.data(), num_heads);
      changes.changed_from = from;
      changes.changed_to = to;
      changes.num_changed = num_changed;
      changes.insert_heads = heads;
      changes.num_insert_heads = num_heads;
    }
    sssp_run<vertex_t, edge_t, weight_t> run;
    run.graph = device_objective_graph(res, g, k);
    run.changes = &changes;
    run.source = st.source;
    run.delta = delta_;
    run.max_weight = max_weight_;
    run.distances = st.device_distances.data();
    run.parents = st.device_parents.data();
    run.cuda_ws = &ws.get();
    sssp_cuda_update(res, run);
    return run.counters;
  }

  sssp::result<vertex_t, distance_t>& result_;
  sssp::stats& out_;
  sssp_state<vertex_t, distance_t>* state_ = nullptr;
  std::int64_t max_weight_ = 1;
  std::int64_t delta_ = 1;
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_sssp_participant(
    sssp::result<vertex_t, distance_t>& r, sssp::stats& out) {
  return std::make_unique<sssp_participant<vertex_t, edge_t, weight_t, distance_t>>(r, out);
}

}  // namespace dyng::detail

namespace dyng::sssp {

// ------------------------------------------------------------------------------------------------
// result
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::result(std::unique_ptr<state_type> state) noexcept
    : impl_(std::move(state)) {}

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::result(result&& other) noexcept = default;

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>& result<vertex_t, distance_t>::operator=(result&& other) noexcept =
    default;

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t>::~result() = default;

template <typename vertex_t, typename distance_t>
vertex_t result<vertex_t, distance_t>::source() const noexcept {
  return impl_ ? impl_->source : invalid_id<vertex_t>();
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::distances() const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_not_poisoned(*impl_, "sssp::result::distances");
  if (impl_->space == memory_space::host) {
    return host_view(impl_->distances);
  }
  return array_view<const distance_t>(impl_->device_distances.data(),
                                      impl_->device_distances.size(), memory_space::device,
                                      impl_->device);
}

template <typename vertex_t, typename distance_t>
array_view<const vertex_t> result<vertex_t, distance_t>::parents() const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_not_poisoned(*impl_, "sssp::result::parents");
  if (impl_->space == memory_space::host) {
    return host_view(impl_->parents);
  }
  return array_view<const vertex_t>(impl_->device_parents.data(), impl_->device_parents.size(),
                                    memory_space::device, impl_->device);
}

template <typename vertex_t, typename distance_t>
const options& result<vertex_t, distance_t>::get_options() const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_not_poisoned(*impl_, "sssp::result::get_options");
  return impl_->opt;
}

template <typename vertex_t, typename distance_t>
void result<vertex_t, distance_t>::set_options(const options& opt) {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_not_poisoned(*impl_, "sssp::result::set_options");
  detail::expect_options(opt);
  DYNG_EXPECTS(opt.objective == impl_->opt.objective,
               "sssp::result::set_options: the objective is fixed at compute() (",
               impl_->opt.objective, "); got ", opt.objective);
  impl_->opt = opt;
}

template <typename vertex_t, typename distance_t>
std::uint64_t result<vertex_t, distance_t>::graph_version() const noexcept {
  return impl_ ? impl_->version : 0;
}

template <typename vertex_t, typename distance_t>
memory_space result<vertex_t, distance_t>::space() const noexcept {
  return impl_ ? impl_->space : memory_space::host;
}

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::clone(const resources& res) const try {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_not_poisoned(*impl_, "sssp::result::clone");
  detail::expect_supported_backend(res, "sssp::result::clone");
  auto copy = std::make_unique<state_type>();
  copy->source = impl_->source;
  copy->opt = impl_->opt;
  copy->version = impl_->version;
  copy->graph_state = impl_->graph_state;
  copy->poisoned = impl_->poisoned;
  const std::size_t n = impl_->num_vertices();
  const bool from_device = impl_->space == memory_space::device;
  const memory_space from = impl_->space;
  // A CUDA result is complete when the call that produced it returns (compute, update and
  // from_arrays synchronize), so its arrays can be read on any stream.
  const stream_ref stream = detail::is_cuda(res) ? res.stream() : stream_ref{};
  const int device = detail::is_cuda(res) ? res.device() : impl_->device;
  if (detail::is_cuda(res)) {
    copy->space = memory_space::device;
    copy->device = res.device();
    copy->device_distances = buffer<distance_t>(res, n);
    copy->device_parents = buffer<vertex_t>(res, n);
    if (n > 0) {
      detail::copy_bytes(copy->device_distances.data(), memory_space::device,
                         impl_->distance_data(), from, n * sizeof(distance_t), stream, device);
      detail::copy_bytes(copy->device_parents.data(), memory_space::device, impl_->parent_data(),
                         from, n * sizeof(vertex_t), stream, device);
      res.synchronize();
    }
    // Size the handle's pooled workspace now, so the first update through `res` allocates no
    // scratch memory.
    (void)detail::lease_cuda_workspace<vertex_t>(res, static_cast<std::int64_t>(n));
  } else {
    if (from_device) {
      copy->distances.resize(n);
      copy->parents.resize(n);
      if (n > 0) {
        detail::copy_bytes(copy->distances.data(), memory_space::host, impl_->distance_data(), from,
                           n * sizeof(distance_t), stream, device);
        detail::copy_bytes(copy->parents.data(), memory_space::host, impl_->parent_data(), from,
                           n * sizeof(vertex_t), stream, device);
        detail::cuda_synchronize(device, stream);
      }
    } else {
      copy->distances = impl_->distances;
      copy->parents = impl_->parents;
    }
    (void)detail::lease_workspace<vertex_t>(res, static_cast<std::int64_t>(n));
  }
  return result(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("sssp::result::clone (", impl_ ? impl_->num_vertices() : 0,
                                  " vertices)")

template <typename vertex_t, typename distance_t>
template <typename edge_t, typename weight_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::from_arrays(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
    array_view<const distance_t> distances, array_view<const vertex_t> parents, bool canonicalize,
    const options& opt) try {
  static_assert(std::is_same_v<distance_t, std::int64_t>, "sssp: distance_t must be int64_t");
  detail::expect_supported_backend(res, "sssp::result::from_arrays");
  detail::expect_options(opt);
  detail::expect_graph(g, opt, "sssp::result::from_arrays");
  detail::graph_access::expect_placement(res, g, "sssp::result::from_arrays");
  const bool cuda = detail::is_cuda(res);
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "sssp::result::from_arrays: source ", source,
               " is out of range [0, ", n, ")");
  DYNG_EXPECTS(static_cast<std::int64_t>(distances.size()) == n,
               "sssp::result::from_arrays: ", distances.size(), " distances for ", n, " vertices");
  DYNG_EXPECTS(static_cast<std::int64_t>(parents.size()) == n,
               "sssp::result::from_arrays: ", parents.size(), " parents for ", n, " vertices");
  DYNG_EXPECTS(n == 0 || cuda ||
                   (is_host_accessible(distances.space()) && is_host_accessible(parents.space())),
               "sssp::result::from_arrays: the arrays must be in host memory for the ",
               to_string(res.get_backend()), " backend");
  const detail::column_summary column =
      detail::summarize_column(res, detail::graph_access::out_view(g).weight_column(opt.objective));
  DYNG_EXPECTS(column.first_bad < 0, "sssp::result::from_arrays: edge ", column.first_bad,
               " has a weight below 1 in objective ", opt.objective);

  auto state = std::make_unique<state_type>();
  state->source = source;
  state->opt = opt;
  state->version = g.version();
  state->graph_state = detail::graph_access::impl(g).state_id;
  // The tree is imported, canonicalized and checked in host memory (for the CUDA backend in host
  // vectors that are uploaded afterwards, as MOSP-CUDA reads its trees on the host and uploads
  // them in "upload").
  std::vector<distance_t> host_distances;
  std::vector<vertex_t> host_parents;
  std::vector<distance_t>& d = cuda ? host_distances : state->distances;
  std::vector<vertex_t>& p = cuda ? host_parents : state->parents;
  {
    // The tree copy (MOSP's mospUpdate copies the initial trees inside its "prepare" stage).
    scoped_stage stage(res, "sssp.import");
    detail::assign_host(res, d, distances);
    detail::assign_host(res, p, parents);
    detail::normalize_imported_tree(res, d, p);
  }
  if (canonicalize) {
    scoped_stage stage(res, "sssp.canonicalize");
    detail::canonicalize_tree(detail::objective_graph(res, g, opt.objective, false), source, d, p);
  }
  // The engines' scratch: the handle's pooled workspace, sized once for every result run through
  // `res` (ADR 0015); on the host backends its state array is also the scratch of the parallel
  // validation.
  std::vector<char> validation_scratch;
  std::vector<char>* scratch = &validation_scratch;
  std::optional<detail::workspace_pool::lease<detail::sssp_workspace<vertex_t>>> host_ws;
  if (cuda) {
    (void)detail::lease_cuda_workspace<vertex_t>(res, n);
  } else {
    host_ws.emplace(detail::lease_workspace<vertex_t>(res, n));
    scratch = &(*host_ws)->state;
  }
  if (opt.validate_inputs) {
    scoped_stage stage(res, "sssp.validate");
    DYNG_EXPECTS(detail::sssp_distances_fit(n, column.max_weight),
                 "sssp::result::from_arrays: distances up to ", column.max_weight, " * ", n - 1,
                 " do not fit in 62 bits");
    detail::validate_tree(res, source, d, p, column.max_weight * std::max<std::int64_t>(n - 1, 1),
                          *scratch);
  }
  if (cuda) {
    // The upload of the tree (part of MOSP-CUDA's "upload" stage).
    scoped_stage stage(res, "sssp.upload");
    state->space = memory_space::device;
    state->device = res.device();
    state->device_distances = buffer<distance_t>(res, static_cast<std::size_t>(n));
    state->device_parents = buffer<vertex_t>(res, static_cast<std::size_t>(n));
    detail::upload(res, state->device_distances.data(), d.data(), d.size());
    detail::upload(res, state->device_parents.data(), p.data(), p.size());
    res.synchronize();
  }
  return result(std::move(state));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("sssp::result::from_arrays (", g.num_vertices(), " vertices)")

// ------------------------------------------------------------------------------------------------
// compute / update
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
result<vertex_t> compute(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                         vertex_t source, const options& opt) try {
  detail::expect_supported_backend(res, "sssp::compute");
  scoped_stage stage(res, "sssp.compute");
  detail::expect_options(opt);
  detail::expect_graph(g, opt, "sssp::compute");
  detail::graph_access::expect_placement(res, g, "sssp::compute");
  detail::select_cuda_engine(res, opt.cuda_engine, "sssp::compute");
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "sssp::compute: source ", source, " is out of range [0, ",
               n, ")");
  const detail::column_summary column =
      detail::summarize_column(res, detail::graph_access::out_view(g).weight_column(opt.objective));
  DYNG_EXPECTS(column.first_bad < 0, "sssp::compute: edge ", column.first_bad,
               " has a weight below 1 in objective ", opt.objective,
               "; sssp needs weights in [1, 2^31 - 1]");
  DYNG_EXPECTS(detail::sssp_distances_fit(n, column.max_weight), "sssp::compute: distances up to ",
               column.max_weight, " * ", n - 1, " do not fit in 62 bits");

  auto state = std::make_unique<detail::sssp_state<vertex_t, std::int64_t>>();
  state->source = source;
  state->opt = opt;
  state->version = g.version();
  state->graph_state = detail::graph_access::impl(g).state_id;
  detail::sssp_run<vertex_t, edge_t, weight_t> run;
  run.source = source;
  run.delta = opt.delta > 0 ? opt.delta
                            : detail::sssp_default_delta(std::max<std::int64_t>(g.num_edges(), 1),
                                                         n, column.weight_sum);
  run.max_weight = column.max_weight;
  if (detail::is_cuda(res)) {
    // MOSP-CUDA's sospFromScratchGpu(): the kernel writes every entry of both arrays.
    state->space = memory_space::device;
    state->device = res.device();
    state->device_distances = buffer<std::int64_t>(res, static_cast<std::size_t>(n));
    state->device_parents = buffer<vertex_t>(res, static_cast<std::size_t>(n));
    auto ws = detail::lease_cuda_workspace<vertex_t>(res, n);  // pooled scratch (ADR 0015)
    run.graph = detail::device_objective_graph(res, g, opt.objective);
    run.distances = state->device_distances.data();
    run.parents = state->device_parents.data();
    run.cuda_ws = &ws.get();
    detail::sssp_cuda_compute(res, run);
    return detail::sssp_access::make(std::move(state));
  }
  state->distances.assign(static_cast<std::size_t>(n), detail::sssp_infinity);
  state->parents.assign(static_cast<std::size_t>(n), vertex_t{-1});
  auto ws = detail::lease_workspace<vertex_t>(res, n);  // the handle's pooled scratch (ADR 0015)
  // The OpenMP engine computes from scratch with pushes only; the sequential engine also reads
  // the in-edges.
  run.graph =
      detail::objective_graph(res, g, opt.objective, res.get_backend() == backend::sequential);
  run.distances = state->distances.data();
  run.parents = state->parents.data();
  run.ws = &ws.get();
  detail::run_compute_engine(res, run);
  return detail::sssp_access::make(std::move(state));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("sssp::compute (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r) try {
  scoped_stage stage(res, "sssp.update");
  stats out;
  detail::sssp_participant<vertex_t, edge_t, weight_t, std::int64_t> participant(r, out);
  detail::update_participant<vertex_t, edge_t, weight_t>* participants[] = {&participant};
  detail::run_update(res, g, batch, participants, 1, "sssp.commit");
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("sssp::update (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges; batch of ", batch.num_insertions(), " insertions, ",
                                  batch.num_deletions(), " deletions)")

// ------------------------------------------------------------------------------------------------
// Explicit instantiations (PLAN Section 4.4.3)
// ------------------------------------------------------------------------------------------------

template class result<std::int32_t, std::int64_t>;
template class result<std::int64_t, std::int64_t>;

#define DYNG_INSTANTIATE_SSSP(V, E, W)                                                             \
  template result<V> compute<V, E, W>(const resources&, const graph<V, E, W>&, V, const options&); \
  template stats update<V, E, W>(const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&,  \
                                 result<V>&);                                                      \
  template result<V> result<V>::from_arrays<E, W>(const resources&, const graph<V, E, W>&, V,      \
                                                  array_view<const std::int64_t>,                  \
                                                  array_view<const V>, bool, const options&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP)
#undef DYNG_INSTANTIATE_SSSP

}  // namespace dyng::sssp

namespace dyng::detail {

#define DYNG_INSTANTIATE_SSSP_DETAIL(V, E, W)           \
  template std::unique_ptr<update_participant<V, E, W>> \
  make_sssp_participant<V, E, W, std::int64_t>(sssp::result<V, std::int64_t>&, sssp::stats&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_DETAIL)
#undef DYNG_INSTANTIATE_SSSP_DETAIL

#if !DYNG_HAS_CUDA
// Without the cuda backend nothing can reach these (resources::cuda() throws); they exist so that
// the backend-neutral code above links.
template <typename vertex_t>
void sssp_cuda_workspace<vertex_t>::reserve(const resources& /*res*/, std::int64_t /*requested*/) {
  throw not_supported_error("dyng: sssp: the cuda backend is not built");
}

template <typename vertex_t>
int sssp_cuda_workspace<vertex_t>::next_generation(const resources& /*res*/) {
  throw not_supported_error("dyng: sssp: the cuda backend is not built");
}

template <typename vertex_t>
std::size_t sssp_cuda_workspace<vertex_t>::bytes() const noexcept {
  return 0;
}

template struct sssp_cuda_workspace<std::int32_t>;
template struct sssp_cuda_workspace<std::int64_t>;

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_update(const resources& /*res*/, sssp_run<vertex_t, edge_t, weight_t>& /*run*/) {
  throw not_supported_error("dyng: sssp: the cuda backend is not built");
}

template <typename vertex_t, typename edge_t, typename weight_t>
void sssp_cuda_compute(const resources& /*res*/, sssp_run<vertex_t, edge_t, weight_t>& /*run*/) {
  throw not_supported_error("dyng: sssp: the cuda backend is not built");
}

#define DYNG_INSTANTIATE_SSSP_CUDA_STUB(V, E, W)                                 \
  template void sssp_cuda_update<V, E, W>(const resources&, sssp_run<V, E, W>&); \
  template void sssp_cuda_compute<V, E, W>(const resources&, sssp_run<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SSSP_CUDA_STUB)
#undef DYNG_INSTANTIATE_SSSP_CUDA_STUB
#endif  // !DYNG_HAS_CUDA

template sssp_state<std::int32_t, std::int64_t>& sssp_access::state(
    sssp::result<std::int32_t, std::int64_t>&);
template sssp_state<std::int64_t, std::int64_t>& sssp_access::state(
    sssp::result<std::int64_t, std::int64_t>&);

}  // namespace dyng::detail
