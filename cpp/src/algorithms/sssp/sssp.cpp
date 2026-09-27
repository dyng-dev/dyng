// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
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
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"

#include <dyng/config.hpp>
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
    // assign + clear keeps the capacity and touches every page once here, so the first update
    // does not take the page faults inside its timed phases (MOSP-OpenMP shares one workspace
    // across the K objectives, so only its first objective pays them).
    list->assign(n, vertex_t{0});
    list->clear();
  }
  capacity = requested;
  generation = 0;
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

/// The backends sssp runs on in this release.
void expect_cpu_backend(const resources& res, const char* what) {
  const backend b = res.get_backend();
  if (b != backend::sequential && b != backend::openmp) {
    throw not_supported_error(std::string("dyng: ") + what + ": the " + std::string(to_string(b)) +
                              " backend is not available for sssp yet; available: sequential, "
                              "openmp");
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
  if (res.get_backend() == backend::openmp) {
    const int threads = std::max(1, res.num_threads());
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

template <typename vertex_t, typename edge_t, typename weight_t>
sssp_graph<vertex_t, edge_t, weight_t> objective_graph(const graph<vertex_t, edge_t, weight_t>& g,
                                                       int objective) {
  const auto view = g.view();
  sssp_graph<vertex_t, edge_t, weight_t> out;
  out.num_vertices = view.num_vertices();
  out.out_row_ptr = view.out.row_ptr.data();
  out.out_col_ind = view.out.col_ind.data();
  out.out_weights = view.out.weight_column(objective).data();
  out.in_row_ptr = view.in.row_ptr.data();
  out.in_col_ind = view.in.col_ind.data();
  out.in_weights = view.in.weight_column(objective).data();
  return out;
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

/// The O(n) checks of options::validate_inputs (rooted at the source, no parent cycle).
template <typename vertex_t>
void validate_tree(vertex_t source, const std::vector<std::int64_t>& distances,
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
    expect_cpu_backend(res, "sssp::update");
    state_ = &sssp_access::state(result_);
    if (state_->poisoned) {
      throw stale_result_error(
          "dyng: sssp::update: the result was left unusable by a failed update; recompute it "
          "with sssp::compute()");
    }
    if (state_->version != g.version()) {
      throw stale_result_error(detail::concat_message(
          "dyng: sssp::update: the result matches graph version ", state_->version,
          " but the graph is at version ", g.version(),
          " (the graph was changed without updating this result; recompute it, or update all "
          "results together with dyng::update(res, g, batch, results...))"));
    }
    const sssp::options& opt = state_->opt;
    expect_graph(g, opt, "sssp::update");
    DYNG_EXPECTS(static_cast<std::int64_t>(state_->distances.size()) ==
                     static_cast<std::int64_t>(g.num_vertices()),
                 "sssp::update: the result has ", state_->distances.size(),
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
    const auto view = g.view();
    const column_summary column = summarize_column(res, view.out.weight_column(k));
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
    const int k = st.opt.objective;
    const auto n = static_cast<std::size_t>(g.num_vertices());
    // New vertices (vertex growth) start unreachable.
    if (n > st.distances.size()) {
      st.distances.resize(n, sssp_infinity);
      st.parents.resize(n, vertex_t{-1});
    }
    DYNG_EXPECTS(sssp_distances_fit(static_cast<std::int64_t>(n), max_weight_),
                 "sssp::update: distances up to ", max_weight_, " * ", n - 1,
                 " do not fit in 62 bits");

    // The change list of objective k: every deletion, then every insertion that raised the
    // objective's weight; the insertion heads are the heads of all insertions.
    changes_.changed_from = delta.delete_src;
    changes_.changed_to = delta.delete_dst;
    const std::size_t num_ins = delta.insert_src.size();
    const auto num_objectives = static_cast<std::size_t>(delta.num_weights);
    for (std::size_t i = 0; i < num_ins; ++i) {
      if (delta.weight_increased[i * num_objectives + static_cast<std::size_t>(k)] != 0) {
        changes_.changed_from.push_back(delta.insert_src[i]);
        changes_.changed_to.push_back(delta.insert_dst[i]);
      }
    }
    changes_.insert_heads = delta.insert_dst.data();
    changes_.num_insert_heads = delta.insert_dst.size();

    sssp_run<vertex_t, edge_t, weight_t> run;
    run.graph = objective_graph(g, k);
    run.changes = &changes_;
    run.source = st.source;
    run.delta = delta_;
    run.max_weight = max_weight_;
    run.distances = st.distances.data();
    run.parents = st.parents.data();
    run.ws = &st.ws;
    run_update_engine(res, run);

    sssp::stats s;
    s.affected = run.counters.affected;
    s.iterations = run.counters.iterations;
    s.frontier_visits = run.counters.pushes;
    s.fallback_used = false;
    s.converged = true;
    s.engine_used = res.get_backend() == backend::openmp ? engine::fused : engine::operators;
    s.batch = summary;
    s.invalidated = run.counters.invalidated;
    s.epochs = run.counters.epochs;
    s.pushes = run.counters.pushes;
    s.packed_parents = run.counters.packed_parents;
    out_ = s;
    st.version = g.version();
  }

  void poison() noexcept override {
    if (state_ != nullptr) {
      state_->poisoned = true;
    }
  }

 private:
  sssp::result<vertex_t, distance_t>& result_;
  sssp::stats& out_;
  sssp_state<vertex_t, distance_t>* state_ = nullptr;
  std::int64_t max_weight_ = 1;
  std::int64_t delta_ = 1;
  sssp_changes<vertex_t> changes_;
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
  return host_view(impl_->distances);
}

template <typename vertex_t, typename distance_t>
array_view<const vertex_t> result<vertex_t, distance_t>::parents() const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  return host_view(impl_->parents);
}

template <typename vertex_t, typename distance_t>
const options& result<vertex_t, distance_t>::get_options() const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  return impl_->opt;
}

template <typename vertex_t, typename distance_t>
void result<vertex_t, distance_t>::set_options(const options& opt) {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
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
  return memory_space::host;
}

template <typename vertex_t, typename distance_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::clone(const resources& res) const {
  DYNG_EXPECTS(impl_ != nullptr, "sssp::result: use of a moved-from result");
  detail::expect_cpu_backend(res, "sssp::result::clone");
  auto copy = std::make_unique<state_type>();
  copy->source = impl_->source;
  copy->opt = impl_->opt;
  copy->version = impl_->version;
  copy->poisoned = impl_->poisoned;
  copy->distances = impl_->distances;
  copy->parents = impl_->parents;
  copy->ws.reserve(static_cast<std::int64_t>(copy->distances.size()));
  return result(std::move(copy));
}

template <typename vertex_t, typename distance_t>
template <typename edge_t, typename weight_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::from_arrays(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
    array_view<const distance_t> distances, array_view<const vertex_t> parents, bool canonicalize,
    const options& opt) {
  static_assert(std::is_same_v<distance_t, std::int64_t>, "sssp: distance_t must be int64_t");
  detail::expect_cpu_backend(res, "sssp::result::from_arrays");
  detail::expect_options(opt);
  detail::expect_graph(g, opt, "sssp::result::from_arrays");
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "sssp::result::from_arrays: source ", source,
               " is out of range [0, ", n, ")");
  DYNG_EXPECTS(static_cast<std::int64_t>(distances.size()) == n,
               "sssp::result::from_arrays: ", distances.size(), " distances for ", n, " vertices");
  DYNG_EXPECTS(static_cast<std::int64_t>(parents.size()) == n,
               "sssp::result::from_arrays: ", parents.size(), " parents for ", n, " vertices");
  DYNG_EXPECTS(
      n == 0 || (is_host_accessible(distances.space()) && is_host_accessible(parents.space())),
      "sssp::result::from_arrays: the arrays must be in host memory");
  const auto view = g.view();
  const detail::column_summary column =
      detail::summarize_column(res, view.out.weight_column(opt.objective));
  DYNG_EXPECTS(column.first_bad < 0, "sssp::result::from_arrays: edge ", column.first_bad,
               " has a weight below 1 in objective ", opt.objective);

  auto state = std::make_unique<state_type>();
  state->source = source;
  state->opt = opt;
  state->version = g.version();
  state->distances.assign(distances.begin(), distances.end());
  state->parents.assign(parents.begin(), parents.end());
  for (std::int64_t v = 0; v < n; ++v) {
    auto& d = state->distances[static_cast<std::size_t>(v)];
    if (d >= detail::sssp_infinity / 2) {
      d = detail::sssp_infinity;  // MOSP writes and reads these as "INF"
    }
    // Always checked (memory safety of the engines), with or without validate_inputs.
    const vertex_t p = state->parents[static_cast<std::size_t>(v)];
    DYNG_EXPECTS(p >= -1 && p < n, "sssp::result::from_arrays: vertex ", v, " has parent ", p,
                 ", outside [-1, ", n, ")");
  }
  if (canonicalize) {
    detail::canonicalize_tree(detail::objective_graph(g, opt.objective), source, state->distances,
                              state->parents);
  }
  state->ws.reserve(n);  // the engines' workspace, once (invariant I9)
  if (opt.validate_inputs) {
    DYNG_EXPECTS(detail::sssp_distances_fit(n, column.max_weight),
                 "sssp::result::from_arrays: distances up to ", column.max_weight, " * ", n - 1,
                 " do not fit in 62 bits");
    detail::validate_tree(source, state->distances, state->parents,
                          column.max_weight * std::max<std::int64_t>(n - 1, 1));
  }
  return result(std::move(state));
}

// ------------------------------------------------------------------------------------------------
// compute / update
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
result<vertex_t> compute(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                         vertex_t source, const options& opt) {
  detail::expect_cpu_backend(res, "sssp::compute");
  scoped_stage stage(res, "sssp.compute");
  detail::expect_options(opt);
  detail::expect_graph(g, opt, "sssp::compute");
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "sssp::compute: source ", source, " is out of range [0, ",
               n, ")");
  const auto view = g.view();
  const detail::column_summary column =
      detail::summarize_column(res, view.out.weight_column(opt.objective));
  DYNG_EXPECTS(column.first_bad < 0, "sssp::compute: edge ", column.first_bad,
               " has a weight below 1 in objective ", opt.objective,
               "; sssp needs weights in [1, 2^31 - 1]");
  DYNG_EXPECTS(detail::sssp_distances_fit(n, column.max_weight), "sssp::compute: distances up to ",
               column.max_weight, " * ", n - 1, " do not fit in 62 bits");

  auto state = std::make_unique<detail::sssp_state<vertex_t, std::int64_t>>();
  state->source = source;
  state->opt = opt;
  state->version = g.version();
  state->distances.assign(static_cast<std::size_t>(n), detail::sssp_infinity);
  state->parents.assign(static_cast<std::size_t>(n), vertex_t{-1});

  detail::sssp_run<vertex_t, edge_t, weight_t> run;
  run.graph = detail::objective_graph(g, opt.objective);
  run.source = source;
  run.delta = opt.delta > 0 ? opt.delta
                            : detail::sssp_default_delta(std::max<std::int64_t>(g.num_edges(), 1),
                                                         n, column.weight_sum);
  run.max_weight = column.max_weight;
  run.distances = state->distances.data();
  run.parents = state->parents.data();
  run.ws = &state->ws;
  detail::run_compute_engine(res, run);
  return detail::sssp_access::make(std::move(state));
}

template <typename vertex_t, typename edge_t, typename weight_t>
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result<vertex_t>& r) {
  scoped_stage stage(res, "sssp.update");
  stats out;
  detail::sssp_participant<vertex_t, edge_t, weight_t, std::int64_t> participant(r, out);
  detail::update_participant<vertex_t, edge_t, weight_t>* participants[] = {&participant};
  detail::run_update(res, g, batch, participants, 1, "sssp.commit");
  return out;
}

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

template sssp_state<std::int32_t, std::int64_t>& sssp_access::state(
    sssp::result<std::int32_t, std::int64_t>&);
template sssp_state<std::int64_t, std::int64_t>& sssp_access::state(
    sssp::result<std::int64_t, std::int64_t>&);

}  // namespace dyng::detail
