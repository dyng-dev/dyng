// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-CUDA@e220ee2:src/mospUpdate.cu (mospUpdate: the K objectives, then Steps 2-3),
// src/combinedGraphGpu.cu (preferenceScale, combinedGraphSospGpu: the scaled preference terms,
// the base weight, the default delta of the combined graph) and MOSP-OpenMP@c352151:
// src/mospUpdate.cpp, src/combinedGraphCpu.cpp
/**
 * @file mosp.cpp
 * @brief mosp: argument validation, the composition of K sssp problems (mosp_problem, the
 *        participant of run_update()), the finalize step (Steps 2-3 of MOSP_Update: the combined
 *        graph, its static solve through sssp, `affected`, the path costs), mosp::result,
 *        compute() / update() / from_arrays(), and the explicit instantiations.
 *
 * The flow is MOSP's mospUpdate(): the batch is applied once (run_update's commit), the K
 * objectives are updated one after the other on the shared workspace (each an sssp problem with
 * its own update_enactor; MOSP's per-objective "obj<k>/sosp_update" regions), then
 * combinedGraphSosp{Gpu,Cpu}: the preference scale L = lcm(Pref) (at most 2^20), the terms
 * L / Pref_i, the base weight L * (K + 1), the combined graph (backend kernels: cuda.cu,
 * openmp.cpp, sequential.cpp), its default delta (sssp's rule on the combined graph's edges and
 * weight sum) and the solve from the source with the packing bound `base` (sospFromScratch*).
 *
 * Changes from the originals: K up to 64 (the commit classifies one byte per insertion and
 * objective; MOSP stops at 32); the K trees are sssp results (separate arrays, not one
 * objective-major array); `affected` (the vertices whose combined distance or parent changed):
 * on cuda the combined solve overwrites the previous MOSP tree in place and counts it in its
 * unpack pass, on the host backends the previous tree is kept for the count (a second pair of
 * arrays, swapped); the path costs are part of compute() and
 * update() (options::compute_path_costs; MOSP computes them while writing mospCosts.txt) and
 * cover the K objectives (MOSP's mospPathCosts covers every weight column of the graph); a
 * missing tree edge throws instead of failing the write.
 */
#include "algorithms/mosp/problem.hpp"
#include "algorithms/sssp/problem.hpp"
#include "core/budget_counters.hpp"
#include "core/cuda_runtime.hpp"
#include "core/resources_access.hpp"
#include "framework/budgets.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/workspace.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "util/allocation.hpp"
#include "util/device_fill.hpp"

#include <dyng/config.hpp>
#include <dyng/core/buffer.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/mosp.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail {

template <typename vertex_t, typename distance_t>
mosp_state<vertex_t, distance_t>& mosp_access::state(mosp::result<vertex_t, distance_t>& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "mosp::result: use of a moved-from result");
  return *r.impl_;
}

template <typename vertex_t, typename distance_t>
mosp::result<vertex_t, distance_t> mosp_access::make(
    std::unique_ptr<mosp_state<vertex_t, distance_t>> state) {
  return mosp::result<vertex_t, distance_t>(std::move(state));
}

namespace {

bool is_cuda(const resources& res) noexcept {
  return res.get_backend() == backend::cuda;
}

/// The backends mosp runs on in this build (sssp's).
void expect_supported_backend(const resources& res, const char* what) {
  if (is_cuda(res) && !DYNG_HAS_CUDA) {
    throw not_supported_error(std::string("dyng: ") + what +
                              ": the cuda backend is not built; available: sequential, openmp");
  }
}

template <typename vertex_t, typename distance_t>
void expect_not_poisoned(const mosp_state<vertex_t, distance_t>& st, const char* what) {
  if (st.poisoned) {
    throw stale_result_error(concat_message(
        "dyng: ", what,
        ": the result was left unusable by a failed update; recompute it with mosp::compute() "
        "(or import valid trees with mosp::result::from_arrays())"));
  }
}

/// The state of a result for an accessor: not moved from, not poisoned.
template <typename state_t>
state_t& checked(state_t* st, const char* what) {
  DYNG_EXPECTS(st != nullptr, "mosp::result: use of a moved-from result");
  expect_not_poisoned(*st, what);
  return *st;
}

/// MOSP's preferenceScale(): L = lcm(preferences), 1 for none; the checks of options (K values,
/// each >= 1, L <= 2^20) throw with the offending value.
std::int64_t preference_scale_of(const std::vector<std::int32_t>& preferences, int num_k,
                                 const char* what) {
  if (preferences.empty()) {
    return 1;
  }
  DYNG_EXPECTS(static_cast<int>(preferences.size()) == num_k, what, ": options.preferences has ",
               preferences.size(), " values for ", num_k,
               " objective(s) (one per objective, or none)");
  std::int64_t scale = 1;
  for (std::size_t i = 0; i < preferences.size(); ++i) {
    const std::int64_t pref = preferences[i];
    DYNG_EXPECTS(pref >= 1, what, ": options.preferences[", i, "] is ", pref,
                 "; a preference must be >= 1 (a lower value is a higher priority)");
    scale = scale / std::gcd(scale, pref) * pref;
    DYNG_EXPECTS(scale <= mosp::max_preference_scale, what,
                 ": lcm(options.preferences) exceeds 2^20 (the combined weights are scaled by it)");
  }
  return scale;
}

/// The objectives the options select: K and L.
struct resolved_options {
  int num_objectives = 0;  ///< K
  std::int64_t scale = 1;  ///< L
};

resolved_options resolve_options(const mosp::options& opt, int num_weights, const char* what) {
  DYNG_EXPECTS(opt.delta >= 0, what, ": options.delta must be >= 0 (0 = automatic), got ",
               opt.delta);
  DYNG_EXPECTS(opt.num_objectives >= 0 && opt.num_objectives <= num_weights, what,
               ": options.num_objectives ", opt.num_objectives, " is out of range [0, ",
               num_weights, "] (the graph has ", num_weights, " weight column(s); 0 = all)");
  resolved_options out;
  out.num_objectives = opt.num_objectives > 0 ? opt.num_objectives : num_weights;
  DYNG_EXPECTS(out.num_objectives >= 1, what,
               ": mosp needs at least one weight column (graph_properties::num_weights)");
  DYNG_EXPECTS(out.num_objectives <= mosp::max_objectives, what, ": ", out.num_objectives,
               " objectives; mosp supports at most ", mosp::max_objectives,
               " (set options.num_objectives)");
  out.scale = preference_scale_of(opt.preferences, out.num_objectives, what);
  return out;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void expect_graph(const graph<vertex_t, edge_t, weight_t>& g, const char* what) {
  DYNG_EXPECTS(g.has_transposed(), what,
               ": mosp needs the in-edges (its sssp updates pull); build the graph with "
               "graph_properties::store_transposed = true");
}

/// The sssp options of objective k.
sssp::options objective_options(const mosp::options& opt, int k) {
  sssp::options out;
  out.delta = opt.delta;
  out.objective = k;
  out.cuda_engine = opt.cuda_engine;
  out.validate_inputs = opt.validate_inputs;
  return out;
}

template <typename vertex_t, typename distance_t>
void expect_result_placement(const resources& res, const mosp_state<vertex_t, distance_t>& st,
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

/// Size the combined arrays (and, on the host, their spares) for `n` vertices: new vertices start
/// unreachable in the current tree; the spares are written by the next solve.
template <typename vertex_t, typename distance_t>
void size_combined(const resources& res, mosp_state<vertex_t, distance_t>& st, std::size_t n) {
  if (st.num_vertices() == n &&
      (st.space == memory_space::device || st.spare_parents.size() == n)) {
    return;
  }
  note_reservation();  // invariant I9: the result grows with the graph
  if (!is_cuda(res)) {
    st.space = memory_space::host;
    st.device = -1;
    st.combined_distances.resize(n, sssp_infinity);
    st.combined_parents.resize(n, vertex_t{-1});
    st.spare_distances.resize(n);
    st.spare_parents.resize(n);
    return;
  }
#if DYNG_HAS_CUDA
  const std::size_t old_n = st.space == memory_space::device ? st.num_vertices() : 0;
  buffer<distance_t> distances(res, n);
  buffer<vertex_t> parents(res, n);
  if (old_n > 0) {
    copy_bytes(distances.data(), memory_space::device, st.device_combined_distances.data(),
               memory_space::device, old_n * sizeof(distance_t), res.stream(), res.device());
    copy_bytes(parents.data(), memory_space::device, st.device_combined_parents.data(),
               memory_space::device, old_n * sizeof(vertex_t), res.stream(), res.device());
  }
  {
    const scoped_device guard(res.device());
    fill_async(res.stream(), distances.data() + old_n, n - old_n, distance_t{sssp_infinity});
    fill_async(res.stream(), parents.data() + old_n, n - old_n, vertex_t{-1});
  }
  // Released on res.stream(), after the copies above (as sssp's grow()).
  st.device_combined_distances.set_stream(res.stream());
  st.device_combined_parents.set_stream(res.stream());
  st.device_combined_distances = std::move(distances);
  st.device_combined_parents = std::move(parents);
  st.space = memory_space::device;
  st.device = res.device();
#else
  DYNG_FAIL("mosp: a device result without the cuda backend");
#endif
}

/// What the finalize step did.
struct finalize_counts {
  std::int64_t affected = 0;        ///< vertices whose combined distance or parent changed
  std::int64_t combined_edges = 0;  ///< edges of the combined graph
  std::int64_t host_syncs = 0;      ///< its host synchronizations (cuda), for the budget
  sssp_solve_outcome solve;         ///< the static solve
};

/// The vertices whose combined distance or parent differ between two host trees.
template <typename vertex_t>
std::int64_t count_changed(const resources& res, std::size_t n, const std::int64_t* old_d,
                           const vertex_t* old_p, const std::int64_t* new_d,
                           const vertex_t* new_p) {
  const auto count = static_cast<std::int64_t>(n);
  std::int64_t changed = 0;
#if DYNG_HAS_OPENMP
  const int threads = resources_access::host_threads(res);
#pragma omp parallel for num_threads(threads) if (threads > 1 && count >= 65536) schedule(static) \
    reduction(+ : changed)
#else
  (void)res;
#endif
  for (std::int64_t v = 0; v < count; ++v) {
    changed += (old_d[v] != new_d[v] || old_p[v] != new_p[v]) ? 1 : 0;
  }
  return changed;
}

/// Steps 2-3 of MOSP_Update and the path costs, on the current K trees of `st` (see the file
/// comment); `count_affected` compares the new MOSP tree with the previous one.
template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
finalize_counts finalize(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
                         mosp_state<vertex_t, distance_t>& st, bool count_affected) {
  finalize_counts out;
  const auto n = static_cast<std::size_t>(g.num_vertices());
  const int num_k = st.num_objectives;
  DYNG_EXPECTS(static_cast<std::uint64_t>(num_k) * n <=
                   static_cast<std::uint64_t>(std::numeric_limits<edge_t>::max()),
               "mosp: the combined graph of ", num_k, " trees on ", n, " vertices can have ",
               static_cast<std::uint64_t>(num_k) * n,
               " edges, more than the graph's edge offsets hold; use 64-bit edge offsets");
  size_combined(res, st, n);
  // The preference terms L / Pref_i and the base weight L * (K + 1) (combinedGraphSosp*).
  mosp_combine_input<vertex_t> in;
  in.num_vertices = static_cast<vertex_t>(n);
  in.source = st.source;
  in.num_objectives = num_k;
  in.base = static_cast<std::int32_t>(st.scale * (num_k + 1));
  for (int k = 0; k < num_k; ++k) {
    in.parents[k] = sssp_access::state(st.objectives[static_cast<std::size_t>(k)]).parent_data();
    in.terms[k] = static_cast<std::int32_t>(
        st.opt.preferences.empty() ? st.scale
                                   : st.scale / st.opt.preferences[static_cast<std::size_t>(k)]);
  }
  const std::int64_t base = in.base;
  const bool costs = st.opt.compute_path_costs;
  st.has_path_costs = false;
  workspace_pool& pool = resources_access::workspaces(res);
  std::optional<workspace_pool::lease<mosp_workspace<vertex_t, edge_t, weight_t>>> host_ws;
  const vertex_t* tree = nullptr;  // the new MOSP tree in host memory (path costs)
  if (is_cuda(res)) {
    auto ws = pool.acquire<mosp_cuda_workspace<vertex_t, edge_t, weight_t>>(res);
    {
      // A reserving run (the graph gained vertices) releases the old pinned copy of the tree
      // before it allocates the larger one, and that release synchronizes the stream
      // (pinned_host_memory_resource::deallocate): part of the reservation, counted (I9).
      const framework::budget_scope reserving;
      ws->reserve(res, static_cast<std::int64_t>(n), num_k);
      out.host_syncs += reserving.used().own_host_syncs();
    }
    mosp_combined<vertex_t, edge_t, weight_t> combined;
    {
      scoped_stage stage(res, "mosp.combine");
      combined = mosp_combine_cuda(res, in, ws.get());
      out.host_syncs += 1;
    }
    out.combined_edges = combined.edges;
    const std::int64_t delta =
        sssp_default_delta(combined.edges, static_cast<std::int64_t>(n), combined.weight_sum);
    {
      // The solve overwrites the previous MOSP tree in place and counts `affected` in its unpack
      // pass (the engines' update counter), so no second pair of arrays is kept on the device.
      scoped_stage stage(res, "mosp.combined_sssp");
      out.solve = sssp_solve_view(res, g, combined.view, st.source, delta, base, st.opt.cuda_engine,
                                  st.device_combined_distances.data(),
                                  st.device_combined_parents.data(), count_affected);
      out.host_syncs += out.solve.host_syncs;  // a static solve's (2 + rounds on operators)
    }
    {
      scoped_stage stage(res, "mosp.finalize");  // `affected` was counted by the solve
      out.affected = count_affected ? out.solve.counters.affected : 0;
    }
    if (costs) {
      tree = ws->host_parents.data();
      host_ws.emplace(pool.acquire<mosp_workspace<vertex_t, edge_t, weight_t>>(res));
      (*host_ws)->reserve(n, num_k);
      // The pinned copy of the tree stays valid while the device workspace is leased (to the end
      // of this function). Its download belongs to the path costs (the originals' Steps 2-3 end
      // with the tree on the device), so it is timed in their stage: one more synchronization.
      {
        scoped_stage stage(res, "mosp.path_costs");
        mosp_download_tree_cuda(res, static_cast<std::int64_t>(n),
                                st.device_combined_parents.data(), ws.get());
        out.host_syncs += 1;
        if (st.path_costs.size() != n * static_cast<std::size_t>(num_k)) {
          note_reservation();
          st.path_costs.resize(n * static_cast<std::size_t>(num_k));
        }
        const vertex_t missing =
            mosp_path_costs_openmp(res, graph_access::out_view(g), tree, st.source, num_k,
                                   st.path_costs.data(), host_ws->get());
        DYNG_EXPECTS(missing < 0, "mosp: the MOSP tree edge (", tree[missing], ", ", missing,
                     ") is not an edge of the graph (do the trees belong to this graph?)");
        st.has_path_costs = true;
      }
    }
    return out;
  }
  host_ws.emplace(pool.acquire<mosp_workspace<vertex_t, edge_t, weight_t>>(res));
  mosp_workspace<vertex_t, edge_t, weight_t>& ws = host_ws->get();
  ws.reserve(n, num_k);
  mosp_combined<vertex_t, edge_t, weight_t> combined;
  {
    scoped_stage stage(res, "mosp.combine");
    if (res.get_backend() == backend::openmp) {
      combined = mosp_combine_openmp(res, in, ws);
    } else {
      // The sequential engine recovers parents over in-edges in the distance-only mode.
      combined =
          mosp_combine_sequential(in, ws, !sssp_packs_parents(static_cast<std::int64_t>(n), base));
    }
  }
  out.combined_edges = combined.edges;
  const std::int64_t delta =
      sssp_default_delta(combined.edges, static_cast<std::int64_t>(n), combined.weight_sum);
  {
    scoped_stage stage(res, "mosp.combined_sssp");
    out.solve = sssp_solve_view(res, g, combined.view, st.source, delta, base, st.opt.cuda_engine,
                                st.spare_distances.data(), st.spare_parents.data());
  }
  {
    scoped_stage stage(res, "mosp.finalize");
    if (count_affected) {
      out.affected = count_changed(res, n, st.combined_distances.data(), st.combined_parents.data(),
                                   st.spare_distances.data(), st.spare_parents.data());
    }
    std::swap(st.combined_distances, st.spare_distances);
    std::swap(st.combined_parents, st.spare_parents);
  }
  if (costs) {
    scoped_stage stage(res, "mosp.path_costs");
    if (st.path_costs.size() != n * static_cast<std::size_t>(num_k)) {
      note_reservation();
      st.path_costs.resize(n * static_cast<std::size_t>(num_k));
    }
    tree = st.combined_parents.data();
    const vertex_t missing =
        res.get_backend() == backend::openmp
            ? mosp_path_costs_openmp(res, graph_access::out_view(g), tree, st.source, num_k,
                                     st.path_costs.data(), ws)
            : mosp_path_costs(graph_access::out_view(g), tree, st.source, num_k,
                              st.path_costs.data(), ws);
    DYNG_EXPECTS(missing < 0, "mosp: the MOSP tree edge (", tree[missing], ", ", missing,
                 ") is not an edge of the graph (do the trees belong to this graph?)");
    st.has_path_costs = true;
  }
  return out;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// mosp_problem: the participant of one mosp result in run_update()
// ------------------------------------------------------------------------------------------------

/**
 * @brief The update of one mosp result: K sssp participants (each an sssp problem with its own
 *        update_enactor) and the finalize step, as one participant of run_update(), so mosp
 *        composes with any other result in dyng::update() (problem.hpp, file comment).
 *
 * Budget (invariant I9): the whole algorithm work of the update (the K sssp halves before and
 * after the commit and the finalize step) is measured and checked against the sum of the K sssp
 * budgets and the finalize step's synchronizations (none on the host backends; on cuda one for
 * the combined graph's size, the combined solve's (a static solve's: sssp_cuda_host_syncs(), which
 * counts `affected` too), with the path costs one for the download of the MOSP tree, and in a
 * reserving run that grew the vertex set the release of the old pinned copy of the tree).
 * @tparam vertex_t   Vertex id type.
 * @tparam edge_t     Edge offset type.
 * @tparam weight_t   Weight type.
 * @tparam distance_t Distance type.
 */
template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
class mosp_problem final : public update_participant<vertex_t, edge_t, weight_t> {
 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;                     ///< the container
  using batch_type = edge_batch_view<vertex_t, weight_t>;                   ///< the batch
  using result_type = mosp::result<vertex_t, distance_t>;                   ///< the result
  using state_type = mosp_state<vertex_t, distance_t>;                      ///< its state
  using participant_type = update_participant<vertex_t, edge_t, weight_t>;  ///< sssp's

  /**
   * @brief The update of `r`; `out` receives the stats when it succeeds.
   * @param[in,out] r   The result (must outlive the participant).
   * @param[out]    out The stats.
   */
  mosp_problem(result_type& r, mosp::stats& out) : result_(&r), out_(out) {
    state_ = &mosp_access::state(r);
    const auto num_k = static_cast<std::size_t>(state_->num_objectives);
    stats_.objectives.resize(num_k);
    objectives_.reserve(num_k);
    for (std::size_t k = 0; k < num_k && k < state_->objectives.size(); ++k) {
      objectives_.push_back(make_sssp_participant<vertex_t, edge_t, weight_t, distance_t>(
          state_->objectives[k], stats_.objectives[k]));
    }
  }

  /// @copydoc update_participant::target
  [[nodiscard]] const void* target() const noexcept override {
    return result_;
  }

  /// @copydoc update_participant::use_normalized
  void use_normalized(const normalized_batch<vertex_t>* normalized) noexcept override {
    for (const auto& p : objectives_) {
      p->use_normalized(normalized);
    }
  }

  /// @copydoc update_participant::before_apply
  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    const framework::budget_scope half;
    expect_supported_backend(res, "mosp::update");
    expect_not_poisoned(*state_, "mosp::update");
    framework::expect_current_result("mosp::update", state_->version, state_->graph_state, g);
    expect_graph(g, "mosp::update");
    graph_access::expect_placement(res, g, "mosp::update");
    expect_result_placement(res, *state_, "mosp::update");
    DYNG_EXPECTS(batch.num_insertions() == 0 || batch.num_weights == g.num_weights(),
                 "mosp::update: the batch has ", batch.num_weights,
                 " weight(s) per insertion but the graph has ", g.num_weights(),
                 " (one per weight column)");
    DYNG_EXPECTS(static_cast<int>(objectives_.size()) == state_->num_objectives,
                 "mosp::update: the result has ", objectives_.size(), " trees for ",
                 state_->num_objectives, " objectives");
    for (const auto& p : objectives_) {
      p->before_apply(res, g, batch);
    }
    before_used_ = half.used();
  }

  /// @copydoc update_participant::after_apply
  void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                   const apply_delta<vertex_t>& delta) override {
    const framework::budget_scope phase;
    std::int64_t syncs = 0;
    bool bounded = true;
    for (const auto& p : objectives_) {
      scoped_stage stage(res, "mosp.objective");
      p->after_apply(res, g, summary, delta);
      const framework::budget limit = framework::last_budget_report().limit;
      bounded = bounded && limit.host_syncs != framework::budget::unlimited;
      syncs += limit.host_syncs;
    }
    const finalize_counts counts = finalize(res, g, *state_, true);
    framework::stamp_result(state_->version, state_->graph_state, g);
    const framework::budget limit = bounded
                                        ? framework::budget::steady_state(syncs + counts.host_syncs)
                                        : framework::budget::unchecked();
    framework::check_budget("mosp", limit, before_used_.plus(phase.used()));
    // The stats (outside the measured phase: copying them may allocate).
    stats_.batch = summary;
    stats_.affected = counts.affected;
    stats_.combined_edges = counts.combined_edges;
    stats_.preference_scale = state_->scale;
    stats_.iterations = counts.solve.counters.iterations;
    stats_.frontier_visits = counts.solve.counters.pushes;
    stats_.converged = true;
    stats_.fallback_used = false;
    for (const sssp::stats& s : stats_.objectives) {
      stats_.iterations += s.iterations;
      stats_.frontier_visits += s.frontier_visits;
      stats_.converged = stats_.converged && s.converged;
      stats_.fallback_used = stats_.fallback_used || s.fallback_used;
    }
    stats_.engine_used = stats_.objectives.empty() ? counts.solve.engine_used
                                                   : stats_.objectives.front().engine_used;
    out_ = stats_;
  }

  /// @copydoc update_participant::poison
  void poison() noexcept override {
    state_->poisoned = true;
    for (const auto& p : objectives_) {
      p->poison();
    }
  }

 private:
  result_type* result_;
  state_type* state_ = nullptr;
  mosp::stats& out_;
  mosp::stats stats_;
  std::vector<std::unique_ptr<participant_type>> objectives_;
  budget_counters before_used_{};
};

template <typename vertex_t, typename edge_t, typename weight_t, typename distance_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_mosp_participant(
    mosp::result<vertex_t, distance_t>& r, mosp::stats& out) {
  static_assert(std::is_same_v<distance_t, std::int64_t>, "mosp: distance_t must be int64_t");
  return std::make_unique<mosp_problem<vertex_t, edge_t, weight_t, distance_t>>(r, out);
}

}  // namespace dyng::detail

namespace dyng::mosp {

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
int result<vertex_t, distance_t>::num_objectives() const noexcept {
  return impl_ ? impl_->num_objectives : 0;
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::distances(int objective) const {
  const auto& st = detail::checked(impl_.get(), "mosp::result::distances");
  DYNG_EXPECTS(objective >= 0 && objective < st.num_objectives,
               "mosp::result::distances: objective ", objective, " is out of range [0, ",
               st.num_objectives, ")");
  return st.objectives[static_cast<std::size_t>(objective)].distances();
}

template <typename vertex_t, typename distance_t>
array_view<const vertex_t> result<vertex_t, distance_t>::parents(int objective) const {
  const auto& st = detail::checked(impl_.get(), "mosp::result::parents");
  DYNG_EXPECTS(objective >= 0 && objective < st.num_objectives, "mosp::result::parents: objective ",
               objective, " is out of range [0, ", st.num_objectives, ")");
  return st.objectives[static_cast<std::size_t>(objective)].parents();
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::combined_distances() const {
  const auto& st = detail::checked(impl_.get(), "mosp::result::combined_distances");
  if (st.space == memory_space::host) {
    return host_view(st.combined_distances);
  }
  return array_view<const distance_t>(st.device_combined_distances.data(),
                                      st.device_combined_distances.size(), memory_space::device,
                                      st.device);
}

template <typename vertex_t, typename distance_t>
array_view<const vertex_t> result<vertex_t, distance_t>::combined_parents() const {
  const auto& st = detail::checked(impl_.get(), "mosp::result::combined_parents");
  if (st.space == memory_space::host) {
    return host_view(st.combined_parents);
  }
  return array_view<const vertex_t>(st.device_combined_parents.data(),
                                    st.device_combined_parents.size(), memory_space::device,
                                    st.device);
}

template <typename vertex_t, typename distance_t>
array_view<const distance_t> result<vertex_t, distance_t>::path_costs() const {
  const auto& st = detail::checked(impl_.get(), "mosp::result::path_costs");
  DYNG_EXPECTS(st.has_path_costs,
               "mosp::result::path_costs: the path costs were not computed "
               "(options::compute_path_costs was false at the last compute() or update()); set it "
               "with set_options() and they are computed by the next update()");
  return host_view(st.path_costs);
}

template <typename vertex_t, typename distance_t>
std::int64_t result<vertex_t, distance_t>::preference_scale() const noexcept {
  return impl_ ? impl_->scale : 0;
}

template <typename vertex_t, typename distance_t>
const options& result<vertex_t, distance_t>::get_options() const {
  return detail::checked(impl_.get(), "mosp::result::get_options").opt;
}

template <typename vertex_t, typename distance_t>
void result<vertex_t, distance_t>::set_options(const options& opt) {
  auto& st = detail::checked(impl_.get(), "mosp::result::set_options");
  DYNG_EXPECTS(opt.delta >= 0, "mosp::result::set_options: options.delta must be >= 0, got ",
               opt.delta);
  DYNG_EXPECTS(opt.preferences == st.opt.preferences,
               "mosp::result::set_options: the preferences are fixed at compute()");
  DYNG_EXPECTS(opt.num_objectives == st.opt.num_objectives,
               "mosp::result::set_options: num_objectives is fixed at compute() (",
               st.opt.num_objectives, "); got ", opt.num_objectives);
  for (int k = 0; k < st.num_objectives; ++k) {
    st.objectives[static_cast<std::size_t>(k)].set_options(detail::objective_options(opt, k));
  }
  st.opt = opt;
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
  const auto& st = detail::checked(impl_.get(), "mosp::result::clone");
  detail::expect_supported_backend(res, "mosp::result::clone");
  auto copy = std::make_unique<state_type>();
  copy->source = st.source;
  copy->opt = st.opt;
  copy->num_objectives = st.num_objectives;
  copy->scale = st.scale;
  copy->version = st.version;
  copy->graph_state = st.graph_state;
  copy->path_costs = st.path_costs;
  copy->has_path_costs = st.has_path_costs;
  copy->objectives.reserve(st.objectives.size());
  for (const auto& tree : st.objectives) {
    copy->objectives.push_back(tree.clone(res));
  }
  const std::size_t n = st.num_vertices();
  const memory_space from = st.space;
  const distance_t* d = from == memory_space::host ? st.combined_distances.data()
                                                   : st.device_combined_distances.data();
  const vertex_t* p =
      from == memory_space::host ? st.combined_parents.data() : st.device_combined_parents.data();
  // A result is complete when the call that produced it returns, so its arrays can be read on
  // any stream.
  const bool cuda = detail::is_cuda(res);
  const stream_ref stream = cuda ? res.stream() : stream_ref{};
  const int device = cuda ? res.device() : st.device;
  if (cuda) {
    copy->space = memory_space::device;
    copy->device = res.device();
    copy->device_combined_distances = buffer<distance_t>(res, n);
    copy->device_combined_parents = buffer<vertex_t>(res, n);
    if (n > 0) {
      detail::copy_bytes(copy->device_combined_distances.data(), memory_space::device, d, from,
                         n * sizeof(distance_t), stream, device);
      detail::copy_bytes(copy->device_combined_parents.data(), memory_space::device, p, from,
                         n * sizeof(vertex_t), stream, device);
      res.synchronize();
    }
  } else {
    copy->combined_distances.resize(n);
    copy->combined_parents.resize(n);
    copy->spare_distances.resize(n);
    copy->spare_parents.resize(n);
    if (n > 0) {
      detail::copy_bytes(copy->combined_distances.data(), memory_space::host, d, from,
                         n * sizeof(distance_t), stream, device);
      detail::copy_bytes(copy->combined_parents.data(), memory_space::host, p, from,
                         n * sizeof(vertex_t), stream, device);
      if (from != memory_space::host) {
        detail::cuda_synchronize(device, stream);
      }
    }
  }
  return result(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("mosp::result::clone (", impl_ ? impl_->num_vertices() : 0,
                                  " vertices)")

template <typename vertex_t, typename distance_t>
template <typename edge_t, typename weight_t>
result<vertex_t, distance_t> result<vertex_t, distance_t>::from_arrays(
    const resources& res, const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
    array_view<const array_view<const distance_t>> distances,
    array_view<const array_view<const vertex_t>> parents, bool canonicalize,
    const options& opt) try {
  static_assert(std::is_same_v<distance_t, std::int64_t>, "mosp: distance_t must be int64_t");
  detail::expect_supported_backend(res, "mosp::result::from_arrays");
  detail::expect_graph(g, "mosp::result::from_arrays");
  detail::graph_access::expect_placement(res, g, "mosp::result::from_arrays");
  const detail::resolved_options resolved =
      detail::resolve_options(opt, g.num_weights(), "mosp::result::from_arrays");
  const int num_k = resolved.num_objectives;
  DYNG_EXPECTS(is_host_accessible(distances.space()) && is_host_accessible(parents.space()),
               "mosp::result::from_arrays: the lists of arrays must be in host memory");
  DYNG_EXPECTS(
      static_cast<int>(distances.size()) == num_k && static_cast<int>(parents.size()) == num_k,
      "mosp::result::from_arrays: ", distances.size(), " distance and ", parents.size(),
      " parent arrays for ", num_k, " objective(s)");
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "mosp::result::from_arrays: source ", source,
               " is out of range [0, ", n, ")");
  auto st = std::make_unique<state_type>();
  st->source = source;
  st->opt = opt;
  st->num_objectives = num_k;
  st->scale = resolved.scale;
  st->objectives.reserve(static_cast<std::size_t>(num_k));
  for (int k = 0; k < num_k; ++k) {
    const auto i = static_cast<std::size_t>(k);
    st->objectives.push_back(sssp::result<vertex_t, distance_t>::from_arrays(
        res, g, source, distances[i], parents[i], canonicalize, detail::objective_options(opt, k)));
  }
  (void)detail::finalize(res, g, *st, false);
  detail::framework::stamp_result(st->version, st->graph_state, g);
  return result(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("mosp::result::from_arrays (", g.num_vertices(), " vertices)")

}  // namespace dyng::mosp

namespace dyng::detail {

// ------------------------------------------------------------------------------------------------
// compute / update
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
mosp::result<vertex_t> mosp_compute(const resources& res,
                                    const graph<vertex_t, edge_t, weight_t>& g, vertex_t source,
                                    const mosp::options& opt) try {
  expect_supported_backend(res, "mosp::compute");
  scoped_stage stage(res, "mosp.compute");
  expect_graph(g, "mosp::compute");
  graph_access::expect_placement(res, g, "mosp::compute");
  const resolved_options resolved = resolve_options(opt, g.num_weights(), "mosp::compute");
  const std::int64_t n = g.num_vertices();
  DYNG_EXPECTS(source >= 0 && source < n, "mosp::compute: source ", source, " is out of range [0, ",
               n, ")");
  const int num_k = resolved.num_objectives;
  auto st = std::make_unique<mosp_state<vertex_t, std::int64_t>>();
  st->source = source;
  st->opt = opt;
  st->num_objectives = num_k;
  st->scale = resolved.scale;
  st->objectives.reserve(static_cast<std::size_t>(num_k));
  for (int k = 0; k < num_k; ++k) {
    scoped_stage objective(res, "mosp.objective");
    st->objectives.push_back(sssp::compute(res, g, source, objective_options(opt, k)));
  }
  (void)finalize(res, g, *st, false);
  framework::stamp_result(st->version, st->graph_state, g);
  return mosp_access::make(std::move(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("mosp::compute (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges)")

template <typename vertex_t, typename edge_t, typename weight_t>
mosp::stats mosp_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                        const edge_batch_view<vertex_t, weight_t>& batch,
                        mosp::result<vertex_t>& r) try {
  scoped_stage stage(res, "mosp.update");
  mosp::stats out;
  mosp_problem<vertex_t, edge_t, weight_t, std::int64_t> participant(r, out);
  update_participant<vertex_t, edge_t, weight_t>* participants[] = {&participant};
  run_update(res, g, batch, participants, 1, "mosp.commit");
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("mosp::update (", g.num_vertices(), " vertices, ", g.num_edges(),
                                  " edges; batch of ", batch.num_insertions(), " insertions, ",
                                  batch.num_deletions(), " deletions)")

#if !DYNG_HAS_CUDA
// Without the cuda backend nothing can reach these (resources::cuda() throws); they exist so that
// the backend-neutral code above links.
template <typename vertex_t, typename edge_t, typename weight_t>
void mosp_cuda_workspace<vertex_t, edge_t, weight_t>::reserve(const resources& /*res*/,
                                                              std::int64_t /*n*/, int /*k*/) {
  throw not_supported_error("dyng: mosp: the cuda backend is not built");
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::size_t mosp_cuda_workspace<vertex_t, edge_t, weight_t>::bytes() const noexcept {
  return 0;
}

template <typename vertex_t, typename edge_t, typename weight_t>
mosp_combined<vertex_t, edge_t, weight_t> mosp_combine_cuda(
    const resources& /*res*/, const mosp_combine_input<vertex_t>& /*in*/,
    mosp_cuda_workspace<vertex_t, edge_t, weight_t>& /*ws*/) {
  throw not_supported_error("dyng: mosp: the cuda backend is not built");
}

template <typename vertex_t, typename edge_t, typename weight_t>
void mosp_download_tree_cuda(const resources& /*res*/, std::int64_t /*n*/,
                             const vertex_t* /*parents*/,
                             mosp_cuda_workspace<vertex_t, edge_t, weight_t>& /*ws*/) {
  throw not_supported_error("dyng: mosp: the cuda backend is not built");
}

#define DYNG_INSTANTIATE_MOSP_CUDA_STUB(V, E, W)                                           \
  template struct mosp_cuda_workspace<V, E, W>;                                            \
  template mosp_combined<V, E, W> mosp_combine_cuda<V, E, W>(                              \
      const resources&, const mosp_combine_input<V>&, mosp_cuda_workspace<V, E, W>&);      \
  template void mosp_download_tree_cuda<V, E, W>(const resources&, std::int64_t, const V*, \
                                                 mosp_cuda_workspace<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_CUDA_STUB)
#undef DYNG_INSTANTIATE_MOSP_CUDA_STUB
#endif  // !DYNG_HAS_CUDA

template mosp_state<std::int32_t, std::int64_t>& mosp_access::state(
    mosp::result<std::int32_t, std::int64_t>&);
template mosp_state<std::int64_t, std::int64_t>& mosp_access::state(
    mosp::result<std::int64_t, std::int64_t>&);

#define DYNG_INSTANTIATE_MOSP_DETAIL(V, E, W)                                                \
  template mosp::result<V> mosp_compute<V, E, W>(const resources&, const graph<V, E, W>&, V, \
                                                 const mosp::options&);                      \
  template mosp::stats mosp_update<V, E, W>(const resources&, graph<V, E, W>&,               \
                                            const edge_batch_view<V, W>&, mosp::result<V>&); \
  template std::unique_ptr<update_participant<V, E, W>>                                      \
  make_mosp_participant<V, E, W, std::int64_t>(mosp::result<V, std::int64_t>&, mosp::stats&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP_DETAIL)
#undef DYNG_INSTANTIATE_MOSP_DETAIL

}  // namespace dyng::detail

namespace dyng::mosp {

// ------------------------------------------------------------------------------------------------
// Explicit instantiations (PLAN Section 4.4.3)
// ------------------------------------------------------------------------------------------------

template class result<std::int32_t, std::int64_t>;
template class result<std::int64_t, std::int64_t>;

#define DYNG_INSTANTIATE_MOSP(V, E, W)                                                         \
  template result<V> result<V>::from_arrays<E, W>(                                             \
      const resources&, const graph<V, E, W>&, V,                                              \
      array_view<const array_view<const std::int64_t>>, array_view<const array_view<const V>>, \
      bool, const options&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_MOSP)
#undef DYNG_INSTANTIATE_MOSP

}  // namespace dyng::mosp
