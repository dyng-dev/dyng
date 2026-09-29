// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_sequential.cpp
// (update_static_histogram, apply_histogram_delta), src/dynamic/update_openmp.cpp
// (update_static_histogram_openmp), src/dynamic/update_cuda.cpp (update_static_histogram_cuda),
// src/core/histogram.cpp (CycleHistogram: increment, merge, total) and
// src/engine/histogram_engine.cpp (count_histogram, update_histogram)
/**
 * @file cycle_count.cpp
 * @brief cycle_count: argument validation, version checks, backend dispatch, the hooks of
 *        update() and result bookkeeping, and the explicit instantiations.
 *
 * update() follows CycleEnumeration-GPU's update_static_histogram(): prepare_batch() (here the
 * normalize hook, on G_t), the delete phase on the initial graph (count_minus), apply_batch() (the
 * commit, graph::apply), the insert phase on the post-batch graph (identify_affected + count_plus)
 * and apply_histogram_delta() (finalize). The OpenMP backend runs the phases of
 * update_static_histogram_openmp(); with one thread it runs the sequential phases, as the original
 * does. The CUDA backend runs the device phases of update_static_histogram_cuda() (cuda.cu) on the
 * resident device graph, whose commit merges the batch on the device under set semantics.
 *
 * Under batch_semantics::as_sets the framework normalizes the batch once (run_update; ADR 0020)
 * and the participant takes those lists instead of computing its own.
 */
#include "algorithms/cycle_count/problem.hpp"
#include "algorithms/cycle_count/work_queue.hpp"
#include "core/resources_access.hpp"
#include "framework/composition.hpp"
#include "framework/context.hpp"
#include "framework/enactor.hpp"
#include "framework/views.hpp"
#include "graph/graph_impl.hpp"
#include "graph/normalized_batch.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail {

// ------------------------------------------------------------------------------------------------
// Workspace and histogram arithmetic
// ------------------------------------------------------------------------------------------------

template <typename vertex_t>
void cycle_count_workspace<vertex_t>::reserve(int thread_count, std::size_t vertices) {
  const auto count = static_cast<std::size_t>(std::max(thread_count, 1));
  if (threads.size() < count) {
    threads.resize(count);
  }
  // The marks, stacks and counters themselves are sized by the thread that uses them, inside the
  // phase (first touch in parallel, as the original's thread_local buffers): see
  // cycle_count_thread::prepare().
  marks_needed = std::max(marks_needed, vertices);
}

template <typename vertex_t>
std::size_t cycle_count_workspace<vertex_t>::bytes() const noexcept {
  std::size_t total = change.deletions.capacity() * sizeof(edge_change<vertex_t>) +
                      change.insertions.capacity() * sizeof(edge_change<vertex_t>) +
                      change.requested.capacity() * sizeof(edge_change<vertex_t>) + index.bytes() +
                      (removed.capacity() + added.capacity()) * sizeof(std::uint64_t) +
                      threads.capacity() * sizeof(cycle_count_thread<vertex_t>);
  for (const cycle_count_thread<vertex_t>& t : threads) {
    total += t.visited.capacity() + t.stack.capacity() * sizeof(cycle_search_frame<vertex_t>) +
             t.partial.capacity() * sizeof(std::uint64_t);
  }
  return total;
}

template struct cycle_count_workspace<std::int32_t>;

void cycle_count_checked_add(std::uint64_t& a, std::uint64_t b) {
  if (b > std::numeric_limits<std::uint64_t>::max() - a) {
    throw capacity_error("dyng: cycle_count: a cycle count exceeds 2^64 - 1");
  }
  a += b;
}

// ------------------------------------------------------------------------------------------------
// Result access
// ------------------------------------------------------------------------------------------------

cycle_count_state& cycle_count_access::state(cycle_count::result& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "cycle_count::result: use of a moved-from result");
  return *r.impl_;
}

const cycle_count_state& cycle_count_access::state(const cycle_count::result& r) {
  DYNG_EXPECTS(r.impl_ != nullptr, "cycle_count::result: use of a moved-from result");
  return *r.impl_;
}

cycle_count::result cycle_count_access::make(std::unique_ptr<cycle_count_state> state) {
  return cycle_count::result(std::move(state));
}

namespace {

/// The backends cycle_count runs on: sequential, openmp (if built) and cuda (if built; resources of
/// a backend that is not built cannot be created).
void expect_supported_backend(const resources& res, const char* what) {
  if (res.get_backend() == backend::cuda && !DYNG_HAS_CUDA) {
    throw not_supported_error(
        concat_message("dyng: ", what, ": the cuda backend is not built; available: sequential",
                       DYNG_HAS_OPENMP ? ", openmp" : ""));
  }
}

/// The CUDA engine of the options (PLAN Section 4.5.4): automatic and fused run the fused kernels;
/// there is no operators engine yet.
void expect_cuda_engine(const cycle_count::options& opt, const char* what) {
  if (opt.cuda_engine == engine::operators) {
    throw not_supported_error(concat_message(
        "dyng: ", what,
        ": cycle_count has no operators engine on the cuda backend in this release; use "
        "engine::automatic or engine::fused (the fused work-queue and per-change kernels)"));
  }
}

/// The bound of the CUDA counters (kMaxDeviceCycleLength): the effective bound max(min(k, n), 2)
/// must be at most 64.
std::int64_t expect_device_length(const cycle_count::options& opt, std::int64_t vertices,
                                  const char* what) {
  const std::int64_t length = cycle_count_effective_length(opt.max_length, vertices);
  DYNG_EXPECTS(length <= cycle_count_max_device_length, what,
               ": the cuda backend counts cycles of up to ", cycle_count_max_device_length,
               " vertices (the paths of its searches live in thread-local arrays); options."
               "max_length = ",
               opt.max_length, " on a graph of ", vertices, " vertices needs ", length,
               ": use max_length <= ", cycle_count_max_device_length, " or a host backend");
  return length;
}

/// A failed update leaves the histogram inconsistent: every later use throws until it is
/// recomputed (PLAN Section 4.7.3).
void expect_not_poisoned(const cycle_count_state& state, const char* what) {
  if (state.poisoned) {
    throw stale_result_error(
        concat_message("dyng: ", what,
                       ": the result was left unusable by a failed update; recompute it with "
                       "cycle_count::compute()"));
  }
}

void expect_options(const cycle_count::options& opt) {
  DYNG_EXPECTS(opt.max_length == -1 || opt.max_length >= 2,
               "cycle_count: options.max_length must be -1 (no bound) or >= 2, got ",
               opt.max_length);
  DYNG_EXPECTS(opt.method == cycle_count::search_method::johnson, "cycle_count: options.method ",
               static_cast<int>(opt.method), " is not a search_method");
  DYNG_EXPECTS(opt.mode == cycle_count::cycle_mode::simple, "cycle_count: options.mode ",
               static_cast<int>(opt.mode),
               " is not a cycle_mode (time-window and temporal cycles follow in 0.4)");
  DYNG_EXPECTS(opt.cuda_engine == engine::automatic || opt.cuda_engine == engine::fused ||
                   opt.cuda_engine == engine::operators,
               "cycle_count: options.cuda_engine ", static_cast<int>(opt.cuda_engine),
               " is not an engine");
  DYNG_EXPECTS(opt.scheduler == cycle_count::cuda_scheduler::work_queue ||
                   opt.scheduler == cycle_count::cuda_scheduler::naive,
               "cycle_count: options.scheduler ", static_cast<int>(opt.scheduler),
               " is not a cuda_scheduler");
  DYNG_EXPECTS(
      static_cast<int>(opt.work_items) <= static_cast<int>(cycle_count::cuda_work_items::two_hop),
      "cycle_count: options.work_items ", static_cast<int>(opt.work_items),
      " is not a cuda_work_items");
}

/// The resident out-edges of a CUDA graph as the CUDA engines read them (uploaded on first use).
template <typename vertex_t, typename edge_t, typename weight_t>
cycle_device_graph<edge_t> device_engine_graph(const device_graph<vertex_t, edge_t, weight_t>& d) {
  cycle_device_graph<edge_t> out;
  out.vertex_count = static_cast<std::int64_t>(d.num_vertices);
  out.edge_count = static_cast<std::int64_t>(d.num_edges);
  out.offsets = d.out_row_ptr.data();
  out.neighbors = d.out_col_ind.data();
  return out;
}

/// The graph requirements of PLAN Section 5.1: the pruning of the searches relies on sorted rows of
/// a simple graph.
template <typename vertex_t, typename edge_t, typename weight_t>
void expect_graph(const graph<vertex_t, edge_t, weight_t>& g, const char* what) {
  const graph_properties& props = g.properties();
  DYNG_EXPECTS(props.parallel_edges == multi_edges::forbid && props.order == row_order::sorted,
               what,
               ": cycle_count needs multi_edges::forbid and row_order::sorted (its searches rely "
               "on sorted rows of a simple graph); build the graph with "
               "graph_properties::cycle_enum_compatible() or the default graph_properties");
}

/// Threads of the engines: the thread count of an OpenMP handle, 1 for the sequential backend.
int engine_threads(const resources& res) noexcept {
  return res.get_backend() == backend::openmp ? resources_access::host_threads(res) : 1;
}

/// The out-edges of a graph as the engines read them.
template <typename vertex_t, typename edge_t, typename weight_t>
cycle_graph<vertex_t, edge_t> engine_graph(const graph<vertex_t, edge_t, weight_t>& g) {
  const auto out = graph_access::out_view(g);
  cycle_graph<vertex_t, edge_t> cg;
  cg.vertex_count = static_cast<std::size_t>(out.num_vertices());
  cg.offsets = out.row_ptr.data();
  cg.neighbors = out.col_ind.data();
  return cg;
}

/// The longest length a histogram of `g` covers: min(max_length, max(n, 2)), or max(n, 2)
/// without a bound. No simple cycle is longer than the vertex count, so the clamp changes no count;
/// it keeps a large bound from costing memory and time (histograms, per-thread counters).
template <typename vertex_t, typename edge_t, typename weight_t>
std::int64_t histogram_bound(const cycle_count::options& opt,
                             const graph<vertex_t, edge_t, weight_t>& g) {
  const auto longest = std::max<std::int64_t>(static_cast<std::int64_t>(g.num_vertices()), 2);
  return opt.max_length >= 0 ? std::min<std::int64_t>(opt.max_length, longest) : longest;
}

/// Sum of a histogram, checked (CycleHistogram::total()).
std::uint64_t checked_total(const std::vector<std::uint64_t>& counts) {
  std::uint64_t sum = 0;
  for (const std::uint64_t c : counts) {
    cycle_count_checked_add(sum, c);
  }
  return sum;
}

/// The normalized lists of the framework as the engines' change lists.
template <typename vertex_t>
void copy_changes(const std::vector<set_change<vertex_t>>& from,
                  std::vector<edge_change<vertex_t>>& to) {
  to.resize(from.size());
  for (std::size_t i = 0; i < from.size(); ++i) {
    to[i] = {from[i].source, from[i].target};
  }
}

/// Under batch_semantics::as_sets the commit's normalized batch (apply_delta) must equal the lists
/// of Step 0 (Debug builds only: an O(batch) check of the graph module's two paths).
template <typename vertex_t, typename edge_t, typename weight_t>
void check_normalized_batch([[maybe_unused]] const graph<vertex_t, edge_t, weight_t>& g,
                            [[maybe_unused]] const apply_delta<vertex_t>& delta,
                            [[maybe_unused]] const cycle_count_workspace<vertex_t>& ws) {
#ifndef NDEBUG
  if (!g.properties().semantics.as_sets) {
    return;
  }
  const auto same = [](const std::vector<edge_change<vertex_t>>& list,
                       const std::vector<vertex_t>& src, const std::vector<vertex_t>& dst) {
    if (list.size() != src.size() || list.size() != dst.size()) {
      return false;
    }
    for (std::size_t i = 0; i < list.size(); ++i) {
      if (list[i].source != src[i] || list[i].target != dst[i]) {
        return false;
      }
    }
    return true;
  };
  if (!same(ws.change.deletions, delta.delete_src, delta.delete_dst) ||
      !same(ws.change.insertions, delta.insert_src, delta.insert_dst)) {
    DYNG_FAIL("cycle_count::update: the normalized batch of Step 0 differs from the commit's");
  }
#endif
}

/// apply_histogram_delta(): counts[len] += added[len] - removed[len] for every length; a bucket
/// that would become negative throws internal_error (the original's std::logic_error) before
/// anything is written. The histogram grows to the new bound (it covers min(max_length,
/// max(n, 2)), and n grows with the graph). Only the lengths the phases reached are visited.
/// @return The number of lengths whose count changed.
std::int64_t apply_histogram_delta(cycle_count_state& st, std::int64_t bound_after,
                                   const std::vector<std::uint64_t>& removed,
                                   const std::vector<std::uint64_t>& added) {
  const std::size_t touched = std::max(removed.size(), added.size());
  const auto delta_at = [&](std::size_t len, std::uint64_t& value, std::uint64_t& plus,
                            std::uint64_t& minus) {
    value = len < st.counts.size() ? st.counts[len] : 0;
    plus = len < added.size() ? added[len] : 0;
    minus = len < removed.size() ? removed[len] : 0;
  };
  // Check every bucket first, so a failure leaves the histogram as it was.
  for (std::size_t len = 0; len < touched; ++len) {
    std::uint64_t value = 0;
    std::uint64_t plus = 0;
    std::uint64_t minus = 0;
    delta_at(len, value, plus, minus);
    const std::uint64_t before = value;
    cycle_count_checked_add(value, plus);
    if (value < minus) {
      DYNG_FAIL("cycle_count::update would make the count of length ", len,
                " negative (the result does not match the graph: ", before, " + ", plus, " - ",
                minus, ")");
    }
  }
  const std::size_t size =
      std::max({static_cast<std::size_t>(bound_after) + 1, st.counts.size(), touched});
  st.counts.resize(size, 0);
  std::int64_t changed = 0;
  for (std::size_t len = 0; len < touched; ++len) {
    const std::uint64_t plus = len < added.size() ? added[len] : 0;
    const std::uint64_t minus = len < removed.size() ? removed[len] : 0;
    st.counts[len] = st.counts[len] + plus - minus;
    changed += plus != minus ? 1 : 0;
  }
  st.bound = static_cast<std::int64_t>(size) - 1;
  return changed;
}

/// The M2 participant of one cycle_count result, kept for the backends that do not run through
/// the framework's enactors yet (M3 migration: one backend after the other).
template <typename vertex_t, typename edge_t, typename weight_t>
class cycle_count_legacy_participant final : public update_participant<vertex_t, edge_t, weight_t> {
 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;
  using batch_type = edge_batch_view<vertex_t, weight_t>;
  using workspace_type = cycle_count_workspace<vertex_t>;

  cycle_count_legacy_participant(cycle_count::result& r, cycle_count::stats& out)
      : result_(r), out_(out) {}

  [[nodiscard]] const void* target() const noexcept override {
    return &result_;
  }

  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return false;  // the engines read the out-edges only (on cuda: graph_access::device_out)
  }

  void use_normalized(const normalized_batch<vertex_t>* normalized) noexcept override {
    normalized_ = normalized;
  }

  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    expect_supported_backend(res, "cycle_count::update");
    state_ = &cycle_count_access::state(result_);
    expect_not_poisoned(*state_, "cycle_count::update");
    if (state_->version != g.version()) {
      throw stale_result_error(concat_message(
          "dyng: cycle_count::update: the result matches graph version ", state_->version,
          " but the graph is at version ", g.version(),
          " (the graph was changed without updating this result; recompute it, or update all "
          "results together with dyng::update(res, g, batch, results...))"));
    }
    if (state_->graph_state != graph_access::impl(g).state_id) {
      throw stale_result_error(concat_message(
          "dyng: cycle_count::update: the result was computed on another graph (or on an earlier "
          "state of a graph variable that was reassigned since), although both are at version ",
          g.version(), "; recompute it on this graph"));
    }
    expect_graph(g, "cycle_count::update");
    graph_access::expect_placement(res, g, "cycle_count::update");
    cuda_ = res.get_backend() == backend::cuda;
    if (cuda_) {
      expect_cuda_engine(state_->opt, "cycle_count::update");
    }
    threads_ = cuda_ ? 1 : engine_threads(res);
    ws_.emplace(resources_access::workspaces(res).acquire<workspace_type>());
    workspace_type& ws = ws_->get();

    // ---- Step 0: the net structural change, on G_t (prepare_batch) ----
    if (normalized_ != nullptr) {
      // Normalized once by the framework (run_update, stage <algo>.normalize).
      copy_changes(normalized_->deletions, ws.change.deletions);
      copy_changes(normalized_->insertions, ws.change.insertions);
    } else {
      cycle_count_hook(res, "cycle_count.normalize", [&] {
        compute_structural_change(graph_access::out_view(g), batch, g.properties(), ws.change);
      });
    }
    if (cuda_) {
      // The effective bound after the batch, with every vertex the batch may add (the original's
      // next_vertex_count), must fit the device counters; checked before anything changes.
      std::int64_t n_after = static_cast<std::int64_t>(g.num_vertices());
      if (normalized_ != nullptr) {
        n_after = normalized_->vertices_after;
      } else {
        for (std::size_t i = 0; i < batch.insert_src.size(); ++i) {
          n_after = std::max<std::int64_t>(
              n_after, std::max<std::int64_t>(batch.insert_src[i], batch.insert_dst[i]) + 1);
        }
      }
      device_length_ = expect_device_length(state_->opt, n_after, "cycle_count::update");
    }

    // ---- before_apply = count(-): the cycles through the deleted edges, on G_t ----
    bound_before_ = histogram_bound(state_->opt, g);
    cycle_count_hook(res, "cycle_count.count_minus", [&] {
      if (cuda_) {
        count_minus_cuda(res, g);
        return;
      }
      const auto max_length = static_cast<std::size_t>(bound_before_);
      ws.reserve(threads_, static_cast<std::size_t>(g.num_vertices()));
      ws.removed.clear();
      ws.index.assign(ws.change.deletions);
      run_phase(engine_graph(g), ws.change.deletions, max_length, ws, ws.removed);
    });
  }

  void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                   const apply_delta<vertex_t>& delta) override {
    cycle_count_state& st = *state_;
    workspace_type& ws = ws_->get();
    check_normalized_batch(g, delta, ws);
    const std::int64_t bound_after = histogram_bound(st.opt, g);
    const auto max_length = static_cast<std::size_t>(bound_after);

    if (cuda_) {
      after_apply_cuda(res, g, bound_after);
    } else {
      // ---- identify_affected: the inserted edges and their ownership index, on G_{t+1} ----
      cycle_count_hook(res, "cycle_count.identify_affected", [&] {
#if !defined(DYNG_MUTATION_SKIP_WORKSPACE_RESIZE)
        ws.reserve(threads_, static_cast<std::size_t>(g.num_vertices()));
#endif
        ws.added.clear();
        ws.index.assign(ws.change.insertions);
      });

      // ---- count(+): the cycles through the inserted edges, on G_{t+1} ----
      cycle_count_hook(res, "cycle_count.count_plus", [&] {
        run_phase(engine_graph(g), ws.change.insertions, max_length, ws, ws.added);
      });
    }

    // ---- finalize: apply the signed delta (apply_histogram_delta) ----
    cycle_count::stats s;
    cycle_count_hook(res, "cycle_count.finalize", [&] {
      s.affected = apply_histogram_delta(st, bound_after, ws.removed, ws.added);
      s.cycles_removed = checked_total(ws.removed);
      s.cycles_added = checked_total(ws.added);
    });
    s.iterations = 0;
    s.frontier_visits =
        static_cast<std::int64_t>(ws.change.deletions.size() + ws.change.insertions.size());
    s.fallback_used = false;
    s.converged = true;
    s.engine_used = cuda_ ? engine::fused : engine::operators;
    s.batch = summary;
    s.deletions = static_cast<std::int64_t>(ws.change.deletions.size());
    s.insertions = static_cast<std::int64_t>(ws.change.insertions.size());
    out_ = s;
    st.version = g.version();
    st.graph_state = graph_access::impl(g).state_id;
    cuda_ws_.reset();
    ws_.reset();  // return the workspaces to the pool
  }

  void poison() noexcept override {
    if (state_ != nullptr) {
      state_->poisoned = true;
    }
  }

 private:
  using cuda_workspace_type = cycle_count_cuda_workspace<edge_t>;

  /// CUDA, before_apply: the change lists on the device (the framework's copy, shared with the
  /// device apply, or the workspace's own) and the delete phase on the resident G_t
  /// (count_update_cycles_device up to its "Delete phase on G_t").
  void count_minus_cuda([[maybe_unused]] const resources& res,
                        [[maybe_unused]] const graph_type& g) {
#if DYNG_HAS_CUDA
    workspace_type& ws = ws_->get();
    cuda_ws_.emplace(resources_access::workspaces(res).acquire<cuda_workspace_type>(res));
    cuda_workspace_type& cws = cuda_ws_->get();
    constexpr std::int64_t id_limit = no_change_id;
    if (static_cast<std::int64_t>(ws.change.deletions.size()) >= id_limit ||
        static_cast<std::int64_t>(ws.change.insertions.size()) >= id_limit) {
      // A size limit of the backend, not a malformed batch (PLAN 4.7.3); nothing is changed yet.
      throw capacity_error(
          "dyng: cycle_count::update: the batch exceeds the 32-bit change ids of the cuda "
          "backend (at most 2,139,062,142 deletions and as many insertions after normalization)");
    }
    if constexpr (sizeof(vertex_t) == 4) {
      device_changes_ = normalized_ != nullptr
                            ? upload_normalized_batch(res, *normalized_)
                            : cycle_count_cuda_upload_changes(res, ws.change.deletions,
                                                              ws.change.insertions, cws);
    }
    cycle_count_cuda_begin_update(res, cws);
    const auto& d = graph_access::device_out(res, g);
    // Under set semantics the deletion marks of G_t are the framework's (computed once, shared
    // with the device apply); otherwise the phase marks them in the workspace.
    const std::int32_t* owner = nullptr;
    if constexpr (sizeof(vertex_t) == 4) {
      if (normalized_ != nullptr) {
        owner = mark_normalized_deletions(res, d, *normalized_);
      }
    }
    cycle_count_cuda_phase(res, device_engine_graph(d), owner, device_changes_,
                           static_cast<std::uint32_t>(ws.change.deletions.size()), device_length_,
                           0, cws);
#else
    throw not_supported_error("dyng: cycle_count::update: the cuda backend is not built");
#endif
  }

  /// CUDA, after_apply: the insert phase on G_{t+1} (the graph's resident copy: merged on the
  /// device by the commit, whose insertion ids are the owner array, or uploaded after a host
  /// commit) and the copy of both histograms.
  void after_apply_cuda([[maybe_unused]] const resources& res, [[maybe_unused]] const graph_type& g,
                        [[maybe_unused]] std::int64_t bound_after) {
#if DYNG_HAS_CUDA
    workspace_type& ws = ws_->get();
    cuda_workspace_type& cws = cuda_ws_->get();
    const device_graph<vertex_t, edge_t, weight_t>* d = nullptr;
    cycle_count_hook(res, "cycle_count.identify_affected",
                     [&] { d = &graph_access::device_out(res, g); });
    cycle_count_hook(res, "cycle_count.count_plus", [&] {
      const bool merged = d->insertion_ids.size() >= static_cast<std::size_t>(d->num_edges) &&
                          !d->insertion_ids.empty() && normalized_ != nullptr;
      const std::uint32_t* insertions =
          device_changes_ + 2 * static_cast<std::size_t>(ws.change.deletions.size());
      cycle_count_cuda_phase(
          res, device_engine_graph(*d), merged ? d->insertion_ids.data() : nullptr, insertions,
          static_cast<std::uint32_t>(ws.change.insertions.size()), device_length_, 1, cws);
      cycle_count_cuda_end_update(res, device_length_, cws, ws.removed, ws.added);
      // Lengths past the bound after the batch hold no cycle (a simple cycle has at most n
      // vertices); the device bound may be larger when the batch names vertices it does not add.
      const auto size = static_cast<std::size_t>(bound_after) + 1;
      for (std::vector<std::uint64_t>* h : {&ws.removed, &ws.added}) {
        for (std::size_t len = size; len < h->size(); ++len) {
          if ((*h)[len] != 0) {
            DYNG_FAIL("cycle_count::update (cuda): a cycle of length ", len,
                      " on a graph whose histogram ends at ", bound_after);
          }
        }
        if (h->size() > size) {
          h->resize(size);
        }
      }
    });
#else
    throw not_supported_error("dyng: cycle_count::update: the cuda backend is not built");
#endif
  }

  /// One phase on the backend of the update: update_static_histogram_openmp() parallelizes a phase
  /// over its change edges when it has more than one thread, and runs the sequential phase
  /// otherwise.
  void run_phase(const cycle_graph<vertex_t, edge_t>& graph,
                 const std::vector<edge_change<vertex_t>>& changes, std::size_t max_length,
                 workspace_type& ws, std::vector<std::uint64_t>& phase) const {
    if (threads_ > 1) {
      cycle_count_openmp_phase(graph, changes, max_length, threads_, ws, phase);
    } else {
      cycle_count_sequential_phase(graph, changes, max_length, ws, phase);
    }
  }

  cycle_count::result& result_;
  cycle_count::stats& out_;
  cycle_count_state* state_ = nullptr;
  int threads_ = 1;
  std::int64_t bound_before_ = 2;
  bool cuda_ = false;                                       ///< the update runs on the cuda backend
  std::int64_t device_length_ = 2;                          ///< the bound of the device phases
  const normalized_batch<vertex_t>* normalized_ = nullptr;  ///< the framework's Step 0, if any
  const std::uint32_t* device_changes_ = nullptr;           ///< the device change lists (pairs)
  std::optional<workspace_pool::lease<workspace_type>> ws_;
  std::optional<workspace_pool::lease<cuda_workspace_type>> cuda_ws_;
};

}  // namespace

// ------------------------------------------------------------------------------------------------
// cycle_count_problem: the hooks (problem.hpp), run by the framework's enactors
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::begin_update(framework::context& ctx,
                                                                   old_graph g,
                                                                   const requested& /*batch*/) {
  const resources& res = ctx.res();
  expect_supported_backend(res, "cycle_count::update");
  state_ = &cycle_count_access::state(*result_);
  expect_not_poisoned(*state_, "cycle_count::update");
  const container_type& graph = g.get();
  framework::expect_current_result("cycle_count::update", state_->version, state_->graph_state,
                                   graph);
  expect_graph(graph, "cycle_count::update");
  graph_access::expect_placement(res, graph, "cycle_count::update");
  cuda_ = ctx.on_cuda();
  threads_ = cuda_ ? 1 : engine_threads(res);
  // The host scratch (change lists, ownership index, per-thread marks and counters), shared with
  // every result updated through `res` (ADR 0015); returned in end_update().
  ws_.emplace(ctx.workspaces().acquire<workspace_type>());
}

/// Step 0 on G_t (prepare_batch).
template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::normalize(framework::context& /*ctx*/,
                                                                old_graph g,
                                                                const requested& batch) {
  workspace_type& ws = ws_->get();
  const container_type& graph = g.get();
  normalized_ = batch.normalized;
  if (normalized_ != nullptr) {
    // Normalized once by the framework (run_update(), stage <algo>.normalize; ADR 0020): its lists
    // become the engines' change lists.
    copy_changes(normalized_->deletions, ws.change.deletions);
    copy_changes(normalized_->insertions, ws.change.insertions);
  } else {
    compute_structural_change(graph_access::out_view(graph), batch.edges, graph.properties(),
                              ws.change);
  }
}

/// before_apply = count(-): the owned cycles through the deleted edges, on G_t (the delete phase
/// of update_static_histogram).
template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::count(framework::context& /*ctx*/,
                                                            old_graph g, frontier& /*f*/,
                                                            framework::sign s,
                                                            ownership_type /*rule*/) {
  (void)s;  // sign::minus: the enactor subtracts on G_t
  workspace_type& ws = ws_->get();
  const container_type& graph = g.get();
  bound_before_ = histogram_bound(state_->opt, graph);
  ws.reserve(threads_, static_cast<std::size_t>(graph.num_vertices()));
  ws.removed.clear();
  ws.index.assign(ws.change.deletions);  // ownership::min_member over the deletions
  run_host_phase(graph, ws.change.deletions, static_cast<std::size_t>(bound_before_), ws.removed);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::resume(framework::context& /*ctx*/,
                                                             new_graph g, const applied& applied) {
  const container_type& graph = g.get();
  check_normalized_batch(graph, applied.delta, ws_->get());
  bound_after_ = histogram_bound(state_->opt, graph);
}

/// The inserted edges and their ownership index, on G_{t+1}.
template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::identify_affected(
    framework::context& /*ctx*/, [[maybe_unused]] new_graph g, const applied& /*applied*/,
    frontier& /*f*/) {
  workspace_type& ws = ws_->get();
#if !defined(DYNG_MUTATION_SKIP_WORKSPACE_RESIZE)
  ws.reserve(threads_, static_cast<std::size_t>(g->num_vertices()));
#endif
  ws.added.clear();
  ws.index.assign(ws.change.insertions);  // ownership::min_member over the insertions
}

/// update(): count(+), the owned cycles through the inserted edges on G_{t+1} (the insert phase);
/// compute(): the static count (count_simple_cycles_johnson[_openmp]).
template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::count(framework::context& ctx, new_graph g,
                                                            frontier& /*f*/, framework::sign s,
                                                            ownership_type /*rule*/) {
  (void)s;  // sign::plus: the enactor adds on G_{t+1} (and counts the whole graph in compute())
  const container_type& graph = g.get();
  if (computing_) {
    const cycle_graph<vertex_t, edge_t> cg = engine_graph(graph);
    if (ctx.get_backend() == backend::openmp) {
      cycle_count_openmp_compute(cg, state_->opt.max_length, engine_threads(ctx.res()),
                                 state_->counts);
    } else {
      cycle_count_sequential_compute(cg, state_->opt.max_length, state_->counts);
    }
    return;
  }
  workspace_type& ws = ws_->get();
  run_host_phase(graph, ws.change.insertions, static_cast<std::size_t>(bound_after_), ws.added);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::finalize(framework::context& /*ctx*/,
                                                               stats_type& stats) {
  if (computing_) {
    framework::stamp_result(state_->version, state_->graph_state, *static_graph_);
    return;
  }
  // apply_histogram_delta: the signed delta, checked before anything is written.
  const workspace_type& ws = ws_->get();
  stats.affected = apply_histogram_delta(*state_, bound_after_, ws.removed, ws.added);
  stats.cycles_removed = checked_total(ws.removed);
  stats.cycles_added = checked_total(ws.added);
  fill_counts(stats);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::fill_counts(stats_type& stats) const {
  const workspace_type& ws = ws_->get();
  stats.iterations = 0;
  stats.frontier_visits =
      static_cast<std::int64_t>(ws.change.deletions.size() + ws.change.insertions.size());
  stats.deletions = static_cast<std::int64_t>(ws.change.deletions.size());
  stats.insertions = static_cast<std::int64_t>(ws.change.insertions.size());
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::end_update(framework::context& /*ctx*/,
                                                                 new_graph g,
                                                                 const stats_type& /*stats*/) {
  framework::stamp_result(state_->version, state_->graph_state, g.get());
  ws_.reset();  // return the workspace to the pool
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::poison() noexcept {
  if (state_ != nullptr) {
    state_->poisoned = true;
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::bind_static(framework::context& /*ctx*/,
                                                                  new_graph g) {
  static_graph_ = &g.get();
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::reset(framework::context& /*ctx*/) {
  state_->counts.assign(static_cast<std::size_t>(state_->bound) + 1, 0);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void cycle_count_problem<vertex_t, edge_t, weight_t>::run_host_phase(
    const container_type& g, const std::vector<edge_change<vertex_t>>& changes,
    std::size_t max_length, std::vector<std::uint64_t>& phase) {
  workspace_type& ws = ws_->get();
  if (threads_ > 1) {
    cycle_count_openmp_phase(engine_graph(g), changes, max_length, threads_, ws, phase);
  } else {
    cycle_count_sequential_phase(engine_graph(g), changes, max_length, ws, phase);
  }
}

namespace {

/// Whether a backend runs cycle_count through the framework's enactors (M3 migration: one backend
/// after the other; the others keep the M2 participant until their own commit).
bool on_framework(const resources& res) noexcept {
  return res.get_backend() == backend::sequential;
}

/// The participant of one result during the M3 migration: the framework's participant of
/// cycle_count_problem on the migrated backends, the M2 participant on the others. The backend is
/// known only in before_apply(), so the choice is made there.
template <typename vertex_t, typename edge_t, typename weight_t>
class cycle_count_migration_participant final
    : public update_participant<vertex_t, edge_t, weight_t> {
 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;
  using batch_type = edge_batch_view<vertex_t, weight_t>;
  using problem_type = cycle_count_problem<vertex_t, edge_t, weight_t>;

  cycle_count_migration_participant(cycle_count::result& r, cycle_count::stats& out)
      : result_(r), out_(out) {}

  [[nodiscard]] const void* target() const noexcept override {
    return &result_;
  }

  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return false;  // both read the out-edges only
  }

  void use_normalized(const normalized_batch<vertex_t>* normalized) noexcept override {
    normalized_ = normalized;
  }

  void before_apply(const resources& res, const graph_type& g, const batch_type& batch) override {
    if (on_framework(res)) {
      chosen_ = &framework_.emplace(out_, result_);
    } else {
      chosen_ = &legacy_.emplace(result_, out_);
    }
    chosen_->use_normalized(normalized_);
    chosen_->before_apply(res, g, batch);
  }

  void after_apply(const resources& res, const graph_type& g, const apply_summary& summary,
                   const apply_delta<vertex_t>& delta) override {
    chosen_->after_apply(res, g, summary, delta);
  }

  void poison() noexcept override {
    if (chosen_ != nullptr) {
      chosen_->poison();
    }
  }

 private:
  cycle_count::result& result_;
  cycle_count::stats& out_;
  const normalized_batch<vertex_t>* normalized_ = nullptr;
  std::optional<framework::problem_participant<problem_type>> framework_;
  std::optional<cycle_count_legacy_participant<vertex_t, edge_t, weight_t>> legacy_;
  update_participant<vertex_t, edge_t, weight_t>* chosen_ = nullptr;
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_cycle_count_participant(
    cycle_count::result& r, cycle_count::stats& out) {
  return std::make_unique<cycle_count_migration_participant<vertex_t, edge_t, weight_t>>(r, out);
}

}  // namespace dyng::detail

namespace dyng::cycle_count {

// ------------------------------------------------------------------------------------------------
// result
// ------------------------------------------------------------------------------------------------

result::result(std::unique_ptr<detail::cycle_count_state> state) noexcept
    : impl_(std::move(state)) {}

result::result(result&& other) noexcept = default;

result& result::operator=(result&& other) noexcept = default;

result::~result() = default;

array_view<const std::uint64_t> result::counts() const {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::counts");
  return host_view(st.counts);
}

std::uint64_t result::count(std::int64_t length) const {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::count");
  if (length < 2 || length >= static_cast<std::int64_t>(st.counts.size())) {
    return 0;
  }
  return st.counts[static_cast<std::size_t>(length)];
}

std::uint64_t result::total() const {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::total");
  return detail::checked_total(st.counts);
}

std::int64_t result::bound() const {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::bound");
  return st.bound;
}

const options& result::get_options() const {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::get_options");
  return st.opt;
}

void result::set_options(const options& opt) {
  detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::set_options");
  detail::expect_options(opt);
  DYNG_EXPECTS(
      opt.max_length == st.opt.max_length && opt.method == st.opt.method && opt.mode == st.opt.mode,
      "cycle_count::result::set_options: max_length, method and mode are fixed at "
      "compute() (the histogram counts cycles under them; recompute to change them); "
      "only cuda_engine, scheduler and work_items can change");
  st.opt = opt;
}

std::uint64_t result::graph_version() const noexcept {
  return impl_ == nullptr ? 0 : impl_->version;
}

memory_space result::space() const noexcept {
  return memory_space::host;
}

result result::clone(const resources& res) const try {
  const detail::cycle_count_state& st = detail::cycle_count_access::state(*this);
  detail::expect_not_poisoned(st, "cycle_count::result::clone");
  detail::expect_supported_backend(res, "cycle_count::result::clone");
  return detail::cycle_count_access::make(std::make_unique<detail::cycle_count_state>(st));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("cycle_count::result::clone")

// ------------------------------------------------------------------------------------------------
// compute (count_histogram) and update (update_histogram)
// ------------------------------------------------------------------------------------------------

}  // namespace dyng::cycle_count

namespace dyng::detail {

template <typename vertex_t, typename edge_t, typename weight_t>
cycle_count::result cycle_count_compute(const resources& res,
                                        const graph<vertex_t, edge_t, weight_t>& g,
                                        const cycle_count::options& opt) try {
  scoped_stage stage(res, "cycle_count.compute");
  detail::expect_supported_backend(res, "cycle_count::compute");
  detail::expect_options(opt);
  detail::expect_graph(g, "cycle_count::compute");
  detail::graph_access::expect_placement(res, g, "cycle_count::compute");

  auto state = std::make_unique<detail::cycle_count_state>();
  state->opt = opt;
  state->bound = detail::histogram_bound(opt, g);
  if (res.get_backend() == backend::cuda) {
    detail::expect_cuda_engine(opt, "cycle_count::compute");
    state->counts.assign(static_cast<std::size_t>(state->bound) + 1, 0);
    if (g.num_edges() == 0) {
      // No edge, no cycle: the original's device counters return an empty histogram before they
      // check the length bound, so an edgeless graph of any size counts (update() checks the
      // bound once a batch adds edges).
      state->version = g.version();
      state->graph_state = detail::graph_access::impl(g).state_id;
      return detail::cycle_count_access::make(std::move(state));
    }
    const std::int64_t length = detail::expect_device_length(
        opt, static_cast<std::int64_t>(g.num_vertices()), "cycle_count::compute");
#if DYNG_HAS_CUDA
    // count_simple_cycles_johnson[_queue]_device on the resident graph (stages cycle_count.reset,
    // cycle_count.count and cycle_count.finalize inside).
    const auto& d = detail::graph_access::device_out(res, g);
    auto ws = detail::resources_access::workspaces(res)
                  .acquire<detail::cycle_count_cuda_workspace<edge_t>>(res);
    detail::cycle_count_cuda_compute(res, detail::device_engine_graph(d), length, opt, ws.get(),
                                     state->counts);
#else
    (void)length;
#endif
    state->version = g.version();
    state->graph_state = detail::graph_access::impl(g).state_id;
    return detail::cycle_count_access::make(std::move(state));
  }
  if (detail::on_framework(res)) {
    // The static enactor: reset -> count -> finalize.
    using problem_type = detail::cycle_count_problem<vertex_t, edge_t, weight_t>;
    problem_type problem(*state);
    detail::framework::context ctx(res, problem_type::name);
    const detail::framework::new_view<graph<vertex_t, edge_t, weight_t>> view(g);
    problem.bind_static(ctx, view);
    (void)detail::framework::static_enactor<problem_type>(problem).run(ctx, view);
    return detail::cycle_count_access::make(std::move(state));
  }
  const detail::cycle_graph<vertex_t, edge_t> cg = detail::engine_graph(g);
  detail::cycle_count_hook(res, "cycle_count.reset", [&] {
    state->counts.assign(static_cast<std::size_t>(state->bound) + 1, 0);
  });
  detail::cycle_count_hook(res, "cycle_count.count", [&] {
    if (res.get_backend() == backend::openmp) {
      detail::cycle_count_openmp_compute(cg, opt.max_length, detail::engine_threads(res),
                                         state->counts);
    } else {
      detail::cycle_count_sequential_compute(cg, opt.max_length, state->counts);
    }
  });
  detail::cycle_count_hook(res, "cycle_count.finalize", [&] {
    state->version = g.version();
    state->graph_state = detail::graph_access::impl(g).state_id;
  });
  return detail::cycle_count_access::make(std::move(state));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("cycle_count::compute (", g.num_vertices(), " vertices, ",
                                  g.num_edges(), " edges, max_length ", opt.max_length, ")")

template <typename vertex_t, typename edge_t, typename weight_t>
cycle_count::stats cycle_count_update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
                                      const edge_batch_view<vertex_t, weight_t>& batch,
                                      cycle_count::result& r) try {
  scoped_stage stage(res, "cycle_count.update");
  cycle_count::stats out;
  detail::cycle_count_migration_participant<vertex_t, edge_t, weight_t> participant(r, out);
  detail::update_participant<vertex_t, edge_t, weight_t>* participants[] = {&participant};
  detail::run_update(res, g, batch, participants, 1, "cycle_count.commit");
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("cycle_count::update (", g.num_vertices(), " vertices, ",
                                  g.num_edges(), " edges; batch of ", batch.num_insertions(),
                                  " insertions, ", batch.num_deletions(), " deletions)")

// The instantiated types (PLAN Section 4.4.3): int32 vertices, int32 or int64 offsets, unweighted
// or int32 weights (ignored).
#define DYNG_FOR_EACH_CYCLE_COUNT_TYPE(X)           \
  X(std::int32_t, std::int32_t, ::dyng::unweighted) \
  X(std::int32_t, std::int64_t, ::dyng::unweighted) \
  X(std::int32_t, std::int32_t, std::int32_t)       \
  X(std::int32_t, std::int64_t, std::int32_t)

#define DYNG_INSTANTIATE_CYCLE_COUNT(V, E, W)                                \
  static_assert(cycle_count_supported_v<V, E, W>);                           \
  template cycle_count::result cycle_count_compute<V, E, W>(                 \
      const resources&, const graph<V, E, W>&, const cycle_count::options&); \
  template cycle_count::stats cycle_count_update<V, E, W>(                   \
      const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, cycle_count::result&);
DYNG_FOR_EACH_CYCLE_COUNT_TYPE(DYNG_INSTANTIATE_CYCLE_COUNT)
#undef DYNG_INSTANTIATE_CYCLE_COUNT

#define DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT(V, E, W)                                      \
  template std::unique_ptr<update_participant<V, E, W>> make_cycle_count_participant<V, E, W>( \
      cycle_count::result&, cycle_count::stats&);
DYNG_FOR_EACH_CYCLE_COUNT_TYPE(DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT)
#undef DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT
#undef DYNG_FOR_EACH_CYCLE_COUNT_TYPE

}  // namespace dyng::detail
