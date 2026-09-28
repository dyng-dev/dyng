// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/update_sequential.cpp
// (update_static_histogram, apply_histogram_delta), src/dynamic/update_openmp.cpp
// (update_static_histogram_openmp), src/core/histogram.cpp (CycleHistogram: increment, merge,
// total) and src/engine/histogram_engine.cpp (count_histogram, update_histogram)
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
 * does.
 */
#include "algorithms/cycle_count/problem.hpp"
#include "core/resources_access.hpp"
#include "graph/graph_impl.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
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
void cycle_count_workspace<vertex_t>::reserve(int threads, std::size_t vertices,
                                              std::size_t max_length) {
  const auto count = static_cast<std::size_t>(std::max(threads, 1));
  if (visited.size() < count) {
    visited.resize(count);
  }
  for (std::size_t t = 0; t < count; ++t) {
    if (visited[t].size() < vertices) {
      visited[t].resize(vertices, 0);
    }
  }
  // One cache line is 8 counters: round up, then keep one spare line between two threads.
  constexpr std::size_t line = 8;
  const std::size_t needed = (max_length + 1 + line - 1) / line * line + line;
  if (stride < needed || counts.size() < count * stride) {
    stride = std::max(stride, needed);
    counts.assign(count * stride, 0);
    partial.assign(count * stride, 0);
  }
}

template <typename vertex_t>
std::size_t cycle_count_workspace<vertex_t>::bytes() const noexcept {
  std::size_t total =
      change.deletions.capacity() * sizeof(edge_change<vertex_t>) +
      change.insertions.capacity() * sizeof(edge_change<vertex_t>) +
      change.requested.capacity() * sizeof(edge_change<vertex_t>) + index.bytes() +
      (counts.capacity() + partial.capacity() + removed.capacity() + added.capacity()) *
          sizeof(std::uint64_t);
  for (const std::vector<char>& marks : visited) {
    total += marks.capacity();
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

/// The backends cycle_count runs on in this release (the cuda backend arrives with M2b).
void expect_supported_backend(const resources& res, const char* what) {
  if (res.get_backend() == backend::cuda) {
    throw not_supported_error(concat_message(
        "dyng: ", what, ": cycle_count has no cuda backend in this release; available: sequential",
        DYNG_HAS_OPENMP ? ", openmp" : ""));
  }
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

/// The longest length a histogram of `g` covers: the bound, or max(n, 2) without one (no simple
/// cycle is longer than the vertex count).
template <typename vertex_t, typename edge_t, typename weight_t>
std::int64_t histogram_bound(const cycle_count::options& opt,
                             const graph<vertex_t, edge_t, weight_t>& g) {
  return opt.max_length >= 0
             ? opt.max_length
             : std::max<std::int64_t>(static_cast<std::int64_t>(g.num_vertices()), 2);
}

/// Sum of a histogram, checked (CycleHistogram::total()).
std::uint64_t checked_total(const std::vector<std::uint64_t>& counts) {
  std::uint64_t sum = 0;
  for (const std::uint64_t c : counts) {
    cycle_count_checked_add(sum, c);
  }
  return sum;
}

/// The participant of one cycle_count result in run_update() (and dyng::update()).
template <typename vertex_t, typename edge_t, typename weight_t>
class cycle_count_participant final : public update_participant<vertex_t, edge_t, weight_t> {
 public:
  using graph_type = graph<vertex_t, edge_t, weight_t>;
  using batch_type = edge_batch_view<vertex_t, weight_t>;
  using workspace_type = cycle_count_workspace<vertex_t>;

  cycle_count_participant(cycle_count::result& r, cycle_count::stats& out)
      : result_(r), out_(out) {}

  [[nodiscard]] const void* target() const noexcept override {
    return &result_;
  }

  [[nodiscard]] bool reads_prepared_graph() const noexcept override {
    return false;  // the engines read the out-edges only
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
    threads_ = engine_threads(res);
    ws_.emplace(resources_access::workspaces(res).acquire<workspace_type>());
    workspace_type& ws = ws_->get();

    // ---- Step 0: the net structural change, on G_t (prepare_batch) ----
    cycle_count_hook(res, "cycle_count.normalize", [&] {
      compute_structural_change(graph_access::out_view(g), batch, g.properties(), ws.change);
    });

    // ---- before_apply = count(-): the cycles through the deleted edges, on G_t ----
    bound_before_ = histogram_bound(state_->opt, g);
    cycle_count_hook(res, "cycle_count.count_minus", [&] {
      const auto max_length = static_cast<std::size_t>(bound_before_);
      ws.reserve(threads_, static_cast<std::size_t>(g.num_vertices()), max_length);
      ws.removed.assign(max_length + 1, 0);
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

    // ---- identify_affected: the inserted edges and their ownership index, on G_{t+1} ----
    cycle_count_hook(res, "cycle_count.identify_affected", [&] {
      ws.reserve(threads_, static_cast<std::size_t>(g.num_vertices()), max_length);
      ws.added.assign(max_length + 1, 0);
      ws.index.assign(ws.change.insertions);
    });

    // ---- count(+): the cycles through the inserted edges, on G_{t+1} ----
    cycle_count_hook(res, "cycle_count.count_plus", [&] {
      run_phase(engine_graph(g), ws.change.insertions, max_length, ws, ws.added);
    });

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
    s.engine_used = engine::operators;
    s.batch = summary;
    s.deletions = static_cast<std::int64_t>(ws.change.deletions.size());
    s.insertions = static_cast<std::int64_t>(ws.change.insertions.size());
    out_ = s;
    st.version = g.version();
    st.graph_state = graph_access::impl(g).state_id;
    ws_.reset();  // return the workspace to the pool
  }

  void poison() noexcept override {
    if (state_ != nullptr) {
      state_->poisoned = true;
    }
  }

 private:
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

  /// Under batch_semantics::as_sets the commit's normalized batch (apply_delta) must equal the
  /// lists of Step 0 (Debug builds only: an O(batch) check of the graph module's two paths).
  static void check_normalized_batch([[maybe_unused]] const graph_type& g,
                                     [[maybe_unused]] const apply_delta<vertex_t>& delta,
                                     [[maybe_unused]] const workspace_type& ws) {
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
  /// anything is written. Without a bound the histogram grows to the new bound.
  /// @return The number of lengths whose count changed.
  static std::int64_t apply_histogram_delta(cycle_count_state& st, std::int64_t bound_after,
                                            const std::vector<std::uint64_t>& removed,
                                            const std::vector<std::uint64_t>& added) {
    const auto size = std::max(static_cast<std::size_t>(bound_after) + 1, st.counts.size());
    const auto delta_at = [&](std::size_t len, std::uint64_t& value, std::uint64_t& plus,
                              std::uint64_t& minus) {
      value = len < st.counts.size() ? st.counts[len] : 0;
      plus = len < added.size() ? added[len] : 0;
      minus = len < removed.size() ? removed[len] : 0;
    };
    // Check every bucket first, so a failure leaves the histogram as it was.
    for (std::size_t len = 0; len < size; ++len) {
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
    std::int64_t changed = 0;
    st.counts.resize(size, 0);
    for (std::size_t len = 0; len < size; ++len) {
      const std::uint64_t plus = len < added.size() ? added[len] : 0;
      const std::uint64_t minus = len < removed.size() ? removed[len] : 0;
      st.counts[len] = st.counts[len] + plus - minus;
      changed += plus != minus ? 1 : 0;
    }
    st.bound = std::max(st.bound, bound_after);
    return changed;
  }

  cycle_count::result& result_;
  cycle_count::stats& out_;
  cycle_count_state* state_ = nullptr;
  int threads_ = 1;
  std::int64_t bound_before_ = 2;
  std::optional<workspace_pool::lease<workspace_type>> ws_;
};

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
std::unique_ptr<update_participant<vertex_t, edge_t, weight_t>> make_cycle_count_participant(
    cycle_count::result& r, cycle_count::stats& out) {
  return std::make_unique<cycle_count_participant<vertex_t, edge_t, weight_t>>(r, out);
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
  return detail::cycle_count_access::state(*this).bound;
}

const options& result::get_options() const {
  return detail::cycle_count_access::state(*this).opt;
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

template <typename vertex_t, typename edge_t, typename weight_t>
result compute(const resources& res, const graph<vertex_t, edge_t, weight_t>& g,
               const options& opt) try {
  scoped_stage stage(res, "cycle_count.compute");
  detail::expect_supported_backend(res, "cycle_count::compute");
  detail::expect_options(opt);
  detail::expect_graph(g, "cycle_count::compute");
  detail::graph_access::expect_placement(res, g, "cycle_count::compute");

  auto state = std::make_unique<detail::cycle_count_state>();
  state->opt = opt;
  state->bound = detail::histogram_bound(opt, g);
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
stats update(const resources& res, graph<vertex_t, edge_t, weight_t>& g,
             const edge_batch_view<vertex_t, weight_t>& batch, result& r) try {
  scoped_stage stage(res, "cycle_count.update");
  stats out;
  detail::cycle_count_participant<vertex_t, edge_t, weight_t> participant(r, out);
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

#define DYNG_INSTANTIATE_CYCLE_COUNT(V, E, W)                                                     \
  template result compute<V, E, W>(const resources&, const graph<V, E, W>&, const options&);      \
  template stats update<V, E, W>(const resources&, graph<V, E, W>&, const edge_batch_view<V, W>&, \
                                 result&);
DYNG_FOR_EACH_CYCLE_COUNT_TYPE(DYNG_INSTANTIATE_CYCLE_COUNT)
#undef DYNG_INSTANTIATE_CYCLE_COUNT

}  // namespace dyng::cycle_count

namespace dyng::detail {

#define DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT(V, E, W)                                      \
  template std::unique_ptr<update_participant<V, E, W>> make_cycle_count_participant<V, E, W>( \
      cycle_count::result&, cycle_count::stats&);
DYNG_FOR_EACH_CYCLE_COUNT_TYPE(DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT)
#undef DYNG_INSTANTIATE_CYCLE_COUNT_PARTICIPANT
#undef DYNG_FOR_EACH_CYCLE_COUNT_TYPE

}  // namespace dyng::detail
