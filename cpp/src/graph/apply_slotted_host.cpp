// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/dynamicGraph.cu (createDeviceGraph,
// launchApplyVertexChanges, launchAppendEdges) and src/dynLP.cu (runDynLP's host pass b1)
/**
 * @file apply_slotted_host.cpp
 * @brief row_layout::slotted on the host (ADR 0069): with_capacity()'s storage, Step 0 of
 *        batch_semantics::net_effect(), the in-place apply, the compacted copy and the integrity
 *        check (slotted_rows.hpp).
 *
 * The original keeps the graph on the device: createDeviceGraph sizes each row for its degree in
 * the full graph, launchApplyVertexChanges sets the deleted ids, then the effective insertions,
 * then the ground truth, and launchAppendEdges appends both half-edges of every new edge whose
 * endpoints are present after the batch, sorted by source with a stable radix sort so that each
 * row receives its entries in batch order. The host apply below produces the same rows, row ends
 * and states (the container replay checks it against the recorded traces, M10a step 3), and
 * checks the input contract before it changes anything instead of after.
 */
#include "graph/apply_common.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "graph/slotted_rows.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

constexpr std::uint8_t mark_deleted = 1;   ///< marks: the id is in this batch's deletions
constexpr std::uint8_t mark_inserted = 2;  ///< marks: the id is in this batch's insertions

/// Clears the Step-0 marks of the listed ids when it goes out of scope (also on a throw).
template <typename vertex_t>
class mark_guard {
 public:
  mark_guard(std::vector<std::uint8_t>& marks, const array_view<const vertex_t>& deleted,
             const array_view<const vertex_t>& inserted) noexcept
      : marks_(marks), deleted_(deleted), inserted_(inserted) {}
  mark_guard(const mark_guard&) = delete;
  mark_guard& operator=(const mark_guard&) = delete;
  ~mark_guard() {
    for (const vertex_t v : deleted_) {
      marks_[static_cast<std::size_t>(v)] = 0;
    }
    for (const vertex_t v : inserted_) {
      marks_[static_cast<std::size_t>(v)] = 0;
    }
  }

 private:
  std::vector<std::uint8_t>& marks_;
  array_view<const vertex_t> deleted_;
  array_view<const vertex_t> inserted_;
};

/// Clears the per-row append counts of the listed edges' endpoints on scope exit.
template <typename vertex_t>
class count_guard {
 public:
  count_guard(std::vector<std::uint32_t>& counts, const array_view<const vertex_t>& src,
              const array_view<const vertex_t>& dst) noexcept
      : counts_(counts), src_(src), dst_(dst) {}
  count_guard(const count_guard&) = delete;
  count_guard& operator=(const count_guard&) = delete;
  ~count_guard() {
    for (std::size_t e = 0; e < src_.size(); ++e) {
      counts_[static_cast<std::size_t>(src_[e])] = 0;
      counts_[static_cast<std::size_t>(dst_[e])] = 0;
    }
  }

 private:
  std::vector<std::uint32_t>& counts_;
  array_view<const vertex_t> src_;
  array_view<const vertex_t> dst_;
};

/// A saturating add of the per-row append counts.
inline void count_append(std::uint32_t& c) noexcept {
  if (c != std::numeric_limits<std::uint32_t>::max()) {
    ++c;
  }
}

}  // namespace

void expect_slotted_properties(const graph_properties& props) {
  if (props.layout == row_layout::compact) {
    DYNG_THROW_INVALID_ARGUMENT(
        "graph::with_capacity builds a graph with slotted rows: graph_properties::layout must be "
        "row_layout::slotted (graph::from_edges / from_csr build compact rows)");
  }
  if (props.layout == row_layout::slack) {
    throw not_supported_error(
        "dyng: row_layout::slack arrives with label_propagation's streaming mode (M10b, 0.4); "
        "use row_layout::slotted with the full graph's degrees as capacities");
  }
  const auto refuse = [](const char* property, const char* needed) {
    throw not_supported_error(concat_message("dyng: graph::with_capacity: slotted rows need ",
                                             property, " = ", needed, " in this release"));
  };
  if (props.directed) {
    refuse("graph_properties::directed", "false (every edge is stored in both rows)");
  }
  if (props.order != row_order::append) {
    refuse("graph_properties::order",
           "row_order::append (rows receive their edges in batch order)");
  }
  if (props.parallel_edges != multi_edges::allow) {
    refuse("graph_properties::parallel_edges",
           "multi_edges::allow (the rows are not checked for duplicates, as in "
           "LabelPropagation-CUDA)");
  }
  const batch_semantics& s = props.semantics;
  const batch_semantics ne = batch_semantics::net_effect();
  if (!s.as_net_effect || s.as_sets || s.on_existing_insert != ne.on_existing_insert ||
      s.on_missing_delete != ne.on_missing_delete || s.on_self_loop != ne.on_self_loop ||
      s.deletions_first != ne.deletions_first || s.allow_vertex_growth != ne.allow_vertex_growth) {
    refuse("graph_properties::semantics", "batch_semantics::net_effect()");
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void build_slotted(std::int64_t n, array_view<const edge_t> capacity, int num_weights,
                   csr<vertex_t, edge_t, weight_t>& slots, slotted_rows<vertex_t, edge_t>& rows,
                   bool fill_slots) {
  DYNG_EXPECTS(n >= 0, "graph::with_capacity: a negative vertex capacity ", n);
  DYNG_EXPECTS(capacity.size() == static_cast<std::size_t>(n),
               "graph::with_capacity: ", capacity.size(), " row capacities for ", n, " vertices");
  const int expected = is_unweighted_v<weight_t> ? 0 : 1;
  DYNG_EXPECTS(
      num_weights == expected, "graph::with_capacity: slotted rows hold ", expected,
      " weight column(s) for this weight type, got graph_properties::num_weights = ", num_weights);
  std::int64_t total = 0;
  const auto limit = static_cast<std::int64_t>(std::numeric_limits<edge_t>::max());
  for (std::size_t v = 0; v < capacity.size(); ++v) {
    const auto c = static_cast<std::int64_t>(capacity[v]);
    DYNG_EXPECTS(c >= 0, "graph::with_capacity: row ", v, " has a negative capacity ", c);
    total += c;
    if (total > limit) {
      throw capacity_error(concat_message(
          "dyng: graph::with_capacity: the row capacities sum to more than ", limit,
          " slots, which does not fit the edge offset type; use a graph with 64-bit edge_t "
          "(int64)"));
    }
  }
  const auto nn = static_cast<std::size_t>(n);
  const auto m = static_cast<std::size_t>(total);
  slots.num_weights = num_weights;
  slots.row_ptr.assign(nn + 1, edge_t{0});
  for (std::size_t v = 0; v < nn; ++v) {
    slots.row_ptr[v + 1] = static_cast<edge_t>(slots.row_ptr[v] + capacity[v]);
  }
  if (fill_slots) {
    slots.col_ind.assign(m, vertex_t{-1});
    slots.weights.assign(m * static_cast<std::size_t>(num_weights), weight_t{});
  } else {
    slots.col_ind.clear();
    slots.weights.clear();
  }
  rows.row_end.assign(slots.row_ptr.begin(), slots.row_ptr.end() - 1);
  rows.status.assign(nn, vertex_status::inactive);
  rows.live_entries = 0;
  rows.active_vertices = 0;
  rows.marks.assign(nn, 0);
  rows.counts.assign(nn, 0);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_vertex_batch(const csr<vertex_t, edge_t, weight_t>& slots,
                            const slotted_rows<vertex_t, edge_t>& rows,
                            const edge_batch_view<vertex_t, weight_t>& batch,
                            normalized_batch<vertex_t>& out, bool check_capacity) {
  out.vertex_step = false;
  const std::int64_t n = slots.num_vertices();
  const std::size_t num_edges = batch.insert_src.size();
  const array_view<const vertex_t> ins = batch.insert_vertices;
  const array_view<const vertex_t> del = batch.delete_vertices;
  const array_view<const std::int8_t> labels = batch.insert_vertex_labels;
  // --- The shape -------------------------------------------------------------------------------
  expect_host(ins, "edge_batch_view::insert_vertices");
  expect_host(labels, "edge_batch_view::insert_vertex_labels");
  expect_host(del, "edge_batch_view::delete_vertices");
  expect_host(batch.insert_src, "edge_batch_view::insert_src");
  expect_host(batch.insert_dst, "edge_batch_view::insert_dst");
  expect_host(batch.insert_weights, "edge_batch_view::insert_weights");
  if (!batch.delete_src.empty() || !batch.delete_dst.empty()) {
    throw not_supported_error(
        "dyng: edge deletions are not supported under batch_semantics::net_effect() (a slotted "
        "graph loses edges only with their vertices: delete_vertices)");
  }
  DYNG_EXPECTS(labels.empty() || labels.size() == ins.size(), "the batch has ", labels.size(),
               " labels for ", ins.size(),
               " inserted vertices: a label needs a vertex being inserted (insert_vertex_labels "
               "is empty or holds one label per insert_vertices entry)");
  DYNG_EXPECTS(batch.insert_dst.size() == num_edges, "the batch has ", num_edges,
               " insertion sources but ", batch.insert_dst.size(), " destinations");
  if (num_edges > 0) {
    DYNG_EXPECTS(batch.num_weights == slots.num_weights, "the batch has ", batch.num_weights,
                 " weights per insertion but the graph has ", slots.num_weights, " weight columns");
    DYNG_EXPECTS(
        batch.insert_weights.size() == num_edges * static_cast<std::size_t>(slots.num_weights),
        "the batch has ", batch.insert_weights.size(), " insertion weights, expected ", num_edges,
        " x ", slots.num_weights);
  }
  // --- Ranges and labels (before any id is used as an index) ------------------------------------
  const auto in_range = [n](vertex_t v) { return v >= 0 && static_cast<std::int64_t>(v) < n; };
  for (std::size_t i = 0; i < del.size(); ++i) {
    DYNG_EXPECTS(in_range(del[i]), "deleted vertex ", i, " (id ", del[i], ") is out of range [0, ",
                 n, ")");
  }
  for (std::size_t i = 0; i < ins.size(); ++i) {
    DYNG_EXPECTS(in_range(ins[i]), "inserted vertex ", i, " (id ", ins[i], ") is out of range [0, ",
                 n, ")");
  }
  for (std::size_t i = 0; i < labels.size(); ++i) {
    DYNG_EXPECTS(labels[i] >= -1 && labels[i] <= 1, "inserted vertex ", ins[i], " has the label ",
                 static_cast<int>(labels[i]), "; a seed label is -1 (none), 0 or 1");
  }
  for (std::size_t e = 0; e < num_edges; ++e) {
    DYNG_EXPECTS(in_range(batch.insert_src[e]) && in_range(batch.insert_dst[e]), "inserted edge ",
                 e, " (", batch.insert_src[e], ", ", batch.insert_dst[e],
                 ") has an endpoint out of range [0, ", n, ")");
  }
  // --- Duplicates, single-use ids, seeds deleted in the same batch -----------------------------
  std::vector<std::uint8_t>& marks = rows.marks;
  const mark_guard<vertex_t> clear_marks(marks, del, ins);
  for (std::size_t i = 0; i < del.size(); ++i) {
    std::uint8_t& m = marks[static_cast<std::size_t>(del[i])];
    DYNG_EXPECTS((m & mark_deleted) == 0, "vertex ", del[i],
                 " is listed twice in the batch's deletions");
    m = static_cast<std::uint8_t>(m | mark_deleted);
  }
  for (std::size_t i = 0; i < ins.size(); ++i) {
    const vertex_t v = ins[i];
    std::uint8_t& m = marks[static_cast<std::size_t>(v)];
    DYNG_EXPECTS((m & mark_inserted) == 0, "vertex ", v,
                 " is listed twice in the batch's insertions (vertex ids are single-use)");
    m = static_cast<std::uint8_t>(m | mark_inserted);
    const vertex_status st = rows.status[static_cast<std::size_t>(v)];
    DYNG_EXPECTS(st == vertex_status::inactive, "inserted vertex ", v, " was ",
                 st == vertex_status::active ? "inserted before" : "deleted before",
                 " (vertex ids are single-use: an id inserted or named in a deletion cannot be "
                 "inserted again)");
    DYNG_EXPECTS(labels.empty() || labels[i] < 0 || (m & mark_deleted) == 0, "inserted vertex ", v,
                 " has a seed label and is deleted in the same batch");
  }
  // --- Capacity: the entries each row receives (both endpoints active after the batch) ---------
  const auto active_after = [&](vertex_t v) {
    const std::uint8_t m = marks[static_cast<std::size_t>(v)];
    if ((m & mark_deleted) != 0) {
      return false;
    }
    return (m & mark_inserted) != 0 ||
           rows.status[static_cast<std::size_t>(v)] == vertex_status::active;
  };
  std::vector<std::uint32_t>& counts = rows.counts;
  const count_guard<vertex_t> clear_counts(
      counts, check_capacity ? batch.insert_src : array_view<const vertex_t>(),
      check_capacity ? batch.insert_dst : array_view<const vertex_t>());
  for (std::size_t e = 0; check_capacity && e < num_edges; ++e) {
    const vertex_t u = batch.insert_src[e];
    const vertex_t v = batch.insert_dst[e];
    if (active_after(u) && active_after(v)) {
      count_append(counts[static_cast<std::size_t>(u)]);
      count_append(counts[static_cast<std::size_t>(v)]);
    }
  }
  for (std::size_t e = 0; check_capacity && e < num_edges; ++e) {
    for (const vertex_t w : {batch.insert_src[e], batch.insert_dst[e]}) {
      const auto ww = static_cast<std::size_t>(w);
      const auto used = static_cast<std::int64_t>(rows.row_end[ww] - slots.row_ptr[ww]);
      const auto capacity = static_cast<std::int64_t>(slots.row_ptr[ww + 1] - slots.row_ptr[ww]);
      if (used + static_cast<std::int64_t>(counts[ww]) > capacity) {
        throw capacity_error(concat_message(
            "dyng: the batch appends ", counts[ww], " entries to the row of vertex ", w,
            ", which holds ", used, " of its ", capacity,
            " slots: an edge was delivered twice, or is not an edge of the graph the "
            "capacities were made for (graph::with_capacity)"));
      }
    }
  }
  // --- The net effect ------------------------------------------------------------------------
  out.inserted_vertices.clear();
  out.inserted_labels.clear();
  out.deleted_vertices.clear();
  for (std::size_t i = 0; i < ins.size(); ++i) {
    if ((marks[static_cast<std::size_t>(ins[i])] & mark_deleted) == 0) {
      out.inserted_vertices.push_back(ins[i]);
      out.inserted_labels.push_back(labels.empty() ? std::int8_t{-1} : labels[i]);
    }
  }
  for (std::size_t i = 0; i < del.size(); ++i) {
    if (rows.status[static_cast<std::size_t>(del[i])] == vertex_status::active) {
      out.deleted_vertices.push_back(del[i]);
    }
  }
  out.vertices_before = n;
  out.vertices_after = n;
  out.vertex_step = true;
}

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_vertex_batch_host(csr<vertex_t, edge_t, weight_t>& slots,
                                      slotted_rows<vertex_t, edge_t>& rows,
                                      const edge_batch_view<vertex_t, weight_t>& batch,
                                      const normalized_batch<vertex_t>& normalized,
                                      apply_delta<vertex_t>* delta) {
  DYNG_EXPECTS(normalized.vertex_step, "apply (slotted): Step 0 of this batch is missing");
  const std::size_t num_edges = batch.insert_src.size();
  const int k = slots.num_weights;
  apply_summary summary;
  // The record first: its allocations are the only ones (strong guarantee).
  if (delta != nullptr) {
    delta->num_weights = k;
    delta->weight_increased.clear();
    delta->delete_src.clear();
    delta->delete_dst.clear();
    delta->insert_src.clear();
    delta->insert_dst.clear();
    delta->insert_index.clear();
    delta->insert_src.reserve(num_edges);
    delta->insert_dst.reserve(num_edges);
    delta->insert_index.reserve(num_edges);
    delta->inserted_vertices = normalized.inserted_vertices;
    delta->inserted_labels = normalized.inserted_labels;
    delta->deleted_vertices = normalized.deleted_vertices;
  }
  // Deletions (Del_eff loses its live entries; every listed id ends deleted), as the original's
  // setStatusKernel. Entries are counted while their endpoints are still active.
  for (const vertex_t v : normalized.deleted_vertices) {
    const auto vv = static_cast<std::size_t>(v);
    edge_t removed = 0;
    for (auto e = static_cast<std::size_t>(slots.row_ptr[vv]);
         e < static_cast<std::size_t>(rows.row_end[vv]); ++e) {
      const vertex_t c = slots.col_ind[e];
      if (c == v) {
        removed += 1;
      } else if (rows.status[static_cast<std::size_t>(c)] == vertex_status::active) {
        removed += 2;
      }
    }
    rows.live_entries -= removed;
    rows.status[vv] = vertex_status::deleted;
    --rows.active_vertices;
  }
  for (const vertex_t v : batch.delete_vertices) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::deleted;  // retired
  }
  // Effective insertions (insertKernel); the seeds are the result's, not the graph's.
  for (const vertex_t v : normalized.inserted_vertices) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::active;
    ++rows.active_vertices;
  }
  // Appends in batch order: half-edge 2e (v into u's row), then 2e + 1 (u into v's row).
  const auto append = [&](vertex_t row, vertex_t neighbor, std::size_t e) {
    const auto r = static_cast<std::size_t>(row);
    const auto slot = static_cast<std::size_t>(rows.row_end[r]);
    slots.col_ind[slot] = neighbor;
    const std::size_t m = slots.col_ind.size();
    for (int c = 0; c < k; ++c) {
      slots.weights[static_cast<std::size_t>(c) * m + slot] =
          batch.insert_weights[e * static_cast<std::size_t>(k) + static_cast<std::size_t>(c)];
    }
    ++rows.row_end[r];
  };
  for (std::size_t e = 0; e < num_edges; ++e) {
    const vertex_t u = batch.insert_src[e];
    const vertex_t v = batch.insert_dst[e];
    if (rows.status[static_cast<std::size_t>(u)] != vertex_status::active ||
        rows.status[static_cast<std::size_t>(v)] != vertex_status::active) {
      summary.dropped_insertions += 2;
      continue;
    }
    append(u, v, e);
    append(v, u, e);
    rows.live_entries += 2;
    summary.inserted_edges += 2;
    if (delta != nullptr) {
      delta->insert_src.push_back(u);
      delta->insert_dst.push_back(v);
      delta->insert_index.push_back(e);
    }
  }
  summary.inserted_vertices = static_cast<std::int64_t>(normalized.inserted_vertices.size());
  summary.deleted_vertices = static_cast<std::int64_t>(normalized.deleted_vertices.size());
  summary.num_vertices_after = slots.num_vertices();
  return summary;
}

template <typename vertex_t, typename edge_t>
apply_summary apply_vertex_states_host(slotted_rows<vertex_t, edge_t>& rows,
                                       array_view<const vertex_t> deleted,
                                       const normalized_batch<vertex_t>& normalized,
                                       apply_delta<vertex_t>* delta) {
  DYNG_EXPECTS(normalized.vertex_step, "apply (slotted, device): Step 0 of this batch is missing");
  apply_summary summary;
  if (delta != nullptr) {
    delta->num_weights = 1;
    delta->weight_increased.clear();
    delta->delete_src.clear();
    delta->delete_dst.clear();
    delta->insert_src.clear();
    delta->insert_dst.clear();
    delta->insert_index.clear();
    delta->inserted_vertices = normalized.inserted_vertices;
    delta->inserted_labels = normalized.inserted_labels;
    delta->deleted_vertices = normalized.deleted_vertices;
    delta->edges_on_device = true;
  }
  // As apply_vertex_batch_host(): Del_eff leaves, every listed id ends deleted, Ins_eff arrives.
  for (const vertex_t v : normalized.deleted_vertices) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::deleted;
    --rows.active_vertices;
  }
  for (const vertex_t v : deleted) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::deleted;
  }
  for (const vertex_t v : normalized.inserted_vertices) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::active;
    ++rows.active_vertices;
  }
  summary.inserted_vertices = static_cast<std::int64_t>(normalized.inserted_vertices.size());
  summary.deleted_vertices = static_cast<std::int64_t>(normalized.deleted_vertices.size());
  summary.num_vertices_after = static_cast<std::int64_t>(rows.status.size());
  return summary;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compact_slotted(const csr<vertex_t, edge_t, weight_t>& slots,
                     const slotted_rows<vertex_t, edge_t>& rows,
                     csr<vertex_t, edge_t, weight_t>& out) {
  const auto n = static_cast<std::size_t>(slots.num_vertices());
  const int k = slots.num_weights;
  const auto active = [&](vertex_t v) {
    return rows.status[static_cast<std::size_t>(v)] == vertex_status::active;
  };
  out = csr<vertex_t, edge_t, weight_t>();
  out.num_weights = k;
  out.row_ptr.assign(n + 1, edge_t{0});
  for (std::size_t u = 0; u < n; ++u) {
    edge_t count = 0;
    if (rows.status[u] == vertex_status::active) {
      for (auto e = static_cast<std::size_t>(slots.row_ptr[u]);
           e < static_cast<std::size_t>(rows.row_end[u]); ++e) {
        count += active(slots.col_ind[e]) ? 1 : 0;
      }
    }
    out.row_ptr[u + 1] = static_cast<edge_t>(out.row_ptr[u] + count);
  }
  const auto m = static_cast<std::size_t>(out.row_ptr[n]);
  const std::size_t slot_count = slots.col_ind.size();
  out.col_ind.resize(m);
  out.weights.resize(m * static_cast<std::size_t>(k));
  for (std::size_t u = 0; u < n; ++u) {
    if (rows.status[u] != vertex_status::active) {
      continue;
    }
    auto at = static_cast<std::size_t>(out.row_ptr[u]);
    for (auto e = static_cast<std::size_t>(slots.row_ptr[u]);
         e < static_cast<std::size_t>(rows.row_end[u]); ++e) {
      if (!active(slots.col_ind[e])) {
        continue;
      }
      out.col_ind[at] = slots.col_ind[e];
      for (int c = 0; c < k; ++c) {
        out.weights[static_cast<std::size_t>(c) * m + at] =
            slots.weights[static_cast<std::size_t>(c) * slot_count + e];
      }
      ++at;
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::string slotted_integrity_violation(const csr<vertex_t, edge_t, weight_t>& slots,
                                        const slotted_rows<vertex_t, edge_t>& rows,
                                        int num_weights) {
  const std::int64_t n = slots.num_vertices();
  const auto nn = static_cast<std::size_t>(n);
  const std::size_t m = slots.col_ind.size();
  if (slots.row_ptr.empty() || slots.row_ptr[0] != 0) {
    return "the slot offsets do not start at 0";
  }
  for (std::size_t u = 0; u < nn; ++u) {
    if (slots.row_ptr[u + 1] < slots.row_ptr[u]) {
      return "the slot offsets are not monotone at row " + std::to_string(u);
    }
  }
  if (static_cast<std::size_t>(slots.row_ptr.back()) != m) {
    return "the slot offsets do not end at the number of slots";
  }
  if (slots.num_weights != num_weights ||
      slots.weights.size() != m * static_cast<std::size_t>(num_weights)) {
    return "the weight slots do not match num_weights x the slots";
  }
  if (rows.row_end.size() != nn || rows.status.size() != nn || rows.marks.size() != nn ||
      rows.counts.size() != nn) {
    return "the row ends, states or scratch do not have one entry per vertex";
  }
  std::int64_t active = 0;
  std::int64_t live = 0;
  std::vector<std::pair<vertex_t, vertex_t>> forward;
  std::vector<std::pair<vertex_t, vertex_t>> backward;
  for (std::size_t u = 0; u < nn; ++u) {
    const vertex_status st = rows.status[u];
    if (st != vertex_status::active && st != vertex_status::deleted &&
        st != vertex_status::inactive) {
      return "vertex " + std::to_string(u) + " has an invalid state";
    }
    if (rows.marks[u] != 0 || rows.counts[u] != 0) {
      return "the scratch of vertex " + std::to_string(u) + " is not clear";
    }
    active += st == vertex_status::active ? 1 : 0;
    const auto begin = static_cast<std::size_t>(slots.row_ptr[u]);
    const auto end = static_cast<std::size_t>(rows.row_end[u]);
    if (rows.row_end[u] < slots.row_ptr[u] || rows.row_end[u] > slots.row_ptr[u + 1]) {
      return "the end of row " + std::to_string(u) + " lies outside its slots";
    }
    if (st == vertex_status::inactive && end != begin) {
      return "vertex " + std::to_string(u) + " was never inserted but its row has entries";
    }
    for (std::size_t e = begin; e < end; ++e) {
      const vertex_t c = slots.col_ind[e];
      if (c < 0 || static_cast<std::int64_t>(c) >= n) {
        return "an entry of row " + std::to_string(u) + " is out of range";
      }
      if (rows.status[static_cast<std::size_t>(c)] == vertex_status::inactive) {
        return "row " + std::to_string(u) + " has an entry to the never-inserted vertex " +
               std::to_string(c);
      }
      if (st == vertex_status::active &&
          rows.status[static_cast<std::size_t>(c)] == vertex_status::active) {
        ++live;
      }
      forward.emplace_back(static_cast<vertex_t>(u), c);
      backward.emplace_back(c, static_cast<vertex_t>(u));
    }
    for (std::size_t e = end; e < static_cast<std::size_t>(slots.row_ptr[u + 1]); ++e) {
      bool clear = slots.col_ind[e] == vertex_t{-1};
      for (int c = 0; c < num_weights; ++c) {
        clear = clear && slots.weights[static_cast<std::size_t>(c) * m + e] == weight_t{};
      }
      if (!clear) {
        return "an unused slot of row " + std::to_string(u) + " is not -1 / 0";
      }
    }
  }
  std::sort(forward.begin(), forward.end());
  std::sort(backward.begin(), backward.end());
  if (forward != backward) {
    return "the entries are not symmetric (an undirected edge is stored in both rows)";
  }
  if (active != static_cast<std::int64_t>(rows.active_vertices)) {
    return "the count of active vertices is " + std::to_string(rows.active_vertices) +
           ", the states have " + std::to_string(active);
  }
  if (live != static_cast<std::int64_t>(rows.live_entries)) {
    return "the count of live entries is " + std::to_string(rows.live_entries) +
           ", the rows have " + std::to_string(live);
  }
  return {};
}

#define DYNG_INSTANTIATE_SLOTTED(V, E, W)                                                       \
  template void build_slotted<V, E, W>(std::int64_t, array_view<const E>, int, csr<V, E, W>&,   \
                                       slotted_rows<V, E>&, bool);                              \
  template void normalize_vertex_batch<V, E, W>(const csr<V, E, W>&, const slotted_rows<V, E>&, \
                                                const edge_batch_view<V, W>&,                   \
                                                normalized_batch<V>&, bool);                    \
  template apply_summary apply_vertex_batch_host<V, E, W>(                                      \
      csr<V, E, W>&, slotted_rows<V, E>&, const edge_batch_view<V, W>&,                         \
      const normalized_batch<V>&, apply_delta<V>*);                                             \
  template void compact_slotted<V, E, W>(const csr<V, E, W>&, const slotted_rows<V, E>&,        \
                                         csr<V, E, W>&);                                        \
  template std::string slotted_integrity_violation<V, E, W>(const csr<V, E, W>&,                \
                                                            const slotted_rows<V, E>&, int);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SLOTTED)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_SLOTTED)
DYNG_FOR_EACH_FLOAT_GRAPH_TYPE(DYNG_INSTANTIATE_SLOTTED)
#undef DYNG_INSTANTIATE_SLOTTED

template apply_summary apply_vertex_states_host<std::int32_t, std::int32_t>(
    slotted_rows<std::int32_t, std::int32_t>&, array_view<const std::int32_t>,
    const normalized_batch<std::int32_t>&, apply_delta<std::int32_t>*);
template apply_summary apply_vertex_states_host<std::int32_t, std::int64_t>(
    slotted_rows<std::int32_t, std::int64_t>&, array_view<const std::int32_t>,
    const normalized_batch<std::int32_t>&, apply_delta<std::int32_t>*);

}  // namespace dyng::detail
