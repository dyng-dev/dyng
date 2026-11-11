// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file apply_slack_host.cpp
 * @brief row_layout::slack on the host (ADR 0103): the slot arena of graph::with_capacity() for
 *        streams without known capacities, Step 0 of a vertex batch with its layout step, the
 *        in-place apply (rows that overflow move to the tail of the arena; a compaction when the
 *        dead slots outgrow the owned ones), vertex growth, and the integrity check.
 *
 * Slack rows are slotted rows (ADR 0069) whose slot ranges are not fixed: row v owns the slots
 * [row_ptr[v], row_limit[v]) of one arena (the graph's host CSR, its row_ptr[n] the arena's
 * size), and its entries are [row_ptr[v], row_end[v]), in the order they were appended. A row
 * gets slack_capacity(k) = ceil(k * (1 + headroom)) slots for its k expected entries (the
 * capacity hint of with_capacity(), or what the batch that overflows it needs); a row that
 * overflows moves to the end of the arena with ALL of its entries in their order (also those of
 * departed vertices, which every reader skips, as in slotted rows), so a move never changes the
 * order in which a reader meets a row's entries; its old slots become dead (-1 / 0). When the dead
 * slots outnumber the owned ones, the batch compacts the rows instead: every row is rewritten in
 * vertex order without the entries of deleted vertices (and a deleted vertex's row is emptied),
 * which changes where a row's live entries sit, and so the order of the lanes that sum them on
 * the device (documented in ADR 0103 and graph_view.hpp). The layout step runs inside apply(),
 * after the vertex states of the batch and before its appends, outside any kernel, once per batch
 * at most (amortized: a compaction follows at least as many dead slots as it rewrites).
 */
#include "graph/apply_common.hpp"
#include "graph/graph_impl.hpp"
#include "graph/instantiate.hpp"
#include "graph/slotted_rows.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/types.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

constexpr std::uint8_t mark_deleted = 1;   ///< marks: the id is in this batch's deletions
constexpr std::uint8_t mark_inserted = 2;  ///< marks: the id is in this batch's insertions

/// The Step-0 marks (as apply_slotted_host.cpp's step_marks: one byte per vertex, the low two bits
/// a batch's flags, the high six the tag of the Step 0 that set them, one more byte the last tag).
class slack_marks {
 public:
  explicit slack_marks(std::vector<std::uint8_t>& bytes) noexcept
      : bytes_(bytes.data()), n_(bytes.size() - 1) {
    advance();
  }
  void advance() noexcept {
    auto tag = static_cast<unsigned>(bytes_[n_]) + 1U;
    if (tag > 63U) {
      std::fill(bytes_, bytes_ + n_, std::uint8_t{0});
      tag = 1U;
    }
    bytes_[n_] = static_cast<std::uint8_t>(tag);
    tag_ = static_cast<std::uint8_t>(tag << 2U);
  }
  [[nodiscard]] std::uint8_t flags(std::size_t v) const noexcept {
    const std::uint8_t b = bytes_[v];
    return (b & 0xFCU) == tag_ ? static_cast<std::uint8_t>(b & 3U) : std::uint8_t{0};
  }
  void set(std::size_t v, std::uint8_t f) noexcept {
    bytes_[v] = static_cast<std::uint8_t>(tag_ | f);
  }

 private:
  std::uint8_t* bytes_;
  std::size_t n_;
  std::uint8_t tag_ = 0;
};

/// Clears the per-row counts of the listed edges' endpoints on scope exit.
template <typename vertex_t>
class counts_cleared {
 public:
  counts_cleared(std::vector<std::uint32_t>& counts, const array_view<const vertex_t>& src,
                 const array_view<const vertex_t>& dst) noexcept
      : counts_(counts), src_(src), dst_(dst) {}
  counts_cleared(const counts_cleared&) = delete;
  counts_cleared& operator=(const counts_cleared&) = delete;
  ~counts_cleared() {
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

/// Grow the scratch of Step 0 to `n` vertices (the marks keep their tag byte last).
template <typename vertex_t, typename edge_t>
void grow_scratch(const slotted_rows<vertex_t, edge_t>& rows, std::size_t n) {
  if (rows.counts.size() < n) {
    const std::uint8_t tag = rows.marks.back();
    rows.marks.back() = 0;  // a zero byte reads as clear under every tag
    rows.marks.resize(n + 1, 0);
    rows.marks.back() = tag;
    rows.counts.resize(n, 0);
  }
}

/// The largest vertex id of a batch's arrays, -1 if none.
template <typename vertex_t, typename weight_t>
std::int64_t largest_id(const edge_batch_view<vertex_t, weight_t>& batch) {
  std::int64_t m = -1;
  for (const auto* ids :
       {&batch.insert_vertices, &batch.delete_vertices, &batch.insert_src, &batch.insert_dst}) {
    for (std::size_t i = 0; i < ids->size(); ++i) {
      m = std::max(m, static_cast<std::int64_t>((*ids)[i]));
    }
  }
  return m;
}

template <typename edge_t>
constexpr std::int64_t edge_limit() noexcept {
  return static_cast<std::int64_t>(std::numeric_limits<edge_t>::max());
}

template <typename edge_t>
[[noreturn]] void throw_arena_limit(std::int64_t needed) {
  throw capacity_error(concat_message("dyng: graph::apply: the slack rows need ", needed,
                                      " slots, more than the edge offset "
                                      "type holds (",
                                      edge_limit<edge_t>(),
                                      "); use a graph with 64-bit edge_t (int64)"));
}

}  // namespace

std::int64_t slack_capacity(std::int64_t entries, double headroom) noexcept {
  if (entries <= 0) {
    return 0;
  }
  const double c = std::ceil(static_cast<double>(entries) * (1.0 + headroom));
  if (!(c < 9.0e18)) {
    return std::numeric_limits<std::int64_t>::max();
  }
  return std::max(static_cast<std::int64_t>(c), entries);
}

template <typename vertex_t, typename edge_t, typename weight_t>
void build_slack(std::int64_t n, array_view<const edge_t> capacity, int num_weights,
                 const graph_properties& props, csr<vertex_t, edge_t, weight_t>& slots,
                 slotted_rows<vertex_t, edge_t>& rows, bool fill_slots) {
  DYNG_EXPECTS(n >= 0, "graph::with_capacity: a negative vertex capacity ", n);
  DYNG_EXPECTS(capacity.size() <= static_cast<std::size_t>(n),
               "graph::with_capacity: ", capacity.size(), " row capacities for ", n,
               " vertices (slack rows take at most one hint per vertex)");
  const int expected = is_unweighted_v<weight_t> ? 0 : 1;
  DYNG_EXPECTS(
      num_weights == expected, "graph::with_capacity: slack rows hold ", expected,
      " weight column(s) for this weight type, got graph_properties::num_weights = ", num_weights);
  const auto nn = static_cast<std::size_t>(n);
  const double h = props.headroom;
  slots.num_weights = num_weights;
  slots.row_ptr.assign(nn + 1, edge_t{0});
  rows.row_limit.assign(nn, edge_t{0});
  std::int64_t total = 0;
  for (std::size_t v = 0; v < nn; ++v) {
    std::int64_t hint = 0;
    if (v < capacity.size()) {
      hint = static_cast<std::int64_t>(capacity[v]);
      DYNG_EXPECTS(hint >= 0, "graph::with_capacity: row ", v, " has a negative capacity ", hint);
    }
    slots.row_ptr[v] = static_cast<edge_t>(total);
    const std::int64_t c = slack_capacity(hint, h);
    if (c > edge_limit<edge_t>() - total) {
      throw capacity_error(concat_message(
          "dyng: graph::with_capacity: the slack rows' capacities (with headroom ", h,
          ") sum to more than ", edge_limit<edge_t>(),
          " slots, which does not fit the edge offset type; use a graph with 64-bit edge_t "
          "(int64)"));
    }
    total += c;
    rows.row_limit[v] = static_cast<edge_t>(total);
  }
  slots.row_ptr[nn] = static_cast<edge_t>(total);
  const auto m = static_cast<std::size_t>(total);
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
  rows.marks.assign(nn + 1, 0);
  rows.counts.assign(nn, 0);
  rows.slack = true;
  rows.headroom = h;
  rows.allow_growth = props.semantics.allow_vertex_growth;
  rows.dead_slots = 0;
  rows.relocations = 0;
  rows.compactions = 0;
  rows.layout_epoch = 0;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_vertex_batch_slack(const csr<vertex_t, edge_t, weight_t>& slots,
                                  const slotted_rows<vertex_t, edge_t>& rows,
                                  const edge_batch_view<vertex_t, weight_t>& batch,
                                  normalized_batch<vertex_t>& out) {
  out.vertex_step = false;
  out.slack_moves.clear();
  out.slack_compact = false;
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
        "dyng: edge deletions are not supported under batch_semantics::net_effect() (a slack "
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
  // --- Ranges (vertex growth: any id up to the vertex type's limit) and labels -----------------
  const std::int64_t id_end =
      rows.allow_growth ? static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max()) : n;
  const char* range_rule = rows.allow_growth
                               ? " (vertex growth: the ids must stay below the vertex type's "
                                 "largest value)"
                               : "";
  const auto in_range = [id_end](vertex_t v) {
    return v >= 0 && static_cast<std::int64_t>(v) < id_end;
  };
  for (std::size_t i = 0; i < del.size(); ++i) {
    DYNG_EXPECTS(in_range(del[i]), "deleted vertex ", i, " (id ", del[i], ") is out of range [0, ",
                 id_end, ")", range_rule);
  }
  for (std::size_t i = 0; i < ins.size(); ++i) {
    DYNG_EXPECTS(in_range(ins[i]), "inserted vertex ", i, " (id ", ins[i], ") is out of range [0, ",
                 id_end, ")", range_rule);
  }
  for (std::size_t i = 0; i < labels.size(); ++i) {
    DYNG_EXPECTS(labels[i] >= -1 && labels[i] <= 1, "inserted vertex ", ins[i], " has the label ",
                 static_cast<int>(labels[i]), "; a seed label is -1 (none), 0 or 1");
  }
  for (std::size_t e = 0; e < num_edges; ++e) {
    DYNG_EXPECTS(in_range(batch.insert_src[e]) && in_range(batch.insert_dst[e]), "inserted edge ",
                 e, " (", batch.insert_src[e], ", ", batch.insert_dst[e],
                 ") has an endpoint out of range [0, ", id_end, ")", range_rule);
  }
  const std::int64_t n_after = std::max(n, largest_id(batch) + 1);
  grow_scratch(rows, static_cast<std::size_t>(n_after));
  const auto status_of = [&](std::size_t v) {
    return static_cast<std::int64_t>(v) < n ? rows.status[v] : vertex_status::inactive;
  };
  // --- Duplicates, single-use ids, seeds deleted in the same batch -----------------------------
  slack_marks marks(rows.marks);
  for (std::size_t i = 0; i < del.size(); ++i) {
    const auto v = static_cast<std::size_t>(del[i]);
    const std::uint8_t m = marks.flags(v);
    DYNG_EXPECTS((m & mark_deleted) == 0, "vertex ", del[i],
                 " is listed twice in the batch's deletions");
    marks.set(v, static_cast<std::uint8_t>(m | mark_deleted));
  }
  out.inserted_vertices.clear();
  out.inserted_labels.clear();
  out.deleted_vertices.clear();
  out.inserted_vertices.reserve(ins.size());
  out.inserted_labels.reserve(ins.size());
  for (std::size_t i = 0; i < ins.size(); ++i) {
    const vertex_t v = ins[i];
    const auto vv = static_cast<std::size_t>(v);
    const std::uint8_t m = marks.flags(vv);
    DYNG_EXPECTS((m & mark_inserted) == 0, "vertex ", v,
                 " is listed twice in the batch's insertions (vertex ids are single-use)");
    marks.set(vv, static_cast<std::uint8_t>(m | mark_inserted));
    const vertex_status st = status_of(vv);
    DYNG_EXPECTS(st == vertex_status::inactive, "inserted vertex ", v, " was ",
                 st == vertex_status::active ? "inserted before" : "deleted before",
                 " (vertex ids are single-use: an id inserted or named in a deletion cannot be "
                 "inserted again)");
    const std::int8_t label = labels.empty() ? std::int8_t{-1} : labels[i];
    DYNG_EXPECTS(label < 0 || (m & mark_deleted) == 0, "inserted vertex ", v,
                 " has a seed label and is deleted in the same batch");
    if ((m & mark_deleted) == 0) {
      out.inserted_vertices.push_back(v);
      out.inserted_labels.push_back(label);
    }
  }
  // --- The layout step: the appends per row, the rows that overflow, a compaction ---------------
  const auto active_after = [&](vertex_t v) {
    const std::uint8_t m = marks.flags(static_cast<std::size_t>(v));
    if ((m & mark_deleted) != 0) {
      return false;
    }
    return (m & mark_inserted) != 0 ||
           status_of(static_cast<std::size_t>(v)) == vertex_status::active;
  };
  std::vector<std::uint32_t>& counts = rows.counts;
  const counts_cleared<vertex_t> clear_counts(counts, batch.insert_src, batch.insert_dst);
  for (std::size_t e = 0; e < num_edges; ++e) {
    const vertex_t u = batch.insert_src[e];
    const vertex_t v = batch.insert_dst[e];
    if (active_after(u) && active_after(v)) {
      for (const vertex_t w : {u, v}) {
        std::uint32_t& c = counts[static_cast<std::size_t>(w)];
        if (c != std::numeric_limits<std::uint32_t>::max()) {
          ++c;
        }
      }
    }
  }
  const std::int64_t arena = static_cast<std::int64_t>(slots.row_ptr[static_cast<std::size_t>(n)]);
  std::int64_t freed = 0;     // the slots of the rows that move
  std::int64_t added = 0;     // the slots of their new places
  std::int64_t ub_extra = 0;  // what a compaction may need beyond the owned slots
  for (std::size_t e = 0; e < num_edges; ++e) {
    for (const vertex_t w : {batch.insert_src[e], batch.insert_dst[e]}) {
      const auto ww = static_cast<std::size_t>(w);
      const std::int64_t k = counts[ww];
      if (k == 0) {
        continue;  // no append, or already planned
      }
      counts[ww] = 0;
      std::int64_t used = 0;
      std::int64_t cap = 0;
      if (static_cast<std::int64_t>(ww) < n) {
        used = static_cast<std::int64_t>(rows.row_end[ww]) -
               static_cast<std::int64_t>(slots.row_ptr[ww]);
        cap = row_capacity_of(slots, rows, ww);
      }
      const std::int64_t need = slack_capacity(used + k, rows.headroom);
      ub_extra += std::max<std::int64_t>(0, need - cap);
      if (used + k > cap) {
        out.slack_moves.push_back(static_cast<std::int64_t>(w));
        out.slack_moves.push_back(need);
        freed += cap;
        added += need;
      }
    }
  }
  // The moves in increasing vertex order (pairs).
  {
    const std::size_t pairs = out.slack_moves.size() / 2;
    std::vector<std::pair<std::int64_t, std::int64_t>> p(pairs);
    for (std::size_t i = 0; i < pairs; ++i) {
      p[i] = {out.slack_moves[2 * i], out.slack_moves[2 * i + 1]};
    }
    std::sort(p.begin(), p.end());
    for (std::size_t i = 0; i < pairs; ++i) {
      out.slack_moves[2 * i] = p[i].first;
      out.slack_moves[2 * i + 1] = p[i].second;
    }
  }
  const std::int64_t dead_after = static_cast<std::int64_t>(rows.dead_slots) + freed;
  const std::int64_t arena_after = arena + added;
  // Compact when the dead slots outnumber the owned ones (ADR 0103), or when the moves would not
  // fit edge_t; a compaction needs at most the owned slots plus each touched row's growth.
  bool compact = dead_after > 0 && 2 * dead_after > arena_after;
  if (!compact && arena_after > edge_limit<edge_t>()) {
    compact = true;
  }
  if (compact) {
    const std::int64_t bound = arena - static_cast<std::int64_t>(rows.dead_slots) + ub_extra;
    if (bound > edge_limit<edge_t>()) {
      throw_arena_limit<edge_t>(bound);
    }
    out.slack_moves.clear();
  }
  out.slack_compact = compact;
  // --- The net effect ---------------------------------------------------------------------------
  for (std::size_t i = 0; i < del.size(); ++i) {
    if (status_of(static_cast<std::size_t>(del[i])) == vertex_status::active) {
      out.deleted_vertices.push_back(del[i]);
    }
  }
  out.vertices_before = n;
  out.vertices_after = n_after;
  out.vertex_step = true;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void grow_slack_rows(csr<vertex_t, edge_t, weight_t>& slots, slotted_rows<vertex_t, edge_t>& rows,
                     std::int64_t n_after) {
  const auto n = static_cast<std::size_t>(slots.num_vertices());
  const auto na = static_cast<std::size_t>(n_after);
  if (na <= n) {
    return;
  }
  // Reserve everything first: the resizes below cannot throw (the strong guarantee of apply()).
  slots.row_ptr.reserve(na + 1);
  rows.row_end.reserve(na);
  rows.row_limit.reserve(na);
  rows.status.reserve(na);
  rows.marks.reserve(na + 1);
  rows.counts.reserve(na);
  const edge_t arena = slots.row_ptr[n];
  slots.row_ptr.resize(na + 1, arena);
  rows.row_end.resize(na, arena);
  rows.row_limit.resize(na, arena);
  rows.status.resize(na, vertex_status::inactive);
  if (rows.counts.size() < na) {
    const std::uint8_t tag = rows.marks.back();
    rows.marks.back() = 0;
    rows.marks.resize(na + 1, 0);
    rows.marks.back() = tag;
    rows.counts.resize(na, 0);
  }
  ++rows.layout_epoch;
}

template <typename vertex_t, typename weight_t>
void slack_append_counts(const std::vector<vertex_status>& status,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         std::vector<std::int64_t>& counts) {
  counts.assign(status.size(), 0);
  for (std::size_t e = 0; e < batch.insert_src.size(); ++e) {
    const auto u = static_cast<std::size_t>(batch.insert_src[e]);
    const auto v = static_cast<std::size_t>(batch.insert_dst[e]);
    if (status[u] == vertex_status::active && status[v] == vertex_status::active) {
      ++counts[u];
      ++counts[v];
    }
  }
}

template <typename vertex_t, typename edge_t, typename weight_t>
void slack_compaction_layout(const csr<vertex_t, edge_t, weight_t>& slots,
                             const slotted_rows<vertex_t, edge_t>& rows,
                             const std::vector<vertex_status>& status,
                             const std::vector<std::int64_t>& kept,
                             const std::vector<std::int64_t>& appended, std::vector<edge_t>& starts,
                             std::vector<edge_t>& limits) {
  const std::size_t n = status.size();
  const std::size_t rows_before = rows.row_limit.size();
  starts.assign(n + 1, edge_t{0});
  limits.assign(n, edge_t{0});
  std::int64_t total = 0;
  for (std::size_t v = 0; v < n; ++v) {
    const std::int64_t entries = kept[v] + appended[v];
    std::int64_t cap = 0;
    if (entries > 0) {
      cap = slack_capacity(entries, rows.headroom);
    } else if (status[v] == vertex_status::inactive && v < rows_before) {
      cap = row_capacity_of(slots, rows, v);  // never written: its hint stays
    }
    if (cap > edge_limit<edge_t>() - total) {
      throw_arena_limit<edge_t>(total + cap);
    }
    starts[v] = static_cast<edge_t>(total);
    total += cap;
    limits[v] = static_cast<edge_t>(total);
  }
  starts[n] = static_cast<edge_t>(total);
}

namespace {

/// The kept entries of every row for a compaction: those whose column is not deleted (the states
/// after the batch's deletions), none of a deleted vertex's own row.
/// The rows of the first `rows_before` vertices exist in `slots` (the rest grew in this batch and
/// are empty); `status` holds the states after the batch.
template <typename vertex_t, typename edge_t, typename weight_t>
std::vector<std::int64_t> kept_entries(const csr<vertex_t, edge_t, weight_t>& slots,
                                       const slotted_rows<vertex_t, edge_t>& rows,
                                       const std::vector<vertex_status>& status) {
  const std::size_t n = status.size();
  const std::size_t rows_before = rows.row_end.size();
  std::vector<std::int64_t> kept(n, 0);
  for (std::size_t v = 0; v < rows_before; ++v) {
    if (status[v] == vertex_status::deleted) {
      continue;
    }
    std::int64_t k = 0;
    for (auto e = static_cast<std::size_t>(slots.row_ptr[v]);
         e < static_cast<std::size_t>(rows.row_end[v]); ++e) {
      k += status[static_cast<std::size_t>(slots.col_ind[e])] != vertex_status::deleted ? 1 : 0;
    }
    kept[v] = k;
  }
  return kept;
}

/// Rewrite the rows (grown and with the states after the batch) into a new arena by `starts` /
/// `limits`, keeping the entries whose column is not deleted in their order.
template <typename vertex_t, typename edge_t, typename weight_t>
void rewrite_rows(csr<vertex_t, edge_t, weight_t>& slots, slotted_rows<vertex_t, edge_t>& rows,
                  std::vector<edge_t>& starts, std::vector<edge_t>& limits,
                  std::vector<vertex_t>& col, std::vector<weight_t>& w) noexcept {
  const std::size_t n = rows.status.size();
  const int k = slots.num_weights;
  for (std::size_t v = 0; v < n; ++v) {
    auto at = static_cast<std::size_t>(starts[v]);
    if (rows.status[v] != vertex_status::deleted) {
      for (auto e = static_cast<std::size_t>(slots.row_ptr[v]);
           e < static_cast<std::size_t>(rows.row_end[v]); ++e) {
        const vertex_t c = slots.col_ind[e];
        if (rows.status[static_cast<std::size_t>(c)] == vertex_status::deleted) {
          continue;
        }
        col[at] = c;
        if (k > 0) {
          w[at] = slots.weights[e];
        }
        ++at;
      }
    }
    rows.row_end[v] = static_cast<edge_t>(at);
  }
  slots.row_ptr.swap(starts);
  rows.row_limit.swap(limits);
  slots.col_ind.swap(col);
  slots.weights.swap(w);
  rows.dead_slots = 0;
  ++rows.compactions;
  ++rows.layout_epoch;
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_vertex_batch_slack_host(csr<vertex_t, edge_t, weight_t>& slots,
                                            slotted_rows<vertex_t, edge_t>& rows,
                                            const edge_batch_view<vertex_t, weight_t>& batch,
                                            const normalized_batch<vertex_t>& normalized,
                                            apply_delta<vertex_t>* delta) {
  DYNG_EXPECTS(normalized.vertex_step, "apply (slack): Step 0 of this batch is missing");
  const std::size_t num_edges = batch.insert_src.size();
  const int k = slots.num_weights;
  apply_summary summary;
  // The record and the arena first: their allocations are the only ones (strong guarantee).
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
  const auto n_after = static_cast<std::size_t>(normalized.vertices_after);
  // A compaction's new arena is laid out and allocated before anything changes, from the states
  // the batch leaves (computed on a copy).
  std::vector<edge_t> starts;
  std::vector<edge_t> limits;
  std::vector<vertex_t> new_col;
  std::vector<weight_t> new_w;
  if (normalized.slack_compact) {
    std::vector<vertex_status> after(rows.status);
    after.resize(n_after, vertex_status::inactive);
    for (const vertex_t v : batch.delete_vertices) {
      after[static_cast<std::size_t>(v)] = vertex_status::deleted;
    }
    for (const vertex_t v : normalized.inserted_vertices) {
      after[static_cast<std::size_t>(v)] = vertex_status::active;
    }
    std::vector<std::int64_t> appended;
    slack_append_counts(after, batch, appended);
    const std::vector<std::int64_t> kept = kept_entries(slots, rows, after);
    slack_compaction_layout(slots, rows, after, kept, appended, starts, limits);
    const auto m = static_cast<std::size_t>(starts.back());
    new_col.assign(m, vertex_t{-1});
    new_w.assign(m * static_cast<std::size_t>(k), weight_t{});
  } else {
    std::int64_t arena_after = static_cast<std::int64_t>(slots.row_ptr.back());
    for (std::size_t i = 1; i < normalized.slack_moves.size(); i += 2) {
      arena_after += normalized.slack_moves[i];
    }
    slots.col_ind.reserve(static_cast<std::size_t>(arena_after));
    slots.weights.reserve(static_cast<std::size_t>(arena_after) * static_cast<std::size_t>(k));
  }
  grow_slack_rows(slots, rows, static_cast<std::int64_t>(n_after));
  // Deletions (Del_eff loses its live entries; every listed id ends deleted), then Ins_eff.
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
  for (const vertex_t v : normalized.inserted_vertices) {
    rows.status[static_cast<std::size_t>(v)] = vertex_status::active;
    ++rows.active_vertices;
  }
  // The layout step.
  if (normalized.slack_compact) {
    rewrite_rows(slots, rows, starts, limits, new_col, new_w);
    summary.compactions = 1;
  } else {
    const std::size_t n = rows.status.size();
    for (std::size_t i = 0; i + 1 < normalized.slack_moves.size(); i += 2) {
      const auto v = static_cast<std::size_t>(normalized.slack_moves[i]);
      const auto cap = static_cast<std::size_t>(normalized.slack_moves[i + 1]);
      const auto old_start = static_cast<std::size_t>(slots.row_ptr[v]);
      const auto old_limit = static_cast<std::size_t>(rows.row_limit[v]);
      const auto used = static_cast<std::size_t>(rows.row_end[v]) - old_start;
      const std::size_t start = slots.col_ind.size();
      slots.col_ind.resize(start + cap, vertex_t{-1});
      slots.weights.resize((start + cap) * static_cast<std::size_t>(k), weight_t{});
      for (std::size_t j = 0; j < used; ++j) {
        slots.col_ind[start + j] = slots.col_ind[old_start + j];
        if (k > 0) {
          slots.weights[start + j] = slots.weights[old_start + j];
        }
      }
      for (std::size_t s = old_start; s < old_limit; ++s) {
        slots.col_ind[s] = vertex_t{-1};
        if (k > 0) {
          slots.weights[s] = weight_t{};
        }
      }
      rows.dead_slots = static_cast<edge_t>(rows.dead_slots + (old_limit - old_start));
      slots.row_ptr[v] = static_cast<edge_t>(start);
      rows.row_limit[v] = static_cast<edge_t>(start + cap);
      rows.row_end[v] = static_cast<edge_t>(start + used);
      ++rows.relocations;
    }
    if (!normalized.slack_moves.empty()) {
      slots.row_ptr[n] = static_cast<edge_t>(slots.col_ind.size());
      ++rows.layout_epoch;
    }
    summary.relocated_rows = static_cast<std::int64_t>(normalized.slack_moves.size() / 2);
  }
  // Appends in batch order: half-edge 2e (v into u's row), then 2e + 1 (u into v's row).
  const auto append = [&](vertex_t row, vertex_t neighbor, std::size_t e) {
    const auto r = static_cast<std::size_t>(row);
    const auto slot = static_cast<std::size_t>(rows.row_end[r]);
    slots.col_ind[slot] = neighbor;
    if (k > 0) {
      slots.weights[slot] = batch.insert_weights[e];
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
  summary.num_vertices_after = static_cast<std::int64_t>(rows.status.size());
  return summary;
}

template <typename vertex_t, typename edge_t, typename weight_t>
void compact_slack_host(csr<vertex_t, edge_t, weight_t>& slots,
                        slotted_rows<vertex_t, edge_t>& rows) {
  const std::vector<std::int64_t> kept = kept_entries(slots, rows, rows.status);
  const std::vector<std::int64_t> none(rows.status.size(), 0);
  std::vector<edge_t> starts;
  std::vector<edge_t> limits;
  slack_compaction_layout(slots, rows, rows.status, kept, none, starts, limits);
  const auto m = static_cast<std::size_t>(starts.back());
  std::vector<vertex_t> col(m, vertex_t{-1});
  std::vector<weight_t> w(m * static_cast<std::size_t>(slots.num_weights), weight_t{});
  rewrite_rows(slots, rows, starts, limits, col, w);
}

template <typename vertex_t, typename edge_t, typename weight_t>
std::string slack_integrity_violation(const csr<vertex_t, edge_t, weight_t>& slots,
                                      const slotted_rows<vertex_t, edge_t>& rows, int num_weights) {
  if (slots.row_ptr.empty()) {
    return "the row starts are missing";
  }
  const std::int64_t n = slots.num_vertices();
  const auto nn = static_cast<std::size_t>(n);
  const auto arena = static_cast<std::int64_t>(slots.row_ptr[nn]);
  const std::size_t m = slots.col_ind.size();
  if (!rows.slack) {
    return "the rows are not marked slack";
  }
  if (static_cast<std::int64_t>(m) != arena) {
    return "the arena holds " + std::to_string(m) + " slots, the row starts say " +
           std::to_string(arena);
  }
  if (slots.num_weights != num_weights ||
      slots.weights.size() != m * static_cast<std::size_t>(num_weights)) {
    return "the weight slots do not match num_weights x the slots";
  }
  if (rows.row_end.size() != nn || rows.row_limit.size() != nn || rows.status.size() != nn ||
      rows.marks.size() < nn + 1 || rows.counts.size() < nn) {
    return "the row ends, limits, states or scratch do not have one entry per vertex";
  }
  if (rows.marks.back() > 63) {
    return "the tag of the Step-0 marks is out of range";
  }
  std::vector<std::pair<std::int64_t, std::int64_t>> ranges;
  std::int64_t owned = 0;
  std::int64_t active = 0;
  std::int64_t live = 0;
  std::vector<std::pair<vertex_t, vertex_t>> forward;
  std::vector<std::pair<vertex_t, vertex_t>> backward;
  for (std::size_t u = 0; u < nn; ++u) {
    const std::int64_t start = slots.row_ptr[u];
    const std::int64_t end = rows.row_end[u];
    const std::int64_t limit = rows.row_limit[u];
    if (start < 0 || start > end || end > limit || limit > arena) {
      return "row " + std::to_string(u) + " lies outside the arena or its ends are disordered";
    }
    if (limit > start) {
      ranges.emplace_back(start, limit);
    }
    owned += limit - start;
    const vertex_status st = rows.status[u];
    if (st != vertex_status::active && st != vertex_status::deleted &&
        st != vertex_status::inactive) {
      return "vertex " + std::to_string(u) + " has an invalid state";
    }
    if (rows.counts[u] != 0) {
      return "the scratch of vertex " + std::to_string(u) + " is not clear";
    }
    active += st == vertex_status::active ? 1 : 0;
    if (st == vertex_status::inactive && end != start) {
      return "vertex " + std::to_string(u) + " was never inserted but its row has entries";
    }
    for (auto e = static_cast<std::size_t>(start); e < static_cast<std::size_t>(end); ++e) {
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
  }
  std::sort(ranges.begin(), ranges.end());
  for (std::size_t i = 1; i < ranges.size(); ++i) {
    if (ranges[i].first < ranges[i - 1].second) {
      return "two rows own the slot " + std::to_string(ranges[i].first);
    }
  }
  if (owned + static_cast<std::int64_t>(rows.dead_slots) != arena) {
    return "the rows own " + std::to_string(owned) + " slots and " +
           std::to_string(rows.dead_slots) + " are dead, but the arena has " +
           std::to_string(arena);
  }
  // Every slot that holds no entry (a row's unused slots, the dead slots) is -1 / 0.
  std::vector<std::uint8_t> holds(m, 0);
  for (std::size_t u = 0; u < nn; ++u) {
    for (auto e = static_cast<std::size_t>(slots.row_ptr[u]);
         e < static_cast<std::size_t>(rows.row_end[u]); ++e) {
      holds[e] = 1;
    }
  }
  for (std::size_t e = 0; e < m; ++e) {
    if (holds[e] != 0) {
      continue;
    }
    bool clear = slots.col_ind[e] == vertex_t{-1};
    for (int c = 0; c < num_weights; ++c) {
      clear = clear && slots.weights[static_cast<std::size_t>(c) * m + e] == weight_t{};
    }
    if (!clear) {
      return "the slot " + std::to_string(e) + " holds no entry but is not -1 / 0";
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

#define DYNG_INSTANTIATE_SLACK(V, E, W)                                                           \
  template void build_slack<V, E, W>(std::int64_t, array_view<const E>, int,                      \
                                     const graph_properties&, csr<V, E, W>&, slotted_rows<V, E>&, \
                                     bool);                                                       \
  template void normalize_vertex_batch_slack<V, E, W>(                                            \
      const csr<V, E, W>&, const slotted_rows<V, E>&, const edge_batch_view<V, W>&,               \
      normalized_batch<V>&);                                                                      \
  template void grow_slack_rows<V, E, W>(csr<V, E, W>&, slotted_rows<V, E>&, std::int64_t);       \
  template void slack_compaction_layout<V, E, W>(                                                 \
      const csr<V, E, W>&, const slotted_rows<V, E>&, const std::vector<vertex_status>&,          \
      const std::vector<std::int64_t>&, const std::vector<std::int64_t>&, std::vector<E>&,        \
      std::vector<E>&);                                                                           \
  template apply_summary apply_vertex_batch_slack_host<V, E, W>(                                  \
      csr<V, E, W>&, slotted_rows<V, E>&, const edge_batch_view<V, W>&,                           \
      const normalized_batch<V>&, apply_delta<V>*);                                               \
  template void compact_slack_host<V, E, W>(csr<V, E, W>&, slotted_rows<V, E>&);                  \
  template std::string slack_integrity_violation<V, E, W>(const csr<V, E, W>&,                    \
                                                          const slotted_rows<V, E>&, int);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_SLACK)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_SLACK)
DYNG_FOR_EACH_FLOAT_GRAPH_TYPE(DYNG_INSTANTIATE_SLACK)
#undef DYNG_INSTANTIATE_SLACK

template void slack_append_counts<std::int32_t, std::int32_t>(
    const std::vector<vertex_status>&, const edge_batch_view<std::int32_t, std::int32_t>&,
    std::vector<std::int64_t>&);
template void slack_append_counts<std::int64_t, std::int32_t>(
    const std::vector<vertex_status>&, const edge_batch_view<std::int64_t, std::int32_t>&,
    std::vector<std::int64_t>&);
template void slack_append_counts<std::int32_t, unweighted>(
    const std::vector<vertex_status>&, const edge_batch_view<std::int32_t, unweighted>&,
    std::vector<std::int64_t>&);
template void slack_append_counts<std::int32_t, float>(const std::vector<vertex_status>&,
                                                       const edge_batch_view<std::int32_t, float>&,
                                                       std::vector<std::int64_t>&);

}  // namespace dyng::detail
