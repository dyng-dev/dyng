// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:src/csrGraph.cpp (applyChangeBatch, transposeCsrGraph)
/**
 * @file apply_host.cpp
 * @brief Host batch application, transposition, construction and integrity checks for the
 *        compact CSR layout.
 *
 * apply_batch_host() is MOSP's applyChangeBatch() ported straight: the changes are grouped by
 * source row keeping batch order, every changed row is rebuilt as a list of (neighbour, weight
 * slot, alive) entries (deletions remove the first alive match, insertions overwrite the first
 * alive match or append), and a new CSR is assembled; the weight-increase classification then
 * compares the first (u,v) of each row before and after. The mechanical changes are: names,
 * templates on the index types, objective-major weight columns (the MOSP CsrGraph is edge-major),
 * K > 32 (one byte per objective instead of a 32-bit mask), exceptions instead of bool + cout,
 * and the batch_semantics switches around the unchanged core (Step 0: self-loops, vertex growth,
 * undirected expansion; ignore/error policies; insertions-first order; sorted rows).
 */
#include "graph/apply_host.hpp"

#include "graph/instantiate.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

/// Throws dyng::invalid_argument_error with a message and file:line (no condition text).
#define DYNG_THROW_INVALID_ARGUMENT(...)                                                           \
  ::dyng::detail::throw_with_location<::dyng::invalid_argument_error>(__FILE__, __LINE__, nullptr, \
                                                                      __VA_ARGS__)

namespace dyng::detail {

namespace {

/// Throws unless a non-empty view is readable on the host.
template <typename value_t>
void expect_host(const array_view<value_t>& view, const char* what) {
  DYNG_EXPECTS(view.empty() || is_host_accessible(view.space()), what,
               " must be in host-accessible memory for a host graph");
}

/// One entry of a row being rebuilt. slot >= 0 is an edge of the original graph; slot < 0 is
/// effective insertion -slot - 1.
template <typename vertex_t>
struct row_entry {
  vertex_t vertex;
  std::int64_t slot;
  bool alive;
};

/// A self-loop in a batch or an input: returns true if it is to be skipped.
bool skip_self_loop(batch_semantics::self_loop policy, const char* what, std::int64_t index,
                    std::int64_t u) {
  switch (policy) {
    case batch_semantics::self_loop::keep:
      return false;
    case batch_semantics::self_loop::drop:
      return true;
    case batch_semantics::self_loop::error:
      break;
  }
  DYNG_THROW_INVALID_ARGUMENT(what, " ", index, " is a self-loop (", u, ", ", u,
                              ") and the graph's batch_semantics::on_self_loop is error");
  return true;
}

}  // namespace

// ------------------------------------------------------------------------------------------------
// apply
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_batch_host(const csr<vertex_t, edge_t, weight_t>& original,
                               const edge_batch_view<vertex_t, weight_t>& batch,
                               const graph_properties& props,
                               csr<vertex_t, edge_t, weight_t>& updated,
                               apply_delta<vertex_t>* delta, int threads) {
  const batch_semantics& semantics = props.semantics;
  const int num_objectives = original.num_weights;
  const auto k_count = static_cast<std::size_t>(num_objectives);
  const std::int64_t n = original.num_vertices();
  const std::size_t m = original.col_ind.size();
  const std::size_t num_inserts = batch.insert_src.size();
  const std::size_t num_deletes = batch.delete_src.size();

  // --- Validation ------------------------------------------------------------------------------
  if (!batch.insert_vertices.empty() || !batch.delete_vertices.empty() ||
      !batch.insert_vertex_labels.empty()) {
    throw not_supported_error(
        "dyng: vertex insertions and deletions are not supported yet (planned for 0.3)");
  }
  DYNG_EXPECTS(batch.insert_dst.size() == num_inserts, "the batch has ", num_inserts,
               " insertion sources but ", batch.insert_dst.size(), " destinations");
  DYNG_EXPECTS(batch.delete_dst.size() == num_deletes, "the batch has ", num_deletes,
               " deletion sources but ", batch.delete_dst.size(), " destinations");
  if (num_inserts > 0) {
    DYNG_EXPECTS(batch.num_weights == num_objectives, "the batch has ", batch.num_weights,
                 " weights per insertion but the graph has ", num_objectives, " weight columns");
    DYNG_EXPECTS(batch.insert_weights.size() == num_inserts * k_count, "the batch has ",
                 batch.insert_weights.size(), " insertion weights, expected ", num_inserts, " x ",
                 num_objectives);
  }
  expect_host(batch.insert_src, "edge_batch_view::insert_src");
  expect_host(batch.insert_dst, "edge_batch_view::insert_dst");
  expect_host(batch.insert_weights, "edge_batch_view::insert_weights");
  expect_host(batch.delete_src, "edge_batch_view::delete_src");
  expect_host(batch.delete_dst, "edge_batch_view::delete_dst");

  apply_summary summary;

  // --- Step 0: the effective operations (self-loops, vertex growth, undirected expansion) -------
  std::vector<vertex_t> ins_from;
  std::vector<vertex_t> ins_to;
  std::vector<std::size_t> ins_weights;  // batch insertion whose weights are used
  std::vector<vertex_t> del_from;
  std::vector<vertex_t> del_to;
  ins_from.reserve(num_inserts);
  ins_to.reserve(num_inserts);
  ins_weights.reserve(num_inserts);
  std::int64_t n_after = n;
  const auto id_limit = static_cast<std::int64_t>(std::numeric_limits<vertex_t>::max());
  for (std::size_t i = 0; i < num_inserts; ++i) {
    const vertex_t u = batch.insert_src[i];
    const vertex_t v = batch.insert_dst[i];
    DYNG_EXPECTS(u >= 0 && v >= 0, "insertion ", i, " (", u, ", ", v, ") has a negative id");
    if (!semantics.allow_vertex_growth) {
      DYNG_EXPECTS(u < n && v < n, "insertion ", i, " (", u, ", ", v, ") names a vertex >= ", n,
                   " and batch_semantics::allow_vertex_growth is false");
    }
    if (u == v &&
        skip_self_loop(semantics.on_self_loop, "insertion", static_cast<std::int64_t>(i), u)) {
      ++summary.dropped_self_loops;
      continue;
    }
    DYNG_EXPECTS(std::max<std::int64_t>(u, v) < id_limit, "insertion ", i,
                 " names the largest representable vertex id");
    n_after = std::max<std::int64_t>(n_after, std::max<std::int64_t>(u, v) + 1);
    ins_from.push_back(u);
    ins_to.push_back(v);
    ins_weights.push_back(i);
    if (!props.directed && u != v) {
      ins_from.push_back(v);
      ins_to.push_back(u);
      ins_weights.push_back(i);
    }
  }
  const auto missing = [&](std::size_t j, std::int64_t u, std::int64_t v) {
    DYNG_EXPECTS(semantics.on_missing_delete == batch_semantics::missing_delete::ignore,
                 "deletion ", j, " (", u, ", ", v,
                 ") names a missing edge and batch_semantics::on_missing_delete is error");
    ++summary.ignored_deletions;
  };
  for (std::size_t j = 0; j < num_deletes; ++j) {
    const vertex_t u = batch.delete_src[j];
    const vertex_t v = batch.delete_dst[j];
    DYNG_EXPECTS(u >= 0 && v >= 0, "deletion ", j, " (", u, ", ", v, ") has a negative id");
    if (u == v &&
        skip_self_loop(semantics.on_self_loop, "deletion", static_cast<std::int64_t>(j), u)) {
      ++summary.dropped_self_loops;
      continue;
    }
    if (u >= n_after || v >= n_after) {
      missing(j, u, v);  // no such vertex, so no such edge
      if (!props.directed && u != v) {
        missing(j, v, u);
      }
      continue;
    }
    del_from.push_back(u);
    del_to.push_back(v);
    if (!props.directed && u != v) {
      del_from.push_back(v);
      del_to.push_back(u);
    }
  }
  const std::size_t num_ins = ins_from.size();
  const std::size_t num_del = del_from.size();
  const auto rows = static_cast<std::size_t>(n_after);

  // --- Group changes by source row, keeping batch order inside each row ------------------------
  std::vector<std::size_t> del_start(rows + 1, 0);
  std::vector<std::size_t> ins_start(rows + 1, 0);
  for (std::size_t j = 0; j < num_del; ++j) {
    ++del_start[static_cast<std::size_t>(del_from[j]) + 1];
  }
  for (std::size_t i = 0; i < num_ins; ++i) {
    ++ins_start[static_cast<std::size_t>(ins_from[i]) + 1];
  }
  for (std::size_t u = 0; u < rows; ++u) {
    del_start[u + 1] += del_start[u];
    ins_start[u + 1] += ins_start[u];
  }
  std::vector<std::size_t> del_order(num_del);
  std::vector<std::size_t> ins_order(num_ins);
  {
    std::vector<std::size_t> cursor(del_start.begin(), del_start.end() - 1);
    for (std::size_t j = 0; j < num_del; ++j) {
      del_order[cursor[static_cast<std::size_t>(del_from[j])]++] = j;
    }
    cursor.assign(ins_start.begin(), ins_start.end() - 1);
    for (std::size_t i = 0; i < num_ins; ++i) {
      ins_order[cursor[static_cast<std::size_t>(ins_from[i])]++] = i;
    }
  }

  // --- Rebuild every changed row ---------------------------------------------------------------
  std::vector<std::vector<row_entry<vertex_t>>> changed_contents;
  std::vector<std::int64_t> row_index(rows, -1);
  for (std::size_t u = 0; u < rows; ++u) {
    if (del_start[u] == del_start[u + 1] && ins_start[u] == ins_start[u + 1]) {
      continue;
    }
    std::vector<row_entry<vertex_t>> row;
    if (static_cast<std::int64_t>(u) < n) {
      for (auto e = static_cast<std::int64_t>(original.row_ptr[u]);
           e < static_cast<std::int64_t>(original.row_ptr[u + 1]); ++e) {
        row.push_back({original.col_ind[static_cast<std::size_t>(e)], e, true});
      }
    }
    const auto apply_deletions = [&] {
      for (std::size_t j = del_start[u]; j < del_start[u + 1]; ++j) {
        const vertex_t v = del_to[del_order[j]];
        bool removed = false;
        for (auto& entry : row) {
          if (entry.alive && entry.vertex == v) {
            entry.alive = false;
            removed = true;
            break;
          }
        }
        if (removed) {
          ++summary.deleted_edges;
        } else {
          missing(del_order[j], static_cast<std::int64_t>(u), v);
        }
      }
    };
    const auto apply_insertions = [&] {
      for (std::size_t j = ins_start[u]; j < ins_start[u + 1]; ++j) {
        const std::size_t i = ins_order[j];
        const vertex_t v = ins_to[i];
        bool replaced = false;
        for (auto& entry : row) {
          if (entry.alive && entry.vertex == v) {
            switch (semantics.on_existing_insert) {
              case batch_semantics::existing_insert::upsert:
                entry.slot = -static_cast<std::int64_t>(i) - 1;
                ++summary.updated_edges;
                break;
              case batch_semantics::existing_insert::ignore:
                ++summary.ignored_insertions;
                break;
              case batch_semantics::existing_insert::error:
                DYNG_THROW_INVALID_ARGUMENT("insertion ", ins_weights[i], " (", u, ", ", v,
                                            ") names an existing edge and "
                                            "batch_semantics::on_existing_insert is error");
            }
            replaced = true;
            break;
          }
        }
        if (!replaced) {
          row.push_back({v, -static_cast<std::int64_t>(i) - 1, true});
          ++summary.inserted_edges;
        }
      }
    };
    if (semantics.deletions_first) {
      apply_deletions();
      apply_insertions();
    } else {
      apply_insertions();
      apply_deletions();
    }
    if (props.order == row_order::sorted) {
      row.erase(std::remove_if(row.begin(), row.end(), [](const auto& e) { return !e.alive; }),
                row.end());
      std::stable_sort(row.begin(), row.end(),
                       [](const auto& a, const auto& b) { return a.vertex < b.vertex; });
    }
    row_index[u] = static_cast<std::int64_t>(changed_contents.size());
    changed_contents.push_back(std::move(row));
  }

  // --- Assemble the updated CSR ------------------------------------------------------------------
  // Every row lands at the same place whatever the thread count: the degrees are computed per row,
  // the offsets by a prefix sum, and each row (or run of unchanged rows) is copied to its offset.
  // With the OpenMP backend the three passes run in parallel; the result is identical.
  updated = csr<vertex_t, edge_t, weight_t>();
  updated.num_weights = num_objectives;
  updated.row_ptr.assign(rows + 1, edge_t{0});
  const auto row_count = static_cast<std::int64_t>(rows);
  std::vector<std::int64_t> degree(rows, 0);
  std::int64_t* deg = degree.data();
  const bool parallel = threads > 1 && row_count >= 65536;
  (void)parallel;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(static)
#endif
  for (std::int64_t u = 0; u < row_count; ++u) {
    const std::int64_t index = row_index[static_cast<std::size_t>(u)];
    std::int64_t d = 0;
    if (index >= 0) {
      for (const auto& entry : changed_contents[static_cast<std::size_t>(index)]) {
        d += entry.alive ? 1 : 0;
      }
    } else if (u < n) {
      d = static_cast<std::int64_t>(original.row_ptr[static_cast<std::size_t>(u) + 1]) -
          static_cast<std::int64_t>(original.row_ptr[static_cast<std::size_t>(u)]);
    }
    deg[u] = d;
  }
  std::int64_t total = 0;
  for (std::size_t u = 0; u < rows; ++u) {
    total += degree[u];
    updated.row_ptr[u + 1] = checked_edge_count<edge_t>(total);
  }
  const auto m_new = static_cast<std::size_t>(total);
  updated.col_ind.resize(m_new);
  updated.weights.resize(m_new * k_count);
  const edge_t* new_row = updated.row_ptr.data();
  vertex_t* new_col = updated.col_ind.data();
  weight_t* new_weights = updated.weights.data();
  // Rows are handed out in blocks; inside a block a maximal run of unchanged rows is contiguous in
  // both graphs and copied at once (the weight columns are objective-major, so a per-row copy
  // would cost K calls per row).
  constexpr std::int64_t block = 4096;
  const std::int64_t blocks = (row_count + block - 1) / block;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(dynamic, 16)
#endif
  for (std::int64_t b = 0; b < blocks; ++b) {
    const std::int64_t first = b * block;
    const std::int64_t stop = std::min(row_count, first + block);
    for (std::int64_t u = first; u < stop; ++u) {
      auto out = static_cast<std::size_t>(new_row[u]);
      const std::int64_t index = row_index[static_cast<std::size_t>(u)];
      if (index < 0) {
        if (u >= n) {
          continue;
        }
        std::int64_t last = u + 1;
        while (last < stop && last < n && row_index[static_cast<std::size_t>(last)] < 0) {
          ++last;
        }
        const auto begin = static_cast<std::size_t>(original.row_ptr[static_cast<std::size_t>(u)]);
        const auto end = static_cast<std::size_t>(original.row_ptr[static_cast<std::size_t>(last)]);
        std::copy(original.col_ind.data() + begin, original.col_ind.data() + end, new_col + out);
        for (std::size_t k = 0; k < k_count; ++k) {
          const auto* column = original.weights.data() + k * m;
          std::copy(column + begin, column + end, new_weights + k * m_new + out);
        }
        u = last - 1;
        continue;
      }
      for (const auto& entry : changed_contents[static_cast<std::size_t>(index)]) {
        if (!entry.alive) {
          continue;
        }
        new_col[out] = entry.vertex;
        if (entry.slot >= 0) {
          const auto e = static_cast<std::size_t>(entry.slot);
          for (std::size_t k = 0; k < k_count; ++k) {
            new_weights[k * m_new + out] = original.weights[k * m + e];
          }
        } else {
          const std::size_t i = ins_weights[static_cast<std::size_t>(-entry.slot - 1)];
          for (std::size_t k = 0; k < k_count; ++k) {
            new_weights[k * m_new + out] = batch.insert_weights[i * k_count + k];
          }
        }
        ++out;
      }
    }
  }
  summary.inserted_vertices = n_after - n;
  summary.num_vertices_after = n_after;

  // --- Classification: weight increases -----------------------------------------------------------
  // Compare the final weight of (u,v) with its weight in the original graph (first occurrence of
  // v in row u in both), as applyChangeBatch() does.
  if (delta != nullptr) {
    delta->num_weights = num_objectives;
    delta->insert_src = ins_from;
    delta->insert_dst = ins_to;
    delta->weight_increased.assign(num_ins * k_count, 0);
    delta->delete_src = std::move(del_from);
    delta->delete_dst = std::move(del_to);
    for (std::size_t i = 0; i < num_ins; ++i) {
      const auto u = static_cast<std::size_t>(ins_from[i]);
      const vertex_t v = ins_to[i];
      if (static_cast<std::int64_t>(u) >= n) {
        continue;
      }
      std::int64_t original_edge = -1;
      for (auto e = static_cast<std::int64_t>(original.row_ptr[u]);
           e < static_cast<std::int64_t>(original.row_ptr[u + 1]); ++e) {
        if (original.col_ind[static_cast<std::size_t>(e)] == v) {
          original_edge = e;
          break;
        }
      }
      if (original_edge < 0) {
        continue;
      }
      std::int64_t final_edge = -1;
      for (auto e = static_cast<std::int64_t>(updated.row_ptr[u]);
           e < static_cast<std::int64_t>(updated.row_ptr[u + 1]); ++e) {
        if (updated.col_ind[static_cast<std::size_t>(e)] == v) {
          final_edge = e;
          break;
        }
      }
      if (final_edge < 0) {
        continue;
      }
      for (std::size_t k = 0; k < k_count; ++k) {
        if (updated.weights[k * m_new + static_cast<std::size_t>(final_edge)] >
            original.weights[k * m + static_cast<std::size_t>(original_edge)]) {
          delta->weight_increased[i * k_count + k] = 1;
        }
      }
    }
  }
  return summary;
}

// ------------------------------------------------------------------------------------------------
// transpose
// ------------------------------------------------------------------------------------------------

#if DYNG_HAS_OPENMP
namespace {

/// transpose_host() with OpenMP: count and fill with atomics, then sort every row's edge indices,
/// so the result equals the sequential transposition exactly (inside a row the sources appear in
/// increasing out-edge index, i.e. in out-edge order).
template <typename vertex_t, typename edge_t, typename weight_t>
void transpose_parallel(const csr<vertex_t, edge_t, weight_t>& graph,
                        csr<vertex_t, edge_t, weight_t>& reverse, int threads) {
  const auto n = static_cast<std::int64_t>(graph.num_vertices());
  const auto k_count = static_cast<std::size_t>(graph.num_weights);
  const auto m = static_cast<std::int64_t>(graph.col_ind.size());
  const vertex_t* col = graph.col_ind.data();
  const edge_t* row = graph.row_ptr.data();
  // Scratch arrays are left uninitialized (every entry is written below, in parallel), so their
  // pages are first touched by the threads instead of by one zero-filling thread.
  std::unique_ptr<vertex_t[]> source(new vertex_t[static_cast<std::size_t>(m)]);
  vertex_t* src = source.get();
  edge_t* counts = reverse.row_ptr.data();
#pragma omp parallel num_threads(threads)
  {
#pragma omp for schedule(dynamic, 1024)
    for (std::int64_t u = 0; u < n; ++u) {
      for (edge_t e = row[u]; e < row[u + 1]; ++e) {
        src[e] = static_cast<vertex_t>(u);
      }
    }
#pragma omp for schedule(static)
    for (std::int64_t e = 0; e < m; ++e) {
      __atomic_fetch_add(&counts[static_cast<std::size_t>(col[e]) + 1], edge_t{1},
                         __ATOMIC_RELAXED);
    }
  }
  for (std::int64_t v = 0; v < n; ++v) {
    counts[v + 1] += counts[v];
  }
  reverse.col_ind.resize(static_cast<std::size_t>(m));
  reverse.weights.resize(static_cast<std::size_t>(m) * k_count);
  std::vector<edge_t> cursor(reverse.row_ptr.begin(), reverse.row_ptr.end() - 1);
  std::unique_ptr<edge_t[]> edge_of(new edge_t[static_cast<std::size_t>(m)]);
  edge_t* next = cursor.data();
  edge_t* slot = edge_of.get();
  const edge_t* in_row = reverse.row_ptr.data();
  vertex_t* in_col = reverse.col_ind.data();
#pragma omp parallel num_threads(threads)
  {
#pragma omp for schedule(static)
    for (std::int64_t e = 0; e < m; ++e) {
      const edge_t pos = __atomic_fetch_add(&next[col[e]], edge_t{1}, __ATOMIC_RELAXED);
      slot[pos] = static_cast<edge_t>(e);
    }
#pragma omp for schedule(dynamic, 1024)
    for (std::int64_t v = 0; v < n; ++v) {
      std::sort(slot + in_row[v], slot + in_row[v + 1]);
    }
#pragma omp for schedule(static)
    for (std::int64_t pos = 0; pos < m; ++pos) {
      in_col[pos] = src[slot[pos]];
    }
    for (std::size_t k = 0; k < k_count; ++k) {
      const weight_t* w_in = graph.weights.data() + k * static_cast<std::size_t>(m);
      weight_t* w_out = reverse.weights.data() + k * static_cast<std::size_t>(m);
#pragma omp for schedule(static)
      for (std::int64_t pos = 0; pos < m; ++pos) {
        w_out[pos] = w_in[slot[pos]];
      }
    }
  }
}

}  // namespace
#endif  // DYNG_HAS_OPENMP

template <typename vertex_t, typename edge_t, typename weight_t>
void transpose_host(const csr<vertex_t, edge_t, weight_t>& graph,
                    csr<vertex_t, edge_t, weight_t>& reverse, int threads) {
  const auto n = static_cast<std::size_t>(graph.num_vertices());
  const auto k_count = static_cast<std::size_t>(graph.num_weights);
  const std::size_t m = graph.col_ind.size();
  reverse = csr<vertex_t, edge_t, weight_t>();
  reverse.num_weights = graph.num_weights;
  reverse.row_ptr.assign(n + 1, edge_t{0});
#if DYNG_HAS_OPENMP
  if (threads > 1 && m >= 4096) {
    transpose_parallel(graph, reverse, threads);
    return;
  }
#else
  (void)threads;
#endif
  for (std::size_t e = 0; e < m; ++e) {
    ++reverse.row_ptr[static_cast<std::size_t>(graph.col_ind[e]) + 1];
  }
  for (std::size_t v = 0; v < n; ++v) {
    reverse.row_ptr[v + 1] += reverse.row_ptr[v];
  }
  reverse.col_ind.resize(m);
  reverse.weights.resize(m * k_count);
  std::vector<edge_t> cursor(reverse.row_ptr.begin(), reverse.row_ptr.end() - 1);
  // The position of every edge in the reverse graph, then one scatter per weight column (the
  // columns are objective-major, so each pass reads its column sequentially).
  std::vector<edge_t> position(m);
  for (std::size_t u = 0; u < n; ++u) {
    for (auto e = static_cast<std::size_t>(graph.row_ptr[u]);
         e < static_cast<std::size_t>(graph.row_ptr[u + 1]); ++e) {
      const edge_t pos = cursor[static_cast<std::size_t>(graph.col_ind[e])]++;
      position[e] = pos;
      reverse.col_ind[static_cast<std::size_t>(pos)] = static_cast<vertex_t>(u);
    }
  }
  for (std::size_t k = 0; k < k_count; ++k) {
    const weight_t* in = graph.weights.data() + k * m;
    weight_t* out = reverse.weights.data() + k * m;
    for (std::size_t e = 0; e < m; ++e) {
      out[static_cast<std::size_t>(position[e])] = in[e];
    }
  }
}

// ------------------------------------------------------------------------------------------------
// construction
// ------------------------------------------------------------------------------------------------

namespace {

/// Builds the compact CSR from (src[i], dst[i]) and a weight accessor weight_of(i, k).
template <typename vertex_t, typename edge_t, typename weight_t, typename weight_of_t>
void build_rows(std::int64_t n, int num_weights, const std::vector<vertex_t>& src,
                const std::vector<vertex_t>& dst, const std::vector<std::size_t>& index,
                const graph_properties& props, const weight_of_t& weight_of,
                csr<vertex_t, edge_t, weight_t>& out) {
  const auto rows = static_cast<std::size_t>(n);
  const auto k_count = static_cast<std::size_t>(num_weights);
  // Counting sort by source (stable: keeps input order inside a row).
  std::vector<std::size_t> start(rows + 1, 0);
  for (const vertex_t u : src) {
    ++start[static_cast<std::size_t>(u) + 1];
  }
  for (std::size_t u = 0; u < rows; ++u) {
    start[u + 1] += start[u];
  }
  std::vector<std::size_t> order(src.size());
  {
    std::vector<std::size_t> cursor(start.begin(), start.end() - 1);
    for (std::size_t i = 0; i < src.size(); ++i) {
      order[cursor[static_cast<std::size_t>(src[i])]++] = i;
    }
  }
  // Per row: (neighbour, item) in final order.
  struct item {
    vertex_t vertex;
    std::size_t first;  // position of the first occurrence (row order)
    std::size_t last;   // item whose weights are used (last occurrence)
  };
  std::vector<item> kept;
  kept.reserve(src.size());
  std::vector<std::int64_t> degree(rows, 0);
  std::vector<item> row;
  for (std::size_t u = 0; u < rows; ++u) {
    row.clear();
    for (std::size_t j = start[u]; j < start[u + 1]; ++j) {
      row.push_back({dst[order[j]], j, order[j]});
    }
    if (props.parallel_edges == multi_edges::forbid) {
      std::stable_sort(row.begin(), row.end(),
                       [](const item& a, const item& b) { return a.vertex < b.vertex; });
      std::size_t w = 0;
      for (std::size_t r = 0; r < row.size(); ++r) {
        if (w > 0 && row[w - 1].vertex == row[r].vertex) {
          row[w - 1].last = row[r].last;  // the last occurrence wins
        } else {
          row[w++] = row[r];
        }
      }
      row.resize(w);
      if (props.order == row_order::append) {
        std::sort(row.begin(), row.end(),
                  [](const item& a, const item& b) { return a.first < b.first; });
      }
    } else if (props.order == row_order::sorted) {
      std::stable_sort(row.begin(), row.end(),
                       [](const item& a, const item& b) { return a.vertex < b.vertex; });
    }
    degree[u] = static_cast<std::int64_t>(row.size());
    kept.insert(kept.end(), row.begin(), row.end());
  }
  out = csr<vertex_t, edge_t, weight_t>();
  out.num_weights = num_weights;
  out.row_ptr.assign(rows + 1, edge_t{0});
  std::int64_t total = 0;
  for (std::size_t u = 0; u < rows; ++u) {
    total += degree[u];
    out.row_ptr[u + 1] = checked_edge_count<edge_t>(total);
  }
  const std::size_t m = kept.size();
  out.col_ind.resize(m);
  out.weights.resize(m * k_count);
  for (std::size_t e = 0; e < m; ++e) {
    out.col_ind[e] = kept[e].vertex;
    for (std::size_t k = 0; k < k_count; ++k) {
      out.weights[k * m + e] = weight_of(index[kept[e].last], k);
    }
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void build_from_edges_host(const edge_list_view<vertex_t, weight_t>& edges,
                           const graph_properties& props, csr<vertex_t, edge_t, weight_t>& out) {
  const std::size_t count = edges.src.size();
  const std::int64_t n = edges.num_vertices;
  const int num_objectives = edges.num_weights;
  DYNG_EXPECTS(n >= 0, "edge_list_view::num_vertices must be >= 0, got ", n);
  DYNG_EXPECTS(num_objectives >= 0, "edge_list_view::num_weights must be >= 0, got ",
               num_objectives);
  DYNG_EXPECTS(edges.dst.size() == count, "the edge list has ", count, " sources but ",
               edges.dst.size(), " destinations");
  DYNG_EXPECTS(edges.weights.size() == count * static_cast<std::size_t>(num_objectives),
               "the edge list has ", edges.weights.size(), " weights, expected ", count, " x ",
               num_objectives);
  expect_host(edges.src, "edge_list_view::src");
  expect_host(edges.dst, "edge_list_view::dst");
  expect_host(edges.weights, "edge_list_view::weights");
  std::vector<vertex_t> src;
  std::vector<vertex_t> dst;
  std::vector<std::size_t> index;
  src.reserve(count);
  dst.reserve(count);
  index.reserve(count);
  for (std::size_t i = 0; i < count; ++i) {
    const vertex_t u = edges.src[i];
    const vertex_t v = edges.dst[i];
    DYNG_EXPECTS(u >= 0 && u < n && v >= 0 && v < n, "edge ", i, " (", u, ", ", v,
                 ") is out of range [0, ", n, ")");
    if (u == v &&
        skip_self_loop(props.semantics.on_self_loop, "edge", static_cast<std::int64_t>(i), u)) {
      continue;
    }
    src.push_back(u);
    dst.push_back(v);
    index.push_back(i);
    if (!props.directed && u != v) {
      src.push_back(v);
      dst.push_back(u);
      index.push_back(i);
    }
  }
  const auto k_count = static_cast<std::size_t>(num_objectives);
  build_rows(
      n, num_objectives, src, dst, index, props,
      [&](std::size_t i, std::size_t k) { return edges.weights[i * k_count + k]; }, out);
}

namespace {

/// What one pass over a CSR finds: whether it is well formed and whether its rows already are
/// what the properties ask for.
struct csr_scan {
  bool malformed = false;      ///< an offset or a column index is invalid
  bool self_loops = false;     ///< some row u lists u
  bool non_decreasing = true;  ///< every row is sorted (equal neighbours allowed)
  bool increasing = true;      ///< every row is sorted without repeated neighbours
};

/// One pass over the rows (in parallel with more than one thread); reads only in-bounds entries.
template <typename vertex_t, typename edge_t, typename weight_t>
csr_scan scan_csr(const csr_view<vertex_t, edge_t, weight_t>& input, int threads) {
  const std::int64_t n = input.num_vertices();
  const auto m = static_cast<std::int64_t>(input.col_ind.size());
  const edge_t* row = input.row_ptr.data();
  const vertex_t* col = input.col_ind.data();
  bool malformed = n > 0 && (row[0] != 0 || static_cast<std::int64_t>(row[n]) != m);
  bool self_loops = false;
  bool non_decreasing = true;
  bool increasing = true;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (threads > 1 && m >= 65536) schedule(static) \
    reduction(|| : malformed, self_loops) reduction(&& : non_decreasing, increasing)
#else
  (void)threads;
#endif
  for (std::int64_t u = 0; u < n; ++u) {
    const auto begin = static_cast<std::int64_t>(row[u]);
    const auto end = static_cast<std::int64_t>(row[u + 1]);
    if (begin < 0 || end < begin || end > m) {
      malformed = true;
      continue;
    }
    for (std::int64_t e = begin; e < end; ++e) {
      const vertex_t v = col[e];
      if (v < 0 || v >= n) {
        malformed = true;
        break;
      }
      self_loops = self_loops || static_cast<std::int64_t>(v) == u;
      if (e > begin) {
        non_decreasing = non_decreasing && col[e - 1] <= v;
        increasing = increasing && col[e - 1] < v;
      }
    }
  }
  return csr_scan{malformed, self_loops, non_decreasing, increasing};
}

/// Whether building the graph from a well-formed CSR would reproduce it unchanged: a directed
/// graph whose self-loops (if any) are kept, and whose rows already have the requested order and
/// no parallel edges where they are forbidden (build_rows() then keeps every row as it is).
bool csr_is_final(const csr_scan& scan, const graph_properties& props) {
  if (!props.directed ||
      (scan.self_loops && props.semantics.on_self_loop != batch_semantics::self_loop::keep)) {
    return false;
  }
  if (props.parallel_edges == multi_edges::allow) {
    return props.order == row_order::append || scan.non_decreasing;
  }
  return scan.increasing;  // forbid: rows without repeats; sorted rows are also in append order
}

/// Parallel copy of `count` values (first touch spread over the threads).
template <typename value_t>
void copy_parallel(const value_t* from, std::size_t count, std::vector<value_t>& to, int threads) {
  to.resize(count);
  value_t* out = to.data();
  const auto size = static_cast<std::int64_t>(count);
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (threads > 1 && size >= 65536) schedule(static)
#else
  (void)threads;
#endif
  for (std::int64_t i = 0; i < size; ++i) {
    out[i] = from[i];
  }
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void build_from_csr_host(const csr_view<vertex_t, edge_t, weight_t>& input,
                         const graph_properties& props, csr<vertex_t, edge_t, weight_t>& out,
                         int threads, csr<vertex_t, edge_t, weight_t>* movable) {
  expect_host(input.row_ptr, "csr_view::row_ptr");
  expect_host(input.col_ind, "csr_view::col_ind");
  expect_host(input.weights, "csr_view::weights");
  const int num_objectives = input.num_weights;
  const std::size_t m = input.col_ind.size();
  DYNG_EXPECTS(num_objectives >= 0, "csr_view::num_weights must be >= 0, got ", num_objectives);
  DYNG_EXPECTS(!input.row_ptr.empty() || m == 0, "csr_view::row_ptr is empty but there are ", m,
               " edges");
  const std::int64_t n = input.num_vertices();
  // Fast path: one (parallel) pass; a CSR that is well formed and already final is copied (or
  // moved) as it is. Anything else takes the general path below, whose checks report the first
  // problem exactly as before.
  const csr_scan scan = scan_csr(input, threads);
  const bool weights_match = input.weights.size() == m * static_cast<std::size_t>(num_objectives);
  if (weights_match && !scan.malformed && csr_is_final(scan, props)) {
    out = csr<vertex_t, edge_t, weight_t>();
    out.num_weights = num_objectives;
    if (movable != nullptr) {
      out.row_ptr = std::move(movable->row_ptr);
      out.col_ind = std::move(movable->col_ind);
      out.weights = std::move(movable->weights);
      *movable = csr<vertex_t, edge_t, weight_t>();
    } else {
      copy_parallel(input.row_ptr.data(), input.row_ptr.size(), out.row_ptr, threads);
      copy_parallel(input.col_ind.data(), m, out.col_ind, threads);
      copy_parallel(input.weights.data(), input.weights.size(), out.weights, threads);
    }
    if (out.row_ptr.empty()) {
      out.row_ptr.assign(1, edge_t{0});
    }
    return;
  }
  if (!input.row_ptr.empty()) {
    DYNG_EXPECTS(input.row_ptr[0] == 0, "csr row_ptr[0] must be 0, got ", input.row_ptr[0]);
    for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
      DYNG_EXPECTS(input.row_ptr[u + 1] >= input.row_ptr[u], "csr row_ptr is not monotone at ",
                   u + 1);
    }
    DYNG_EXPECTS(static_cast<std::size_t>(input.row_ptr[static_cast<std::size_t>(n)]) == m,
                 "csr row_ptr[n] = ", input.row_ptr[static_cast<std::size_t>(n)],
                 " does not match the ", m, " column indices");
  }
  DYNG_EXPECTS(input.weights.size() == m * static_cast<std::size_t>(num_objectives), "csr has ",
               input.weights.size(), " weights, expected ", num_objectives, " x ", m);
  std::vector<vertex_t> src(m);
  std::vector<vertex_t> dst(m);
  for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
    for (auto e = static_cast<std::size_t>(input.row_ptr[u]);
         e < static_cast<std::size_t>(input.row_ptr[u + 1]); ++e) {
      const vertex_t v = input.col_ind[e];
      DYNG_EXPECTS(v >= 0 && v < n, "csr column index ", v, " of edge ", e, " is out of range [0, ",
                   n, ")");
      src[e] = static_cast<vertex_t>(u);
      dst[e] = v;
    }
  }
  if (!props.directed) {
    std::vector<std::pair<vertex_t, vertex_t>> forward(m);
    std::vector<std::pair<vertex_t, vertex_t>> backward(m);
    for (std::size_t e = 0; e < m; ++e) {
      forward[e] = {src[e], dst[e]};
      backward[e] = {dst[e], src[e]};
    }
    std::sort(forward.begin(), forward.end());
    std::sort(backward.begin(), backward.end());
    DYNG_EXPECTS(forward == backward,
                 "an undirected graph needs a symmetric CSR (every edge in both directions)");
  }
  std::vector<vertex_t> kept_src;
  std::vector<vertex_t> kept_dst;
  std::vector<std::size_t> index;
  kept_src.reserve(m);
  kept_dst.reserve(m);
  index.reserve(m);
  for (std::size_t e = 0; e < m; ++e) {
    if (src[e] == dst[e] && skip_self_loop(props.semantics.on_self_loop, "edge",
                                           static_cast<std::int64_t>(e), src[e])) {
      continue;
    }
    kept_src.push_back(src[e]);
    kept_dst.push_back(dst[e]);
    index.push_back(e);
  }
  build_rows(
      n, num_objectives, kept_src, kept_dst, index, props,
      [&](std::size_t e, std::size_t k) { return input.weights[k * m + e]; }, out);
}

// ------------------------------------------------------------------------------------------------
// integrity
// ------------------------------------------------------------------------------------------------

template <typename vertex_t, typename edge_t, typename weight_t>
std::string integrity_violation(const graph_impl<vertex_t, edge_t, weight_t>& impl) {
  const auto& g = impl.out;
  const std::int64_t n = g.num_vertices();
  const std::size_t m = g.col_ind.size();
  if (g.row_ptr.empty()) {
    return "row_ptr is empty";
  }
  if (g.row_ptr[0] != 0) {
    return "row_ptr[0] != 0";
  }
  for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
    if (g.row_ptr[u + 1] < g.row_ptr[u]) {
      return "row_ptr is not monotone at row " + std::to_string(u);
    }
  }
  if (static_cast<std::size_t>(g.row_ptr.back()) != m) {
    return "row_ptr[n] does not match the number of column indices";
  }
  if (g.num_weights != impl.props.num_weights || g.num_weights < 0 ||
      g.weights.size() != m * static_cast<std::size_t>(g.num_weights)) {
    return "the weight columns do not match num_weights x num_edges";
  }
  for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
    for (auto e = static_cast<std::size_t>(g.row_ptr[u]);
         e < static_cast<std::size_t>(g.row_ptr[u + 1]); ++e) {
      const vertex_t v = g.col_ind[e];
      if (v < 0 || v >= n) {
        return "column index out of range in row " + std::to_string(u);
      }
      if (e > static_cast<std::size_t>(g.row_ptr[u])) {
        const vertex_t previous = g.col_ind[e - 1];
        if (impl.props.order == row_order::sorted && previous > v) {
          return "row " + std::to_string(u) + " is not sorted";
        }
        if (impl.props.parallel_edges == multi_edges::forbid &&
            impl.props.order == row_order::sorted && previous == v) {
          return "row " + std::to_string(u) + " has parallel edges";
        }
      }
    }
  }
  if (impl.props.parallel_edges == multi_edges::forbid && impl.props.order == row_order::append) {
    std::vector<vertex_t> row;
    for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
      row.assign(g.col_ind.begin() + static_cast<std::ptrdiff_t>(g.row_ptr[u]),
                 g.col_ind.begin() + static_cast<std::ptrdiff_t>(g.row_ptr[u + 1]));
      std::sort(row.begin(), row.end());
      if (std::adjacent_find(row.begin(), row.end()) != row.end()) {
        return "row " + std::to_string(u) + " has parallel edges";
      }
    }
  }
  if (impl.props.semantics.on_self_loop == batch_semantics::self_loop::drop ||
      impl.props.semantics.on_self_loop == batch_semantics::self_loop::error) {
    for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
      for (auto e = static_cast<std::size_t>(g.row_ptr[u]);
           e < static_cast<std::size_t>(g.row_ptr[u + 1]); ++e) {
        if (static_cast<std::size_t>(g.col_ind[e]) == u) {
          return "self-loop in row " + std::to_string(u) + " although self-loops are not kept";
        }
      }
    }
  }
  if (!impl.props.directed) {
    csr<vertex_t, edge_t, weight_t> reverse;
    transpose_host(g, reverse);
    std::vector<std::pair<vertex_t, vertex_t>> forward;
    std::vector<std::pair<vertex_t, vertex_t>> backward;
    for (std::size_t u = 0; u < static_cast<std::size_t>(n); ++u) {
      for (auto e = static_cast<std::size_t>(g.row_ptr[u]);
           e < static_cast<std::size_t>(g.row_ptr[u + 1]); ++e) {
        forward.emplace_back(static_cast<vertex_t>(u), g.col_ind[e]);
        backward.emplace_back(g.col_ind[e], static_cast<vertex_t>(u));
      }
    }
    std::sort(forward.begin(), forward.end());
    std::sort(backward.begin(), backward.end());
    if (forward != backward) {
      return "an undirected graph is not stored symmetric";
    }
  }
  if (impl.props.store_transposed && impl.has_in_edges()) {  // not built: nothing stored yet
    csr<vertex_t, edge_t, weight_t> reverse;
    transpose_host(g, reverse);
    const csr<vertex_t, edge_t, weight_t>& in = impl.in_edges(1);
    if (reverse.row_ptr != in.row_ptr || reverse.col_ind != in.col_ind ||
        reverse.weights != in.weights || reverse.num_weights != in.num_weights) {
      return "the stored in-edges are not the transpose of the out-edges";
    }
  }
  return {};
}

// ------------------------------------------------------------------------------------------------
// instantiation
// ------------------------------------------------------------------------------------------------

#define DYNG_INSTANTIATE_APPLY_HOST(V, E, W)                                                     \
  template apply_summary apply_batch_host<V, E, W>(                                              \
      const csr<V, E, W>&, const edge_batch_view<V, W>&, const graph_properties&, csr<V, E, W>&, \
      apply_delta<V>*, int);                                                                     \
  template void transpose_host<V, E, W>(const csr<V, E, W>&, csr<V, E, W>&, int);                \
  template void build_from_edges_host<V, E, W>(const edge_list_view<V, W>&,                      \
                                               const graph_properties&, csr<V, E, W>&);          \
  template void build_from_csr_host<V, E, W>(const csr_view<V, E, W>&, const graph_properties&,  \
                                             csr<V, E, W>&, int, csr<V, E, W>*);                 \
  template std::string integrity_violation<V, E, W>(const graph_impl<V, E, W>&);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_APPLY_HOST)
#undef DYNG_INSTANTIATE_APPLY_HOST

}  // namespace dyng::detail
