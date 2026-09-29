// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from CycleEnumeration-GPU@0a976ad:src/dynamic/directed_graph.cpp (apply_batch,
// prepare_batch, has_edge) and src/dynamic/edge_change.cpp (sort_and_dedup, normalize)
/**
 * @file apply_set_host.cpp
 * @brief Host batch application under batch_semantics::as_sets (batch_semantics::set()).
 *
 * Step 0 is CycleEnumeration-GPU's prepare_batch(): both lists sorted by (source, target) and
 * deduplicated (sort_and_dedup), self-loops dropped, deletions of absent edges dropped, insertions
 * of present edges dropped unless the same edge is also deleted (a delete-then-reinsert pair stays
 * in both lists). The application is its apply_batch(): every touched row is the merge of its
 * surviving entries with its inserted targets, both sorted; untouched rows are copied. The
 * mechanical changes are: names, templates on the index types, weight columns (an entry keeps the
 * weights of its original edge, an inserted one takes the weights of the first insertion of the
 * pair in batch order), the batch_semantics switches around the unchanged core (errors instead of
 * silent drops where the graph asks for them, undirected expansion, no vertex growth), and the
 * assembly of the new CSR: the row offsets are the old ones shifted by the degree changes of the
 * touched rows before them, and the rows are written in parallel blocks (the bytes are the same
 * for every thread count, and equal to the original's sequential push_back loop).
 */
#include "graph/apply_common.hpp"
#include "graph/apply_host.hpp"
#include "graph/device_graph.hpp"
#include "graph/instantiate.hpp"
#include "graph/normalized_batch.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace dyng::detail {

namespace {

/// One requested change (normalized_batch.hpp).
template <typename vertex_t>
using change = set_change<vertex_t>;

/// Lexicographic order by source, then target (EdgeChange's operator<).
template <typename vertex_t>
bool edge_less(const change<vertex_t>& a, const change<vertex_t>& b) noexcept {
  return a.source != b.source ? a.source < b.source : a.target < b.target;
}

/// Order by source, target, then position in the batch: a total order (the positions are unique),
/// so an unstable sort gives what a stable sort by (source, target) gives.
template <typename vertex_t>
bool edge_index_less(const change<vertex_t>& a, const change<vertex_t>& b) noexcept {
  if (a.source != b.source) {
    return a.source < b.source;
  }
  return a.target != b.target ? a.target < b.target : a.index < b.index;
}

/// Number of bits of a non-negative value (0 for 0).
inline int bit_width(std::uint64_t value) noexcept {
  int bits = 0;
  while (value != 0) {
    ++bits;
    value >>= 1;
  }
  return bits;
}

/// Lists shorter than this are sorted with std::sort (the radix sort's counters cost more).
constexpr std::size_t radix_sort_threshold = 256;
/// Digit width of the radix sort (2048 counters per pass).
constexpr int radix_digit_bits = 11;

/// Sort a requested list by (source, target, position in the batch).
///
/// Precondition: the list is in batch order (positions non-decreasing), as normalize_set_batch()
/// builds it. A least-significant-digit radix sort on the key (source, target) is stable, so equal
/// pairs stay in batch order: the result is the one of sorting by (source, target, position), a
/// total order, whatever the order of the input. Its cost does not depend on the order of the input
/// (the original's std::sort of 8-byte EdgeChange records is fast on sorted input and slow on
/// shuffled input; std::sort of these 16-byte records with a three-key comparator is slower than
/// the original's on both). Short lists, and keys wider than 64 bits (64-bit ids), use std::sort
/// with the same total order.
/// @param[in,out] changes  The list.
/// @param[in,out] scratch  A second list of the same capacity class (swapped with `changes`).
/// @param[in,out] counters The digit counters (reused).
template <typename vertex_t>
void sort_changes(std::vector<change<vertex_t>>& changes, std::vector<change<vertex_t>>& scratch,
                  std::vector<std::uint32_t>& counters) {
  const std::size_t size = changes.size();
  if (size < radix_sort_threshold ||
      size > static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max())) {
    std::sort(changes.begin(), changes.end(), edge_index_less<vertex_t>);
    return;
  }
#ifndef NDEBUG
  for (std::size_t i = 1; i < size; ++i) {
    if (changes[i].index < changes[i - 1].index) {
      DYNG_FAIL("apply (set semantics): a requested list is not in batch order");
    }
  }
#endif
  std::uint64_t max_source = 0;
  std::uint64_t max_target = 0;
  for (const change<vertex_t>& c : changes) {  // ids are validated non-negative
    max_source = std::max(max_source, static_cast<std::uint64_t>(c.source));
    max_target = std::max(max_target, static_cast<std::uint64_t>(c.target));
  }
  const int target_bits = bit_width(max_target);
  const int key_bits = bit_width(max_source) + target_bits;
  if (key_bits > 64) {
    std::sort(changes.begin(), changes.end(), edge_index_less<vertex_t>);
    return;
  }
  const auto key = [target_bits](const change<vertex_t>& c) noexcept {
    return target_bits == 64 ? static_cast<std::uint64_t>(c.target)
                             : (static_cast<std::uint64_t>(c.source) << target_bits) |
                                   static_cast<std::uint64_t>(c.target);
  };
  constexpr std::size_t radix = std::size_t{1} << radix_digit_bits;
  constexpr std::uint64_t digit_mask = radix - 1;
  const int passes = (key_bits + radix_digit_bits - 1) / radix_digit_bits;
  counters.assign(static_cast<std::size_t>(passes) * radix, 0);
  for (const change<vertex_t>& c : changes) {
    const std::uint64_t k = key(c);
    for (int p = 0; p < passes; ++p) {
      ++counters[static_cast<std::size_t>(p) * radix +
                 ((k >> (p * radix_digit_bits)) & digit_mask)];
    }
  }
  scratch.resize(size);
  for (int p = 0; p < passes; ++p) {
    std::uint32_t* count = counters.data() + static_cast<std::size_t>(p) * radix;
    const int shift = p * radix_digit_bits;
    if (count[(key(changes[0]) >> shift) & digit_mask] == size) {
      continue;  // every key has the same digit here: the pass would not move anything
    }
    std::uint32_t sum = 0;
    for (std::size_t d = 0; d < radix; ++d) {
      const std::uint32_t c = count[d];
      count[d] = sum;
      sum += c;
    }
    const change<vertex_t>* from = changes.data();
    change<vertex_t>* to = scratch.data();
    for (std::size_t i = 0; i < size; ++i) {
      to[count[(key(from[i]) >> shift) & digit_mask]++] = from[i];
    }
    changes.swap(scratch);  // O(1); both keep their capacity for the next update
  }
}

/// sort_and_dedup(): sorted by (source, target); of equal pairs the first in batch order stays
/// (its position supplies the weights of an insertion and names it in error messages).
template <typename vertex_t>
void sort_and_dedup(std::vector<change<vertex_t>>& changes, std::vector<change<vertex_t>>& scratch,
                    std::vector<std::uint32_t>& counters) {
  sort_changes(changes, scratch, counters);
  changes.erase(std::unique(changes.begin(), changes.end(),
                            [](const change<vertex_t>& a, const change<vertex_t>& b) {
                              return a.source == b.source && a.target == b.target;
                            }),
                changes.end());
}

/// has_edge(): binary search in the sorted row of `source` (false for a vertex >= n).
template <typename vertex_t, typename edge_t, typename weight_t>
bool has_edge(const csr<vertex_t, edge_t, weight_t>& g, std::int64_t n, vertex_t source,
              vertex_t target) {
  if (static_cast<std::int64_t>(source) >= n) {
    return false;
  }
  const auto begin = g.col_ind.begin() + static_cast<std::ptrdiff_t>(g.row_ptr[source]);
  const auto end = g.col_ind.begin() + static_cast<std::ptrdiff_t>(g.row_ptr[source + 1]);
  return std::binary_search(begin, end, target);
}

/// Step 0 on a graph of n vertices and m edges, whatever holds it: `present` fills
/// out.present[i] (1 if the i-th requested change, deletions first, then insertions, names an
/// edge of G_t) for the sorted, duplicate-free requested lists; the host CSR answers with has_edge,
/// a resident device copy with one kernel (graph_impl keeps the host copy stale then).
template <typename edge_t, typename vertex_t, typename weight_t, typename present_t>
void normalize_set_batch_with(std::int64_t n, std::size_t m, int num_weights,
                              const edge_batch_view<vertex_t, weight_t>& batch,
                              const graph_properties& props, normalized_batch<vertex_t>& out,
                              present_t&& present) {
  const batch_semantics& semantics = props.semantics;
  expect_supported_semantics(props);  // checked at construction as well
  validate_batch_shape(batch, num_weights);
  const std::size_t num_inserts = batch.insert_src.size();
  const std::size_t num_deletes = batch.delete_src.size();
  apply_summary summary;
  out.device_current = false;
  out.deletions_marked = false;

  // --- Step 0: the requested operations (ids, self-loops, vertex growth, undirected expansion) --
  // Both lists are built in batch order (sort_changes relies on it).
  std::vector<change<vertex_t>>& insertions = out.requested_insertions;
  std::vector<change<vertex_t>>& deletions = out.requested_deletions;
  insertions.clear();
  deletions.clear();
  insertions.reserve(props.directed ? num_inserts : 2 * num_inserts);
  deletions.reserve(props.directed ? num_deletes : 2 * num_deletes);
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
    insertions.push_back({u, v, i});
    if (!props.directed && u != v) {
      insertions.push_back({v, u, i});
    }
  }
  for (std::size_t j = 0; j < num_deletes; ++j) {
    const vertex_t u = batch.delete_src[j];
    const vertex_t v = batch.delete_dst[j];
    DYNG_EXPECTS(u >= 0 && v >= 0, "deletion ", j, " (", u, ", ", v, ") has a negative id");
    if (u == v &&
        skip_self_loop(semantics.on_self_loop, "deletion", static_cast<std::int64_t>(j), u)) {
      ++summary.dropped_self_loops;
      continue;
    }
    deletions.push_back({u, v, j});
    if (!props.directed && u != v) {
      deletions.push_back({v, u, j});
    }
  }

  // --- prepare_batch(): sorted, duplicate-free lists without no-ops ------------------------------
  sort_and_dedup(deletions, out.sort_scratch, out.sort_counters);
  sort_and_dedup(insertions, out.sort_scratch, out.sort_counters);
  out.present.resize(deletions.size() + insertions.size());
  present(deletions, insertions, out.present);
  const std::uint8_t* deletion_present = out.present.data();
  const std::uint8_t* insertion_present = out.present.data() + deletions.size();
  std::vector<change<vertex_t>>& del_kept = out.deletions;
  del_kept.clear();
  del_kept.reserve(deletions.size());
  for (std::size_t d = 0; d < deletions.size(); ++d) {
    if (deletion_present[d] != 0) {
      del_kept.push_back(deletions[d]);
      continue;
    }
    DYNG_EXPECTS(semantics.on_missing_delete == batch_semantics::missing_delete::ignore,
                 "deletion ", deletions[d].index, " (", deletions[d].source, ", ",
                 deletions[d].target,
                 ") names a missing edge and batch_semantics::on_missing_delete is error");
    ++summary.ignored_deletions;
  }
  std::vector<change<vertex_t>>& ins_kept = out.insertions;
  ins_kept.clear();
  ins_kept.reserve(insertions.size());
  for (std::size_t i = 0; i < insertions.size(); ++i) {
    const change<vertex_t>& c = insertions[i];
    if (insertion_present[i] != 0) {
      if (!std::binary_search(del_kept.begin(), del_kept.end(), c, edge_less<vertex_t>)) {
        if (semantics.on_existing_insert == batch_semantics::existing_insert::error) {
          DYNG_THROW_INVALID_ARGUMENT("insertion ", c.index, " (", c.source, ", ", c.target,
                                      ") names an existing edge and "
                                      "batch_semantics::on_existing_insert is error");
        }
        ++summary.ignored_insertions;
        continue;
      }
      ++summary.cancelled_pairs;  // deleted and re-inserted: removed, then added back
    }
    ins_kept.push_back(c);
  }
  summary.deleted_edges = static_cast<std::int64_t>(del_kept.size());
  summary.inserted_edges = static_cast<std::int64_t>(ins_kept.size());
  summary.inserted_vertices = n_after - n;
  summary.num_vertices_after = n_after;
  out.summary = summary;
  out.vertices_before = n;
  out.vertices_after = n_after;
  out.edges_before = static_cast<std::int64_t>(m);
  out.edges_after = static_cast<std::int64_t>(m) + static_cast<std::int64_t>(ins_kept.size()) -
                    static_cast<std::int64_t>(del_kept.size());
  (void)checked_edge_count<edge_t>(out.edges_after);
}

}  // namespace

template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_set_batch(const csr<vertex_t, edge_t, weight_t>& original,
                         const edge_batch_view<vertex_t, weight_t>& batch,
                         const graph_properties& props, normalized_batch<vertex_t>& out) {
  const std::int64_t n = original.num_vertices();
  normalize_set_batch_with<edge_t>(
      n, original.col_ind.size(), original.num_weights, batch, props, out,
      [&](const std::vector<change<vertex_t>>& deletions,
          const std::vector<change<vertex_t>>& insertions, std::vector<std::uint8_t>& present) {
        std::size_t k = 0;
        for (const change<vertex_t>& d : deletions) {
          present[k++] = has_edge(original, n, d.source, d.target) ? 1 : 0;
        }
        for (const change<vertex_t>& c : insertions) {
          present[k++] = has_edge(original, n, c.source, c.target) ? 1 : 0;
        }
      });
}

template <typename vertex_t, typename edge_t, typename weight_t>
void normalize_set_batch_device(const resources& res,
                                const device_graph<vertex_t, edge_t, weight_t>& base,
                                const edge_batch_view<vertex_t, weight_t>& batch,
                                const graph_properties& props, normalized_batch<vertex_t>& out) {
  normalize_set_batch_with<edge_t>(
      static_cast<std::int64_t>(base.num_vertices), static_cast<std::size_t>(base.num_edges),
      base.num_weights, batch, props, out,
      [&](const std::vector<change<vertex_t>>& deletions,
          const std::vector<change<vertex_t>>& insertions, std::vector<std::uint8_t>& present) {
#if DYNG_HAS_CUDA
        device_edge_membership(res, base, deletions, insertions, out, present);
#else
        (void)res;
        (void)deletions;
        (void)insertions;
        (void)present;
        throw not_supported_error("dyng: the cuda backend is not built");
#endif
      });
}

template <typename vertex_t, typename edge_t, typename weight_t>
apply_summary apply_set_batch_host(const csr<vertex_t, edge_t, weight_t>& original,
                                   const edge_batch_view<vertex_t, weight_t>& batch,
                                   const graph_properties& props,
                                   csr<vertex_t, edge_t, weight_t>& updated,
                                   apply_delta<vertex_t>* delta, int threads,
                                   const normalized_batch<vertex_t>* normalized) {
  normalized_batch<vertex_t> local;
  if (normalized == nullptr) {
    normalize_set_batch(original, batch, props, local);
    normalized = &local;
  } else {
    DYNG_EXPECTS(normalized->vertices_before == original.num_vertices() &&
                     normalized->edges_before == static_cast<std::int64_t>(original.col_ind.size()),
                 "apply (set semantics): the normalized batch belongs to another graph state");
  }
  const int num_objectives = original.num_weights;
  const auto k_count = static_cast<std::size_t>(num_objectives);
  const std::int64_t n = original.num_vertices();
  const std::size_t m = original.col_ind.size();
  const std::int64_t n_after = normalized->vertices_after;
  const std::vector<change<vertex_t>>& del_kept = normalized->deletions;
  const std::vector<change<vertex_t>>& ins_kept = normalized->insertions;
  const apply_summary summary = normalized->summary;

  // --- apply_batch(): the touched rows and their change ranges -----------------------------------
  // Both lists are sorted by source, so the changes of a row are contiguous in each list.
  std::vector<vertex_t> touched;
  std::vector<std::size_t> del_first;  // per touched row: first deletion, first insertion
  std::vector<std::size_t> ins_first;
  {
    std::size_t d = 0;
    std::size_t i = 0;
    while (d < del_kept.size() || i < ins_kept.size()) {
      const vertex_t u = i >= ins_kept.size()   ? del_kept[d].source
                         : d >= del_kept.size() ? ins_kept[i].source
                                                : std::min(del_kept[d].source, ins_kept[i].source);
      touched.push_back(u);
      del_first.push_back(d);
      ins_first.push_back(i);
      while (d < del_kept.size() && del_kept[d].source == u) {
        ++d;
      }
      while (i < ins_kept.size() && ins_kept[i].source == u) {
        ++i;
      }
    }
    del_first.push_back(d);
    ins_first.push_back(i);
  }
  const std::size_t num_touched = touched.size();
  // shift[j]: the change of the edge count of the touched rows before touched row j.
  std::vector<std::int64_t> shift(num_touched + 1, 0);
  for (std::size_t j = 0; j < num_touched; ++j) {
    shift[j + 1] = shift[j] + static_cast<std::int64_t>(ins_first[j + 1] - ins_first[j]) -
                   static_cast<std::int64_t>(del_first[j + 1] - del_first[j]);
  }
  const std::int64_t total = static_cast<std::int64_t>(m) + shift[num_touched];

  // --- Assemble the updated CSR ------------------------------------------------------------------
  updated = csr<vertex_t, edge_t, weight_t>();
  updated.num_weights = num_objectives;
  const auto rows = static_cast<std::size_t>(n_after);
  const auto m_new = static_cast<std::size_t>(checked_edge_count<edge_t>(total));
  updated.row_ptr.resize(rows + 1);
  updated.col_ind.resize(m_new);
  updated.weights.resize(m_new * k_count);
  const edge_t zero_offset = 0;
  const edge_t* old_row = original.row_ptr.empty() ? &zero_offset : original.row_ptr.data();
  const vertex_t* old_col = original.col_ind.data();
  const weight_t* old_weights = original.weights.data();
  edge_t* new_row = updated.row_ptr.data();
  vertex_t* new_col = updated.col_ind.data();
  weight_t* new_weights = updated.weights.data();
  const vertex_t* touched_rows = touched.data();
  const auto old_offset = [&](std::int64_t u) {
    return static_cast<std::int64_t>(old_row[std::min(u, n)]);
  };
  const auto first_touched_at_or_after = [&](std::int64_t u) {
    return static_cast<std::size_t>(std::lower_bound(touched_rows, touched_rows + num_touched, u,
                                                     [](vertex_t row, std::int64_t value) {
                                                       return static_cast<std::int64_t>(row) <
                                                              value;
                                                     }) -
                                    touched_rows);
  };
  const auto row_count = static_cast<std::int64_t>(rows);
  const bool parallel = threads > 1 && row_count >= 65536;
  (void)parallel;
  constexpr std::int64_t block = 4096;

  // Row offsets: the old offset shifted by the changes of the touched rows before the row.
  const std::int64_t offset_blocks = (row_count + 1 + block - 1) / block;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(static)
#endif
  for (std::int64_t b = 0; b < offset_blocks; ++b) {
    const std::int64_t first = b * block;
    const std::int64_t stop = std::min(row_count + 1, first + block);
    std::size_t j = first_touched_at_or_after(first);
    for (std::int64_t u = first; u < stop; ++u) {
      while (j < num_touched && static_cast<std::int64_t>(touched_rows[j]) < u) {
        ++j;
      }
      new_row[u] = static_cast<edge_t>(old_offset(u) + shift[j]);
    }
  }

  // Rows: runs of untouched rows are copied at once; a touched row is apply_batch()'s merge.
  std::atomic<bool> degree_mismatch{false};
  const std::int64_t blocks = (row_count + block - 1) / block;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(threads) if (parallel) schedule(dynamic, 16)
#endif
  for (std::int64_t b = 0; b < blocks; ++b) {
    const std::int64_t first = b * block;
    const std::int64_t stop = std::min(row_count, first + block);
    std::size_t j = first_touched_at_or_after(first);
    for (std::int64_t u = first; u < stop; ++u) {
      if (j < num_touched && static_cast<std::int64_t>(touched_rows[j]) == u) {
        // Merge the kept old entries with the insertions, both sorted.
        auto a = static_cast<std::size_t>(old_offset(u));
        const auto old_end = static_cast<std::size_t>(old_offset(u + 1));
        std::size_t del = del_first[j];
        const std::size_t d_end = del_first[j + 1];
        std::size_t ins = ins_first[j];
        const std::size_t n_end = ins_first[j + 1];
        const auto row_begin = static_cast<std::size_t>(new_row[u]);
        std::size_t out = row_begin;
        const auto push = [&](vertex_t target, std::size_t old_edge, std::size_t insertion) {
          if (out == row_begin || new_col[out - 1] != target) {
            new_col[out] = target;
            for (std::size_t k = 0; k < k_count; ++k) {
              new_weights[k * m_new + out] = insertion == static_cast<std::size_t>(-1)
                                                 ? old_weights[k * m + old_edge]
                                                 : batch.insert_weights[insertion * k_count + k];
            }
            ++out;
          }
        };
        while (a < old_end || ins < n_end) {
          if (a < old_end) {
            const vertex_t target = old_col[a];
            while (del < d_end && del_kept[del].target < target) {
              ++del;
            }
            if (del < d_end && del_kept[del].target == target) {
              ++a;
              continue;  // deleted (unless re-inserted below)
            }
            if (ins >= n_end || target <= ins_kept[ins].target) {
              push(target, a, static_cast<std::size_t>(-1));
              ++a;
              continue;
            }
          }
          push(ins_kept[ins].target, 0, ins_kept[ins].index);
          ++ins;
        }
        if (out != static_cast<std::size_t>(new_row[u + 1])) {
          degree_mismatch.store(true, std::memory_order_relaxed);
        }
        ++j;
        continue;
      }
      if (u >= n) {
        continue;  // a new vertex without insertions: an empty row
      }
      std::int64_t last = u + 1;
      const std::int64_t next_touched =
          j < num_touched ? static_cast<std::int64_t>(touched_rows[j]) : stop;
      last = std::min({stop, n, next_touched});
      const auto begin = static_cast<std::size_t>(old_offset(u));
      const auto end = static_cast<std::size_t>(old_offset(last));
      const auto out = static_cast<std::size_t>(new_row[u]);
      std::copy(old_col + begin, old_col + end, new_col + out);
      for (std::size_t k = 0; k < k_count; ++k) {
        std::copy(old_weights + k * m + begin, old_weights + k * m + end,
                  new_weights + k * m_new + out);
      }
      u = last - 1;
    }
  }
  if (degree_mismatch.load()) {
    DYNG_FAIL("apply (set semantics): a merged row does not have the computed degree");
  }

  if (delta != nullptr) {
    fill_set_delta(*normalized, *delta, num_objectives);
  }
  return summary;
}

#define DYNG_INSTANTIATE_APPLY_SET_HOST(V, E, W)                                                 \
  template void normalize_set_batch<V, E, W>(const csr<V, E, W>&, const edge_batch_view<V, W>&,  \
                                             const graph_properties&, normalized_batch<V>&);     \
  template apply_summary apply_set_batch_host<V, E, W>(                                          \
      const csr<V, E, W>&, const edge_batch_view<V, W>&, const graph_properties&, csr<V, E, W>&, \
      apply_delta<V>*, int, const normalized_batch<V>*);
DYNG_FOR_EACH_GRAPH_TYPE(DYNG_INSTANTIATE_APPLY_SET_HOST)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_APPLY_SET_HOST)
#undef DYNG_INSTANTIATE_APPLY_SET_HOST

// The device Step 0 reads 32-bit vertex ids (device_set_apply_supported_v), as the device apply.
#define DYNG_INSTANTIATE_NORMALIZE_SET_DEVICE(V, E, W)                              \
  template void normalize_set_batch_device<V, E, W>(                                \
      const resources&, const device_graph<V, E, W>&, const edge_batch_view<V, W>&, \
      const graph_properties&, normalized_batch<V>&);
DYNG_INSTANTIATE_NORMALIZE_SET_DEVICE(std::int32_t, std::int32_t, std::int32_t)
DYNG_INSTANTIATE_NORMALIZE_SET_DEVICE(std::int32_t, std::int64_t, std::int32_t)
DYNG_FOR_EACH_UNWEIGHTED_GRAPH_TYPE(DYNG_INSTANTIATE_NORMALIZE_SET_DEVICE)
#undef DYNG_INSTANTIATE_NORMALIZE_SET_DEVICE

}  // namespace dyng::detail
