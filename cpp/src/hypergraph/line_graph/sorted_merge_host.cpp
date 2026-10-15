// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/graph_maintenance.cu (updateLineGraph, mergeAndRebuild:
// the algorithm)
// Derived from ESCHER-GPU@abcf9b2:kernel/delta_merge.cu (mergedRowSlots, writeMergedRows: the
// layout)
/**
 * @file sorted_merge_host.cpp
 * @brief The host form of ESCHER-GPU's sorted delta merge of the line graph (ADR 0042): the same
 *        delta (the rows of the inserted keys, the removals of the deleted keys, the mirrored
 *        additions) merged row by row into a new payload of the same layout.
 */
#include "hypergraph/line_graph/host_line_graph.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace dyng::detail::hyper {

namespace {

// Pairs are packed as (row key << 32) | value, so a sorted pair array groups the entries of each
// row in ascending value order (delta_merge.cu).
std::uint64_t pack_pair(std::int64_t key, std::int64_t value) {
  return (static_cast<std::uint64_t>(key) << 32) | static_cast<std::uint32_t>(value);
}
int pair_value(std::uint64_t p) {
  return static_cast<int>(p & 0xffffffffULL);
}
int pair_key(std::uint64_t p) {
  return static_cast<int>(p >> 32);
}

// Per key 0..n: the first index of its pairs in a sorted pair array (offsets[n + 1] = size).
std::vector<std::int64_t> pair_ranges(const std::vector<std::uint64_t>& pairs, std::int64_t n) {
  std::vector<std::int64_t> offsets(static_cast<std::size_t>(n) + 2, 0);
  for (const std::uint64_t p : pairs) ++offsets[static_cast<std::size_t>(pair_key(p)) + 1];
  for (std::size_t k = 0; k + 1 < offsets.size(); ++k) offsets[k + 1] += offsets[k];
  return offsets;
}

}  // namespace

void merge_line_graph(line_graph_layout& g, std::int64_t num_keys,
                      const std::vector<std::int64_t>& deleted,
                      const std::vector<std::int64_t>& inserted, const member_rows& inserted_rows,
                      const std::vector<std::vector<std::int64_t>>& v2h, int threads,
                      std::int64_t limit) {
  const std::int64_t old_keys = g.num_keys();
  DYNG_EXPECTS(num_keys >= old_keys && num_keys <= INT_MAX, "line graph: ", num_keys,
               " keys after a batch, ", old_keys, " before");
  const int num_threads = threads > 0 ? threads : 1;
  (void)num_threads;
  // The frontiers (A's DeviceFrontier): a flag per key 0..num_keys.
  std::vector<unsigned char> is_deleted(static_cast<std::size_t>(num_keys) + 1, 0);
  std::vector<unsigned char> is_inserted(static_cast<std::size_t>(num_keys) + 1, 0);
  for (const std::int64_t id : deleted) is_deleted[static_cast<std::size_t>(id) + 1] = 1;
  for (const std::int64_t id : inserted) is_inserted[static_cast<std::size_t>(id) + 1] = 1;

  // Rows of the inserted hyperedges: the union of their vertices' v2h rows (sorted, unique).
  std::vector<std::uint64_t> rows;
  for (std::size_t i = 0; i < inserted.size(); ++i) {
    const std::int64_t e = inserted[i] + 1;
    for (std::int64_t k = inserted_rows.offsets[i]; k < inserted_rows.offsets[i + 1]; ++k) {
      for (const std::int64_t id :
           v2h[static_cast<std::size_t>(inserted_rows.members[static_cast<std::size_t>(k)])]) {
        if (id + 1 != e) rows.push_back(pack_pair(e, id + 1));
      }
    }
  }
  std::sort(rows.begin(), rows.end());
  rows.erase(std::unique(rows.begin(), rows.end()), rows.end());

  // Deleted hyperedges leave the rows of their surviving neighbours: (u, d) for every u of the
  // old row of d that is not deleted.
  std::vector<std::uint64_t> removals;
  for (const std::int64_t id : deleted) {
    const std::int64_t d = id + 1;
    if (d > old_keys) continue;
    const auto k = static_cast<std::size_t>(d) - 1;
    for (int p = g.starts[k]; p < g.starts[k] + g.counts[k]; ++p) {
      const int u = g.payload[static_cast<std::size_t>(p)];
      if (!is_deleted[static_cast<std::size_t>(u)]) removals.push_back(pack_pair(u, d));
    }
  }
  std::sort(removals.begin(), removals.end());

  // Inserted hyperedges join the rows of their non-inserted neighbours: (h, e) for every entry
  // (e, h) of a new row with h not inserted.
  std::vector<std::uint64_t> additions;
#if !defined(DYNG_MUTATION_HYPERGRAPH_MIRROR_ADDITIONS_DROPPED)
  for (const std::uint64_t p : rows) {
    const int h = pair_value(p);
    if (!is_inserted[static_cast<std::size_t>(h)]) additions.push_back(pack_pair(h, pair_key(p)));
  }
#endif
  // (MUTATION, PLAN 8.4, M8b criterion 20: DYNG_MUTATION_HYPERGRAPH_MIRROR_ADDITIONS_DROPPED drops
  // the mirrored additions: the neighbours of an inserted hyperedge do not gain it.)
  std::sort(additions.begin(), additions.end());

  const std::vector<std::int64_t> row_off = pair_ranges(rows, num_keys);
  const std::vector<std::int64_t> rem_off = pair_ranges(removals, num_keys);
  const std::vector<std::int64_t> add_off = pair_ranges(additions, num_keys);

  // mergedRowSlots: the length of every merged row and its padded slots; the 64-bit scan.
  line_graph_layout next;
  next.counts.assign(static_cast<std::size_t>(num_keys), 0);
  next.starts.assign(static_cast<std::size_t>(num_keys), 0);
  std::int64_t total = 0;
  for (std::int64_t u = 1; u <= num_keys; ++u) {
    const auto k = static_cast<std::size_t>(u);
    std::int64_t n = 0;
    if (is_deleted[k] || is_inserted[k]) {
      n = row_off[k + 1] - row_off[k];
    } else {
      n = (u <= old_keys ? g.counts[k - 1] : 0) - (rem_off[k + 1] - rem_off[k]) +
          (add_off[k + 1] - add_off[k]);
    }
    next.counts[k - 1] = static_cast<std::int32_t>(n);
    next.starts[k - 1] = static_cast<std::int32_t>(std::min<std::int64_t>(total, INT_MAX));
    total += ((n + 4) / 4) * 4;
  }
  if (total > limit) {
    throw capacity_error(
        "dyng: hypergraph: the line graph (h2h) needs " + std::to_string(total) +
        " payload entries after the batch, more than its capacity policy allows (" +
        std::to_string(limit) + "; the store's offsets are int32)");
  }
  next.payload.assign(static_cast<std::size_t>(total), 0);

  // writeMergedRows: replaced rows take their new row, the others their old entries minus the
  // removals plus the additions, merged in order; then the terminator. A row whose delta breaks
  // the merge's requirements (removals are entries of the old row, additions are not) would not
  // fill its slots exactly: it is counted and reported, and nothing is written outside the row.
  std::int64_t broken = 0;
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(num_threads) if (num_threads > 1 && num_keys > 4096) \
    schedule(dynamic, 1024) reduction(+ : broken)
#endif
  for (std::int64_t u = 1; u <= num_keys; ++u) {
    const auto k = static_cast<std::size_t>(u);
    int* row = next.payload.data() + next.starts[k - 1];
    const int n = next.counts[k - 1];
    if (is_deleted[k] || is_inserted[k]) {
      for (std::int64_t i = row_off[k]; i < row_off[k + 1]; ++i)
        row[i - row_off[k]] = pair_value(rows[static_cast<std::size_t>(i)]);
    } else if (u <= old_keys) {
      const int* src = g.payload.data() + g.starts[k - 1];
      const int old_len = g.counts[k - 1];
      std::int64_t r = rem_off[k], a = add_off[k];
      int w = 0;
      const auto put = [&](int value) {
        if (w < n) row[w] = value;
        ++w;
      };
      for (int i = 0; i < old_len; ++i) {
        const int x = src[i];
        while (a < add_off[k + 1] && pair_value(additions[static_cast<std::size_t>(a)]) < x)
          put(pair_value(additions[static_cast<std::size_t>(a++)]));
        while (r < rem_off[k + 1] && pair_value(removals[static_cast<std::size_t>(r)]) < x) ++r;
        if (r < rem_off[k + 1] && pair_value(removals[static_cast<std::size_t>(r)]) == x) {
          ++r;  // removed
          continue;
        }
        put(x);
      }
      while (a < add_off[k + 1]) put(pair_value(additions[static_cast<std::size_t>(a++)]));
      if (w != n) ++broken;
    } else {
      for (std::int64_t i = add_off[k]; i < add_off[k + 1]; ++i)
        row[i - add_off[k]] = pair_value(additions[static_cast<std::size_t>(i)]);
    }
    const int padded = ((n + 4) / 4) * 4;
    row[padded - 1] = INT_MIN;
  }
  if (broken != 0) {
    DYNG_FAIL("line graph merge: ", broken,
              " rows have a delta that is not a subset of the old row or overlaps it (the h2v and "
              "v2h stores disagree with the line graph)");
  }
  g = std::move(next);
}

}  // namespace dyng::detail::hyper
