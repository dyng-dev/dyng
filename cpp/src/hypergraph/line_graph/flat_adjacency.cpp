// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/utils.cpp (buildFlatRows, flatHyperedgeAdjacency)
// Derived from ESCHER-GPU@abcf9b2:kernel/locator.cu (buildRowLocatorKernel: the out offsets)
/**
 * @file flat_adjacency.cpp
 * @brief The initial line graph in ESCHER-GPU's construct layout (flatHyperedgeAdjacency) and
 *        the locator arrays of a line graph on the host (ADR 0042).
 */
#include "hypergraph/line_graph/host_line_graph.hpp"
#include "util/region_exceptions.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <string>
#include <vector>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace dyng::detail::hyper {

namespace {

// Builds sorted, de-duplicated adjacency rows in the flattened CBST layout
// (same layout as flatten2DVector: each row padded to a multiple of 4 with
// zeros and an INT_MIN terminator). Rows are independent: each thread
// gathers the candidate neighbours of a row into a buffer, sorts and
// de-duplicates it and appends the result to a per-thread arena; the rows
// are then copied to their offsets. The gather allocates (the buffer, the
// arena): an exception of a row is kept and rethrown after the region (D13,
// ADR 0061); the copy region does not allocate.
template <class gather_t>
line_graph_layout build_flat_rows(std::int64_t num_rows, gather_t gather, int threads,
                                  std::int64_t limit) {
  const int num_threads = threads > 0 ? threads : 1;
  std::vector<std::vector<int>> arena(static_cast<std::size_t>(num_threads));
  std::vector<int> row_thread(static_cast<std::size_t>(num_rows));
  std::vector<std::int64_t> row_arena_pos(static_cast<std::size_t>(num_rows));
  line_graph_layout g;
  g.counts.assign(static_cast<std::size_t>(num_rows), 0);
  region_exceptions errors;
#if DYNG_HAS_OPENMP
#pragma omp parallel num_threads(num_threads) if (num_threads > 1)
#endif
  {
#if DYNG_HAS_OPENMP
    const int tid = omp_get_thread_num();
#else
    const int tid = 0;
#endif
    std::vector<int>& out = arena[static_cast<std::size_t>(tid)];
    std::vector<int> buffer;
#if DYNG_HAS_OPENMP
#pragma omp for schedule(dynamic, 256)
#endif
    for (std::int64_t row = 0; row < num_rows; ++row) {
      errors.run([&] {
        buffer.clear();
        gather(row, buffer);
        std::sort(buffer.begin(), buffer.end());
        buffer.erase(std::unique(buffer.begin(), buffer.end()), buffer.end());
        row_thread[static_cast<std::size_t>(row)] = tid;
        row_arena_pos[static_cast<std::size_t>(row)] = static_cast<std::int64_t>(out.size());
        g.counts[static_cast<std::size_t>(row)] = static_cast<std::int32_t>(buffer.size());
        out.insert(out.end(), buffer.begin(), buffer.end());
      });
    }
  }
  errors.rethrow();
  // The offsets in 64 bits; A ended the process at more than 2^31 - 1 entries, dynG throws
  // before it allocates the payload.
  g.starts.assign(static_cast<std::size_t>(num_rows), 0);
  std::int64_t total = 0;
  for (std::int64_t row = 0; row < num_rows; ++row) {
    g.starts[static_cast<std::size_t>(row)] =
        static_cast<std::int32_t>(std::min<std::int64_t>(total, INT_MAX));
    total += ((g.counts[static_cast<std::size_t>(row)] + 4) / 4) * 4;
  }
  if (total > limit) {
    throw capacity_error("dyng: hypergraph: the line graph (h2h) needs " + std::to_string(total) +
                         " payload entries, more than its capacity policy allows (" +
                         std::to_string(limit) + "; the store's offsets are int32)");
  }
  g.payload.assign(static_cast<std::size_t>(total), 0);
#if DYNG_HAS_OPENMP
#pragma omp parallel for num_threads(num_threads) if (num_threads > 1) schedule(dynamic, 1024)
#endif
  for (std::int64_t row = 0; row < num_rows; ++row) {
    const auto r = static_cast<std::size_t>(row);
    const int* src = arena[static_cast<std::size_t>(row_thread[r])].data() + row_arena_pos[r];
    std::copy(src, src + g.counts[r], g.payload.begin() + g.starts[r]);
    g.payload[static_cast<std::size_t>(g.starts[r]) +
              static_cast<std::size_t>(((g.counts[r] + 4) / 4) * 4) - 1] = INT_MIN;
  }
  return g;
}

}  // namespace

line_graph_layout flat_hyperedge_adjacency(const std::vector<std::int64_t>& h2v_ptr,
                                           const std::vector<std::int32_t>& h2v_cols,
                                           std::int32_t num_vertices, std::int64_t num_keys,
                                           int threads, std::int64_t limit) {
  const std::int64_t num_rows = h2v_ptr.empty() ? 0 : static_cast<std::int64_t>(h2v_ptr.size()) - 1;
  DYNG_EXPECTS(num_keys >= num_rows, "line graph: ", num_keys, " keys for ", num_rows, " rows");
  DYNG_EXPECTS(num_keys <= INT_MAX, "line graph: ", num_keys, " keys exceed the int32 keys");
  // vertex2hyperedge: the keys of every vertex, ascending (the order does not matter: every row
  // is sorted and deduplicated).
  std::vector<std::int64_t> v_ptr(static_cast<std::size_t>(num_vertices) + 1, 0);
  for (const std::int32_t v : h2v_cols) ++v_ptr[static_cast<std::size_t>(v) + 1];
  for (std::size_t v = 0; v < static_cast<std::size_t>(num_vertices); ++v) v_ptr[v + 1] += v_ptr[v];
  std::vector<int> v_keys(h2v_cols.size());
  {
    std::vector<std::int64_t> fill(v_ptr.begin(), v_ptr.end() - 1);
    for (std::int64_t id = 0; id < num_rows; ++id) {
      for (std::int64_t k = h2v_ptr[static_cast<std::size_t>(id)];
           k < h2v_ptr[static_cast<std::size_t>(id) + 1]; ++k) {
        v_keys[static_cast<std::size_t>(
            fill[static_cast<std::size_t>(h2v_cols[static_cast<std::size_t>(k)])]++)] =
            static_cast<int>(id + 1);
      }
    }
  }
  return build_flat_rows(
      num_keys,
      [&](std::int64_t hyperedge, std::vector<int>& out) {
        if (hyperedge >= num_rows) return;
        const int self = static_cast<int>(hyperedge) + 1;
        for (std::int64_t k = h2v_ptr[static_cast<std::size_t>(hyperedge)];
             k < h2v_ptr[static_cast<std::size_t>(hyperedge) + 1]; ++k) {
          const auto vertex = static_cast<std::size_t>(h2v_cols[static_cast<std::size_t>(k)]);
          for (std::int64_t p = v_ptr[vertex]; p < v_ptr[vertex + 1]; ++p) {
            const int other = v_keys[static_cast<std::size_t>(p)];
            if (other != self) out.push_back(other);
          }
        }
      },
      threads, limit);
}

void line_graph_locator(const line_graph_layout& g, std::vector<int>& loc, std::vector<int>& len,
                        std::vector<int>& out) {
  const std::int64_t n = g.num_keys();
  loc.assign(static_cast<std::size_t>(n) + 1, 0);
  len.assign(static_cast<std::size_t>(n) + 1, 0);
  out.assign(static_cast<std::size_t>(n) + 1, 0);
  for (std::int64_t key = 1; key <= n; ++key) {
    const auto k = static_cast<std::size_t>(key);
    const int base = g.starts[k - 1];
    const int count = g.counts[k - 1];
    loc[k] = base;
    len[k] = count;
    // first position whose entry is greater than key (rows are sorted)
    const auto first = g.payload.begin() + base;
    out[k] = base + static_cast<int>(std::upper_bound(first, first + count, static_cast<int>(key)) -
                                     first);
  }
}

}  // namespace dyng::detail::hyper
