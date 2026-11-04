// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu (buildCsr, applyDeltaPairs,
// expandDeltaKeysKernel, applyRowsWarp, scatterWeightsKernel)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (lineGraph)
/**
 * @file slack_line_graph.cpp
 * @brief The slack CSR line graph of slack_csr storage on the host (ADR 0086).
 */
#include "hypergraph/slack_csr/slack_line_graph.hpp"

#include "hypergraph/line_graph/host_sort.hpp"
#include "operators/execution.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail::hyper {

namespace {

namespace ops = dyng::detail::operators;

// The neighbours of a live id (lineGraph's gather): the union of its members' hyperedge lists
// without itself, sorted and deduplicated.
void gather_row(const std::vector<std::vector<std::int32_t>>& rows,
                const std::vector<std::vector<std::int64_t>>& v2h, std::int64_t id,
                std::vector<std::int32_t>& out) {
  out.clear();
  for (const std::int32_t v : rows[static_cast<std::size_t>(id)])
    for (const std::int64_t other : v2h[static_cast<std::size_t>(v)])
      if (other != id) out.push_back(static_cast<std::int32_t>(other));
  std::sort(out.begin(), out.end());
  out.erase(std::unique(out.begin(), out.end()), out.end());
}

// A chunk of rows gathered into its own arena (the regions' allocations stay in their chunk;
// for_each carries an exception out, ADR 0083).
struct gather_chunk {
  const std::vector<std::vector<std::int32_t>>* rows;
  const std::vector<std::vector<std::int64_t>>* v2h;
  const std::int64_t* bound;  // chunk c: ids [bound[c], bound[c + 1])
  std::int64_t live_rows;     // ids at or above have no row
  std::vector<std::int32_t>* arena;
  std::int32_t* degree;
  void operator()(std::int64_t c) const {
    std::vector<std::int32_t>& out = arena[c];
    std::vector<std::int32_t> row;
    for (std::int64_t id = bound[c]; id < bound[c + 1]; ++id) {
      if (id >= live_rows) {
        degree[id] = 0;
        continue;
      }
      gather_row(*rows, *v2h, id, row);
      degree[id] = static_cast<std::int32_t>(row.size());
      out.insert(out.end(), row.begin(), row.end());
    }
  }
};

struct copy_chunk {
  const std::int64_t* bound;
  const std::vector<std::int32_t>* arena;
  const std::int64_t* row_start;
  const std::int32_t* degree;
  std::int32_t* col_ind;
  void operator()(std::int64_t c) const {
    const std::int32_t* src = arena[c].data();
    for (std::int64_t id = bound[c]; id < bound[c + 1]; ++id) {
      std::copy(src, src + degree[id], col_ind + row_start[id]);
      src += degree[id];
    }
  }
};

// The rows of a batch's directed keys (row << 33 | insert << 32 | col), one run per row.
struct row_run {
  std::int64_t row = 0, begin = 0, deletions = 0, insertions = 0;
};

// Phase 1 of applyRowsWarp: a row's deletions, each the swap-remove of the first occurrence.
struct delete_in_rows {
  const row_run* runs;
  const std::uint64_t* keys;
  const std::int64_t* row_start;
  std::int32_t* col_ind;
  std::int32_t* degree_after;  // per run
  const std::int32_t* deg;
  void operator()(std::int64_t j) const {
    const row_run& run = runs[j];
    const std::int64_t base = row_start[run.row];
    std::int32_t d = deg[run.row];
    std::int64_t deletions = run.deletions;
#if defined(DYNG_MUTATION_HSOSP_CSR_SKIPS_LAST_DELETION)
    // MUTATION (PLAN 8.4; M11a criterion 16 #3, the host-store twin): the CSR apply skips the
    // last deletion of each row.
    if (deletions > 0) --deletions;
#endif
    for (std::int64_t k = 0; k < deletions; ++k) {
      const auto v = static_cast<std::int32_t>(keys[run.begin + k] & 0xffffffffULL);
      for (std::int32_t e = 0; e < d; ++e) {
        if (col_ind[base + e] == v) {
          col_ind[base + e] = col_ind[base + d - 1];
          --d;
          break;
        }
      }
    }
    degree_after[j] = d;
  }
};

// Phase 3: relocation (planned in row order) and the appended insertions.
struct insert_in_rows {
  const row_run* runs;
  const std::uint64_t* keys;
  const std::int32_t* degree_after;
  const std::int64_t* new_start;  // per run: the tail position of a relocated row, else -1
  const std::int32_t* new_cap;
  std::int64_t* row_start;
  std::int32_t* deg;
  std::int32_t* cap;
  std::int32_t* col_ind;
  void operator()(std::int64_t j) const {
    const row_run& run = runs[j];
    const std::int64_t r = run.row;
    const std::int32_t d = degree_after[j];
    std::int64_t base = row_start[r];
    if (new_start[j] >= 0) {
      std::copy(col_ind + base, col_ind + base + d, col_ind + new_start[j]);
      row_start[r] = new_start[j];
      cap[r] = new_cap[j];
      base = new_start[j];
    }
    for (std::int64_t k = 0; k < run.insertions; ++k)
      col_ind[base + d + k] =
          static_cast<std::int32_t>(keys[run.begin + run.deletions + k] & 0xffffffffULL);
    deg[r] = static_cast<std::int32_t>(d + run.insertions);
  }
};

}  // namespace

template <typename exec_t>
void slack_line_graph::build(const exec_t& exec, const std::vector<std::vector<std::int32_t>>& rows,
                             const std::vector<std::vector<std::int64_t>>& v2h,
                             std::int64_t num_nodes, std::int64_t node_capacity,
                             double entry_headroom, const std::vector<std::int64_t>& weights,
                             std::int64_t max_entries) {
  // !(x >= 1) also rejects NaN (buildCsr).
  DYNG_EXPECTS(entry_headroom >= 1.0,
               "hypergraph: the slack_csr line graph's entry headroom must "
               "be at least 1, got ",
               entry_headroom);
  DYNG_EXPECTS(num_nodes <= node_capacity, "hypergraph: ", num_nodes,
               " nodes exceed the node capacity ", node_capacity);
  const std::int64_t live_rows =
      std::min<std::int64_t>(num_nodes, static_cast<std::int64_t>(rows.size()));
  // The rows, gathered by chunks of ids.
  const std::int64_t chunks =
      std::max<std::int64_t>(1, std::min<std::int64_t>(num_nodes / 4096 + 1, 64 * exec.threads()));
  std::vector<std::int64_t> bound(static_cast<std::size_t>(chunks) + 1);
  for (std::int64_t c = 0; c <= chunks; ++c)
    bound[static_cast<std::size_t>(c)] = num_nodes * c / chunks;
  std::vector<std::vector<std::int32_t>> arena(static_cast<std::size_t>(chunks));
  std::vector<std::int32_t> degree(static_cast<std::size_t>(num_nodes));
  exec.for_each(chunks,
                gather_chunk{&rows, &v2h, bound.data(), live_rows, arena.data(), degree.data()});
  // The layout (buildCsr): every node below num_nodes a slot of row_capacity_for(deg).
  std::vector<std::int64_t> row_start(static_cast<std::size_t>(node_capacity), 0);
  std::vector<std::int32_t> cap(static_cast<std::size_t>(node_capacity), 0);
  std::int64_t cursor = 0, entries = 0;
  for (std::int64_t id = 0; id < num_nodes; ++id) {
    const std::int32_t d = degree[static_cast<std::size_t>(id)];
    row_start[static_cast<std::size_t>(id)] = cursor;
    cap[static_cast<std::size_t>(id)] = static_cast<std::int32_t>(slack::row_capacity_for(d));
    cursor += cap[static_cast<std::size_t>(id)];
    entries += d;
  }
  const std::int64_t capacity = slack::entry_capacity(cursor, entry_headroom);
  if (capacity > max_entries) {
    throw capacity_error("dyng: hypergraph: the slack_csr line graph needs " +
                         std::to_string(capacity) + " entries, more than " +
                         std::to_string(max_entries) + " (the int32 counts of its entries)");
  }
  std::vector<std::int32_t> col_ind(static_cast<std::size_t>(capacity), 0);
  exec.for_each(chunks, copy_chunk{bound.data(), arena.data(), row_start.data(), degree.data(),
                                   col_ind.data()});
  degree.resize(static_cast<std::size_t>(node_capacity), 0);
  std::vector<std::int64_t> node_weight = node_weight_;
  node_weight.resize(static_cast<std::size_t>(node_capacity), 0);
  for (std::size_t id = 0; id < weights.size() && id < node_weight.size(); ++id)
    node_weight[id] = weights[id];
  // Commit (no allocation below).
  row_start_.swap(row_start);
  deg_.swap(degree);
  cap_.swap(cap);
  col_ind_.swap(col_ind);
  node_weight_.swap(node_weight);
  num_nodes_ = num_nodes;
  used_entries_ = cursor;
  num_entries_ = entries;
}

template <typename exec_t>
bool slack_line_graph::apply(const exec_t& exec, const line_graph_delta& delta,
                             std::int64_t num_nodes, const std::vector<std::int64_t>& new_ids,
                             const std::vector<std::int64_t>& new_weights) {
  DYNG_EXPECTS(num_nodes <= node_capacity(), "hypergraph: ", num_nodes,
               " nodes exceed the node capacity ", node_capacity());
  num_nodes_ = num_nodes;
  const auto deleted = static_cast<std::int64_t>(delta.deleted_a.size());
  const auto inserted = static_cast<std::int64_t>(delta.inserted_a.size());
  num_entries_ += 2 * (inserted - deleted);
  const std::int64_t pairs = deleted + inserted;
  if (pairs > 0) {
    if (2 * pairs > INT_MAX) {
      throw capacity_error("dyng: hypergraph: a batch of " + std::to_string(pairs) +
                           " line-graph pair changes exceeds the int32 counts of the CSR apply");
    }
    // expandDeltaKeysKernel: both directions of every pair, deletions before insertions per row.
    std::vector<std::uint64_t> keys(static_cast<std::size_t>(2 * pairs));
    std::size_t k = 0;
    const auto add = [&](std::int64_t a, std::int64_t b, std::uint64_t insert) {
      keys[k++] =
          (static_cast<std::uint64_t>(a) << 33) | (insert << 32) | static_cast<std::uint64_t>(b);
      keys[k++] =
          (static_cast<std::uint64_t>(b) << 33) | (insert << 32) | static_cast<std::uint64_t>(a);
    };
    for (std::int64_t i = 0; i < deleted; ++i)
      add(delta.deleted_a[static_cast<std::size_t>(i)],
          delta.deleted_b[static_cast<std::size_t>(i)], 0);
    for (std::int64_t i = 0; i < inserted; ++i)
      add(delta.inserted_a[static_cast<std::size_t>(i)],
          delta.inserted_b[static_cast<std::size_t>(i)], 1);
    sort_keys(exec, keys);
    std::vector<row_run> runs;
    for (std::size_t i = 0; i < keys.size();) {
      row_run run;
      run.row = static_cast<std::int64_t>(keys[i] >> 33);
      run.begin = static_cast<std::int64_t>(i);
      std::size_t j = i;
      while (j < keys.size() && static_cast<std::int64_t>(keys[j] >> 33) == run.row) {
        ((keys[j] >> 32) & 1ULL ? run.insertions : run.deletions) += 1;
        ++j;
      }
      runs.push_back(run);
      i = j;
    }
    const auto num_runs = static_cast<std::int64_t>(runs.size());
    std::vector<std::int32_t> degree_after(runs.size());
    exec.for_each(num_runs, delete_in_rows{runs.data(), keys.data(), row_start_.data(),
                                           col_ind_.data(), degree_after.data(), deg_.data()});
    // The relocations, planned in row order: a row whose insertions do not fit its slot gets a
    // slot of row_capacity_for(need) at the tail. The device takes the positions with atomicAdd in
    // any order; the tail overflows (and the CSR is rebuilt) exactly when their sum does not fit.
    std::vector<std::int64_t> new_start(runs.size(), -1);
    std::vector<std::int32_t> new_cap(runs.size(), 0);
    const std::int64_t tail_before = used_entries_;
    std::int64_t tail = used_entries_;
    bool overflow = false;
    for (std::size_t j = 0; j < runs.size(); ++j) {
      const std::int64_t need = degree_after[j] + runs[j].insertions;
      if (runs[j].insertions == 0 || need <= cap_[static_cast<std::size_t>(runs[j].row)]) continue;
      const std::int64_t c = slack::row_capacity_for(need);
      if (tail + c > cap_entries()) overflow = true;
      new_start[j] = tail;
      new_cap[j] = static_cast<std::int32_t>(c);
      tail += c;
    }
    used_entries_ = tail;
    if (overflow) return false;
    exec.for_each(num_runs, insert_in_rows{runs.data(), keys.data(), degree_after.data(),
                                           new_start.data(), new_cap.data(), row_start_.data(),
                                           deg_.data(), cap_.data(), col_ind_.data()});
    for (const std::int64_t s : new_start) relocations_ += s >= 0 ? 1 : 0;
    tail_entries_ += tail - tail_before;
  }
  // scatterWeightsKernel: the weights of the new ids.
  for (std::size_t i = 0; i < new_ids.size(); ++i)
    node_weight_[static_cast<std::size_t>(new_ids[i])] = new_weights[i];
  return true;
}

void slack_line_graph::grow_nodes(std::int64_t needed) {
  if (needed <= node_capacity()) return;
  const std::int64_t to = std::max(needed, node_capacity() + node_capacity() / 2);
  row_start_.resize(static_cast<std::size_t>(to), 0);
  deg_.resize(static_cast<std::size_t>(to), 0);
  cap_.resize(static_cast<std::size_t>(to), 0);
  node_weight_.resize(static_cast<std::size_t>(to), 0);
  ++node_growths_;
}

void slack_line_graph::sorted_rows(std::int64_t num_rows, std::vector<std::int64_t>& row_ptr,
                                   std::vector<std::int64_t>& col_ind,
                                   std::vector<std::int64_t>& out) const {
  row_ptr.assign(static_cast<std::size_t>(num_rows) + 1, 0);
  for (std::int64_t r = 0; r < num_rows; ++r) {
    const std::int32_t d = r < num_nodes_ ? deg_[static_cast<std::size_t>(r)] : 0;
    row_ptr[static_cast<std::size_t>(r) + 1] = row_ptr[static_cast<std::size_t>(r)] + d;
  }
  col_ind.resize(static_cast<std::size_t>(row_ptr.back()));
  out.assign(static_cast<std::size_t>(num_rows), 0);
  for (std::int64_t r = 0; r < num_rows; ++r) {
    const auto begin = col_ind.begin() + row_ptr[static_cast<std::size_t>(r)];
    const auto end = col_ind.begin() + row_ptr[static_cast<std::size_t>(r) + 1];
    if (begin != end) {
      const std::int32_t* src = col_ind_.data() + row_start_[static_cast<std::size_t>(r)];
      std::copy(src, src + (end - begin), begin);
      std::sort(begin, end);
    }
    out[static_cast<std::size_t>(r)] =
        row_ptr[static_cast<std::size_t>(r)] + (std::upper_bound(begin, end, r) - begin);
  }
}

std::vector<std::string> slack_line_graph::check(
    const std::vector<std::vector<std::int32_t>>& rows,
    const std::vector<std::vector<std::int64_t>>& v2h) const {
  std::vector<std::string> problems;
  std::int64_t entries = 0;
  std::vector<std::int32_t> want, got;
  for (std::int64_t r = 0; r < num_nodes_ && problems.size() < 20; ++r) {
    const auto i = static_cast<std::size_t>(r);
    const std::int32_t d = deg_[i];
    entries += d;
    if (d < 0 || d > cap_[i] || (d > 0 && row_start_[i] + cap_[i] > used_entries_)) {
      problems.push_back("slack line graph: node " + std::to_string(r) + " overruns its slot");
      continue;
    }
    if (r < static_cast<std::int64_t>(rows.size())) {
      gather_row(rows, v2h, r, want);
    } else {
      want.clear();
    }
    got.assign(col_ind_.begin() + row_start_[i], col_ind_.begin() + row_start_[i] + d);
    std::sort(got.begin(), got.end());
    if (got != want)
      problems.push_back("slack line graph: the row of node " + std::to_string(r) + " has " +
                         std::to_string(got.size()) + " neighbours, the incidence lists give " +
                         std::to_string(want.size()));
  }
  for (std::int64_t r = num_nodes_; r < node_capacity(); ++r)
    if (deg_[static_cast<std::size_t>(r)] != 0) {
      problems.push_back("slack line graph: node " + std::to_string(r) +
                         " above the nodes has a row");
      break;
    }
  if (problems.empty() && entries != num_entries_)
    problems.push_back("slack line graph: " + std::to_string(num_entries_) +
                       " entries counted, the rows hold " + std::to_string(entries));
  if (used_entries_ > cap_entries()) problems.push_back("slack line graph: the tail overruns");
  return problems;
}

std::int64_t slack_line_graph::bytes() const noexcept {
  return static_cast<std::int64_t>(row_start_.capacity() * sizeof(std::int64_t) +
                                   (deg_.capacity() + cap_.capacity() + col_ind_.capacity()) *
                                       sizeof(std::int32_t) +
                                   node_weight_.capacity() * sizeof(std::int64_t));
}

#define DYNG_INSTANTIATE_SLACK_LINE_GRAPH(EXEC)                                                   \
  template void slack_line_graph::build<EXEC>(                                                    \
      const EXEC&, const std::vector<std::vector<std::int32_t>>&,                                 \
      const std::vector<std::vector<std::int64_t>>&, std::int64_t, std::int64_t, double,          \
      const std::vector<std::int64_t>&, std::int64_t);                                            \
  template bool slack_line_graph::apply<EXEC>(const EXEC&, const line_graph_delta&, std::int64_t, \
                                              const std::vector<std::int64_t>&,                   \
                                              const std::vector<std::int64_t>&);
DYNG_INSTANTIATE_SLACK_LINE_GRAPH(ops::sequential_exec)
DYNG_INSTANTIATE_SLACK_LINE_GRAPH(ops::openmp_exec)
#undef DYNG_INSTANTIATE_SLACK_LINE_GRAPH

}  // namespace dyng::detail::hyper
