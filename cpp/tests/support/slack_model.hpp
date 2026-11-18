// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file slack_model.hpp
 * @brief The kit's host model of row_layout::slack (ADR 0103): the vertex states and, per vertex,
 *        the sequence of its row's entries (every appended entry in order, also those to departed
 *        vertices, until a compaction drops them), independent of the graph's slot arena; and its
 *        comparison with a slotted or slack graph after every batch (the rows as sequences of
 *        (column, weight bits), which is stronger than the multisets of PLAN 6.4.7's row oracle).
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_view.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test {

/// The rows of a slotted or slack graph as sequences, its states, and their model.
class slack_model {
 public:
  explicit slack_model(std::int64_t n)
      : status_(static_cast<std::size_t>(n), vertex_status::inactive),
        rows_(static_cast<std::size_t>(n)) {}

  /// Apply a vertex batch under net_effect(): Ins \ Del become active, every id of Del ends
  /// deleted, a compaction when `compacted` (after the states, before the appends), then the
  /// appends in batch order of every edge whose two endpoints are active after the batch.
  void apply(const edge_batch_view<std::int32_t, float>& b, bool compacted) {
    std::int64_t largest = static_cast<std::int64_t>(status_.size()) - 1;
    for (const auto* ids : {&b.insert_vertices, &b.delete_vertices, &b.insert_src, &b.insert_dst}) {
      for (std::size_t i = 0; i < ids->size(); ++i) {
        largest = std::max<std::int64_t>(largest, (*ids)[i]);
      }
    }
    status_.resize(static_cast<std::size_t>(largest + 1), vertex_status::inactive);
    rows_.resize(status_.size());
    std::vector<char> del(status_.size(), 0);
    for (std::size_t i = 0; i < b.delete_vertices.size(); ++i) {
      del[static_cast<std::size_t>(b.delete_vertices[i])] = 1;
    }
    for (std::size_t i = 0; i < b.delete_vertices.size(); ++i) {
      status_[static_cast<std::size_t>(b.delete_vertices[i])] = vertex_status::deleted;
    }
    for (std::size_t i = 0; i < b.insert_vertices.size(); ++i) {
      const auto v = static_cast<std::size_t>(b.insert_vertices[i]);
      if (del[v] == 0) {
        status_[v] = vertex_status::active;
      }
    }
    if (compacted) {
      compact();
    }
    for (std::size_t e = 0; e < b.insert_src.size(); ++e) {
      const auto u = static_cast<std::size_t>(b.insert_src[e]);
      const auto v = static_cast<std::size_t>(b.insert_dst[e]);
      if (status_[u] != vertex_status::active || status_[v] != vertex_status::active) {
        continue;
      }
      std::uint32_t bits = 0;
      std::memcpy(&bits, &b.insert_weights[e], 4);
      rows_[u].emplace_back(static_cast<std::int32_t>(v), bits);
      rows_[v].emplace_back(static_cast<std::int32_t>(u), bits);
    }
  }

  /// A compaction: the entries to deleted vertices dropped, a deleted vertex's row emptied.
  void compact() {
    for (std::size_t v = 0; v < rows_.size(); ++v) {
      if (status_[v] == vertex_status::deleted) {
        rows_[v].clear();
        continue;
      }
      std::vector<std::pair<std::int32_t, std::uint32_t>> kept;
      for (const auto& e : rows_[v]) {
        if (status_[static_cast<std::size_t>(e.first)] != vertex_status::deleted) {
          kept.push_back(e);
        }
      }
      rows_[v].swap(kept);
    }
  }

  /// The live directed entries (both endpoints active).
  [[nodiscard]] std::int64_t live_entries() const {
    std::int64_t live = 0;
    for (std::size_t v = 0; v < rows_.size(); ++v) {
      if (status_[v] != vertex_status::active) {
        continue;
      }
      for (const auto& e : rows_[v]) {
        live += status_[static_cast<std::size_t>(e.first)] == vertex_status::active ? 1 : 0;
      }
    }
    return live;
  }

  /// Compare with the graph's view: the states, every row as a sequence, the counts.
  template <typename graph_t>
  void expect_equal(const graph_t& g, const std::string& where) const {
    const auto view = g.view();
    ASSERT_EQ(static_cast<std::size_t>(g.num_vertices()), status_.size()) << where;
    const auto& out = view.out;
    std::int64_t active = 0;
    for (std::size_t v = 0; v < status_.size(); ++v) {
      ASSERT_EQ(view.status[v], status_[v]) << where << " vertex " << v;
      active += status_[v] == vertex_status::active ? 1 : 0;
      const auto start = static_cast<std::size_t>(out.row_ptr[v]);
      const auto end = static_cast<std::size_t>(view.row_end[v]);
      ASSERT_EQ(end - start, rows_[v].size()) << where << " row " << v;
      for (std::size_t j = 0; j < rows_[v].size(); ++j) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &out.weights[start + j], 4);
        ASSERT_EQ(out.col_ind[start + j], rows_[v][j].first)
            << where << " row " << v << " slot " << j;
        ASSERT_EQ(bits, rows_[v][j].second) << where << " row " << v << " slot " << j;
      }
    }
    EXPECT_EQ(g.num_active_vertices(), active) << where;
    EXPECT_EQ(static_cast<std::int64_t>(g.num_edges()), live_entries()) << where;
  }

  [[nodiscard]] const std::vector<vertex_status>& status() const noexcept {
    return status_;
  }

 private:
  std::vector<vertex_status> status_;
  std::vector<std::vector<std::pair<std::int32_t, std::uint32_t>>> rows_;
};

}  // namespace dyng::test
