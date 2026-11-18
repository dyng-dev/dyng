// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_operators_cases.hpp
 * @brief The inputs of label_propagation's operator tests (operators/; ADR 0102), shared by the
 *        host test (sequential_exec) and the CUDA test (cuda_exec): slotted rows with soft-deleted
 *        entries, unused slots and deleted vertices, whose hubs have the bin-edge degrees 8, 9,
 *        32, 33, 1024, 1025, 3000 and 6000, and a vertex list; an edge list over a vertex list for
 *        connected_components.
 */
#pragma once

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

namespace dyng::test::lp_operators {

/// The hub degrees (the degree bins' edges: 32 and 1024, and above).
inline const std::vector<int> hub_degrees = {8, 9, 32, 33, 1024, 1025, 3000, 6000};

/// Slotted rows on the host.
struct host_rows {
  std::int32_t n = 0;                   ///< vertices
  std::vector<std::int32_t> row_start;  ///< n
  std::vector<std::int32_t> row_end;    ///< n
  std::vector<std::int32_t> col_ind;    ///< the slots (-1: soft-deleted or unused)
  std::vector<float> weights;           ///< the slots' weights
  std::vector<std::int32_t> status;     ///< 0 active, 1 deleted, 2 inactive
  std::vector<float> x;                 ///< the values
  std::vector<std::int32_t> hubs;       ///< the hubs, in hub_degrees' order
};

/**
 * @brief The test rows: `small` ordinary vertices of degree 0-12 and one hub per hub degree, every
 *        degree counting the entries of the row (soft-deleted ones included, about 1 in 9; the bin
 *        rule), unused slots after each row, about 1 in 23 vertices deleted (their entries kept,
 *        as the graph keeps them), weights in (0, 1], values in [0, 1].
 * @param[in] small The ordinary vertices.
 * @param[in] seed  The draws.
 * @return The rows.
 */
inline host_rows make_rows(std::int32_t small, std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  std::uniform_real_distribution<float> unit(0.0F, 1.0F);
  host_rows g;
  g.n = small + static_cast<std::int32_t>(hub_degrees.size());
  std::vector<int> degree(static_cast<std::size_t>(g.n));
  for (std::int32_t v = 0; v < small; ++v) {
    degree[static_cast<std::size_t>(v)] = static_cast<int>(rng() % 13);
  }
  for (std::size_t h = 0; h < hub_degrees.size(); ++h) {
    const auto v = static_cast<std::int32_t>(small + static_cast<std::int32_t>(h));
    degree[static_cast<std::size_t>(v)] = hub_degrees[h];
    g.hubs.push_back(v);
  }
  g.status.assign(static_cast<std::size_t>(g.n), 0);
  for (std::int32_t v = 0; v < small; ++v) {
    if (rng() % 23 == 0) {
      g.status[static_cast<std::size_t>(v)] = 1;
    }
  }
  std::int32_t slot = 0;
  for (std::int32_t v = 0; v < g.n; ++v) {
    g.row_start.push_back(slot);
    for (int k = 0; k < degree[static_cast<std::size_t>(v)]; ++k) {
      const bool soft_deleted = rng() % 9 == 0;
      g.col_ind.push_back(soft_deleted ? -1 : static_cast<std::int32_t>(rng() % g.n));
      g.weights.push_back(1.0F - unit(rng));  // (0, 1]
    }
    slot += degree[static_cast<std::size_t>(v)];
    g.row_end.push_back(slot);
    const int unused = static_cast<int>(rng() % 3);
    for (int k = 0; k < unused; ++k) {
      g.col_ind.push_back(-1);
      g.weights.push_back(0.0F);
    }
    slot += unused;
  }
  g.x.resize(static_cast<std::size_t>(g.n));
  for (auto& value : g.x) {
    value = unit(rng);
  }
  return g;
}

/// Every vertex, hubs first (so a hub lands at position 0 of every mapping too).
inline std::vector<std::int32_t> all_vertices(const host_rows& g) {
  std::vector<std::int32_t> list(g.hubs);
  for (std::int32_t v = 0; v < g.n; ++v) {
    if (std::find(g.hubs.begin(), g.hubs.end(), v) == g.hubs.end()) {
      list.push_back(v);
    }
  }
  return list;
}

/// An edge list over a vertex list (components of every size, isolated vertices included).
struct edge_case {
  std::int32_t n = 0;                  ///< the id space
  std::vector<std::int32_t> vertices;  ///< the list (distinct, shuffled)
  std::vector<std::int32_t> src;       ///< first endpoints
  std::vector<std::int32_t> dst;       ///< second endpoints
};

/**
 * @brief A random edge list: `listed` of the `n` ids form the list, `edges` edges among them.
 * @param[in] n      The id space.
 * @param[in] listed The listed vertices.
 * @param[in] edges  The edges.
 * @param[in] seed   The draws.
 * @return The case.
 */
inline edge_case make_edges(std::int32_t n, std::int32_t listed, std::int32_t edges,
                            std::uint64_t seed) {
  std::mt19937_64 rng(seed);
  edge_case c;
  c.n = n;
  std::vector<std::int32_t> ids(static_cast<std::size_t>(n));
  std::iota(ids.begin(), ids.end(), 0);
  std::shuffle(ids.begin(), ids.end(), rng);
  c.vertices.assign(ids.begin(), ids.begin() + listed);
  for (std::int32_t e = 0; e < edges; ++e) {
    c.src.push_back(c.vertices[rng() % c.vertices.size()]);
    c.dst.push_back(c.vertices[rng() % c.vertices.size()]);
  }
  return c;
}

}  // namespace dyng::test::lp_operators
