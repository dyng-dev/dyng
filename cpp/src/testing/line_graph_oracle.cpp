// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HypergraphOracle.cpp (rebuildLineGraph)
/**
 * @file line_graph_oracle.cpp
 * @brief dyng::testing::rebuild_line_graph (MOSP_ESCHER's HypergraphOracle algorithm). Standard
 *        library only: it shares no code with the hypergraph container (line_graph_oracle_
 *        isolation).
 */
#include <dyng/testing/line_graph_oracle.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace dyng::testing {

line_graph_rows rebuild_line_graph(const std::vector<std::int64_t>& h2v_offsets,
                                   const std::vector<std::int32_t>& h2v_members,
                                   std::int32_t num_vertices) {
  if (num_vertices < 0) throw std::invalid_argument("rebuild_line_graph: negative vertex count");
  if (h2v_offsets.empty() || h2v_offsets.front() != 0 ||
      h2v_offsets.back() != static_cast<std::int64_t>(h2v_members.size())) {
    throw std::invalid_argument(
        "rebuild_line_graph: the offsets must start at 0 and end at the member count");
  }
  const std::int64_t m = static_cast<std::int64_t>(h2v_offsets.size()) - 1;
  if (m > INT32_MAX) throw std::invalid_argument("rebuild_line_graph: more than 2^31 - 1 ids");
  const auto n = static_cast<std::size_t>(num_vertices);
  for (std::int64_t id = 0; id < m; ++id) {
    if (h2v_offsets[static_cast<std::size_t>(id) + 1] < h2v_offsets[static_cast<std::size_t>(id)])
      throw std::invalid_argument("rebuild_line_graph: the offsets decrease at id " +
                                  std::to_string(id));
  }
  for (const std::int32_t v : h2v_members) {
    if (v < 0 || v >= num_vertices)
      throw std::invalid_argument("rebuild_line_graph: member " + std::to_string(v) +
                                  " is out of range");
  }

  // Vertex -> live hyperedges, from the rows only (a repeated member counts once).
  std::vector<std::vector<std::int64_t>> by_vertex(n);
  for (std::int64_t id = 0; id < m; ++id) {
    for (std::int64_t k = h2v_offsets[static_cast<std::size_t>(id)];
         k < h2v_offsets[static_cast<std::size_t>(id) + 1]; ++k) {
      auto& hs = by_vertex[static_cast<std::size_t>(h2v_members[static_cast<std::size_t>(k)])];
      if (hs.empty() || hs.back() != id) hs.push_back(id);
    }
  }

  // Every ordered pair (a, b), a != b, of hyperedges sharing a vertex, emitted per vertex, then
  // sorted and deduplicated.
  std::vector<std::uint64_t> pairs;
  for (const auto& hs : by_vertex) {
    for (std::size_t i = 0; i < hs.size(); ++i)
      for (std::size_t j = 0; j < hs.size(); ++j)
        if (i != j)
          pairs.push_back((static_cast<std::uint64_t>(hs[i]) << 32) |
                          static_cast<std::uint32_t>(hs[j]));
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

  line_graph_rows lg;
  lg.offsets.assign(static_cast<std::size_t>(m) + 1, 0);
  lg.neighbours.resize(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    ++lg.offsets[static_cast<std::size_t>(pairs[k] >> 32) + 1];
    lg.neighbours[k] = static_cast<std::int64_t>(pairs[k] & 0xffffffffULL);
  }
  for (std::size_t id = 0; id < static_cast<std::size_t>(m); ++id)
    lg.offsets[id + 1] += lg.offsets[id];
  return lg;
}

}  // namespace dyng::testing
