// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file oracle.hpp
 * @brief Brute-force host counts of the tutorial's patterns (patterns.hpp), from a hypergraph's
 *        live hyperedges (hypergraph::to_hyperedge_list): every triple or pair of hyperedges
 *        checked directly, independent of the line graph and of the wedge engine.
 *
 * Host-only and small-input only (cubic in the hyperedges). The tutorial
 * (docs/tutorials/count_a_new_hypergraph_pattern.md) quotes it between its markers.
 */
#pragma once

#include <dyng/core/resources.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <vector>

namespace tutorial {

/// The members of every live hyperedge, sorted (the order of the hyperedges does not matter).
using member_rows = std::vector<std::vector<std::int32_t>>;

/**
 * @brief The live hyperedges of a hypergraph, on the host.
 * @tparam graph_t A dyng::hypergraph.
 * @param[in] res Resources of the hypergraph's backend.
 * @param[in] h   The hypergraph.
 * @return Their member lists.
 */
template <typename graph_t>
member_rows live_rows(const dyng::resources& res, const graph_t& h) {
  const auto list = h.to_hyperedge_list(res);
  member_rows out;
  for (std::size_t e = 0; e + 1 < list.offsets.size(); ++e) {
    std::vector<std::int32_t> row(list.members.begin() + list.offsets[e],
                                  list.members.begin() + list.offsets[e + 1]);
    std::sort(row.begin(), row.end());
    out.push_back(std::move(row));
  }
  return out;
}

/// |A ∩ B| of two sorted rows.
inline int shared(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b) {
  std::vector<std::int32_t> both;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(both));
  return static_cast<int>(both.size());
}

/// |A ∩ B ∩ C| of three sorted rows.
inline int core(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                const std::vector<std::int32_t>& c) {
  std::vector<std::int32_t> ab;
  std::set_intersection(a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ab));
  return shared(ab, c);
}

// [tutorial: oracle]
/// The triangles of the line graph by brute force: every triple of hyperedges that pairwise
/// share a vertex, by their core size (bins 0, 1, 2 or more); their sum is the triangle count.
inline std::array<std::uint64_t, 3> brute_core_triads(const member_rows& rows) {
  std::array<std::uint64_t, 3> bins{};
  const std::size_t m = rows.size();
  for (std::size_t i = 0; i < m; ++i)
    for (std::size_t j = i + 1; j < m; ++j)
      for (std::size_t k = j + 1; k < m; ++k)
        if (shared(rows[i], rows[j]) > 0 && shared(rows[j], rows[k]) > 0 &&
            shared(rows[i], rows[k]) > 0)
          ++bins[static_cast<std::size_t>(std::min(core(rows[i], rows[j], rows[k]), 2))];
  return bins;
}

/// The pairs of hyperedges that share at least k vertices, by brute force.
inline std::uint64_t brute_strong_pairs(const member_rows& rows, int k) {
  std::uint64_t n = 0;
  for (std::size_t i = 0; i < rows.size(); ++i)
    for (std::size_t j = i + 1; j < rows.size(); ++j) n += shared(rows[i], rows[j]) >= k ? 1 : 0;
  return n;
}
// [tutorial: oracle end]

/// The triangles of the line graph by brute force.
inline std::uint64_t brute_triangles(const member_rows& rows) {
  const std::array<std::uint64_t, 3> bins = brute_core_triads(rows);
  return bins[0] + bins[1] + bins[2];
}

/// The pairs of hyperedges that both contain `hub`, by brute force.
inline std::uint64_t brute_hub_pairs(const member_rows& rows, std::int32_t hub) {
  std::uint64_t with_hub = 0;
  for (const auto& r : rows) with_hub += std::binary_search(r.begin(), r.end(), hub) ? 1 : 0;
  return with_hub * (with_hub - (with_hub > 0 ? 1 : 0)) / 2;
}

}  // namespace tutorial
