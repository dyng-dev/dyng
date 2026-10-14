// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (groupByVertex)
/**
 * @file store.cpp
 * @brief The shared helpers of the hypergraph stores.
 */
#include "hypergraph/store.hpp"

#include <algorithm>
#include <cstdint>
#include <vector>

namespace dyng::detail::hyper {

void group_by_vertex(const member_rows& rows, const std::vector<std::int64_t>& ids,
                     grouped_incidences& out) {
  // ESCHER-GPU's groupByVertex: the incidences packed as (vertex << 32 | row), sorted, split at
  // vertex boundaries; the values are the rows' ids. Inside a vertex the rows keep their order.
  std::vector<std::uint64_t> incidences;
  incidences.reserve(rows.members.size());
  for (std::int64_t i = 0; i < rows.size(); ++i) {
    for (std::int64_t k = rows.offsets[static_cast<std::size_t>(i)];
         k < rows.offsets[static_cast<std::size_t>(i) + 1]; ++k) {
      incidences.push_back(
          (static_cast<std::uint64_t>(static_cast<std::uint32_t>(rows.members[k])) << 32) |
          static_cast<std::uint32_t>(i));
    }
  }
  std::sort(incidences.begin(), incidences.end());
  out.vertices.clear();
  out.ids.clear();
  out.prefix.clear();
  out.ids.reserve(incidences.size());
  for (std::size_t k = 0; k < incidences.size(); ++k) {
    const auto v = static_cast<std::int32_t>(incidences[k] >> 32);
    if (k == 0 || v != out.vertices.back()) {
      if (k > 0) out.prefix.push_back(static_cast<std::int32_t>(out.ids.size()));
      out.vertices.push_back(v);
    }
    out.ids.push_back(ids[static_cast<std::size_t>(incidences[k] & 0xffffffffULL)]);
  }
  if (!incidences.empty()) out.prefix.push_back(static_cast<std::int32_t>(out.ids.size()));
}

}  // namespace dyng::detail::hyper
