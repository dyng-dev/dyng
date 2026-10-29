// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (groupByVertex)
/**
 * @file store.cpp
 * @brief The shared helpers of the hypergraph stores.
 */
#include "hypergraph/store.hpp"

#include <dyng/core/error.hpp>

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

void group_by_hyperedge(const std::vector<std::int64_t>& ids,
                        const std::vector<std::int32_t>& vertices, grouped_members& out) {
  // MOSP_ESCHER's flushGrouped: a stable sort by the row key, so the values of a row keep their
  // order (that order decides which values of a fill go to an overflow segment).
  std::vector<std::size_t> order(ids.size());
  for (std::size_t i = 0; i < order.size(); ++i) order[i] = i;
  std::stable_sort(order.begin(), order.end(),
                   [&](std::size_t a, std::size_t b) { return ids[a] < ids[b]; });
  out.ids.clear();
  out.members.clear();
  out.prefix.clear();
  for (const std::size_t i : order) {
    if (out.ids.empty() || out.ids.back() != ids[i]) {
      out.ids.push_back(ids[i]);
      out.prefix.push_back(out.prefix.empty() ? 0 : out.prefix.back());
    }
    out.members.push_back(vertices[i]);
    ++out.prefix.back();
  }
}

// The slack_csr calls on a store that does not take them: the container never makes them (it
// refuses incidence edits under escher storage before the store changes).
void incidence_store::h2v_remove(const grouped_members& /*g*/) {
  DYNG_FAIL("hypergraph store: an incidence removal reached a store without incidence edits");
}

void incidence_store::h2v_append(const grouped_members& /*g*/) {
  DYNG_FAIL("hypergraph store: an incidence addition reached a store without incidence edits");
}

void incidence_store::begin_line_graph_delta(const std::vector<std::int32_t>& /*vertices*/,
                                             const std::vector<std::int64_t>& /*ids*/) {
  DYNG_FAIL("hypergraph store: the pair delta reached a store without a slack line graph");
}

void incidence_store::line_graph_delta_of(const std::vector<std::int32_t>& /*vertices*/,
                                          const std::vector<std::int64_t>& /*ids*/,
                                          const std::vector<std::int8_t>& /*signs*/,
                                          const std::vector<std::int64_t>& /*inserted_ids*/,
                                          line_graph_delta& /*out*/) {
  DYNG_FAIL("hypergraph store: the pair delta reached a store without a slack line graph");
}

void incidence_store::finish_line_graph(const line_graph_delta& /*delta*/,
                                        const std::vector<std::int64_t>& /*new_ids*/,
                                        const std::vector<std::int64_t>& /*new_weights*/) {
  DYNG_FAIL("hypergraph store: the pair delta reached a store without a slack line graph");
}

}  // namespace dyng::detail::hyper
