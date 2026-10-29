// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HypergraphOracle.cpp (rebuildLineGraph,
// lineGraphDelta, referenceDistances, checkSosp; lines 25-171)
/**
 * @file hyper_sssp_reference.cpp
 * @brief dyng::testing::hyper_sssp_reference, check_hyper_sssp_tree and line_graph_delta
 *        (MOSP_ESCHER's HypergraphOracle algorithms). Standard library only: it shares no code
 *        with hyper_sssp or the hypergraph container (hyper_sssp_reference_isolation).
 */
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <queue>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace dyng::testing {

namespace {

// MOSP_ESCHER's HostHypergraph::INF: the distance of a dead or unreachable hyperedge.
constexpr std::int64_t infinity = INT64_MAX / 4;

// A line graph as a CSR over ids (internal: 32-bit neighbour ids keep coauth-DBLP's 250M entries
// at 1 GB).
struct line_graph {
  std::vector<std::int64_t> offsets;  // ids + 1
  std::vector<std::int32_t> neighbours;

  [[nodiscard]] std::int64_t ids() const { return static_cast<std::int64_t>(offsets.size()) - 1; }
};

// The hyperedges of a call, checked.
void check_rows(const char* what, const std::vector<std::int64_t>& offsets,
                const std::vector<std::int32_t>& members, std::int32_t num_vertices) {
  const std::string fn(what);
  if (num_vertices < 0) throw std::invalid_argument(fn + ": negative vertex count");
  if (offsets.empty() || offsets.front() != 0 ||
      offsets.back() != static_cast<std::int64_t>(members.size()))
    throw std::invalid_argument(fn + ": the offsets must start at 0 and end at the member count");
  if (offsets.size() - 1 > static_cast<std::size_t>(INT32_MAX))
    throw std::invalid_argument(fn + ": more than 2^31 - 1 ids");
  for (std::size_t id = 0; id + 1 < offsets.size(); ++id) {
    if (offsets[id + 1] < offsets[id])
      throw std::invalid_argument(fn + ": the offsets decrease at id " + std::to_string(id));
  }
  for (const std::int32_t v : members) {
    if (v < 0 || v >= num_vertices)
      throw std::invalid_argument(fn + ": member " + std::to_string(v) + " is out of range");
  }
}

// rebuildLineGraph (HypergraphOracle.cpp:25-80): vertex -> live hyperedges from the rows only (a
// repeated member counts once), every ordered pair (a, b), a != b, of the hyperedges of a vertex,
// sorted and deduplicated.
line_graph rebuild(const std::vector<std::int64_t>& offsets,
                   const std::vector<std::int32_t>& members, std::int32_t num_vertices) {
  const std::size_t m = offsets.size() - 1;
  const auto n = static_cast<std::size_t>(num_vertices);
  std::vector<std::int64_t> v_off(n + 1, 0);
  std::vector<std::int32_t> last(n, -1);
  for (std::size_t id = 0; id < m; ++id) {
    for (std::int64_t k = offsets[id]; k < offsets[id + 1]; ++k) {
      const auto v = static_cast<std::size_t>(members[static_cast<std::size_t>(k)]);
      if (last[v] != static_cast<std::int32_t>(id)) {
        last[v] = static_cast<std::int32_t>(id);
        ++v_off[v + 1];
      }
    }
  }
  for (std::size_t v = 0; v < n; ++v) v_off[v + 1] += v_off[v];
  std::vector<std::int32_t> v_he(static_cast<std::size_t>(v_off[n]));
  {
    std::vector<std::int64_t> cur(v_off.begin(), v_off.end() - 1);
    std::fill(last.begin(), last.end(), -1);
    for (std::size_t id = 0; id < m; ++id) {
      for (std::int64_t k = offsets[id]; k < offsets[id + 1]; ++k) {
        const auto v = static_cast<std::size_t>(members[static_cast<std::size_t>(k)]);
        if (last[v] != static_cast<std::int32_t>(id)) {
          last[v] = static_cast<std::int32_t>(id);
          v_he[static_cast<std::size_t>(cur[v]++)] = static_cast<std::int32_t>(id);
        }
      }
    }
  }
  std::size_t num_pairs = 0;
  for (std::size_t v = 0; v < n; ++v) {
    const auto d = static_cast<std::size_t>(v_off[v + 1] - v_off[v]);
    num_pairs += d * (d > 0 ? d - 1 : 0);
  }
  std::vector<std::uint64_t> pairs;
  pairs.reserve(num_pairs);
  for (std::size_t v = 0; v < n; ++v) {
    for (std::int64_t i = v_off[v]; i < v_off[v + 1]; ++i)
      for (std::int64_t j = v_off[v]; j < v_off[v + 1]; ++j)
        if (i != j)
          pairs.push_back(
              (static_cast<std::uint64_t>(v_he[static_cast<std::size_t>(i)]) << 32) |
              static_cast<std::uint32_t>(v_he[static_cast<std::size_t>(j)]));
  }
  std::sort(pairs.begin(), pairs.end());
  pairs.erase(std::unique(pairs.begin(), pairs.end()), pairs.end());

  line_graph lg;
  lg.offsets.assign(m + 1, 0);
  lg.neighbours.resize(pairs.size());
  for (std::size_t k = 0; k < pairs.size(); ++k) {
    ++lg.offsets[static_cast<std::size_t>(pairs[k] >> 32) + 1];
    lg.neighbours[k] = static_cast<std::int32_t>(pairs[k] & 0xffffffffULL);
  }
  for (std::size_t id = 0; id < m; ++id) lg.offsets[id + 1] += lg.offsets[id];
  return lg;
}

void check_weights(const char* what, const std::vector<std::int64_t>& weights, std::size_t ids) {
  const std::string fn(what);
  if (weights.size() != ids)
    throw std::invalid_argument(fn + ": " + std::to_string(weights.size()) + " weights for " +
                                std::to_string(ids) + " ids");
  for (std::size_t id = 0; id < ids; ++id) {
    if (weights[id] < 0)
      throw std::invalid_argument(fn + ": negative weight of id " + std::to_string(id));
  }
}

// The reference of one call: distances, canonical parents.
struct reference {
  std::vector<std::int64_t> distances;
  std::vector<std::int32_t> parents;
};

reference solve(const char* what, const std::vector<std::int64_t>& offsets,
                const std::vector<std::int32_t>& members, std::int32_t num_vertices,
                const std::vector<std::int64_t>& weights, std::int64_t source,
                bool source_is_vertex) {
  const std::string fn(what);
  check_rows(what, offsets, members, num_vertices);
  const std::size_t m = offsets.size() - 1;
  check_weights(what, weights, m);
  if (source_is_vertex ? (source < 0 || source >= num_vertices)
                       : (source < 0 || static_cast<std::size_t>(source) >= m))
    throw std::invalid_argument(fn + ": source " + std::to_string(source) + " is out of range");
  const line_graph lg = rebuild(offsets, members, num_vertices);
  const auto live = [&](std::size_t id) { return offsets[id + 1] > offsets[id]; };

  reference r;
  r.distances.assign(m, infinity);
  r.parents.assign(m, -1);
  // The hyperedges that hold the source vertex (a vertex source).
  std::vector<char> seeded(m, 0);
  using entry = std::pair<std::int64_t, std::int32_t>;
  std::priority_queue<entry, std::vector<entry>, std::greater<entry>> pq;
  const auto seed = [&](std::size_t id, std::int64_t d) {
    if (d >= infinity) throw std::overflow_error(fn + ": a distance reaches INT64_MAX / 4");
    if (d < r.distances[id]) {
      r.distances[id] = d;
      pq.push({d, static_cast<std::int32_t>(id)});
    }
  };
  if (source_is_vertex) {
    for (std::size_t id = 0; id < m; ++id) {
      for (std::int64_t k = offsets[id]; k < offsets[id + 1]; ++k) {
        if (members[static_cast<std::size_t>(k)] == source) {
          seeded[id] = 1;
          seed(id, weights[id]);
          break;
        }
      }
    }
  } else if (live(static_cast<std::size_t>(source))) {
    seed(static_cast<std::size_t>(source), 0);
  }
  // referenceDistances (HypergraphOracle.cpp:107-132): stepping into v costs its weight.
  while (!pq.empty()) {
    const auto [d, u] = pq.top();
    pq.pop();
    if (d != r.distances[static_cast<std::size_t>(u)]) continue;
    for (std::int64_t k = lg.offsets[static_cast<std::size_t>(u)];
         k < lg.offsets[static_cast<std::size_t>(u) + 1]; ++k) {
      const auto v = static_cast<std::size_t>(lg.neighbours[static_cast<std::size_t>(k)]);
      if (weights[v] >= infinity - d)
        throw std::overflow_error(fn + ": a distance reaches INT64_MAX / 4");
      const std::int64_t nd = d + weights[v];
      if (nd < r.distances[v]) {
        r.distances[v] = nd;
        pq.push({nd, static_cast<std::int32_t>(v)});
      }
    }
  }
  // The canonical parents (checkSosp, HypergraphOracle.cpp:154-169): the lowest-id tight live
  // neighbour (rows are sorted, so the first tight one); the source, and with a vertex source a
  // hyperedge reached from the source itself, have none.
  for (std::size_t id = 0; id < m; ++id) {
    const std::int64_t d = r.distances[id];
    if (d >= infinity) continue;
    if (!source_is_vertex && id == static_cast<std::size_t>(source)) continue;
    if (source_is_vertex && seeded[id] && d == weights[id]) continue;
    for (std::int64_t k = lg.offsets[id]; k < lg.offsets[id + 1]; ++k) {
      const auto p = static_cast<std::size_t>(lg.neighbours[static_cast<std::size_t>(k)]);
      if (r.distances[p] < infinity && r.distances[p] + weights[id] == d) {
        r.parents[id] = static_cast<std::int32_t>(p);
        break;
      }
    }
  }
  return r;
}

}  // namespace

std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> hyper_sssp_reference(
    const std::vector<std::int64_t>& h2v_offsets, const std::vector<std::int32_t>& h2v_members,
    std::int32_t num_vertices, const std::vector<std::int64_t>& weights, std::int64_t source,
    bool source_is_vertex) {
  reference r = solve("hyper_sssp_reference", h2v_offsets, h2v_members, num_vertices, weights,
                      source, source_is_vertex);
  return {std::move(r.distances), std::move(r.parents)};
}

std::pair<std::int64_t, std::int64_t> check_hyper_sssp_tree(
    const std::vector<std::int64_t>& h2v_offsets, const std::vector<std::int32_t>& h2v_members,
    std::int32_t num_vertices, const std::vector<std::int64_t>& weights, std::int64_t source,
    bool source_is_vertex, const std::vector<std::int64_t>& distances,
    const std::vector<std::int32_t>& parents) {
  const reference r = solve("check_hyper_sssp_tree", h2v_offsets, h2v_members, num_vertices,
                            weights, source, source_is_vertex);
  const std::size_t n =
      std::max({r.distances.size(), distances.size(), parents.size()});
  std::int64_t distance_mismatches = 0;
  std::int64_t parent_errors = 0;
  for (std::size_t id = 0; id < n; ++id) {
    const std::int64_t want_d = id < r.distances.size() ? r.distances[id] : infinity;
    const std::int32_t want_p = id < r.parents.size() ? r.parents[id] : -1;
    const std::int64_t got_d = id < distances.size() ? distances[id] : infinity;
    const std::int32_t got_p = id < parents.size() ? parents[id] : -1;
    distance_mismatches += got_d != want_d ? 1 : 0;
    parent_errors += got_p != want_p ? 1 : 0;
  }
  return {distance_mismatches, parent_errors};
}

std::pair<std::vector<std::pair<std::int64_t, std::int64_t>>,
          std::vector<std::pair<std::int64_t, std::int64_t>>>
line_graph_delta(const std::vector<std::int64_t>& pre_offsets,
                 const std::vector<std::int32_t>& pre_members,
                 const std::vector<std::int64_t>& post_offsets,
                 const std::vector<std::int32_t>& post_members, std::int32_t num_vertices) {
  check_rows("line_graph_delta (pre)", pre_offsets, pre_members, num_vertices);
  check_rows("line_graph_delta (post)", post_offsets, post_members, num_vertices);
  const line_graph pre = rebuild(pre_offsets, pre_members, num_vertices);
  const line_graph post = rebuild(post_offsets, post_members, num_vertices);
  std::vector<std::pair<std::int64_t, std::int64_t>> deleted, inserted;
  // lineGraphDelta (HypergraphOracle.cpp:82-105): the merge of the two sorted rows of every id,
  // keeping b > a.
  const std::int64_t m = std::max(pre.ids(), post.ids());
  for (std::int64_t a = 0; a < m; ++a) {
    const std::int32_t* r0 = nullptr;
    const std::int32_t* e0 = nullptr;
    const std::int32_t* r1 = nullptr;
    const std::int32_t* e1 = nullptr;
    if (a < pre.ids()) {
      r0 = pre.neighbours.data() + pre.offsets[static_cast<std::size_t>(a)];
      e0 = pre.neighbours.data() + pre.offsets[static_cast<std::size_t>(a) + 1];
    }
    if (a < post.ids()) {
      r1 = post.neighbours.data() + post.offsets[static_cast<std::size_t>(a)];
      e1 = post.neighbours.data() + post.offsets[static_cast<std::size_t>(a) + 1];
    }
    while (r0 != e0 || r1 != e1) {
      if (r1 == e1 || (r0 != e0 && *r0 < *r1)) {
        if (*r0 > a) deleted.emplace_back(a, *r0);
        ++r0;
      } else if (r0 == e0 || *r1 < *r0) {
        if (*r1 > a) inserted.emplace_back(a, *r1);
        ++r1;
      } else {
        ++r0;
        ++r1;
      }
    }
  }
  return {std::move(deleted), std::move(inserted)};
}

}  // namespace dyng::testing
