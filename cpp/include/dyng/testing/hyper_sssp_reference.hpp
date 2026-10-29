// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hyper_sssp_reference.hpp
 * @brief dyng::testing::hyper_sssp_reference, check_hyper_sssp_tree and line_graph_delta: the
 *        independent oracle of hyper_sssp (M11a; PLAN Sections 5.8 and 6.4.8), node-weighted
 *        shortest paths over the line graph of a hypergraph and the line-graph change of a batch.
 * @ingroup testing
 *
 * The oracle follows MOSP_ESCHER\@4b86159's test oracle (hypergraph/src/HypergraphOracle.cpp,
 * lines 25-171: rebuildLineGraph, lineGraphDelta, referenceDistances, checkSosp):
 *
 * - **The line graph** (h2h) is rebuilt from the hyperedges alone: the hyperedges of every vertex,
 *   every ordered pair (a, b), a != b, of them emitted per vertex, the pairs sorted and
 *   deduplicated. A hyperedge with an empty row is dead: it has an empty line-graph row and is
 *   nobody's neighbour. A repeated member of a row counts once.
 * - **Distances** are those of a textbook binary-heap Dijkstra in which stepping INTO hyperedge h
 *   costs its weight w(h): D(h) = min over line-graph neighbours p of D(p) + w(h). The source is
 *   either a hyperedge, at distance 0 (the original's virtual source hyperedge), or a vertex s,
 *   where every live hyperedge h containing s starts at D(h) = w(h). Any non-negative weights
 *   are accepted, 0 included (the original's virtual rows weigh 0). A dead or unreachable
 *   hyperedge has distance INT64_MAX / 4 (the original's HostHypergraph::INF) and parent -1.
 * - **The canonical parent** of a reachable live hyperedge h that is not the source is the
 *   lowest-id live neighbour p that is tight, D(p) + w(h) == D(h); with a vertex source, a
 *   hyperedge h that contains the source vertex and has D(h) == w(h) has parent -1 (it is
 *   reached from the source itself, as from the original's virtual source row, the lowest id).
 *   The source hyperedge has parent -1.
 *
 * Ids are 0-based hyperedge ids (positions of the rows); parents are 0-based ids, -1 for none.
 *
 * It is an oracle (PLAN Section 5.8): it uses the C++ standard library only, shares no code with
 * hyper_sssp (cpp/src/algorithms/hyper_sssp), the hypergraph container or its line graph
 * (cpp/src/hypergraph), and includes no other project header; an include-graph and link test
 * (hyper_sssp_reference_isolation) checks this, and a planted violation proves the check fails
 * (hyper_sssp_reference_isolation_planted). It holds the whole line graph on the host and is
 * meant for test sizes and the parity harness (coauth-DBLP's line graph has 250,891,568 entries).
 */
#pragma once

#include <cstdint>
#include <utility>
#include <vector>

namespace dyng::testing {

/**
 * @brief Distances and canonical parents of the node-weighted shortest paths over the line graph
 *        of a hypergraph (MOSP_ESCHER's referenceDistances and canonical parents).
 * @param[in] h2v_offsets      The hyperedges by id: id i has the members
 *                             `h2v_members[h2v_offsets[i] .. h2v_offsets[i + 1])` (empty = dead).
 * @param[in] h2v_members      The members, in [0, num_vertices).
 * @param[in] num_vertices     Vertices.
 * @param[in] weights          One weight per id, each >= 0.
 * @param[in] source           The source: a hyperedge id in [0, ids) (`source_is_vertex` false;
 *                             a dead source hyperedge reaches nothing), or a vertex in
 *                             [0, num_vertices) (`source_is_vertex` true).
 * @param[in] source_is_vertex Whether `source` is a vertex.
 * @return {distances, parents}: one distance (INT64_MAX / 4 for dead or unreachable) and one
 *         parent (0-based, -1 for none) per id.
 * @throws std::invalid_argument for malformed offsets, a member or a source out of range, a
 *         weight count other than the id count or a negative weight.
 * @throws std::overflow_error   if a distance would reach INT64_MAX / 4.
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>>
hyper_sssp_reference(const std::vector<std::int64_t>& h2v_offsets,
                     const std::vector<std::int32_t>& h2v_members, std::int32_t num_vertices,
                     const std::vector<std::int64_t>& weights, std::int64_t source,
                     bool source_is_vertex = false);

/**
 * @brief Checks distances and parents of a hyper_sssp result against the oracle (MOSP_ESCHER's
 *        checkSosp, with the parents checked even where the distances differ).
 *
 * The distance of every id must equal hyper_sssp_reference()'s; the parent of every id must be
 * its canonical parent, computed from the REFERENCE distances, so a parent error is counted on
 * its own even when distances mismatch. Ids past the end of `distances` or `parents` count as
 * INT64_MAX / 4 and -1 (the canonical encodings of ADR 0080: a longer result is checked on the
 * extra ids too, which must be dead or unreachable).
 * @param[in] h2v_offsets      The hyperedges by id (see hyper_sssp_reference()).
 * @param[in] h2v_members      The members.
 * @param[in] num_vertices     Vertices.
 * @param[in] weights          One weight per id.
 * @param[in] source           The source (see hyper_sssp_reference()).
 * @param[in] source_is_vertex Whether `source` is a vertex.
 * @param[in] distances        The distances to check, by id.
 * @param[in] parents          The parents to check, by id (0-based, -1 for none).
 * @return {distance mismatches, parent errors} over max(ids, distances.size(), parents.size())
 *         ids.
 * @throws std::invalid_argument, std::overflow_error, std::bad_alloc as hyper_sssp_reference().
 * @ingroup testing
 */
[[nodiscard]] std::pair<std::int64_t, std::int64_t> check_hyper_sssp_tree(
    const std::vector<std::int64_t>& h2v_offsets, const std::vector<std::int32_t>& h2v_members,
    std::int32_t num_vertices, const std::vector<std::int64_t>& weights, std::int64_t source,
    bool source_is_vertex, const std::vector<std::int64_t>& distances,
    const std::vector<std::int32_t>& parents);

/**
 * @brief The net change of the line graph between two states of a hypergraph (MOSP_ESCHER's
 *        lineGraphDelta on the two rebuilt line graphs): the pairs adjacent only before
 *        (deleted) and only after (inserted).
 * @param[in] pre_offsets   The hyperedges before, by id (see hyper_sssp_reference()).
 * @param[in] pre_members   Their members.
 * @param[in] post_offsets  The hyperedges after, by id (the id counts may differ: a missing id
 *                          is dead).
 * @param[in] post_members  Their members.
 * @param[in] num_vertices  Vertices (of both states).
 * @return {deleted, inserted}: (a, b) id pairs with a < b, each list sorted.
 * @throws std::invalid_argument for malformed offsets or a member out of range.
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] std::pair<std::vector<std::pair<std::int64_t, std::int64_t>>,
                        std::vector<std::pair<std::int64_t, std::int64_t>>>
line_graph_delta(const std::vector<std::int64_t>& pre_offsets,
                 const std::vector<std::int32_t>& pre_members,
                 const std::vector<std::int64_t>& post_offsets,
                 const std::vector<std::int32_t>& post_members, std::int32_t num_vertices);

}  // namespace dyng::testing
