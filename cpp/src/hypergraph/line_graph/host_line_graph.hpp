// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file host_line_graph.hpp
 * @brief The line graph (h2h) of a hypergraph on the host, in ESCHER-GPU's layout: the initial
 *        build (A's flatHyperedgeAdjacency) and the host form of the sorted delta merge (A's
 *        updateLineGraph), used by the host stores of the sequential and openmp backends; the
 *        build is also the initial H2H of the CUDA store (ADR 0042).
 *
 * Keys are CBST keys (hyperedge id + 1); a row lists the keys of the hyperedges that share a
 * vertex with its own, sorted and without repeats. Both functions give exactly the rows and the
 * layout (offsets, padding) of ESCHER-GPU@abcf9b2, so the host stores and the CUDA store agree on
 * every payload entry.
 */
#pragma once

#include "hypergraph/store.hpp"

#include <cstdint>
#include <vector>

namespace dyng::detail::hyper {

/**
 * @brief The line graph of a hypergraph from its rows (ESCHER-GPU's flatHyperedgeAdjacency over
 *        vertex2hyperedge, utils/utils.cpp@abcf9b2:143): row k lists the keys of every other
 *        hyperedge sharing a vertex with the hyperedge of key k, sorted and deduplicated, in the
 *        construct layout.
 * @param[in] h2v_ptr      The rows by id: row i is h2v_cols[h2v_ptr[i] .. h2v_ptr[i + 1]) (a dead
 *                         id has an empty row).
 * @param[in] h2v_cols     The members (0-based vertices in [0, num_vertices)).
 * @param[in] num_vertices Vertices.
 * @param[in] num_keys     The keys of the line graph (at least the rows; the rows above are
 *                         empty).
 * @param[in] threads      Host threads.
 * @param[in] limit        The largest payload (capacity_policy's payload limit).
 * @return The layout.
 * @throws capacity_error if the padded rows need more than `limit` payload entries (ESCHER-GPU
 *         ended the process above 2^31 - 1).
 */
[[nodiscard]] line_graph_layout flat_hyperedge_adjacency(const std::vector<std::int64_t>& h2v_ptr,
                                                         const std::vector<std::int32_t>& h2v_cols,
                                                         std::int32_t num_vertices,
                                                         std::int64_t num_keys, int threads,
                                                         std::int64_t limit);

/**
 * @brief The host form of ESCHER-GPU's sorted delta merge (updateLineGraph,
 *        structure/graph_maintenance.cu@abcf9b2:156-231): the line graph after a batch from the
 *        one before it, out of place, in the same layout.
 *
 * The rows of the inserted keys are the union of their members' v2h rows (after the batch), the
 * deleted keys leave the rows of their surviving neighbours, the inserted keys join the rows of
 * their neighbours that were not inserted, and every deleted or inserted key takes its new row
 * (empty when it was not inserted again).
 * @param[in,out] g            The line graph before the batch (keys 1..g.num_keys()), replaced
 *                             by the one after it (keys 1..num_keys).
 * @param[in] num_keys         The keys after the batch (>= g.num_keys()).
 * @param[in] deleted          The deleted ids.
 * @param[in] inserted         The ids of the inserted rows.
 * @param[in] inserted_rows    Their members (sorted, distinct).
 * @param[in] v2h              The ids of every vertex after the batch.
 * @param[in] threads          Host threads.
 * @param[in] limit            The largest payload.
 * @throws capacity_error if the merged rows need more than `limit` payload entries (nothing
 *         changes then).
 */
void merge_line_graph(line_graph_layout& g, std::int64_t num_keys,
                      const std::vector<std::int64_t>& deleted,
                      const std::vector<std::int64_t>& inserted, const member_rows& inserted_rows,
                      const std::vector<std::vector<std::int64_t>>& v2h, int threads,
                      std::int64_t limit);

/**
 * @brief The row-view arrays of a line graph (A's RowLocator::build with out): loc, len and out by
 *        key 0..num_keys (key 0 unused).
 * @param[in]  g   The line graph.
 * @param[out] loc Offsets.
 * @param[out] len Entries.
 * @param[out] out The offset of the first entry greater than the key.
 */
void line_graph_locator(const line_graph_layout& g, std::vector<int>& loc, std::vector<int>& len,
                        std::vector<int>& out);

}  // namespace dyng::detail::hyper
