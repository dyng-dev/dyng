// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:headers/graphTypes.h (DynLPConfig, BatchData)
/**
 * @file vertex_arrival.hpp
 * @brief Vertex-arrival streams for label propagation: a weighted similarity graph with binary
 *        classes, revealed in batches of arriving (some labelled) and departing vertices.
 *
 * generators::legacy::dynlp_vertex_arrival_stream() reproduces LabelPropagation-CUDA\@a276a3a's
 * (DynLP's) generator byte for byte for a seed (ADR 0067): the Erdős–Rényi graph with a random
 * spanning tree and similarity weights of generateSparseGraph, the batches of generateBatches and
 * the auto threshold of computeAvgEdgeWeight. The stream is what label_propagation's parity runs
 * replay: graph::with_capacity(stream.row_capacity), then the batches in order, with
 * stream.similarity_threshold as the published threshold (PLAN 6.4.7).
 *
 * Provisional (M10a): the batches hold plain arrays until edge_batch carries vertex arrivals
 * (M10a step 3); the library's own generator of the same stream shape arrives in M10b.
 * @ingroup generators
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace dyng::generators {

/**
 * @brief Options of the vertex-arrival generators (the fields of LabelPropagation-CUDA's
 *        DynLPConfig that the generator reads; its defaults).
 * @ingroup generators
 */
struct vertex_arrival_options {
  std::int64_t total_vertices = 50000;  ///< --totalNodes: vertices of the full graph (> 1)
  std::int32_t average_degree = 5;      ///< --avgDegree: target average degree (> 0)
  std::int32_t num_batches = 10;        ///< --numBatches: batches b0 .. b(num_batches - 1) (> 0)
  std::uint32_t seed = 42;              ///< --seed: the graph's std::mt19937 seed (the batches use
                                        ///< seed + 12345)
  float ground_truth_fraction = 0.01F;  ///< labelled share of the arrivals (b0: of all vertices)
  float deleted_fraction = 0.09F;       ///< departures per batch, as a share of its arrivals
};

/**
 * @brief One batch of a vertex-arrival stream: arrivals (with their seed labels), the edges they
 *        bring and departures, applied with the net effect of the batch (an id that arrives and
 *        departs in the same batch never enters the graph).
 * @tparam vertex_t Vertex id type.
 * @ingroup generators
 */
template <typename vertex_t>
struct vertex_arrival_batch {
  std::vector<vertex_t> insert_vertices;          ///< arriving vertices, in the generator's order
  std::vector<std::int8_t> insert_vertex_labels;  ///< per arrival: -1 (unlabelled), or its class
                                                  ///< (0 or 1) for a ground-truth vertex
  std::vector<vertex_t> edge_sources;       ///< new edges (u, v) with u < v, in the generator's
                                            ///< order: every full-graph edge whose later endpoint
                                            ///< arrives in this batch
  std::vector<vertex_t> edge_destinations;  ///< see edge_sources
  std::vector<float> edge_weights;          ///< the weight of each new edge
  std::vector<vertex_t> delete_vertices;    ///< departing vertices, in the generator's order
};

/**
 * @brief A generated vertex-arrival stream: the full graph, its classes and the batches.
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @ingroup generators
 */
template <typename vertex_t, typename edge_t>
struct vertex_arrival_stream {
  /**
   * @brief The full graph: both directions of every edge, rows sorted by column, one float
   *        weight column (num_weights = 1).
   */
  csr<vertex_t, edge_t, float> graph;
  std::vector<std::int8_t> true_labels;                 ///< the class (0 or 1) of every vertex
  std::vector<vertex_arrival_batch<vertex_t>> batches;  ///< b0 .. b(num_batches - 1)
  /**
   * @brief The degree of every vertex in the full graph: the slot capacity of its row (what a
   *        slotted graph reserves for it).
   */
  std::vector<edge_t> row_capacity;
  /**
   * @brief The published similarity threshold tau: the mean weight over all directed entries of
   *        the full graph in row order, summed in double and rounded to float (the original's
   *        computeAvgEdgeWeight).
   */
  float similarity_threshold = 0.5F;
  std::size_t max_batch_vertices = 0;  ///< the largest insert_vertices.size() of a batch
  std::size_t max_batch_edges = 0;     ///< the largest edge_sources.size() of a batch
};

namespace legacy {

/**
 * @brief LabelPropagation-CUDA\@a276a3a's DynLP generator, byte for byte for a seed: the graph
 *        (generateSparseGraph), the batches (generateBatches) and the published threshold
 *        (computeAvgEdgeWeight).
 *
 * The graph comes from std::mt19937(seed): balanced classes shuffled, a random spanning tree
 * through a shuffled permutation, then random edges until total_vertices * average_degree / 2
 * (at least total_vertices - 1) undirected edges exist or 10 times that many consecutive draws
 * fail; same-class edges get a weight uniform in [0.5, 1), cross-class ones in [0.01, 0.5). The
 * batches come from std::mt19937(seed + 12345): b0 holds max(2, int(total_vertices *
 * ground_truth_fraction)) labelled vertices, balanced between the classes; the remaining vertices
 * arrive in batches b1 .. b(num_batches - 1), each with max(1, int(arrivals *
 * ground_truth_fraction)) labelled ones, and int(arrivals * deleted_fraction) unlabelled vertices
 * present by then depart (an arrival of the same batch included).
 *
 * The draws are libstdc++ 12's std::shuffle, std::uniform_int_distribution and
 * std::uniform_real_distribution<float> with std::mt19937, reproduced by the library (so the
 * stream is the same with any standard library; standard_library_matches_legacy_streams() checks
 * that this build's standard library agrees), with exact float arithmetic (no contraction). The
 * graph construction and the batch edge lists run on the host threads of `res` and do not depend
 * on their number.
 *
 * @tparam vertex_t Vertex id type (32-bit, as the original's).
 * @tparam edge_t   Edge offset type (std::int32_t or std::int64_t).
 * @param[in] res  Resources: any backend (the generator runs on the host; its threads).
 * @param[in] opt  The options.
 * @return The stream.
 * @throws invalid_argument_error if total_vertices <= 1 or does not fit 32 bits, average_degree
 *         <= 0, num_batches <= 0, a fraction is not in [0, 1], or the graph would have more than
 *         2^31 - 1 directed entries (the original's 32-bit offsets).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup generators
 */
template <typename vertex_t = std::int32_t, typename edge_t = std::int32_t>
[[nodiscard]] vertex_arrival_stream<vertex_t, edge_t> dynlp_vertex_arrival_stream(
    const resources& res, const vertex_arrival_options& opt);

}  // namespace legacy

}  // namespace dyng::generators
