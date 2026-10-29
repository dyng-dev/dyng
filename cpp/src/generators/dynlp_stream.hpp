// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:headers/graphTypes.h (CSRGraph, BatchData)
/**
 * @file dynlp_stream.hpp
 * @brief The DynLP legacy generator in the original's own shapes (internal; ADR 0067).
 *
 * dynlp_legacy_generate() is generators::legacy::dynlp_vertex_arrival_stream() before the
 * conversion into the public stream: the arrays of LabelPropagation-CUDA's CSRGraph, true labels
 * (floats 0 / 1) and BatchData (with the ground-truth ids in the original's order and their float
 * labels), which the public stream folds into one label per arrival. The parity tests compare
 * these arrays with the original's recorded ones field by field (and by the SHA-256 the recorder's
 * digest mode stores, at 5M and 50M), so nothing of the original's output is left unchecked.
 */
#pragma once

#include <dyng/generators/vertex_arrival.hpp>

#include <cstdint>
#include <vector>

namespace dyng::detail {

/// One BatchData of the original: the arrays exactly as generateBatches() fills them.
struct dynlp_legacy_batch {
  std::vector<std::int32_t> inserted;  ///< insertedNodeIds
  std::vector<std::int32_t> gt_ids;    ///< groundTruthNodeIds (the original's order)
  std::vector<float> gt_labels;        ///< groundTruthLabels (0.0f / 1.0f)
  std::vector<std::int32_t> deleted;   ///< deletedNodeIds
  std::vector<std::int32_t> edge_src;  ///< newEdgeSrc
  std::vector<std::int32_t> edge_dst;  ///< newEdgeDst
  std::vector<float> edge_weight;      ///< newEdgeWeight
};

/// The original's generator output: the CSRGraph, the true labels, the batches and tau.
struct dynlp_legacy_output {
  std::int32_t num_nodes = 0;
  std::vector<std::int32_t> row_ptr;  ///< rowPtr (32-bit, as the original's)
  std::vector<std::int32_t> col_ind;  ///< colInd
  std::vector<float> values;          ///< values
  std::vector<float> true_labels;     ///< trueLabels (0.0f / 1.0f)
  std::vector<dynlp_legacy_batch> batches;
  float tau = 0.5F;                   ///< computeAvgEdgeWeight(full graph)
  std::int64_t undirected_edges = 0;  ///< edgeList.size() of generateSparseGraph
};

/**
 * @brief generateSparseGraph + generateBatches + computeAvgEdgeWeight of the original.
 * @param[in] opt     The options (validated here).
 * @param[in] threads Host threads for the row sorts and the batch edge lists (>= 1); the output
 *                    does not depend on it.
 * @return The original's arrays.
 * @throws invalid_argument_error for the options generators::legacy::dynlp_vertex_arrival_stream
 *         rejects.
 */
[[nodiscard]] dynlp_legacy_output dynlp_legacy_generate(
    const generators::vertex_arrival_options& opt, int threads);

/**
 * @brief Whether this build's std::shuffle (std::mt19937, small and large counts) and
 *        std::uniform_real_distribution<float> draw exactly as the twins in util/rng.hpp
 *        (compiled without floating-point contraction, like the twins). False without libstdc++.
 * @return True if every compared draw is equal.
 */
[[nodiscard]] bool dynlp_draws_match_standard_library();

}  // namespace dyng::detail
