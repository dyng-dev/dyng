// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/checker.cu (DynLPChecker::harmonicSolution, checkLabels)
/**
 * @file harmonic_solution.hpp
 * @brief dyng::testing::harmonic_solution: the converged harmonic labels F* of a graph with seed
 *        labels, the reference of label_propagation's oracle (PLAN 5.1, 5.8, 6.4.7; O27; AD7).
 * @ingroup testing
 *
 * Label propagation converges to the harmonic function of the graph: every unlabelled vertex's
 * label is the weighted mean of its neighbours' labels, the seeds keep theirs. On the unknowns U
 * (present, unlabelled and in a component that contains a seed) this is the linear system
 *
 *     (D - W_UU) x = W_UL y_L,     D = diag(sum of the weights to present neighbours),
 *
 * solved here by Jacobi-preconditioned conjugate gradients in double precision from x0 = 0, to a
 * relative residual |r| <= relative_tolerance * |b| (default 1e-12) within max_iterations
 * (default 100000): LabelPropagation-CUDA\@a276a3a's --check oracle (its kPcgRelTol and
 * kPcgMaxIter), with one change: every reduction (the dot products and norms) sums fixed chunks of
 * vertices in a fixed order, so the result is the same for every thread count (the original's
 * OpenMP reductions are not). Components without a seed have no harmonic solution: their vertices
 * are excluded and counted (BFS from the present seeds over present unlabelled vertices).
 *
 * label_error_against() is the oracle's comparison: the mean and the maximum of |F - F*| over the
 * present unlabelled vertices of seeded components, and the class flips ((F >= 0.5f) differs
 * from (F* >= 0.5)). class_agreement_with() adds a margin: a flip counts against the margin only where
 * |F* - 0.5| exceeds it (AD7: no flip where |F* - 0.5| > 3 m).
 *
 * Host-only and standard library only (std::thread for the optional threads): it shares no code
 * with label_propagation (PLAN 5.8; ADR 0068); an include-graph and link test checks this.
 */
#pragma once

#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief Options of harmonic_solution().
 * @ingroup testing
 */
struct harmonic_options {
  double relative_tolerance = 1e-12;     ///< stop when |r| <= relative_tolerance * |b|
  std::int64_t max_iterations = 100000;  ///< the PCG iteration cap (> 0)
  int num_threads = 1;                   ///< host threads (>= 1); the result does not depend on it
};

/**
 * @brief The harmonic solution and how it was reached.
 * @ingroup testing
 */
struct harmonic_result {
  /// F* per vertex: a present seed holds its label (0 or 1), a determined vertex the solution,
  /// every other vertex 0.5.
  std::vector<double> solution;
  /// 1 for the determined vertices: present, unlabelled, in a component with a present seed.
  std::vector<std::uint8_t> determined;
  bool converged = false;           ///< relative_residual <= relative_tolerance
  double relative_residual = 0.0;   ///< |r| / |b| at the end (0 when b = 0: x = 0 exactly)
  std::int64_t iterations = 0;      ///< PCG iterations
  std::int64_t num_seeds = 0;       ///< present vertices with a seed label
  std::int64_t num_unlabelled = 0;  ///< present vertices without one
  std::int64_t num_determined = 0;  ///< unlabelled vertices of seeded components
  std::int64_t num_excluded = 0;    ///< unlabelled vertices of seed-free components
};

/**
 * @brief Solve for the harmonic labels of the graph induced by the present vertices.
 *
 * Entries with a negative column (unused slots) and entries to vertices that are not present are
 * ignored, so a slotted graph's rows (row_ptr[v] .. the row's end, the rest -1) can be passed as
 * they are. The graph must be symmetric (an undirected graph stored in both directions).
 * @tparam edge_t Offset type (std::int32_t or std::int64_t).
 * @param[in] row_ptr     num_vertices + 1 offsets into col_ind.
 * @param[in] col_ind     The neighbour of each entry (negative: no entry).
 * @param[in] weights     The weight of each entry (non-negative).
 * @param[in] present     1 for the vertices of the current graph.
 * @param[in] seed_labels Per vertex -1 (unlabelled) or its seed label 0 or 1.
 * @param[in] opt         Options.
 * @return The solution.
 * @throws std::invalid_argument if the sizes disagree, a column is out of range, a seed label is
 *         not -1, 0 or 1, or an option is invalid.
 * @ingroup testing
 */
template <typename edge_t>
[[nodiscard]] harmonic_result harmonic_solution(const std::vector<edge_t>& row_ptr,
                                                const std::vector<std::int32_t>& col_ind,
                                                const std::vector<float>& weights,
                                                const std::vector<std::uint8_t>& present,
                                                const std::vector<std::int8_t>& seed_labels,
                                                const harmonic_options& opt = {});

/**
 * @brief The comparison of labels with the harmonic solution (the oracle's per-batch line).
 * @ingroup testing
 */
struct label_error {
  std::int64_t compared = 0;  ///< present unlabelled vertices of seeded components
  std::int64_t excluded = 0;  ///< present unlabelled vertices of seed-free components
  double mean = 0.0;          ///< mean |F - F*| over the compared vertices (0 if none)
  double max = 0.0;           ///< max |F - F*|
  std::int64_t flips = 0;     ///< compared vertices with (F >= 0.5f) != (F* >= 0.5)
};

/**
 * @brief Mean and max |F - F*| and the class flips over the compared vertices, summed in vertex
 *        order (as LabelPropagation-CUDA's checkLabels).
 * @param[in] labels      F, one float per vertex.
 * @param[in] f           The harmonic solution of the same graph.
 * @param[in] present     The present vertices (as given to harmonic_solution()).
 * @param[in] seed_labels The seed labels (as given to harmonic_solution()).
 * @return The comparison.
 * @throws std::invalid_argument if the sizes disagree.
 * @ingroup testing
 */
[[nodiscard]] label_error label_error_against(const std::vector<float>& labels,
                                              const harmonic_result& f,
                                              const std::vector<std::uint8_t>& present,
                                              const std::vector<std::int8_t>& seed_labels);

/**
 * @brief Class agreement of labels with the harmonic solution, with a margin.
 * @ingroup testing
 */
struct class_agreement {
  std::int64_t compared = 0;             ///< present unlabelled vertices of seeded components
  std::int64_t flips = 0;                ///< (F >= 0.5f) != (F* >= 0.5)
  std::int64_t flips_beyond_margin = 0;  ///< flips with |F* - 0.5| > margin
  double largest_flip_distance = 0.0;    ///< the largest |F* - 0.5| of a flipped vertex (0: none)
};

/**
 * @brief Count the class flips and those where F* is farther than `margin` from 0.5.
 * @param[in] labels      F, one float per vertex.
 * @param[in] f           The harmonic solution.
 * @param[in] present     The present vertices.
 * @param[in] seed_labels The seed labels.
 * @param[in] margin      The margin (>= 0).
 * @return The agreement.
 * @throws std::invalid_argument if the sizes disagree or the margin is negative.
 * @ingroup testing
 */
[[nodiscard]] class_agreement class_agreement_with(const std::vector<float>& labels,
                                                   const harmonic_result& f,
                                                   const std::vector<std::uint8_t>& present,
                                                   const std::vector<std::int8_t>& seed_labels,
                                                   double margin);

}  // namespace dyng::testing
