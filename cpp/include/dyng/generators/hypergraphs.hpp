// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraphs.hpp
 * @brief Seeded generators of hypergraphs and hyperedge batches with distributions owned by the
 *        library: bit-exact per (seed, major version) on every platform (PLAN Section 5.8).
 * @ingroup generators
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_view.hpp>

#include <cstdint>
#include <string_view>

namespace dyng::generators {

/**
 * @brief Options of uniform_hypergraph().
 * @ingroup generators
 */
struct uniform_hypergraph_options {
  std::int64_t num_vertices = 0;    ///< n (>= 1 when there are hyperedges)
  std::int64_t num_hyperedges = 0;  ///< hyperedges to draw (>= 0)
  /**
   * @brief The cardinality of every hyperedge is uniform in [min_cardinality, max_cardinality], clamped to n; its members are distinct vertices, uniform among the subsets of that size.
   */
  std::int64_t min_cardinality = 1;
  std::int64_t max_cardinality = 4;  ///< see min_cardinality
  std::int64_t min_weight = 1;    ///< weighted types: weights uniform in [min_weight, max_weight]
  std::int64_t max_weight = 100;  ///< see min_weight
  std::uint64_t seed = 1;         ///< the seed of the stream
};

/**
 * @brief A uniform random hypergraph: hyperedge i (id i of hypergraph::from_hyperedges()) has a
 *        uniform cardinality and uniformly drawn distinct members, sorted.
 *
 * The stream is a function of (options, the library's major version) alone: std::mt19937_64(seed)
 * (specified by the C++ standard) with the library's own uniform integer and sampling
 * algorithms, so it is the same with every compiler and standard library. Golden hashes in the
 * tests pin it.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] opt The options.
 * @return The hyperedges (num_vertices = n).
 * @throws invalid_argument_error if a count is negative, there are hyperedges but no vertex, the
 *                                cardinality range is empty or below 1, n does not fit vertex_t,
 *                                or the weight range is empty, below 1 or does not fit weight_t.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup generators
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyperedge_list<vertex_t, weight_t> uniform_hypergraph(
    const uniform_hypergraph_options& opt);

/**
 * @brief The kinds of hyperedge batches random_hyperedge_batch() draws (the degenerate cases of
 *        the conformance kit among them). `k` is hyperedge_batch_options::size.
 * @ingroup generators
 */
enum class hyperedge_batch_mix : std::uint8_t {
  insert_only,   ///< k fresh hyperedges
  delete_only,   ///< k distinct live hyperedges deleted
  mixed,         ///< (k + 1) / 2 live hyperedges deleted and (k + 1) / 2 fresh ones inserted
  reuse,         ///< delete-then-reuse within one batch: the k largest live hyperedges deleted
                 ///< and k fresh ones no larger than the smallest of them inserted (they fit the
                 ///< freed slots: under id_reuse::erase_first they take their ids)
  heavy_delete,  ///< nine tenths of the live hyperedges deleted, k fresh ones inserted
  delete_all,    ///< every live hyperedge deleted
  reinsert,      ///< k live hyperedges deleted and their member sets (and weights) inserted again
  larger,        ///< half of the live hyperedges deleted and 2 x live + 3 fresh ones inserted:
                 ///< a batch larger than the hypergraph
  duplicates,    ///< k fresh hyperedges with every member twice (shuffled), and a copy of the
                 ///< live hyperedge with the smallest id (parallel hyperedges)
  dead_ids,      ///< min(k, 2) live hyperedges deleted, the same ids again, a dead id below the
                 ///< id bound (if there is one), the bound and -1 (all skipped by apply()), one
                 ///< fresh insertion
  skewed,        ///< min(k, 2) live hyperedges deleted, one fresh hyperedge of
                 ///< min(n, skewed_cardinality) members and one of 1 member
  limits,        ///< ids at the index limits: the largest live id deleted, deletions of
                 ///< INT64_MIN, INT64_MAX and the id bound, insertions {0, n - 1}, {n - 1} and
                 ///< {0} (with the smallest and the largest weight)
  empty,         ///< no operation
};

/**
 * @brief The name of a hyperedge_batch_mix ("insert_only", "delete_only", ...).
 * @param[in] mix The mix.
 * @return Its name.
 * @ingroup generators
 */
[[nodiscard]] std::string_view to_string(hyperedge_batch_mix mix) noexcept;

/**
 * @brief Options of random_hyperedge_batch().
 * @ingroup generators
 */
struct hyperedge_batch_options {
  hyperedge_batch_mix mix = hyperedge_batch_mix::mixed;  ///< the kind of batch
  std::int64_t size = -1;  ///< k of the mix (>= 0); -1: max(1, live hyperedges / 10)
  /**
   * @brief Fresh hyperedges have a cardinality uniform in [min_cardinality, max_cardinality] (clamped to n) and distinct members.
   */
  std::int64_t min_cardinality = 1;
  std::int64_t max_cardinality = 4;        ///< see min_cardinality
  std::int64_t skewed_cardinality = 1100;  ///< hyperedge_batch_mix::skewed's large hyperedge
  std::int64_t min_weight = 1;             ///< weighted types: weights of fresh hyperedges
  std::int64_t max_weight = 100;           ///< see min_weight
  std::uint64_t seed = 1;                  ///< the seed of the stream
};

/**
 * @brief A random hyperedge batch for a hypergraph state given by its h2v rows: row i lists the
 *        members of the hyperedge with id i (an empty row is a dead id; the number of rows is
 *        the id bound), as hypergraph::h2v() returns them.
 *
 * Live hyperedges are chosen uniformly (in the order of a seeded shuffle of the live ids);
 * deleted ids appear in the batch in that order, insertions after them in generation order. The
 * stream is a function of (state, options, the library's major version) alone, as for
 * uniform_hypergraph(). The batch names ids of the given state, so it must be applied to that
 * state; its inserted ids are the hypergraph's choice (hyper_apply_summary::inserted_ids).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (dyng::unweighted, or integral).
 * @param[in] num_vertices The vertex count n.
 * @param[in] h2v          The rows by id (host memory).
 * @param[in] weights      The weights by id (host memory; empty for unweighted or unknown:
 *                         hyperedge_batch_mix::reinsert then uses weight min_weight).
 * @param[in] opt          The options.
 * @return The batch.
 * @throws invalid_argument_error if the options are invalid (as uniform_hypergraph()'s), size is
 *                                below -1, or the rows are not in host memory or malformed.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup generators
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyperedge_batch<vertex_t, weight_t> random_hyperedge_batch(
    vertex_t num_vertices, const incidence_view<vertex_t>& h2v, array_view<const weight_t> weights,
    const hyperedge_batch_options& opt);

/**
 * @brief A random hyperedge batch for the current state of a hypergraph (its h2v rows and
 *        weights, copied to the host for a CUDA hypergraph).
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type.
 * @param[in] res Execution resources of the hypergraph's backend.
 * @param[in] h   The hypergraph.
 * @param[in] opt The options.
 * @return The batch (see the other overload).
 * @throws invalid_argument_error if the options are invalid, the hypergraph was moved from or is
 *                                poisoned, or the resources belong to another backend.
 * @throws out_of_memory_error    if memory cannot be allocated.
 * @throws cuda_error             if the CUDA runtime reports an error.
 * @sync Reads the hypergraph's rows (a copy to the host for a CUDA hypergraph).
 * @ingroup generators
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyperedge_batch<vertex_t, weight_t> random_hyperedge_batch(
    const resources& res, const hypergraph<vertex_t, weight_t>& h,
    const hyperedge_batch_options& opt);

}  // namespace dyng::generators
