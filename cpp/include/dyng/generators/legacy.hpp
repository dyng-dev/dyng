// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP-OpenMP@c352151:headers/changeGenerator.h (ChangeMode, ChangeGeneratorOptions,
// generateChangeBatch)
// Derived from CycleEnumeration-GPU@0a976ad:include/cycle_enum/dynamic/batch_generator.hpp
// (BatchParams, generate_batch)
// Derived from ESCHER-GPU@abcf9b2:include/batch.hpp (Batch, BatchGenerator, InsertMode)
/**
 * @file legacy.hpp
 * @brief generators::legacy: bit-exact reproductions of the original repositories' generators,
 *        for the parity harness and for reproducing published experiments.
 * @ingroup generators
 */
#pragma once

#include <dyng/core/resources.hpp>
#include <dyng/graph/csr.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>
#include <random>
#include <string>
#include <string_view>

/**
 * @defgroup generators Generators
 * @brief Seeded generators of graphs and batches.
 *
 * generators::legacy reproduces the random streams of the original tools for fixed seeds, so the
 * inputs of published experiments can be regenerated without the original code. The streams
 * depend on std::mt19937 (specified by the C++ standard) and on the draws of libstdc++'s
 * std::uniform_int_distribution, which dynG reproduces itself (so they are the same with any
 * standard library), and, for the modes that shuffle, on libstdc++'s std::shuffle, which needs a
 * build against libstdc++ (the others throw not_supported_error). The pinned originals were built
 * with libstdc++ 11 or later; standard_library_matches_legacy_streams() checks at run time that
 * this build's standard library draws the same (true under libstdc++ 11+), which is what makes a
 * stream regenerated here equal to the original's (docs/api/cpp/generators.md).
 *
 * The other generators of the namespace dyng::generators (generators/hypergraphs.hpp) use
 * distributions owned by the library: their streams are bit-exact per (seed, major version) on
 * every platform.
 */

namespace dyng::generators::legacy {

/**
 * @brief How MOSP's change generator draws a batch (MOSP's ChangeMode).
 * @ingroup generators
 */
enum class mosp_change_mode : std::uint8_t {
  /// Insertions join two random vertices with random weights in [weight_min, weight_max];
  /// deletions are sampled with replacement from the existing edges.
  uniform,
  /// The thesis workload: insertions join two random vertices with weights below the graph's
  /// average weight of each objective; every deletion removes a distinct edge of an SOSP tree (the
  /// Dijkstra tree of any objective from `source`). Needs libstdc++ (std::shuffle).
  targeted,
  /// Every change overwrites the weights of an existing edge (an insertion of that edge) with new
  /// random weights in [weight_min, weight_max].
  reweight,
  /// Every change raises all weights of a distinct SOSP-tree edge by a random amount in
  /// [1, weight_max], saturating at 2^31 - 1. Needs libstdc++ (std::shuffle).
  increase,
};

/**
 * @brief Options of mosp_changes() (MOSP's ChangeGeneratorOptions; the `mospPrep changes` flags).
 * @ingroup generators
 */
struct mosp_change_options {
  std::int64_t num_changes = 0;        ///< --changes: number of changes
  double insertion_percentage = 50.0;  ///< --ins: share of insertions (uniform, targeted)
  mosp_change_mode mode = mosp_change_mode::uniform;  ///< --mode
  std::int32_t weight_min = 1;                        ///< --wmin: smallest drawn weight (>= 1)
  std::int32_t weight_max = 100;                      ///< --wmax: largest drawn weight
  std::uint32_t seed = 1;                             ///< --seed: std::mt19937 seed
  std::int64_t local_hops = 0;  ///< --local: if positive, every endpoint lies within this many
                                ///< hops (ignoring direction) of a random centre (a local batch)
  bool safe_deletions = false;  ///< --safe: drop deletions that would disconnect a vertex that is
                                ///< reachable from `source`
  std::int64_t source = 0;      ///< --source: the source of the trees and of the safe filter
};

/**
 * @brief What mosp_changes() did (the report line of `mospPrep changes`).
 * @ingroup generators
 */
struct mosp_change_report {
  std::int64_t inserts = 0;            ///< insertions of new edges
  std::int64_t reweights = 0;          ///< insertions that overwrite existing edges
  std::int64_t deletes = 0;            ///< deletions kept
  std::int64_t requested_deletes = 0;  ///< deletions before the safe filter
  int safe_rounds = 0;                 ///< rounds of the safe filter (0 without it)
  bool safe = false;                   ///< the safe filter ran
  std::int64_t centre = -1;            ///< centre of a local batch, -1 otherwise
  std::int64_t local_hops = 0;         ///< radius of a local batch
  std::int64_t region = 0;             ///< vertices of the local region

  /**
   * @brief The report in `mospPrep changes` format, e.g.
   *        "inserts=50 reweights=0 deletes=50 (safe: kept 50 of 50 after 1 rounds)".
   * @return The text (without the "changes: " prefix).
   */
  [[nodiscard]] std::string summary() const;
};

/**
 * @brief MOSP's change generator (generateChangeBatch of MOSP-OpenMP c352151 and MOSP-CUDA
 *        e220ee2, the same code; `mospPrep changes`), bit-exact for a fixed seed.
 *
 * The batch has one weight per objective of `graph` (all of its columns) and follows MOSP's file
 * order: the insertions, then the re-weighted edges (as insertions), then the deletions. Written
 * with io::write_legacy_batch() it is byte-identical to `mospPrep changes` on the same CSR files
 * (tested against the originals' outputs). The safe filter applies the batch under
 * graph_properties::mosp_compatible(), as MOSP's applyChangeBatch() does.
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (32-bit integer weights, as MOSP's).
 * @param[in]  graph  The graph (host memory), e.g. from io::read_csr_triplet() or graph::to_csr().
 * @param[in]  opt    The options.
 * @param[out] report Optional: what the generator did.
 * @return The batch.
 * @throws invalid_argument_error if the options are invalid (fewer than 2 vertices, no weight
 *         column, a negative count, weight_min > weight_max or < 1, source out of range), a local
 *         region has fewer than 2 vertices, or no edge is available for deletions or re-weights
 *         (MOSP's error cases).
 * @throws not_supported_error    for the targeted and increase modes in a build without libstdc++.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup generators
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] edge_batch<vertex_t, weight_t> mosp_changes(
    const csr_view<vertex_t, edge_t, weight_t>& graph, const mosp_change_options& opt,
    mosp_change_report* report = nullptr);

/**
 * @brief Options of cycle_enum_batch() (CycleEnumeration-GPU's BatchParams; the `cycle-enum`
 *        flags --deletes, --inserts, --batch-seed and --batch-locality).
 * @ingroup generators
 */
struct cycle_enum_batch_options {
  std::int64_t num_deletions = 0;   ///< existing edges to delete (sampled without replacement)
  std::int64_t num_insertions = 0;  ///< new edges to insert (non-edges, no self-loops)
  std::uint64_t seed = 0;           ///< seed of std::mt19937_64
  /**
   * @brief Locality window: when >= 0 and smaller than the vertex count, both endpoints of every
   *        change lie in a window of max(window, 2) consecutive vertex ids whose start is drawn
   *        from the seed. The default, a negative value, means no window (the original's
   *        std::nullopt).
   */
  std::int64_t locality_window = -1;
};

/**
 * @brief CycleEnumeration-GPU's batch generator (generate_batch of commit 0a976ad; `cycle-enum
 *        --task update`), bit-exact for a fixed seed.
 *
 * Deletions are sampled without replacement from the existing edges (in window), insertions from
 * the directed pairs of the window that are neither edges nor self-loops, as in the original:
 * one std::mt19937_64(seed) draws the window start, shuffles the candidate deletions (libstdc++'s
 * std::shuffle) and draws the insertion endpoints (libstdc++'s std::uniform_int_distribution);
 * dynG reproduces both algorithms, so the batch does not depend on the standard library. The
 * batch is normalized: the deletions, then the insertions, each sorted by (source, destination).
 * Applied under graph_properties::cycle_enum_compatible() it changes the graph exactly as the
 * original's apply_batch(). For a weighted graph every insertion carries the weight 1 in each
 * weight column (the original's batches have no weights).
 *
 * @tparam vertex_t Vertex id type.
 * @tparam edge_t   Edge offset type.
 * @tparam weight_t Weight type (`unweighted` or integral).
 * @param[in] graph The graph (host memory) with sorted rows and no parallel edges, e.g. the view
 *                  of a graph built under graph_properties::cycle_enum_compatible().
 * @param[in] opt   The options.
 * @return The batch (num_weights of the graph).
 * @throws invalid_argument_error if a count is negative, the graph has fewer than 2 vertices (and
 *         a count is positive), its rows are not sorted and duplicate-free, or there are not
 *         enough existing edges or non-edges in the window for the requested counts (the
 *         original's errors).
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @sync
 * @ingroup generators
 */
template <typename vertex_t, typename edge_t, typename weight_t>
[[nodiscard]] edge_batch<vertex_t, weight_t> cycle_enum_batch(
    const csr_view<vertex_t, edge_t, weight_t>& graph, const cycle_enum_batch_options& opt);

/**
 * @brief Whether this build's standard library draws std::uniform_int_distribution (with
 *        std::mt19937 and std::mt19937_64) and std::shuffle exactly as the reproductions of the
 *        legacy generators do: checked at run time by drawing from both.
 *
 * The legacy streams are computed by dynG's own reproduction of libstdc++'s algorithms, so they do
 * not depend on the standard library; this check confirms that the standard library of the build
 * is the one the originals were built with (libstdc++ 11 or later: true), i.e. that a program
 * linked with it, like the originals, draws the same streams. Other standard libraries give false.
 * @return True if every compared draw is equal.
 * @throws out_of_memory_error if host memory cannot be allocated.
 * @ingroup generators
 */
[[nodiscard]] bool standard_library_matches_legacy_streams();

/**
 * @brief ESCHER-GPU's synthetic hypergraph (its positional arguments `n m r1 r2`), in dynG's
 *        0-based vertex ids.
 * @ingroup generators
 */
struct escher_hypergraph_options {
  std::int64_t num_hyperedges = 0;  ///< n: hyperedges to draw (>= 0)
  /**
   * @brief m: the size of every hyperedge is drawn from [1, m] (and clamped to the vertex range).
   */
  std::int32_t max_cardinality = 1;
  std::int32_t min_vertex = 0;  ///< r1 - 1: the smallest vertex drawn (>= 0)
  std::int32_t max_vertex = 0;  ///< r2 - 1: the largest vertex drawn (the hypergraph has
                                ///< max_vertex + 1 vertices, ESCHER-GPU's maxVertexId)
};

/**
 * @brief How escher_batch_generator draws the inserted hyperedges (ESCHER-GPU's InsertMode).
 * @ingroup generators
 */
enum class escher_insert_mode : std::uint8_t {
  random,  ///< fresh hyperedges from escher_hypergraph()'s distribution (A's default)
  sample,  ///< copies of uniformly sampled live hyperedges (A's default with --input)
};

/**
 * @brief The name of an escher_insert_mode ("random", "sample").
 * @param[in] mode The mode.
 * @return Its name.
 * @ingroup generators
 */
[[nodiscard]] std::string_view to_string(escher_insert_mode mode) noexcept;

/**
 * @brief Options of escher_batch_generator (ESCHER-GPU's --deletions, --insertions,
 *        --batch-size, --insert-mode and the distribution of its random insertions).
 * @ingroup generators
 */
struct escher_batch_options {
  /// Deletions per batch (at most the live count). -1 together with num_insertions == -1:
  /// ESCHER-GPU's demo batch without batch options, min(2, live) deletions and one more insertion.
  std::int64_t num_deletions = -1;
  std::int64_t num_insertions = -1;  ///< insertions per batch (see num_deletions)
  escher_insert_mode insert_mode = escher_insert_mode::random;  ///< --insert-mode
  /**
   * @brief The distribution of random insertions (A's graph parameters; for a hypergraph read from a file, its largest cardinality, min_vertex 0 and its largest vertex). max_vertex is raised to the hypergraph's num_vertices() - 1 if that is larger, as A does.
   */
  escher_hypergraph_options graph;

  /**
   * @brief ESCHER-GPU's --batch-size N: N / 2 deletions and N - N / 2 insertions.
   * @param[in] size N (>= 0).
   * @param[in] mode The insert mode.
   * @param[in] graph The distribution of random insertions.
   * @return The options.
   */
  [[nodiscard]] static escher_batch_options batch_size(std::int64_t size, escher_insert_mode mode,
                                                       const escher_hypergraph_options& graph) {
    escher_batch_options o;
    o.num_deletions = size / 2;
    o.num_insertions = size - size / 2;
    o.insert_mode = mode;
    o.graph = graph;
    return o;
  }
};

/**
 * @brief ESCHER-GPU's synthetic hypergraph (hyperedge2vertex of ESCHER-GPU\@abcf9b2), bit-exact
 *        for the engine's state: hyperedge i has a size drawn from [1, max_cardinality] (clamped
 *        to the vertex range) and distinct members drawn from [min_vertex, max_vertex] until it
 *        has that many, sorted.
 *
 * The engine is the single std::mt19937_64 of ESCHER-GPU's run (std::mt19937_64(seed) for its
 * --seed); escher_batch_generator continues from it, as A's batches continue A's stream. Built
 * with hypergraph::from_hyperedges() under {id_reuse::erase_first, capacity_policy::
 * escher_gpu()}, hyperedge i has A's key i + 1 and vertex v is A's vertex v + 1.
 * @tparam vertex_t Vertex id type.
 * @tparam weight_t Hyperedge weight type (a weighted list gets weight 1 everywhere).
 * @param[in,out] rng The engine (advanced by the draws).
 * @param[in]     opt The parameters.
 * @return The hyperedges, num_vertices = max_vertex + 1.
 * @throws invalid_argument_error if num_hyperedges < 0, max_cardinality < 1, min_vertex < 0 or
 *                                max_vertex < min_vertex.
 * @throws out_of_memory_error    if host memory cannot be allocated.
 * @ingroup generators
 */
template <typename vertex_t, typename weight_t>
[[nodiscard]] hyperedge_list<vertex_t, weight_t> escher_hypergraph(
    std::mt19937_64& rng, const escher_hypergraph_options& opt);

/**
 * @brief ESCHER-GPU's batch generator (BatchGenerator::next of ESCHER-GPU\@abcf9b2), bit-exact
 *        for the engine's state, reading the live state of a dynG hypergraph.
 *
 * Each next() reads the live ids in increasing order (A's live keys in key order) and their
 * rows, chooses the deletions by a partial Fisher-Yates shuffle of the live ids (sorted
 * afterwards) and the insertions as random hyperedges (escher_hypergraph()'s distribution) or as
 * copies of live hyperedges indexed into the partly shuffled live ids (sample mode), all from the
 * shared engine in A's draw order. Driven against a hypergraph under {id_reuse::erase_first,
 * capacity_policy::escher_gpu()} that applies every batch, it reproduces A's batches: from the
 * second batch on that needs dynG's ids to equal A's keys, so it is also an end-to-end check of
 * the id parity.
 * @ingroup generators
 */
class escher_batch_generator {
 public:
  /**
   * @brief A generator drawing from `rng` (which must outlive it).
   * @param[in,out] rng The shared engine.
   * @param[in]     opt The options.
   * @throws invalid_argument_error if exactly one of the counts is -1 or a count is below -1, or
   *                                the random-insertion parameters are invalid.
   */
  escher_batch_generator(std::mt19937_64& rng, const escher_batch_options& opt);

  /**
   * @brief The next batch for the hypergraph's live state.
   * @tparam vertex_t Vertex id type.
   * @tparam weight_t Hyperedge weight type (insertions of a weighted batch get weight 1).
   * @param[in] res Execution resources of the hypergraph's backend (its rows are read on the
   *                host).
   * @param[in] h   The hypergraph.
   * @return The batch: the deletions (sorted ids), then the insertions in A's order.
   * @throws invalid_argument_error if the hypergraph was moved from or is poisoned, or the
   *                                resources belong to another backend.
   * @throws out_of_memory_error    if memory cannot be allocated.
   * @throws cuda_error             if the CUDA runtime reports an error.
   * @sync Reads the hypergraph's rows (a copy to the host for a CUDA hypergraph).
   */
  template <typename vertex_t, typename weight_t>
  [[nodiscard]] hyperedge_batch<vertex_t, weight_t> next(const resources& res,
                                                         const hypergraph<vertex_t, weight_t>& h);

 private:
  std::mt19937_64* rng_;
  escher_batch_options opt_;
};

}  // namespace dyng::generators::legacy
