// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_reference.hpp
 * @brief dyng::testing::triad_reference: brute-force triad counts of a hypergraph (the 30-bin
 *        h-motif counts and StatHyper's triad types 1-3), the independent oracle of triad_count
 *        (M9a; PLAN Sections 5.8 and 8.2).
 * @ingroup testing
 *
 * The counts follow ESCHER-GPU\@abcf9b2's CPU reference (utils/reference.cpp, the oracle behind
 * its `--reference` flag): the line graph is rebuilt from the hyperedges (hyperedges i and j are
 * adjacent when they share a vertex), every triangle i < j < k of it is enumerated, and 64-bit
 * accumulators add up
 *
 * - **h_motif** (30 bins): each triangle in the bin of its h-motif class, decided by which of
 *   the seven regions of the Venn diagram of its three hyperedges A, B, C are non-empty;
 * - **stathyper_type1** (1 value): sum over the hyperedges of C(|e|, 3);
 * - **stathyper_type2** (1 value): sum over the adjacent pairs {e, f} of C(a, 2) * b * c, where
 *   a is the size of their intersection, b and c the numbers of members of e only and of f only,
 *   for the pairs with a >= 2, b > 0 and c > 0;
 * - **stathyper_type3** (1 value): sum over the triangles of x * y * z, where x, y, z are the
 *   sizes of the pairwise-only overlaps (the members of A and B but not C, ...), for the
 *   triangles where all three are positive.
 *
 * **The h-motif classes** are derived here, not copied from the original's table. Bit 0, 1, 2 of
 * a Venn id say that the regions "A only", "B only", "C only" are non-empty; bit 3, 4, 5 the
 * regions "A and B only", "B and C only", "C and A only"; bit 6 the region "A, B and C". A class is an
 * orbit of the 128 Venn ids under the six permutations of (A, B, C) (triad_venn_canonical(): the
 * smallest id of the orbit). Ids with fewer than two intersecting pairs (a pair intersects when
 * its pairwise-only region or the triple region is non-empty) are in no class (class 0). The 30
 * remaining orbits are numbered 1-30 in ESCHER's (MoCHy's) order by the list of their canonical
 * ids, triad_class_representatives():
 *
 * | Class | Canonical id | Non-empty regions | Class | Canonical id | Non-empty regions |
 * |---|---|---|---|---|---|
 * | 1 | 64 | ABC | 16 | 95 | ABC, AB, BC, A, B, C |
 * | 2 | 65 | ABC, A | 17 | 120 | ABC, AB, BC, CA |
 * | 3 | 67 | ABC, A, B | 18 | 121 | ABC, AB, BC, CA, A |
 * | 4 | 71 | ABC, A, B, C | 19 | 123 | ABC, AB, BC, CA, A, B |
 * | 5 | 72 | ABC, AB | 20 | 127 | all seven |
 * | 6 | 73 | ABC, AB, A | 21 | 24 | AB, BC (open) |
 * | 7 | 76 | ABC, AB, C | 22 | 26 | AB, BC, B (open) |
 * | 8 | 75 | ABC, AB, A, B | 23 | 25 | AB, BC, A (open) |
 * | 9 | 77 | ABC, AB, A, C | 24 | 27 | AB, BC, A, B (open) |
 * | 10 | 79 | ABC, AB, A, B, C | 25 | 29 | AB, BC, A, C (open) |
 * | 11 | 88 | ABC, AB, BC | 26 | 31 | AB, BC, A, B, C (open) |
 * | 12 | 90 | ABC, AB, BC, B | 27 | 56 | AB, BC, CA |
 * | 13 | 89 | ABC, AB, BC, A | 28 | 57 | AB, BC, CA, A |
 * | 14 | 91 | ABC, AB, BC, A, B | 29 | 59 | AB, BC, CA, A, B |
 * | 15 | 93 | ABC, AB, BC, A, C | 30 | 63 | AB, BC, CA, A, B, C |
 *
 * (ABC means the region "A, B and C", AB the region "A and B only", A the region "A only".) Class k is bin k - 1 of the 30-bin
 * counts. Classes 21-26 (bins 20-25) are the open triads, where one pair does not intersect; a
 * triangle of the line graph never is one, so those bins are always 0, and 24 bins (0-19 and
 * 26-29) can be reached.
 *
 * It is an oracle (PLAN Section 5.8): it uses the C++ standard library (and OpenMP, when built
 * with it) only, shares no code with triad_count, its engines or the hypergraph container, and
 * includes no other project header; an include-graph and link test (triad_reference_isolation)
 * checks this. It is meant for test sizes (and the parity harness): the line graph is held on the
 * host and every triangle is visited.
 */
#pragma once

#include <array>
#include <cstdint>
#include <vector>

namespace dyng::testing {

/**
 * @brief The triad patterns the oracle counts (the oracle's own list, so that it does not depend
 *        on the algorithm it checks).
 * @ingroup testing
 */
enum class triad_pattern : std::uint8_t {
  h_motif,          ///< the 30-bin h-motif counts of the line graph's triangles
  stathyper_type1,  ///< StatHyper type 1: sum over the hyperedges of C(|e|, 3)
  stathyper_type2,  ///< StatHyper type 2: adjacent pairs, C(a, 2) * b * c
  stathyper_type3,  ///< StatHyper type 3: triangles, x * y * z of the pairwise-only overlaps
};

/**
 * @brief Number of h-motif classes (and bins of the h_motif pattern).
 * @ingroup testing
 */
inline constexpr int triad_num_classes = 30;

/**
 * @brief Number of Venn ids (7 region bits).
 * @ingroup testing
 */
inline constexpr int triad_num_venn_ids = 128;

/**
 * @brief The name of a pattern ("h_motif", "stathyper_type1", ...).
 * @param[in] pattern The pattern.
 * @return Its name.
 * @ingroup testing
 */
[[nodiscard]] const char* triad_pattern_name(triad_pattern pattern) noexcept;

/**
 * @brief Number of values a pattern counts: 30 for h_motif, 1 for the StatHyper types.
 * @param[in] pattern The pattern.
 * @return The number of bins.
 * @ingroup testing
 */
[[nodiscard]] int triad_num_bins(triad_pattern pattern) noexcept;

/**
 * @brief The canonical id of each h-motif class in ESCHER's numbering: element k - 1 is the
 *        smallest Venn id of class k (the table of the file comment).
 * @return The 30 representatives.
 * @ingroup testing
 */
[[nodiscard]] const std::array<std::uint8_t, triad_num_classes>&
triad_class_representatives() noexcept;

/**
 * @brief The Venn id after a permutation of (A, B, C).
 * @param[in] venn_id An id in [0, 128).
 * @param[in] image   image[x] is the position (0 = A, 1 = B, 2 = C) hyperedge x moves to; a
 *                    permutation of {0, 1, 2}.
 * @return The id of the permuted triple.
 * @throws std::invalid_argument if venn_id is out of range or image is not a permutation.
 * @ingroup testing
 */
[[nodiscard]] int triad_venn_permuted(int venn_id, const std::array<int, 3>& image);

/**
 * @brief The canonical id of a Venn id: the smallest id over the six permutations of (A, B, C).
 * @param[in] venn_id An id in [0, 128).
 * @return The canonical id.
 * @throws std::invalid_argument if venn_id is out of range.
 * @ingroup testing
 */
[[nodiscard]] int triad_venn_canonical(int venn_id);

/**
 * @brief The h-motif class of a Venn id: 1-30, or 0 when fewer than two pairs intersect.
 * @param[in] venn_id An id in [0, 128).
 * @return The class.
 * @throws std::invalid_argument if venn_id is out of range.
 * @ingroup testing
 */
[[nodiscard]] int triad_motif_class(int venn_id);

/**
 * @brief The class of every Venn id (triad_motif_class() over [0, 128)).
 * @return The 128 classes.
 * @ingroup testing
 */
[[nodiscard]] std::array<std::int8_t, triad_num_venn_ids> triad_motif_class_map();

/**
 * @brief The Venn id of three member sets.
 * @param[in] a The members of A (any order; repeated members count once).
 * @param[in] b The members of B.
 * @param[in] c The members of C.
 * @return The id (bit 0 "A only" ... bit 6 "A, B and C").
 * @throws std::bad_alloc if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] int triad_venn_id(const std::vector<std::int32_t>& a,
                                const std::vector<std::int32_t>& b,
                                const std::vector<std::int32_t>& c);

/**
 * @brief How the oracle runs.
 * @ingroup testing
 */
struct triad_reference_options {
  /// Threads: 1 runs sequentially, 0 uses OpenMP's default (sequential in a build without
  /// OpenMP). Every thread count gives the same counts.
  int threads = 0;
};

/**
 * @brief The counts of all four patterns.
 * @ingroup testing
 */
struct triad_reference_counts {
  std::array<std::uint64_t, triad_num_classes> h_motif{};  ///< bin k - 1 = class k
  std::uint64_t type1 = 0;                                 ///< StatHyper type 1
  std::uint64_t type2 = 0;                                 ///< StatHyper type 2
  std::uint64_t type3 = 0;                                 ///< StatHyper type 3

  /**
   * @brief The counts of one pattern.
   * @param[in] pattern The pattern.
   * @return 30 values for h_motif, else 1.
   */
  [[nodiscard]] std::vector<std::uint64_t> of(triad_pattern pattern) const;
};

/**
 * @brief The counts of the requested patterns of a hypergraph (the others stay 0).
 *
 * The hypergraph is given by id: id i has the members `members[offsets[i] .. offsets[i + 1])`
 * (the layout of hypergraph::h2v() or of a hyperedge_list, copied to the host); an empty row is a
 * dead id and takes part in nothing. Repeated members of a row count once. The line graph is
 * built only when a requested pattern needs it (all but type 1).
 * @param[in] offsets      Row offsets (ids + 1 values, from 0 to members.size()).
 * @param[in] members      The members, in [0, num_vertices).
 * @param[in] num_vertices Vertices.
 * @param[in] patterns     The patterns to count.
 * @param[in] options      Threads.
 * @return The counts.
 * @throws std::invalid_argument for malformed offsets, a member out of range, more than 2^31 - 1
 *         ids or a negative vertex count.
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] triad_reference_counts triad_reference_all(
    const std::vector<std::int64_t>& offsets, const std::vector<std::int32_t>& members,
    std::int32_t num_vertices, const std::vector<triad_pattern>& patterns,
    const triad_reference_options& options = {});

/**
 * @brief The counts of one pattern of a hypergraph (triad_reference_all() for one pattern).
 * @param[in] offsets      Row offsets (ids + 1 values).
 * @param[in] members      The members, in [0, num_vertices).
 * @param[in] num_vertices Vertices.
 * @param[in] pattern      The pattern.
 * @param[in] options      Threads.
 * @return 30 values for h_motif, else 1.
 * @throws std::invalid_argument as triad_reference_all().
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] std::vector<std::uint64_t> triad_reference(
    const std::vector<std::int64_t>& offsets, const std::vector<std::int32_t>& members,
    std::int32_t num_vertices, triad_pattern pattern, const triad_reference_options& options = {});

/**
 * @brief removed(G, D): per bin, the triads of G that contain a hyperedge of D, as
 *        counts(G) - counts(G \ D) (for the statistics of small test sets).
 * @param[in] offsets      G's row offsets by id.
 * @param[in] members      G's members.
 * @param[in] num_vertices Vertices.
 * @param[in] ids          D: ids of G (dead, out-of-range and repeated ids are ignored).
 * @param[in] pattern      The pattern.
 * @param[in] options      Threads.
 * @return 30 values for h_motif, else 1.
 * @throws std::invalid_argument as triad_reference_all().
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] std::vector<std::uint64_t> triad_reference_removed(
    const std::vector<std::int64_t>& offsets, const std::vector<std::int32_t>& members,
    std::int32_t num_vertices, const std::vector<std::int64_t>& ids, triad_pattern pattern,
    const triad_reference_options& options = {});

/**
 * @brief added(G', I): per bin, the triads of G' (the hypergraph after a batch) that contain an
 *        inserted hyperedge of I, as counts(G') - counts(G' \ I).
 * @param[in] offsets      G''s row offsets by id.
 * @param[in] members      G''s members.
 * @param[in] num_vertices Vertices.
 * @param[in] ids          I: the inserted ids (hyper_apply_summary::inserted_ids).
 * @param[in] pattern      The pattern.
 * @param[in] options      Threads.
 * @return 30 values for h_motif, else 1.
 * @throws std::invalid_argument as triad_reference_all().
 * @throws std::bad_alloc        if host memory cannot be allocated.
 * @ingroup testing
 */
[[nodiscard]] std::vector<std::uint64_t> triad_reference_added(
    const std::vector<std::int64_t>& offsets, const std::vector<std::int32_t>& members,
    std::int32_t num_vertices, const std::vector<std::int64_t>& ids, triad_pattern pattern,
    const triad_reference_options& options = {});

}  // namespace dyng::testing
