// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file escher_dump.hpp
 * @brief The batch dumps of ESCHER-GPU@abcf9b2 (A) on the parity side (M8b, criterion 14; PLAN
 *        6.4.6 and 8.3): a decoder of the batch records of A's CBST traces (M8a, ADR 0036; its
 *        `escher_gpu.build` and `escher_gpu.update` records), A's command line as the run used it,
 *        and the by-content resolver of deletions.
 *
 * Ids are dynG's: A's vertex v is vertex v - 1 and A's key k is hyperedge id k - 1.
 *
 * **Replay by content** (PLAN 6.4.6). A's generator chooses its deletions by key in the state it
 * reads, so a replay that gave other ids than A would delete other hyperedges by key; the dump
 * also carries the member set of every deleted hyperedge (`deletedRows`), and by_content_resolver
 * turns each set into a live id of the replaying hypergraph: **the smallest live id with that
 * (sorted, deduplicated) member set that the batch has not taken yet**. The choice is
 * deterministic and needs no ids from A; when several live hyperedges share the set it can differ
 * from A's key (an "ambiguous" resolution), which changes ids but not the live member sets.
 *
 * Host-only and dynG-free (the standard library and the trace reader), so the converter of
 * tools/compat/escher links it without the test libraries.
 */
#pragma once

#include "parity/escher/cbst_trace.hpp"

#include <cstdint>
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

namespace dyng::test::escher_dump {

/// Rows of vertex ids (one hyperedge each).
using rows = std::vector<std::vector<std::int32_t>>;

/// The parts of ESCHER-GPU's command line (utils/options.cpp) that shape its hypergraph and
/// batches.
struct escher_options {
  bool from_file = false;                  ///< --input FILE (else the synthetic generator)
  std::int64_t num_hyperedges = 0;         ///< synthetic: n
  std::int32_t max_vertices_per_edge = 1;  ///< synthetic: m (per-hyperedge size in [1, m])
  std::int32_t min_vertex_id = 1;          ///< synthetic: r1 (1-based)
  std::int32_t max_vertex_id = 1;          ///< synthetic: r2 (1-based)
  std::int64_t payload_capacity = 0;       ///< [payload_capacity]
  std::int64_t batches = 1;                ///< --batches
  bool composition_set = false;            ///< --batch-size / --deletions / --insertions given
  std::int64_t deletions = 0;              ///< deletions per batch (when composition_set)
  std::int64_t insertions = 0;             ///< insertions per batch (when composition_set)
  bool sample_inserts = false;             ///< --insert-mode sample (the default with --input)
  std::uint64_t seed = 1;                  ///< --seed: the single std::mt19937_64

  /**
   * @brief Parse A's argv (as the trace's header records it) with A's rules.
   * @param[in] argv The command line, program name first.
   * @return The options.
   * @throws cbst_trace::trace_error if the command line is not one A accepts.
   */
  static escher_options parse(const std::vector<std::string>& argv);
};

/// A's `DynamicHypergraph::build`: the loaded hypergraph.
struct build_record {
  std::int32_t num_vertices = 0;  ///< A's maxVertexId (vertices 0 .. n - 1 in dynG)
  rows loaded;                    ///< the hyperedges by id (0-based vertices)
  std::int64_t num_hyperedge_keys = 0;
  std::int64_t device_bytes = -1;  ///< A's deviceBytes() after the build (-1: not recorded)
};

/// One of A's batches: what its generator produced and what its CBST insert gave.
struct batch_record {
  std::int64_t index = 0;                  ///< A's batch number (1-based)
  std::vector<std::int64_t> deleted_ids;   ///< deletedKeys - 1 (sorted)
  rows deleted_rows;                       ///< deletedRows (0-based vertices, same order)
  rows inserted_rows;                      ///< insertedRows (0-based vertices, as generated)
  std::vector<std::int64_t> inserted_ids;  ///< insertedKeys - 1
  std::int64_t num_hyperedge_keys = 0;
  std::int64_t device_bytes = -1;  ///< A's deviceBytes() after the batch (-1: not recorded)
};

/**
 * @brief Decodes A's batch records from the records of a trace, in order.
 *
 * feed() every record; it reports the header, a completed build and a completed batch (the
 * update's enter and exit records joined), and an exception of the original.
 */
class decoder {
 public:
  /// What a record completed.
  enum class event { none, header, build, batch, exception };

  /**
   * @brief Take the next record of the trace.
   * @param[in]     rec The record.
   * @param[in,out] in  The trace reader (for side-file arrays).
   * @return What it completed.
   * @throws cbst_trace::trace_error on a malformed record.
   */
  event feed(const cbst_trace::record& rec, cbst_trace::reader& in);

  /// The repository of the trace's header ("ESCHER-GPU" for A's traces).
  [[nodiscard]] const std::string& repo() const noexcept {
    return repo_;
  }
  /// The command line of the trace's header.
  [[nodiscard]] const std::vector<std::string>& argv() const noexcept {
    return argv_;
  }
  /// The last completed build.
  [[nodiscard]] const build_record& build() const noexcept {
    return build_;
  }
  /// The last completed batch.
  [[nodiscard]] const batch_record& batch() const noexcept {
    return batch_;
  }
  /// The type of the original's exception (event::exception).
  [[nodiscard]] const std::string& exception() const noexcept {
    return exception_;
  }

 private:
  std::string repo_;
  std::vector<std::string> argv_;
  build_record build_;
  batch_record batch_;
  std::string exception_;
};

/**
 * @brief The rows of a batch record's array pair `name.values` / `name.prefix` (inclusive
 *        prefix ends), minus `shift` (1 for A's 1-based vertices).
 * @param[in,out] in    The trace reader.
 * @param[in]     rec   The record.
 * @param[in]     name  The field name.
 * @param[in]     shift Subtracted from every value.
 * @return The rows.
 */
rows read_rows(cbst_trace::reader& in, const cbst_trace::record& rec, const std::string& name,
               int shift);

/**
 * @brief Resolves deleted member sets to live hyperedge ids (the rule of the file comment) and
 *        follows the replaying hypergraph's state.
 */
class by_content_resolver {
 public:
  /**
   * @brief Start from the hyperedges by id (an empty row is a dead id).
   * @param[in] by_id The rows.
   */
  void reset(const rows& by_id);

  /**
   * @brief The live ids of a batch's deleted member sets, in order: each the smallest live id
   *        with that member set not taken earlier in the batch, or -1 if there is none.
   * @param[in]  deleted_rows The member sets (any order of members, repeats allowed).
   * @param[out] ambiguous    Optional: per set, whether more than one live id had it.
   * @return The ids.
   */
  std::vector<std::int64_t> resolve(const rows& deleted_rows,
                                    std::vector<bool>* ambiguous = nullptr) const;

  /**
   * @brief Record a commit: the deleted ids are gone, the inserted rows live at their ids.
   * @param[in] deleted_ids  The ids deleted (resolve()'s; -1 entries are ignored).
   * @param[in] inserted     The inserted member sets.
   * @param[in] inserted_ids Their ids (hyper_apply_summary::inserted_ids).
   */
  void commit(const std::vector<std::int64_t>& deleted_ids, const rows& inserted,
              const std::vector<std::int64_t>& inserted_ids);

  /// The live hyperedges.
  [[nodiscard]] std::size_t size() const noexcept {
    return content_of_.size();
  }

 private:
  std::map<std::vector<std::int32_t>, std::set<std::int64_t>> ids_of_;
  std::unordered_map<std::int64_t, std::vector<std::int32_t>> content_of_;
};

/// The sorted, deduplicated copy of a row.
std::vector<std::int32_t> normalized(std::vector<std::int32_t> row);

}  // namespace dyng::test::escher_dump
