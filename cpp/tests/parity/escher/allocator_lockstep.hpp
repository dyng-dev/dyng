// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocator_lockstep.hpp
 * @brief The host id allocator (cpp/src/hypergraph/id_allocator) in lockstep with the CBST trees
 *        MOSP_ESCHER@4b86159 recorded (its CBST traces, ADR 0036) and with
 *        dyng::testing::cbst_host_model (M11a criterion 8; ADR 0085).
 *
 * Every operation of a tree of a MOSP_ESCHER trace (construct, insert, erase, fill, unfill,
 * compact) runs on a production id_allocator under the mosp_escher policy and on the host model;
 * after it, the allocator must agree with the state the original recorded and with the model:
 * the keys and their state (live or a deleted slot), the slot length of every key, the value
 * count, the tail occupancy, the tail capacity and whether the tail is an overflow segment of
 * every live key, the bump pointer, the payload capacity, the compactions, every InsertMapping
 * and every exception. The fills and unfills of the vertex-kind batches (the incidence edits of
 * slack_csr storage) are what this adds to M8b's id replays.
 *
 * The allocator models rows that are sets (a hypergraph's H2V rows): an unfill removes values that
 * are members, a fill adds values that are not. The harness keeps every row as the operations
 * write it; an unfill value that is not in its row is dropped before the call (an unfill of an
 * absent value changes nothing in the original either), and a tree whose rows stop being sets is
 * reported (`not_sets`) and no longer compared. Only the trees `trees` selects are run ("h2v": the
 * H2V tree of MOSP_ESCHER's DynamicHypergraph; "all": every tree, the targeted traces').
 */
#pragma once

#include "parity/escher/cbst_trace.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::test::allocator_lockstep {

/// What a lockstep run does.
struct options {
  std::string trees = "h2v";        ///< "h2v" or "all"
  std::size_t max_mismatches = 20;  ///< mismatches recorded in full
};

/// The result of one trace.
struct report {
  std::string trace, repo;
  std::int64_t trees = 0;                  ///< trees run
  std::int64_t ops = 0;                    ///< operations run
  std::int64_t states = 0;                 ///< states compared with the trace and the model
  std::int64_t keys_compared = 0;          ///< key records compared
  std::int64_t inserts = 0;                ///< InsertMappings compared
  std::int64_t fills = 0;                  ///< fill operations
  std::int64_t unfills = 0;                ///< unfill operations
  std::int64_t multi_segment_unfills = 0;  ///< unfill items on a row with an overflow segment
  std::int64_t overflow_fills = 0;         ///< fill items that opened an overflow segment
  std::int64_t compactions = 0;            ///< compactions (the allocator's)
  std::int64_t exceptions = 0;             ///< operations the original rejected (compared)
  std::int64_t not_sets = 0;      ///< trees whose rows stopped being sets (not compared after)
  std::int64_t unreplayable = 0;  ///< operations whose inputs are hashes only
  std::int64_t mismatch_count = 0;
  std::vector<std::string> mismatches;
  double seconds = 0;
  /// @brief No mismatch. @return mismatch_count == 0.
  [[nodiscard]] bool ok() const {
    return mismatch_count == 0;
  }
  /// @brief The report as a JSON object. @return The JSON text.
  [[nodiscard]] std::string to_json() const;
};

/**
 * @brief Runs the trees of one trace in lockstep (see the file comment).
 * @param[in,out] in   The trace.
 * @param[in] opt      What to run.
 * @param[in] name     The trace's name for the report.
 * @return The report.
 */
report run(cbst_trace::reader& in, const options& opt, const std::string& name);

}  // namespace dyng::test::allocator_lockstep
