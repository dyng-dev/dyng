// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_results.hpp
 * @brief The result side of hyper_sssp's parity (M11a criterion 14; HD2; ADR 0091): the sessions
 *        of MOSP_ESCHER@4b86159's H-SOSP dumps (the golden set hyper_sssp_mosp_escher, ADR 0081)
 *        replayed WHOLE through hyper_sssp, every hsospUpdate and hsospRecompute B recorded
 *        compared with dynG's result.
 *
 * The container is loaded and every batch converted as the by-id container replay
 * (hsosp_replay.hpp) does it: B's rows with its virtual source and target rows at their positions
 * (weight 1 in the container; dynG id = B id - 1), slack_csr storage and line graph,
 * id_reuse::insert_first, capacity_policy::mosp_escher(extra), B's CSR entry headroom and node
 * capacity. The operations naming a virtual row are filtered (B skips them). A batch is applied
 * by the update of the results B updates after it (one hyper_sssp::update, or one dyng::update of
 * two), each with B's UpdateConfig of that call (work_budget, max_iterations, the recompute
 * kernels' block size); a batch no result follows is applied with hypergraph::apply. A recompute
 * B recorded is a recompute of that state into its own arrays (the bench's static row). The modes:
 *  - by_id: hyper_sssp's internal virtual-endpoint mode (source = B's source row, the source and
 *    target rows weigh 0 inside hyper_sssp): distances and parents equal B's by id, the canonical
 *    encodings of ADR 0080 (full arrays, or the digest of a digest case);
 *  - by_id_public: the same container, the public API from B's source vertex s: the distances
 *    equal B's on every real hyperedge; a parent differs only where B's parent is its source row
 *    (dynG: -1) or its target row (counted), and every parent is checked against the oracle's
 *    canonical rule (ADR 0082) from the distances and the host incidence;
 *  - by_key: a container loaded as a user would (no virtual rows, id_reuse::erase_first,
 *    slack_csr storage), the public API, B's batches resolved by content (B's ids tracked to
 *    dynG's: the loaded order, then every batch's inserted ids): distances equal B's per
 *    hyperedge, parents form the canonical tree (the same check).
 * With check_compute, every update is also compared with a fresh compute() (update == compute,
 * bitwise). Every call's digests are reported (sequential == cuda, and the repeat runs).
 */
#pragma once

#include "parity/hsosp/hsosp_dump.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::test::hsosp_results {

/// The parity mode (HD2).
enum class mode { by_id, by_id_public, by_key };

/// What a replay checks.
struct options {
  hsosp_results::mode mode = mode::by_id;
  bool check_compute = false;  ///< every update also against a fresh compute()
  /// The CUDA engine of every compute() (M11b: engine::operators replays the shared engine).
  engine cuda_engine = engine::automatic;
  /// The shared engine's knobs of every result (ADR 0095; parse_operators_knobs' text, or empty).
  std::string operators_knobs;
  std::size_t max_mismatches = 20;
};

/// One call's digests (the canonical encodings over the call's own id bound).
struct call_digest {
  std::int64_t session = 0, state = 0, batches = 0;
  std::string kind, distances, parents;
  bool b_fallback = false, fallback = false;  ///< B's and dynG's fallback decision (updates)
  std::string reason;                         ///< dynG's fallback_reason
};

/// The result of one dump.
struct report {
  std::string file, backend, mode;
  std::int64_t sessions = 0;             ///< sessions replayed
  std::int64_t stopped_sessions = 0;     ///< sessions not replayed to their end (must be 0)
  std::int64_t batches = 0;              ///< batches applied
  std::int64_t updates = 0;              ///< B updates compared
  std::int64_t recomputes = 0;           ///< B recomputes compared
  std::int64_t digest_calls = 0;         ///< calls compared by digest (a digest case)
  std::int64_t ids_compared = 0;         ///< distances compared
  std::int64_t original_exceptions = 0;  ///< calls B rejected (dynG must reject them too)
  std::int64_t source_row_parents = 0;   ///< by_id_public: B's parent its source row (dynG -1)
  std::int64_t target_row_parents = 0;   ///< by_id_public: B's parent its target row
  std::int64_t canonical_checked = 0;    ///< public modes: ids checked by the oracle
  std::int64_t keys_compared = 0;        ///< by_key: inserted hyperedges checked by content
  std::int64_t compute_checked = 0;      ///< updates compared with a fresh compute()
  std::int64_t fallback_equal = 0;       ///< updates whose fallback decision equals B's (HD7)
  std::int64_t fallback_differ = 0;      ///< (tabulated, not a parity target)
  std::int64_t dyng_fallbacks = 0, b_fallbacks = 0;  ///< the decisions' counts
  std::int64_t not_checked = 0;     ///< calls on a state the replay could not follow (a failure)
  std::int64_t digest_skipped = 0;  ///< public modes: digest-only calls (full arrays are T, P50)
  std::int64_t mismatch_count = 0;
  std::vector<std::string> mismatches;
  std::vector<call_digest> calls;
  double seconds = 0;
  /// @brief No mismatch, every session to its end and every call of B's checked. @return The
  /// verdict.
  [[nodiscard]] bool ok() const {
    return mismatch_count == 0 && stopped_sessions == 0 && not_checked == 0;
  }
  /// @brief The report as JSON (with the per-call digests). @return The text.
  [[nodiscard]] std::string to_json() const;
};

/**
 * @brief Replays the sessions of one dump through hyper_sssp (see the file comment).
 * @param[in] records The dump (a paper case with its loaded rows spliced back in).
 * @param[in] res     The backend's resources (sequential or cuda).
 * @param[in] opt     The mode and the checks.
 * @param[in] name    The dump's name for the report.
 * @return The report.
 */
report replay(const std::vector<hsosp_dump::record>& records, const resources& res,
              const options& opt, const std::string& name);

/// The mode's name ("by_id", "by_id_public", "by_key").
const char* to_string(mode m) noexcept;

}  // namespace dyng::test::hsosp_results
