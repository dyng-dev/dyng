// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_properties.hpp
 * @brief hypergraph_properties: the storage, line-graph, id-reuse and capacity policies of a
 *        hypergraph.
 * @ingroup hypergraph
 */
#pragma once

#include <cstdint>
#include <string_view>

/**
 * @defgroup hypergraph Hypergraphs
 * @brief The dynamic hypergraph container, its properties, views, batches and the summary of an
 *        applied batch.
 */

namespace dyng {

/**
 * @brief How the incidences of a hypergraph are stored.
 * @ingroup hypergraph
 */
enum class hypergraph_storage : std::uint8_t {
  escher,     ///< the ESCHER store (ESCHER, IPDPS'26; best-fit reuse, ESCHER+, TKDE'26)
  slack_csr,  ///< slack rows (H-SOSP's DeviceIncidence); planned for 0.4, not_supported_error now
};

/**
 * @brief How the line graph (h2h: hyperedge to overlapping hyperedges) is kept.
 * @ingroup hypergraph
 */
enum class line_graph_policy : std::uint8_t {
  none,          ///< no line graph: hypergraph::h2h() throws
  automatic,     ///< built when it is first read (hypergraph::h2h(), or an algorithm that
                 ///< reads it), then maintained as sorted_merge
  sorted_merge,  ///< built with the hypergraph and maintained by every batch with ESCHER-GPU's
                 ///< sorted delta merge (needs v2h)
  slack_csr,     ///< H-SOSP's slack h2h CSR; planned for 0.4, not_supported_error now
};

/**
 * @brief The order in which one batch's deletions and insertions reach the store, which decides
 *        the ids that best-fit reuse gives the inserted hyperedges.
 * @ingroup hypergraph
 */
enum class id_reuse : std::uint8_t {
  erase_first,   ///< deletions first (ESCHER's order): a slot freed by the batch can be reused
  insert_first,  ///< insertions first (H-SOSP's order): only slots freed before the batch
};

/**
 * @brief How a hypergraph's storage is sized and grows (the payload of each of its trees).
 *
 * A tree updated in place starts with `min(max(min_initial_payload, size + floor(size *
 * initial_headroom) + initial_slack), max_payload)` payload entries, where `size` is the payload
 * its rows need; capacity_error if that is below `size`. A payload that does not fit is first
 * compacted (compact_before_grow, at MOSP_ESCHER's trigger points), then grows to
 * `max(needed + needed / 4, floor(capacity * grow_factor))` entries, at most max_payload, or
 * raises capacity_error when grow_factor is 0. The two presets reproduce the originals exactly;
 * the default is dynG's policy (compact first, then grow, with headroom after a compaction).
 * @ingroup hypergraph
 */
struct capacity_policy {
  double initial_headroom = 0.125;       ///< slack of a new tree as a fraction of its size
  std::int64_t initial_slack = 1024;     ///< slack entries added to every new tree
  std::int64_t min_initial_payload = 0;  ///< lower bound of the initial payload (0: none)
  double grow_factor = 1.5;              ///< growth of a full payload (1.5), or 0: never grow
  bool compact_before_grow = true;       ///< compact a full payload before growing it
  std::int64_t max_payload = 0;          ///< largest payload per tree (0: 2^31 - 1, int32 offsets)
  std::int32_t row_alignment = 4;  ///< an inserted row takes a multiple of this, plus a terminator
  std::int32_t compaction_headroom = 8;  ///< after a compaction, grow while fewer than
                                         ///< capacity / compaction_headroom entries are free
                                         ///< (0: never)
  bool exact_scratch = false;            ///< operation buffers sized exactly (ESCHER-GPU) or grown
                                         ///< geometrically (MOSP_ESCHER)

  /**
   * @brief ESCHER-GPU\@abcf9b2's policy: trees start at `size + size / 8 + 1024` entries (at
   *        least `min_initial_payload`), never compact, and grow by its rule up to 2^31 - 1.
   * @param[in] min_initial_payload ESCHER-GPU's `--payload-capacity` (a lower bound; 0: none).
   * @return The preset.
   */
  [[nodiscard]] static constexpr capacity_policy escher_gpu(
      std::int64_t min_initial_payload = 0) noexcept {
    capacity_policy p;
    p.initial_headroom = 0.125;
    p.initial_slack = 1024;
    p.min_initial_payload = min_initial_payload;
    p.grow_factor = 1.5;
    p.compact_before_grow = false;
    p.max_payload = 0;
    p.row_alignment = 4;
    p.compaction_headroom = 0;
    p.exact_scratch = true;
    return p;
  }

  /**
   * @brief MOSP_ESCHER\@4b86159's policy: trees start at `floor(size * 1.3) + extra` entries (at
   *        most 2.0e9), compact at its trigger points and never grow (capacity_error).
   * @param[in] extra The extra entries of every tree (MOSP_ESCHER's `Caps::extraPayloadInts`;
   *                  its default 2^20).
   * @return The preset.
   */
  [[nodiscard]] static constexpr capacity_policy mosp_escher(std::int64_t extra = std::int64_t{1}
                                                                                  << 20) noexcept {
    capacity_policy p;
    p.initial_headroom = 0.3;
    p.initial_slack = extra;
    p.min_initial_payload = 0;
    p.grow_factor = 0.0;
    p.compact_before_grow = true;
    p.max_payload = 2000000000;
    p.row_alignment = 4;
    p.compaction_headroom = 0;
    p.exact_scratch = false;
    return p;
  }
};

/**
 * @brief The properties of a hypergraph, fixed at construction.
 * @ingroup hypergraph
 */
struct hypergraph_properties {
  hypergraph_storage storage = hypergraph_storage::escher;      ///< the incidence storage
  line_graph_policy line_graph = line_graph_policy::automatic;  ///< the line graph (h2h)
  id_reuse reuse = id_reuse::erase_first;  ///< the order of a batch (decides the reused ids)
  capacity_policy capacity{};              ///< sizing and growth of the storage
  bool store_v2h = true;                   ///< keep vertex -> hyperedges (v2h(); triad_count)
};

/**
 * @brief The enumerator's name ("escher", "slack_csr").
 * @param[in] storage The value.
 * @return Its name, or "unknown".
 * @ingroup hypergraph
 */
[[nodiscard]] std::string_view to_string(hypergraph_storage storage) noexcept;

/**
 * @brief The enumerator's name ("none", "automatic", "sorted_merge", "slack_csr").
 * @param[in] policy The value.
 * @return Its name, or "unknown".
 * @ingroup hypergraph
 */
[[nodiscard]] std::string_view to_string(line_graph_policy policy) noexcept;

/**
 * @brief The enumerator's name ("erase_first", "insert_first").
 * @param[in] reuse The value.
 * @return Its name, or "unknown".
 * @ingroup hypergraph
 */
[[nodiscard]] std::string_view to_string(id_reuse reuse) noexcept;

}  // namespace dyng
