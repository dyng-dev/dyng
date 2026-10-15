// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file capacity.cpp
 * @brief capacity_policy in the store's terms (ADR 0041, section 6).
 */
#include "hypergraph/capacity.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <climits>
#include <cmath>
#include <string>

namespace dyng::detail::hyper {

void validate_capacity(const capacity_policy& c) {
  DYNG_EXPECTS(std::isfinite(c.initial_headroom) && c.initial_headroom >= 0.0,
               "capacity_policy::initial_headroom must be finite and >= 0, got ",
               c.initial_headroom);
  DYNG_EXPECTS(c.initial_slack >= 0, "capacity_policy::initial_slack must be >= 0, got ",
               c.initial_slack);
  DYNG_EXPECTS(c.min_initial_payload >= 0,
               "capacity_policy::min_initial_payload must be >= 0, got ", c.min_initial_payload);
  DYNG_EXPECTS(c.max_payload >= 0, "capacity_policy::max_payload must be >= 0, got ",
               c.max_payload);
  DYNG_EXPECTS(c.row_alignment >= 1, "capacity_policy::row_alignment must be >= 1, got ",
               c.row_alignment);
  DYNG_EXPECTS(c.compaction_headroom >= 0,
               "capacity_policy::compaction_headroom must be >= 0, got ", c.compaction_headroom);
  DYNG_EXPECTS(std::isfinite(c.grow_factor), "capacity_policy::grow_factor must be finite");
  if (c.grow_factor != 1.5 && c.grow_factor != 0.0) {
    throw not_supported_error("dyng: capacity_policy::grow_factor " +
                              std::to_string(c.grow_factor) +
                              " is not supported: this release grows a full payload by 1.5 "
                              "(ESCHER-GPU's rule) or never (grow_factor 0)");
  }
}

void validate_properties(const hypergraph_properties& props) {
  if (props.storage == hypergraph_storage::slack_csr) {
    throw not_supported_error(
        "dyng: hypergraph_storage::slack_csr is planned for 0.4; use hypergraph_storage::escher");
  }
  DYNG_EXPECTS(props.storage == hypergraph_storage::escher,
               "hypergraph_properties::storage has an unknown value");
  switch (props.line_graph) {
    case line_graph_policy::none:
    case line_graph_policy::automatic:
      break;
    case line_graph_policy::sorted_merge:
      DYNG_EXPECTS(props.store_v2h,
                   "hypergraph_properties: line_graph_policy::sorted_merge needs v2h (the merge "
                   "reads the vertices' hyperedges); set store_v2h = true");
      break;
    case line_graph_policy::slack_csr:
      throw not_supported_error(
          "dyng: line_graph_policy::slack_csr is planned for 0.4; use line_graph_policy::none "
          "or automatic");
    default:
      DYNG_FAIL("hypergraph_properties::line_graph has an unknown value");
  }
  DYNG_EXPECTS(props.reuse == id_reuse::erase_first || props.reuse == id_reuse::insert_first,
               "hypergraph_properties::reuse has an unknown value");
  validate_capacity(props.capacity);
}

std::int64_t payload_limit(const capacity_policy& c) noexcept {
  return c.max_payload > 0 ? std::min<std::int64_t>(c.max_payload, INT_MAX) : INT_MAX;
}

escher::store_policy to_store_policy(const capacity_policy& c) {
  escher::store_policy p;
  if (c.grow_factor == 0.0) {
    p.growth = escher::store_policy::growth_rule::none;
    // A payload that never grows is bounded by its initial capacity: max_payload only bounds the
    // initial sizing (initial_payload()).
    p.max_payload = INT_MAX;
  } else {
    p.growth = escher::store_policy::growth_rule::escher_gpu;
    p.max_payload = payload_limit(c);
  }
  p.compaction = c.compact_before_grow ? escher::store_policy::compaction_rule::mosp_escher
                                       : escher::store_policy::compaction_rule::never;
  p.scratch = c.exact_scratch ? escher::store_policy::scratch_rule::exact
                              : escher::store_policy::scratch_rule::geometric;
  p.alignment = c.row_alignment;
  p.compaction_headroom = c.compaction_headroom;
  p.headroom = 0.0;  // the container sizes its trees itself (initial_payload())
  p.extra = 0;
  const capacity_policy a = capacity_policy::escher_gpu(c.min_initial_payload);
  const capacity_policy b = capacity_policy::mosp_escher(c.initial_slack);
  const capacity_policy d{};
  const auto same = [](const capacity_policy& x, const capacity_policy& y) {
    return x.initial_headroom == y.initial_headroom && x.initial_slack == y.initial_slack &&
           x.min_initial_payload == y.min_initial_payload && x.grow_factor == y.grow_factor &&
           x.compact_before_grow == y.compact_before_grow && x.max_payload == y.max_payload &&
           x.row_alignment == y.row_alignment && x.compaction_headroom == y.compaction_headroom &&
           x.exact_scratch == y.exact_scratch;
  };
  p.name = same(c, a)   ? "escher_gpu"
           : same(c, b) ? "mosp_escher"
           : same(c, d) ? "dyng_default"
                        : "capacity_policy";
  return p;
}

std::int64_t initial_payload(const capacity_policy& c, std::int64_t size) {
  // size + floor(size * h) equals ESCHER-GPU's size + size / 8 for h = 0.125 and MOSP_ESCHER's
  // floor(size * 1.3) for h = 0.3 for every size below 2^31 (CapacityPolicy tests).
  // Each term saturates at the payload limit (at most 2^31 - 1), so a huge initial_headroom or
  // initial_slack clamps to max_payload as documented instead of overflowing (M8b review: the
  // double-to-int64 conversion and the sum were undefined beyond int64).
  const std::int64_t limit = payload_limit(c);
  const double scaled = static_cast<double>(size) * c.initial_headroom;
  const std::int64_t slack =
      scaled < static_cast<double>(limit) ? static_cast<std::int64_t>(scaled) : limit;
  std::int64_t capacity = size + slack + std::min(c.initial_slack, limit);
  capacity = std::max(capacity, c.min_initial_payload);
  capacity = std::min(capacity, limit);
  if (capacity < size) {
    throw capacity_error("dyng: hypergraph: a tree of " + std::to_string(size) +
                         " payload entries exceeds the largest payload of its capacity policy (" +
                         std::to_string(payload_limit(c)) + " entries)");
  }
  return capacity;
}

}  // namespace dyng::detail::hyper
