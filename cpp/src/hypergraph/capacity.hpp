// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file capacity.hpp
 * @brief The public capacity_policy and hypergraph_properties in the store's terms: the
 *        conversion to the internal store_policy, the initial payload of a tree, and the checks of
 *        the properties (ADR 0041, section 6).
 */
#pragma once

#include "hypergraph/escher/store_policy.hpp"
#include "hypergraph/slack_csr/slack_layout.hpp"

#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <cstdint>

namespace dyng::detail::hyper {

/**
 * @brief Check a capacity policy.
 * @param[in] c The policy.
 * @throws invalid_argument_error for a negative or non-finite field.
 * @throws not_supported_error    for a grow_factor other than 1.5 (ESCHER-GPU's rule) or 0.
 */
void validate_capacity(const capacity_policy& c);

/**
 * @brief Check the properties of a new hypergraph (storage, line graph, capacity): the matrix of
 *        HD9 (ADR 0086). escher storage takes the line graphs none, automatic (maintained as
 *        sorted_merge) and sorted_merge; slack_csr storage takes none, automatic (maintained as
 *        slack_csr) and slack_csr, and needs v2h.
 * @param[in] props The properties.
 * @throws invalid_argument_error for an invalid capacity policy, sorted_merge without v2h, or
 *         slack_csr storage without v2h.
 * @throws not_supported_error    for escher storage with the slack_csr line graph, slack_csr
 *         storage with sorted_merge (each message names the alternative), an unsupported
 *         grow_factor.
 */
void validate_properties(const hypergraph_properties& props);

/**
 * @brief Check the internal knobs of a slack_csr hypergraph.
 * @param[in] k The knobs.
 * @throws invalid_argument_error for an entry headroom below 1 or NaN, a negative node capacity
 *         or candidate limit.
 */
void validate_knobs(const slack::knobs& k);

/**
 * @brief The store policy that a capacity policy maps to (escher_gpu(), mosp_escher(extra) and
 *        the default give the presets of the same names: growth, compaction, scratch, alignment,
 *        hysteresis and growth bound).
 * @param[in] c The (validated) capacity policy.
 * @return The store policy; its initial sizing is not used (the container sizes its trees with
 *         initial_payload()).
 */
[[nodiscard]] escher::store_policy to_store_policy(const capacity_policy& c);

/**
 * @brief The initial payload capacity of a tree updated in place:
 *        `min(max(min_initial_payload, size + floor(size * initial_headroom) + initial_slack),
 *        max_payload)`.
 * @param[in] c    The (validated) capacity policy.
 * @param[in] size The payload entries the tree's rows take.
 * @return The capacity (>= size).
 * @throws capacity_error if the capacity is below `size` (the rows exceed max_payload).
 */
[[nodiscard]] std::int64_t initial_payload(const capacity_policy& c, std::int64_t size);

/**
 * @brief The largest payload of one tree under a policy.
 * @param[in] c The policy.
 * @return max_payload, or 2^31 - 1 (the int32 offsets) when it is 0 or larger.
 */
[[nodiscard]] std::int64_t payload_limit(const capacity_policy& c) noexcept;

}  // namespace dyng::detail::hyper
