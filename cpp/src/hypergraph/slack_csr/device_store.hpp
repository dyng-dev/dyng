// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file device_store.hpp
 * @brief The store of slack_csr storage on the CUDA backend (ADR 0089): MOSP_ESCHER's H-SOSP
 *        device graph (the incidence mirror, the slack CSR line graph with its node weights,
 *        the order-free pair delta on the device; slack_csr_store.hpp) over the host incidence
 *        model and id allocator that stay authoritative, as MOSP_ESCHER keeps its HostHypergraph.
 *
 * Host-only: the factories are defined in device_store.cu, and throw not_supported_error in a
 * build without CUDA (hypergraph.cpp).
 */
#pragma once

#include "hypergraph/slack_csr/slack_layout.hpp"
#include "hypergraph/store.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <cstdint>
#include <memory>

namespace dyng::detail::hyper {

/**
 * @brief A slack_csr store on the device of `res` over `rows` (row i is hyperedge id i; sorted,
 *        distinct, in range, non-empty). The line graph is built by enable_line_graph().
 * @param[in] res          CUDA resources.
 * @param[in] num_vertices Vertices.
 * @param[in] rows         The rows.
 * @param[in] capacity     The capacity policy (the allocator's presets).
 * @param[in] knobs        The internal knobs (ADR 0086).
 * @return The store.
 * @throws not_supported_error in a build without CUDA.
 */
[[nodiscard]] std::unique_ptr<incidence_store> make_slack_device_store(
    const resources& res, std::int32_t num_vertices, const member_rows& rows,
    const capacity_policy& capacity, const slack::knobs& knobs);

/**
 * @brief A slack_csr store on the device of `res` from a snapshot of any backend's slack_csr
 *        store (the same ids, rows and allocator state; the line graph rebuilt when the snapshot
 *        maintains one).
 * @param[in] res      CUDA resources.
 * @param[in] snapshot The state.
 * @param[in] capacity The capacity policy.
 * @param[in] knobs    The internal knobs.
 * @return The store.
 * @throws not_supported_error in a build without CUDA.
 */
[[nodiscard]] std::unique_ptr<incidence_store> make_slack_device_store(
    const resources& res, const store_snapshot& snapshot, const capacity_policy& capacity,
    const slack::knobs& knobs);

}  // namespace dyng::detail::hyper
