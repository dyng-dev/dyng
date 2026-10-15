// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file escher_store.hpp
 * @brief The device store of the hypergraph container on the CUDA backend: the ESCHER CBST
 *        stores of M8a for H2V and V2H behind the internal store interface (ADR 0041, section 4).
 *
 * Host-only: the factories are defined in escher_store.cu, and throw not_supported_error in a
 * build without CUDA.
 */
#pragma once

#include "hypergraph/store.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <cstdint>
#include <memory>

namespace dyng::detail::hyper {

/**
 * @brief A device store over `rows` (row i is hyperedge id i; sorted, distinct, in range,
 *        non-empty): the H2V and V2H trees constructed on the device of `res` (stages
 *        hypergraph.build for the host layout, hypergraph.upload for the trees and the H2V row
 *        view).
 * @param[in] res          CUDA resources.
 * @param[in] num_vertices Vertices.
 * @param[in] rows         The rows.
 * @param[in] capacity     The capacity policy.
 * @param[in] has_v2h      Whether v2h is kept.
 * @param[in] line_graph   Whether the line graph is built and maintained from the start
 *                         (line_graph_policy::sorted_merge).
 * @return The store.
 * @throws not_supported_error in a build without CUDA.
 */
[[nodiscard]] std::unique_ptr<incidence_store> make_escher_store(const resources& res,
                                                                 std::int32_t num_vertices,
                                                                 const member_rows& rows,
                                                                 const capacity_policy& capacity,
                                                                 bool has_v2h, bool line_graph);

/**
 * @brief A device store from the state of another backend's store (the same ids, deleted slots,
 *        slot lengths and payload accounting).
 * @param[in] res      CUDA resources.
 * @param[in] snapshot The state.
 * @param[in] capacity The capacity policy.
 * @return The store.
 * @throws not_supported_error in a build without CUDA.
 */
[[nodiscard]] std::unique_ptr<incidence_store> make_escher_store(const resources& res,
                                                                 const store_snapshot& snapshot,
                                                                 const capacity_policy& capacity);

}  // namespace dyng::detail::hyper
