// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file virtual_endpoints.hpp
 * @brief hyper_sssp's internal virtual-endpoint mode, for the parity harness and the compat tool
 *        only (HD2, decided by the author on 2026-10-28; ADR 0091). No public name.
 *
 * MOSP_ESCHER@4b86159 (B) adds two virtual hyperedges to its hypergraph, a source row {s} (B id 1)
 * and a target row {t} (B id m + 2), both of weight 0, and runs its SSSP from the source ROW. The
 * by-id parity mode loads them as ordinary hyperedges at B's positions (container weight 1, dynG
 * id = B id - 1). This mode runs hyper_sssp as B does: the source is a hyperedge at distance 0
 * (it is never relaxed), and up to two ids weigh 0 inside hyper_sssp (its weight reads: B's source
 * and target rows), while the container keeps their weight 1. Everything else (the update, the
 * budget, the engines) is the public algorithm's. Also here: the internal knobs of B's
 * UpdateConfig that are not options (the recompute kernels' block size), the static recompute
 * into an existing result (the bench's static row: no allocation), and the engine's own counters
 * of the last call (logged by the harness, never compared: HD7).
 */
#pragma once

#include "algorithms/hyper_sssp/problem.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>

namespace dyng::detail::hyper_sssp_virtual {

/**
 * @brief compute() in the virtual-endpoint mode.
 * @param[in] res        Execution resources (sequential or cuda).
 * @param[in] h          The hypergraph (as hyper_sssp::compute()).
 * @param[in] endpoints  source_hyperedge (a live id, distance 0) and zero_weight_a / _b (ids, or
 *                       -1); source_vertex is ignored.
 * @param[in] opt        Options.
 * @param[in] block_size The recompute kernels' threads per block (B's UpdateConfig::blockSize):
 *                       a multiple of 32 in [32, 1024].
 * @return The result (its source() is -1: it has no source vertex).
 * @throws invalid_argument_error for a source row outside the ids or a bad block size; otherwise
 *         as hyper_sssp::compute().
 */
hyper_sssp::result<std::int32_t> compute(const resources& res,
                                         const hypergraph<std::int32_t, std::int32_t>& h,
                                         const hyper_sssp_endpoints& endpoints,
                                         const hyper_sssp::options& opt, int block_size = 256);

/**
 * @brief Recompute `r` from scratch on `h` into its own arrays (hsospRecompute into B's
 *        pre-allocated state B: the bench's static row). Stage hyper_sssp.compute.
 * @param[in]     res Execution resources.
 * @param[in]     h   The hypergraph `r` belongs to (any version: `r` is stamped with it).
 * @param[in,out] r   A result of compute() (either mode).
 * @throws as hyper_sssp::compute().
 */
void recompute(const resources& res, const hypergraph<std::int32_t, std::int32_t>& h,
               hyper_sssp::result<std::int32_t>& r);

/**
 * @brief Set the recompute kernels' block size of a result (B's UpdateConfig::blockSize of the
 *        next calls; the update kernels always use 256, as B's).
 * @param[in,out] r          The result.
 * @param[in]     block_size A multiple of 32 in [32, 1024] (checked by the next call, as B checks
 *                           it in both entry points before anything changes).
 */
void set_block_size(hyper_sssp::result<std::int32_t>& r, int block_size);

/**
 * @brief The engine's counters of the last compute() / update() / recompute() of `r` (B's
 *        UpdateStats: logged only).
 * @param[in] r The result.
 * @return The counters.
 */
const hyper_sssp_engine_counters& last_counters(const hyper_sssp::result<std::int32_t>& r);

/**
 * @brief The bytes of a result in MOSP_ESCHER's graph_mb scope (its HsospState::deviceBytes():
 *        the memory gate's accounting of the two states, HD6).
 * @param[in] r The result.
 * @return The device bytes (0 for a host result).
 */
std::int64_t state_bytes(const hyper_sssp::result<std::int32_t>& r);

}  // namespace dyng::detail::hyper_sssp_virtual
