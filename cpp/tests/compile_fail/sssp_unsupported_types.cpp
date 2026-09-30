// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_unsupported_types.cpp
 * @brief Must NOT compile: sssp on an unweighted graph (a type cycle_count is instantiated for).
 *        The CTest sssp.static_assert.* requires the static_assert's message, not a link error.
 *        DYNG_COMPILE_FAIL_UPDATE selects sssp::update(), DYNG_COMPILE_FAIL_MULTI dyng::update(),
 *        DYNG_COMPILE_FAIL_DISTANCE a result with another distance type; the default is
 *        sssp::compute().
 */
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <cstdint>

int main() {
  using graph_t = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>;
  const dyng::resources res = dyng::resources::sequential();
  graph_t g;
#if defined(DYNG_COMPILE_FAIL_UPDATE)
  dyng::sssp::result<std::int32_t>* r = nullptr;
  dyng::edge_batch<std::int32_t, dyng::unweighted> batch;
  (void)dyng::sssp::update(res, g, batch.view(), *r);
#elif defined(DYNG_COMPILE_FAIL_MULTI)
  dyng::sssp::result<std::int32_t>* r = nullptr;
  dyng::edge_batch<std::int32_t, dyng::unweighted> batch;
  (void)dyng::update(res, g, batch.view(), *r);
#elif defined(DYNG_COMPILE_FAIL_DISTANCE)
  dyng::sssp::result<std::int32_t, std::int32_t>* r = nullptr;
  (void)r->source();
#else
  (void)dyng::sssp::compute(res, g, 0);
#endif
  return 0;
}
