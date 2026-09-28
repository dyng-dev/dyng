// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_int64_vertices.cpp
 * @brief Must NOT compile: cycle_count on a graph with int64_t vertex ids (a type sssp is
 *        instantiated for). The CTest cycle_count.static_assert.* requires the static_assert's
 *        message, not a link error. DYNG_COMPILE_FAIL_UPDATE selects dyng::update() instead of
 *        cycle_count::compute().
 */
#include <dyng/cycle_count.hpp>
#include <dyng/update.hpp>

#include <cstdint>

int main() {
  using graph_t = dyng::graph<std::int64_t, std::int64_t, std::int32_t>;
  const dyng::resources res = dyng::resources::sequential();
  graph_t g;
#if defined(DYNG_COMPILE_FAIL_UPDATE)
  dyng::cycle_count::result* hist = nullptr;
  dyng::edge_batch<std::int64_t, std::int32_t> batch;
  (void)dyng::update(res, g, batch.view(), *hist);
#else
  (void)dyng::cycle_count::compute(res, g);
#endif
  return 0;
}
