// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file update_arguments.cpp
 * @brief Compile-failure tests of dyng::update()'s argument checks: compiled with -fsyntax-only
 *        and one of the defines below (cpp/tests/CMakeLists.txt checks the message); without a
 *        define it compiles.
 *
 *   DYNG_CF_OWNING_BATCH     the owning edge_batch instead of its view (batch.view())
 *   DYNG_CF_NOT_A_CONTAINER  a type that is not a container of dyng::update()
 */
#include <dyng/core/resources.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/sssp.hpp>
#include <dyng/update.hpp>

#include <cstdint>

void call(const dyng::resources& res, dyng::graph<std::int32_t, std::int32_t, std::int32_t>& g,
          const dyng::edge_batch<std::int32_t, std::int32_t>& b,
          dyng::sssp::result<std::int32_t>& r) {
#if defined(DYNG_CF_OWNING_BATCH)
  (void)dyng::update(res, g, b, r);
#elif defined(DYNG_CF_NOT_A_CONTAINER)
  int not_a_graph = 0;
  (void)dyng::update(res, not_a_graph, b.view(), r);
#else
  (void)dyng::update(res, g, b.view(), r);
#endif
}
