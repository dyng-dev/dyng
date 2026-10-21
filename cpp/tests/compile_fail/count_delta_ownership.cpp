// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file count_delta_ownership.cpp
 * @brief Compile-failure tests of invariant I2 (exactly-once counting has no default ownership
 *        rule): operators::count_delta and count_owned (operators/count_delta.hpp) and the triad
 *        host engine's phases (algorithms/triad_count/host_engine.hpp) take the rule as a required
 *        argument. Compiled with -fsyntax-only and one of the defines below
 *        (cpp/tests/CMakeLists.txt checks the message); without a define it compiles.
 *
 *   DYNG_CF_DELTA_NO_OWNERSHIP     count_delta without the rule
 *   DYNG_CF_OWNED_NO_OWNERSHIP     count_owned without the rule
 *   DYNG_CF_NOT_AN_OWNERSHIP       count_delta with a value that is not an ownership rule
 *   DYNG_CF_TRIAD_NO_OWNERSHIP     the triad host engine's count_phase without the rule
 */
#include "algorithms/triad_count/host_engine.hpp"
#include "framework/policies.hpp"
#include "operators/count_delta.hpp"
#include "operators/execution.hpp"

#include <dyng/core/resources.hpp>

#include <cstdint>

namespace fw = ::dyng::detail::framework;
namespace ops = ::dyng::detail::operators;

namespace {

struct work {
  template <typename bins_t>
  void operator()(std::int64_t i, fw::ownership::min_member /*rule*/, bins_t& bins) const {
    bins.add(0, static_cast<std::uint64_t>(i));
  }
};

[[maybe_unused]] std::uint32_t use() {
  const ops::sequential_exec exec(dyng::resources::sequential());
  std::uint64_t scratch[2] = {};
  std::int64_t out[1] = {};
  std::uint64_t totals[1] = {};
#if defined(DYNG_CF_DELTA_NO_OWNERSHIP)
  return ops::count_delta<1>(exec, fw::sign::plus, 10, work{}, scratch, out);
#elif defined(DYNG_CF_OWNED_NO_OWNERSHIP)
  return ops::count_owned<1>(exec, 10, work{}, scratch, totals);
#elif defined(DYNG_CF_NOT_AN_OWNERSHIP)
  return ops::count_delta<1>(exec, fw::sign::plus, 1, 10, work{}, scratch, out);
#elif defined(DYNG_CF_TRIAD_NO_OWNERSHIP)
  dyng::detail::hyper::row_views views;
  dyng::detail::triad_phase_bins bins;
  (void)dyng::detail::triad_host::count_phase(exec, dyng::triad_count::pattern::h_motif, views,
                                              nullptr, 0, nullptr, fw::sign::plus, scratch, bins);
  return 0;
#else
  (void)totals;
  return ops::count_delta<1>(exec, fw::sign::plus, fw::ownership::min_member{}, 10, work{}, scratch,
                             out) |
         ops::count_owned<1>(exec, fw::ownership::min_member{}, 10, work{}, scratch, totals);
#endif
}

}  // namespace
