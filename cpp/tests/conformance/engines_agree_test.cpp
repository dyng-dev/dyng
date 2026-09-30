// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file engines_agree_test.cpp
 * @brief The kit's check C4 (fused engine == operators engine) exercised on a host backend: no
 *        0.1 algorithm has both engines on one backend, so C4 is skipped in their suites (ADR
 *        0022). The framework's fake levels_problem has both (Tier A hooks and enact_fused), so
 *        test_traits over it make C4 compare, pass for a correct fused engine, and fail for a
 *        broken one.
 */
#include "conformance/conformance.hpp"
#include "framework/fake_problems.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/edge_batch.hpp>
#include <dyng/graph/graph_properties.hpp>

#include <gtest/gtest-spi.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <string_view>
#include <vector>

namespace {

namespace fakes = ::dyng::test::framework_fakes;
namespace fw = ::dyng::detail::framework;
using graph_t = fakes::graph_type;

/// A levels result that remembers the engine it was computed with (the kit's chain updates it
/// with the same engine).
struct levels_run {
  fakes::levels_result r;  ///< the result
  fakes::hook_log log;     ///< the hook calls (unused)
  bool fused = false;      ///< the engine of compute() and of every update
};

/// test_traits of the fake (the members C4 and the kit's chain use). `broken` corrupts the fused
/// engine's result.
template <bool broken>
struct levels_traits {
  static constexpr std::string_view name = "test_levels";
  static constexpr dyng::oracle_kind oracle = dyng::oracle_kind::compute;
  static constexpr dyng::determinism level = dyng::determinism::bitwise;
  template <typename graph_type>
  using result = levels_run;
  using stats = fakes::levels_stats;
  using snapshot = std::vector<std::int32_t>;

  static void require(dyng::graph_properties& /*p*/) {}
  static dyng::conformance::graph_shape shape(dyng::conformance::size_class /*size*/) {
    return {40, 90, 9};
  }
  template <typename graph_type>
  static levels_run compute(const dyng::resources& res, const graph_type& g, dyng::engine e) {
    levels_run run;
    run.fused = e == dyng::engine::fused;
    fakes::levels_options opt;
    opt.fused = run.fused;
    run.r = fakes::compute_levels(res, g, 0, run.log, opt);
    return run;
  }
  template <typename graph_type>
  static stats update(const dyng::resources& res, graph_type& g,
                      const dyng::edge_batch_view<std::int32_t, dyng::unweighted>& batch,
                      levels_run& run) {
    fakes::levels_options opt;
    opt.fused = run.fused;
    stats s = fw::update_one<fakes::levels_problem>(res, g, batch, run.r, run.log, opt);
    if (broken && run.fused && run.r.level.size() > 1) {
      run.r.level[1] += 1;  // a fused engine with a bug
    }
    return s;
  }
  template <typename graph_type>
  static snapshot take(const dyng::resources& /*res*/, const levels_run& run) {
    return run.r.level;
  }
  static std::vector<std::int64_t> deterministic(const stats& s) {
    return {s.affected};
  }
};

template <bool broken>
struct levels_case {
  using traits = levels_traits<broken>;
  using graph_type = graph_t;
};

bool engines_agree_on_sequential_with_a_broken_fused_engine() {
  return dyng::conformance::kit_detail::engines_agree<levels_case<true>>(
      {dyng::backend::sequential});
}

TEST(ConformanceKit, C4ComparesTheTwoEnginesOfAHostBackend) {
  EXPECT_TRUE(
      dyng::conformance::kit_detail::engines_agree<levels_case<false>>({dyng::backend::sequential}))
      << "C4 did not compare: the fake has both engines on the sequential backend";
}

TEST(ConformanceKit, C4FailsWhenTheEnginesDisagree) {
  EXPECT_NONFATAL_FAILURE((void)engines_agree_on_sequential_with_a_broken_fused_engine(),
                          "same<traits>(fused->take(), operators->take())");
}

}  // namespace
