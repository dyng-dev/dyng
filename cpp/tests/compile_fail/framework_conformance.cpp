// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file framework_conformance.cpp
 * @brief Compile-failure tests of framework/conformance.hpp (PLAN Section 8.2): a problem that
 *        misses a required declaration or hook must stop at a static_assert with a plain-English
 *        message, not deep inside the enactor. Compiled with -fsyntax-only and one of the defines
 *        below (cpp/tests/CMakeLists.txt checks the message); without a define it compiles.
 *
 *   DYNG_CF_NO_BASE       a problem that does not derive from problem_base
 *   DYNG_CF_FP_NO_LOOP    a fixed_point problem without loop (and without enact_fused)
 *   DYNG_CF_AG_NO_OWNER   an aggregate_delta problem without ownership_type (invariant I2)
 *   DYNG_CF_AG_NO_MINUS   an aggregate_delta problem without count on the old view (I1)
 *   DYNG_CF_NO_TARGET     a problem in run_update() without target()
 */
#include "framework/composition.hpp"
#include "framework/enactor.hpp"
#include "framework/problem.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/core/stats.hpp>
#include <dyng/graph/graph.hpp>

#include <cstdint>
#include <string_view>

namespace fw = ::dyng::detail::framework;

namespace {

using graph_type = dyng::graph<std::int32_t, std::int32_t, dyng::unweighted>;
using applied_type = fw::applied_batch<std::int32_t>;

struct stats_type : dyng::update_stats {};

#if defined(DYNG_CF_NO_BASE)
struct problem {
  static constexpr std::string_view name = "cf";
  using container_type = graph_type;
  using stats_type = ::stats_type;
};
#elif defined(DYNG_CF_FP_NO_LOOP)
struct problem : fw::problem_base<problem, fw::family::fixed_point> {
  static constexpr std::string_view name = "cf";
  using container_type = graph_type;
  using stats_type = ::stats_type;
};
#elif defined(DYNG_CF_AG_NO_OWNER)
struct problem : fw::problem_base<problem, fw::family::aggregate_delta> {
  static constexpr std::string_view name = "cf";
  using container_type = graph_type;
  using stats_type = ::stats_type;
};
#else
struct problem : fw::problem_base<problem, fw::family::aggregate_delta> {
  static constexpr std::string_view name = "cf";
  using container_type = graph_type;
  using stats_type = ::stats_type;
  using ownership_type = fw::ownership::min_member;
#if !defined(DYNG_CF_AG_NO_MINUS)
  void count(fw::context&, fw::old_view<graph_type>, fw::internal_frontier&, fw::sign,
             ownership_type) {}
#endif
  void count(fw::context&, fw::new_view<graph_type>, fw::internal_frontier&, fw::sign,
             ownership_type) {}
#if !defined(DYNG_CF_NO_TARGET)
  [[nodiscard]] const void* target() const noexcept {
    return this;
  }
#endif
};
#endif

}  // namespace

/// Instantiates both enactors (and the participant adapter) for `problem`.
void instantiate(const dyng::resources& res, const graph_type& g, const applied_type& applied) {
  problem p;
  fw::context ctx(res, "cf");
  stats_type stats;
  fw::update_enactor<problem> update(p);
  update.after_commit(ctx, fw::new_view<graph_type>(g), applied, stats);
  fw::static_enactor<problem> compute(p);
  (void)compute.run(ctx, fw::new_view<graph_type>(g));
#if defined(DYNG_CF_NO_TARGET)
  fw::problem_participant<problem> participant(stats);
  (void)participant;
#endif
}
