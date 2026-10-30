// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file label_propagation_traits.hpp
 * @brief The conformance-kit traits of label_propagation (PLAN Section 8.2; ADR 0076): the float
 *        slotted axis with vertex batches, the oracle kind "reference" (the harmonic solution),
 *        and both schedules.
 *
 * label_propagation runs on float graphs with row_layout::slotted and vertex batches, read
 * through the kit's slotted axis (conformance/slotted_generators.hpp: DynLP graphs of at least
 * 5,000 vertices at average degree 5, the seeds by the kit's rule kit_seed_label()). Two suites
 * run: `label_propagation` (schedule::in_place, the default) and the test-only tag
 * `label_propagation_damped` (schedule::damped; its registry entry is label_propagation's), each
 * the other's partner of C10, so one dyng::update() carries a damped and an in_place result.
 *
 * - near_reference (C2, C8): the harmonic solution of the current graph
 *   (dyng::testing::harmonic_solution over the present vertices and the present seeds): the PCG
 *   converged (0 skips), mean |F - F*| <= 2000 delta at delta = 1e-6 (AD7), and no class flip
 *   where |F* - 0.5| > 3m (m = 2.9217e-3, parity/results/M10a-class-margin.json); for the update
 *   chain and for compute() (IrLP).
 * - compare() (C1, C6, C10, C11): damped bitwise (the labels' bits and the seeds); in_place the
 *   seeds exactly and the labels by mean |dF| <= 2 x 2000 delta over the unlabelled vertices that
 *   either side moved from 0.5. compare_across_backends() (C3, C12): the in_place rule for both
 *   schedules (damped is bitwise per backend, not across: the host and the device sum in other
 *   orders).
 * - deterministic() (C3, C6, C10) leaves out iterations and frontier_visits, and under in_place
 *   also affected, converged and fallback_used (AD20: schedule-dependent there); the
 *   cross-backend counters leave out converged, fallback_used and affected (deterministic per
 *   backend only). Components, sparsified_edges, initial_frontier and the batch summary are exact
 *   everywhere.
 * - history_independent = false: C5 does not apply (vertex ids are single-use and a departure
 *   cannot be undone; the labels depend on the path, DynLP stops at a delta-approximate point).
 */
#pragma once

#include "conformance/slotted_generators.hpp"
#include "conformance/test_traits.hpp"
#include "conformance/type_list.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/graph/graph_view.hpp>
#include <dyng/label_propagation.hpp>
#include <dyng/testing/harmonic_solution.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <limits>
#include <ostream>
#include <string_view>
#include <vector>

namespace dyng::conformance {

namespace tags {
struct label_propagation;  // conformance/registry.hpp declares the tags of the manifests
/// The test-only tag of label_propagation's damped suite (its registry entry is
/// label_propagation's).
struct label_propagation_damped;
}  // namespace tags

namespace lp_kit {

inline constexpr float delta = 1e-6F;              ///< AD7's delta of CI
inline constexpr float tau = 0.5F;                 ///< the kit's sparsification threshold
inline constexpr double class_margin = 2.9217e-3;  ///< m (M10a-class-margin.json)
inline constexpr double bound = 2000.0 * static_cast<double>(delta);  ///< AD7's bound

/// A host copy of a result: the labels and the seed labels.
struct snapshot {
  std::vector<float> labels;       ///< one per vertex
  std::vector<std::int8_t> seeds;  ///< one per vertex: -1, 0 or 1

  /// Bitwise equality.
  friend bool operator==(const snapshot& a, const snapshot& b) {
    return a.seeds == b.seeds && a.labels.size() == b.labels.size() &&
           std::memcmp(a.labels.data(), b.labels.data(), a.labels.size() * sizeof(float)) == 0;
  }
};

/// For gtest messages.
inline void PrintTo(const snapshot& s, std::ostream* os) {  // NOLINT(readability-identifier-naming)
  *os << "{" << s.labels.size() << " labels}";
}

/// The damped rule: bitwise.
inline ::testing::AssertionResult bitwise(const snapshot& a, const snapshot& b) {
  if (a.seeds != b.seeds) {
    return ::testing::AssertionFailure() << "the seed labels differ";
  }
  if (a.labels.size() != b.labels.size()) {
    return ::testing::AssertionFailure() << "the label counts differ";
  }
  for (std::size_t v = 0; v < a.labels.size(); ++v) {
    if (std::memcmp(&a.labels[v], &b.labels[v], sizeof(float)) != 0) {
      return ::testing::AssertionFailure() << "the label bits of vertex " << v << " differ ("
                                           << a.labels[v] << " against " << b.labels[v] << ")";
    }
  }
  return ::testing::AssertionSuccess();
}

/// The in_place rule: the seeds exactly, mean |dF| <= 2 x 2000 delta over the unlabelled
/// vertices that either side moved from 0.5.
inline ::testing::AssertionResult near(const snapshot& a, const snapshot& b) {
  if (a.seeds != b.seeds) {
    return ::testing::AssertionFailure() << "the seed labels differ";
  }
  if (a.labels.size() != b.labels.size()) {
    return ::testing::AssertionFailure() << "the label counts differ";
  }
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::size_t v = 0; v < a.labels.size(); ++v) {
    if (a.seeds[v] >= 0) {
      if (a.labels[v] != b.labels[v]) {
        return ::testing::AssertionFailure() << "the label of seed " << v << " differs";
      }
      continue;
    }
    if (a.labels[v] == 0.5F && b.labels[v] == 0.5F) {
      continue;
    }
    sum += std::fabs(static_cast<double>(a.labels[v]) - static_cast<double>(b.labels[v]));
    ++count;
  }
  const double mean = count > 0 ? sum / static_cast<double>(count) : 0.0;
  if (mean > 2.0 * bound) {
    return ::testing::AssertionFailure() << "mean |dF| = " << mean << " over " << count
                                         << " vertices, above 2 x 2000 delta = " << 2.0 * bound;
  }
  return ::testing::AssertionSuccess();
}

/// The kit's graph properties of label_propagation: slotted rows under net_effect().
inline void require_slotted(graph_properties& p) {
  p.directed = false;
  p.layout = row_layout::slotted;
  p.order = row_order::append;
  p.parallel_edges = multi_edges::allow;
  p.semantics = batch_semantics::net_effect();
}

/**
 * @brief The traits of one schedule.
 * @tparam rule The schedule of the suite's results.
 */
template <label_propagation::schedule rule>
struct traits_of_schedule {
  static constexpr std::string_view name = "label_propagation";  ///< as in the manifest
  static constexpr oracle_kind oracle = oracle_kind::reference;  ///< the harmonic solution
  static constexpr determinism level = determinism::tolerance;   ///< the manifest's level
  static constexpr bool history_independent = false;             ///< C5 does not apply
  /// AD20: `affected` is deterministic per backend under damped, schedule-dependent in place.
  static constexpr bool affected_deterministic = rule == label_propagation::schedule::damped;
  /// The instantiated graph types (float similarity graphs).
  using graph_types =
      type_list<graph<std::int32_t, std::int32_t, float>, graph<std::int32_t, std::int64_t, float>>;
  /// The kit reads them through the float slotted axis.
  template <typename graph_t>
  using container_axis = slotted_axis<graph_t>;
  template <typename graph_t>
  using result = label_propagation::result<std::int32_t>;  ///< the result
  using stats = label_propagation::stats;                  ///< update()'s stats
  using snapshot = lp_kit::snapshot;                       ///< a host copy

  /// The graph requirements.
  static void require(graph_properties& p) {
    require_slotted(p);
  }

  /// The kit's graphs: DynLP graphs of at least 5,000 vertices (M10a criterion 15).
  static slotted_shape shape(size_class size) {
    switch (size) {
      case size_class::tiny:
        return {5000, 5};
      case size_class::small:
        return {6000, 5};
      case size_class::medium:
        return {8000, 5};
    }
    return {};
  }

  /// The kit's options: delta 1e-6, tau 0.5, the suite's schedule.
  static label_propagation::options options(engine e) {
    label_propagation::options opt;
    opt.similarity_threshold = tau;
    opt.tolerance = delta;
    opt.update_schedule = rule;
    opt.cuda_engine = e;
    return opt;
  }

  /// The active vertices of a graph (its statuses, downloaded).
  template <typename graph_t>
  static std::vector<std::uint8_t> present(const graph_t& g) {
    const auto view = g.view();
    std::vector<std::uint8_t> out(static_cast<std::size_t>(g.num_vertices()), 0);
    for (std::size_t v = 0; v < out.size(); ++v) {
      out[v] = view.status[v] == vertex_status::active ? 1 : 0;
    }
    return out;
  }

  /// compute(): IrLP from the active seeds of the kit's rule, reserved for the kit's batches.
  template <typename graph_t>
  static result<graph_t> compute(const resources& res, const graph_t& g, engine e) {
    const std::vector<std::uint8_t> on = present(g);
    std::vector<std::int32_t> seed_vertices;
    std::vector<std::int8_t> seed_labels;
    for (std::size_t v = 0; v < on.size(); ++v) {
      const std::int8_t label = kit_seed_label(static_cast<std::int64_t>(v));
      if (on[v] != 0 && label >= 0) {
        seed_vertices.push_back(static_cast<std::int32_t>(v));
        seed_labels.push_back(label);
      }
    }
    const std::vector<std::int32_t>& sv = seed_vertices;
    const std::vector<std::int8_t>& sl = seed_labels;
    result<graph_t> r =
        label_propagation::compute(res, g, host_view(sv), host_view(sl), options(e));
    const auto n = static_cast<std::size_t>(g.num_vertices());
    r.reserve(res, n, 3 * n);
    return r;
  }

  /// update().
  template <typename graph_t>
  static stats update(const resources& res, graph_t& g,
                      const edge_batch_view<std::int32_t, float>& batch, result<graph_t>& r) {
    return label_propagation::update(res, g, batch, r);
  }

  /// A host copy of the labels and the seeds.
  template <typename graph_t>
  static snapshot take(const resources& res, const result<graph_t>& r) {
    snapshot s;
    const auto labels = r.labels();
    s.labels = is_host_accessible(labels.space()) ? std::vector<float>(labels.begin(), labels.end())
                                                  : to_vector(res, labels);
    const auto seeds = r.seed_labels();
    s.seeds.assign(seeds.begin(), seeds.end());
    return s;
  }

  /// The suite's rule: damped bitwise, in_place near (see the file comment).
  static ::testing::AssertionResult compare(const snapshot& a, const snapshot& b) {
    if constexpr (rule == label_propagation::schedule::damped) {
      return bitwise(a, b);
    } else {
      return near(a, b);
    }
  }

  /// Across backends: the in_place rule for both schedules.
  static ::testing::AssertionResult compare_across_backends(const snapshot& a, const snapshot& b) {
    return near(a, b);
  }

  /// The deterministic counters (iterations and frontier_visits left out; in place also
  /// converged and fallback_used; AD20).
  static std::vector<std::int64_t> deterministic(const stats& s) {
    std::vector<std::int64_t> out = {s.components, s.sparsified_edges, s.initial_frontier};
    if constexpr (rule == label_propagation::schedule::damped) {
      out.push_back(s.converged ? 1 : 0);
      out.push_back(s.fallback_used ? 1 : 0);
    }
    return out;
  }

  /// The counters the backends agree on.
  static std::vector<std::int64_t> deterministic_across_backends(const stats& s) {
    return {s.components, s.sparsified_edges, s.initial_frontier};
  }

  /// Bad options: each throws invalid_argument_error.
  template <typename graph_t>
  static std::vector<std::function<void()>> invalid_options(const resources& res,
                                                            const graph_t& g) {
    const auto with = [&res, &g](label_propagation::options opt) {
      return [&res, &g, opt]() {
        (void)label_propagation::compute(res, g, array_view<const std::int32_t>(),
                                         array_view<const std::int8_t>(), opt);
      };
    };
    label_propagation::options zero = options(engine::automatic);
    zero.tolerance = 0.0F;
    label_propagation::options nan_delta = options(engine::automatic);
    nan_delta.tolerance = std::numeric_limits<float>::quiet_NaN();
    label_propagation::options no_iterations = options(engine::automatic);
    no_iterations.max_iterations = 0;
    label_propagation::options nan_tau = options(engine::automatic);
    nan_tau.similarity_threshold = std::numeric_limits<float>::quiet_NaN();
    return {with(zero), with(nan_delta), with(no_iterations), with(nan_tau)};
  }

  /// C8: no synchronization on the host; on CUDA the fused engine's run-dependent ones.
  static std::int64_t host_sync_budget(backend b) {
    return b == backend::cuda ? run_dependent_budget : 0;
  }

  /// C2, C8: near the harmonic solution of the current graph (see the file comment).
  template <typename graph_t>
  static ::testing::AssertionResult near_reference(const resources& res, const graph_t& g,
                                                   const snapshot& s) {
    const auto csr = g.to_csr(res);
    const std::vector<std::uint8_t> on = present(g);
    std::vector<std::int8_t> seeds = s.seeds;
    for (std::size_t v = 0; v < seeds.size(); ++v) {
      if (on[v] == 0) {
        seeds[v] = -1;  // a deleted seed keeps its label in the result; it is not in the graph
      }
    }
    const dyng::testing::harmonic_result h =
        dyng::testing::harmonic_solution(csr.row_ptr, csr.col_ind, csr.weights, on, seeds);
    if (!h.converged) {
      return ::testing::AssertionFailure()
             << "the PCG of the harmonic solution did not converge (a skip; residual "
             << h.relative_residual << ")";
    }
    const dyng::testing::label_error err =
        dyng::testing::label_error_against(s.labels, h, on, seeds);
    if (err.mean > bound) {
      return ::testing::AssertionFailure()
             << "mean |F - F*| = " << err.mean << " over " << err.compared
             << " vertices, above 2000 delta = " << bound << " (max " << err.max << ")";
    }
    const dyng::testing::class_agreement ca =
        dyng::testing::class_agreement_with(s.labels, h, on, seeds, 3.0 * class_margin);
    if (ca.flips_beyond_margin > 0) {
      return ::testing::AssertionFailure()
             << ca.flips_beyond_margin
             << " class flips where |F* - 0.5| > 3m = " << 3.0 * class_margin
             << " (largest flip distance " << ca.largest_flip_distance << ")";
    }
    for (std::size_t v = 0; v < s.labels.size(); ++v) {
      if (!(s.labels[v] >= 0.0F && s.labels[v] <= 1.0F)) {
        return ::testing::AssertionFailure()
               << "the label of vertex " << v << " is " << s.labels[v] << ", outside [0, 1]";
      }
      if (on[v] != 0 && seeds[v] >= 0 && s.labels[v] != static_cast<float>(seeds[v])) {
        return ::testing::AssertionFailure() << "the seed " << v << " lost its class";
      }
    }
    return ::testing::AssertionSuccess();
  }
};

}  // namespace lp_kit

/// label_propagation under schedule::in_place (the default); C10's partner is the damped suite.
template <>
struct test_traits<tags::label_propagation>
    : lp_kit::traits_of_schedule<label_propagation::schedule::in_place> {
  using test_partners = type_list<tags::label_propagation_damped>;  ///< C10's partner
};

/// label_propagation under schedule::damped (a test-only tag; C10's partner is the in_place
/// suite, a registered algorithm).
template <>
struct test_traits<tags::label_propagation_damped>
    : lp_kit::traits_of_schedule<label_propagation::schedule::damped> {
  /// label_propagation's registry entry.
  static const algorithm_info* registry_entry() {
    return find_algorithm("label_propagation");
  }
};

}  // namespace dyng::conformance
