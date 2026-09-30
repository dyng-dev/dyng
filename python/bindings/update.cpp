// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file update.cpp
 * @brief dyng::update() over a run-time list of results of mixed algorithms.
 *
 * The C++ dyng::update() is variadic (its result types are known at compile time); Python passes
 * a list. As ADR 0023 (note 1) prescribes, the binding builds the type-erased participants of the
 * results itself, with the factories each algorithm header declares for dyng::update()
 * (detail::make_sssp_participant, detail::make_cycle_count_participant), and runs them through
 * detail::participant_of<graph>::run(), the path dyng::update() takes: every before-apply step on
 * G_t, one commit, then every after-apply step on G_{t+1}.
 */
#include "types.hpp"

#include <dyng/update.hpp>

#include <nanobind/stl/vector.h>

#include <memory>
#include <new>
#include <stdexcept>
#include <variant>
#include <vector>

namespace dyng::python {
namespace {

template <typename vertex_t, typename edge_t, typename weight_t>
void bind_update_type(nb::module_& m) {
  using graph_t = graph_holder<vertex_t, edge_t, weight_t>;
  using container_t = graph<vertex_t, edge_t, weight_t>;
  using participant_t = typename detail::participant_of<container_t>::type;
  constexpr bool sssp_ok = detail::sssp_supported_v<vertex_t, edge_t, weight_t>;
  constexpr bool cycle_ok = detail::cycle_count_supported_v<vertex_t, edge_t, weight_t>;

  m.def(
      "update",
      [](const resources& res, graph_t& g, const batch_arrays<vertex_t, weight_t>& b,
         const nb::list& results) {
        DYNG_EXPECTS(nb::len(results) > 0,
                     "dyng.update: no results (applying a batch without updating any result "
                     "would leave every result of the graph stale; use Graph.apply())");
        // One slot per result: its stats, filled by its participant. The holders are found with
        // the GIL; the participants are made under the locks, from the states the update may
        // change (result_holder::for_update(): a copy when an exported array still views the
        // current state).
        using slot = std::variant<sssp::stats, cycle_count::stats>;
        std::vector<slot> stats(nb::len(results));
        std::vector<sssp_holder<vertex_t>*> sssp_results(stats.size(), nullptr);
        std::vector<cycle_count_holder*> cycle_results(stats.size(), nullptr);
        for (std::size_t i = 0; i < stats.size(); ++i) {
          nb::handle item = results[i];
          if (nb::isinstance<sssp_holder<vertex_t>>(item)) {
            if constexpr (sssp_ok) {
              sssp_results[i] = &nb::cast<sssp_holder<vertex_t>&>(item);
              stats[i] = sssp::stats{};
              continue;
            }
          }
          if (nb::isinstance<cycle_count_holder>(item)) {
            if constexpr (cycle_ok) {
              cycle_results[i] = &nb::cast<cycle_count_holder&>(item);
              stats[i] = cycle_count::stats{};
              continue;
            }
          }
          throw invalid_argument_error(
              "dyng.update: result " + std::to_string(i) +
              " is not a result of an algorithm that supports this graph type");
        }
        without_gil([&] {
          lock_set locks;
          locks.add(g.mutex, true);
          for (std::size_t i = 0; i < stats.size(); ++i) {
            locks.add(sssp_results[i] != nullptr ? sssp_results[i]->mutex : cycle_results[i]->mutex,
                      true);
          }
          locks.lock();
          std::vector<std::unique_ptr<participant_t>> owned;
          std::vector<participant_t*> raw;
          try {
            owned.reserve(stats.size());
            raw.reserve(stats.size());
            for (std::size_t i = 0; i < stats.size(); ++i) {
              if (sssp_results[i] != nullptr) {
                if constexpr (sssp_ok) {
                  owned.push_back(
                      detail::make_sssp_participant<vertex_t, edge_t, weight_t, std::int64_t>(
                          sssp_results[i]->for_update(res), std::get<sssp::stats>(stats[i])));
                }
              } else {
                if constexpr (cycle_ok) {
                  owned.push_back(detail::make_cycle_count_participant<vertex_t, edge_t, weight_t>(
                      cycle_results[i]->for_update(res), std::get<cycle_count::stats>(stats[i])));
                }
              }
              raw.push_back(owned.back().get());
            }
          } catch (const std::bad_alloc& e) {
            detail::throw_host_allocation_failure("dyng.update (the participants)", e.what());
          }
          (void)detail::participant_of<container_t>::run(res, g.value, b.view(), raw.data(),
                                                         raw.size(), "update.commit");
        });
        nb::list out;
        for (const slot& s : stats) {
          if (std::holds_alternative<sssp::stats>(s)) {
            out.append(nb::cast(std::get<sssp::stats>(s)));
          } else {
            out.append(nb::cast(std::get<cycle_count::stats>(s)));
          }
        }
        return out;
      },
      nb::arg("resources"), nb::arg("graph"), nb::arg("batch"), nb::arg("results"),
      "Apply one batch and update every result (a list of result objects).");
}

}  // namespace

void bind_update(nb::module_& m) {
#define DYNG_PY_BIND_UPDATE(V, E, W) bind_update_type<V, E, W>(m);
  DYNG_PY_FOR_EACH_GRAPH_TYPE(DYNG_PY_BIND_UPDATE)
#undef DYNG_PY_BIND_UPDATE
}

}  // namespace dyng::python
