// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file profiler.cpp
 * @brief dyng::profiler: stages, samples and counters, read as copies.
 *
 * The records are written during native calls, which run without the GIL (ADR 0023, note 5).
 * They are therefore copied while the process-wide native-call lock is held exclusively, so no
 * call through any handle can be recording at the same time.
 */
#include "common.hpp"

#include <dyng/core/profiler.hpp>

#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/tuple.h>
#include <nanobind/stl/vector.h>

#include <sstream>
#include <tuple>

namespace dyng::python {
namespace {

/// Run `f` while no native call runs (the native-call lock held exclusively, without the GIL).
template <typename function_t>
decltype(auto) while_idle(function_t&& f) {
  nb::gil_scoped_release release;
  std::unique_lock<std::shared_mutex> lock(native_call_mutex());
  return std::forward<function_t>(f)();
}

using stage_row = std::tuple<std::string, int, std::int64_t, double, double>;
using sample_row = std::tuple<std::string, int, double, double>;
using counter_row = std::tuple<std::string, std::int64_t>;

}  // namespace

void bind_profiler(nb::module_& m) {
  nb::class_<profiler>(m, "Profiler", "Records the library's stages and counters.")
      .def(
          "__init__",
          [](profiler* p, bool sync_stages, bool nvtx, bool cuda_events) {
            profiler_options opt;
            opt.sync_stages = sync_stages;
            opt.nvtx = nvtx;
            opt.cuda_events = cuda_events;
            new (p) profiler(opt);
          },
          nb::arg("sync_stages") = false, nb::arg("nvtx") = false, nb::arg("cuda_events") = false)
      .def(
          "stages",
          [](const profiler& p) {
            return while_idle([&] {
              std::vector<stage_row> rows;
              for (const stage_record& s : p.stages()) {
                rows.emplace_back(s.name, s.depth, s.calls, s.host_ms, s.device_ms);
              }
              return rows;
            });
          },
          "(name, depth, calls, host_ms, device_ms) per stage, in first-call order.")
      .def(
          "samples",
          [](const profiler& p) {
            return while_idle([&] {
              std::vector<sample_row> rows;
              for (const stage_sample& s : p.samples()) {
                rows.emplace_back(s.name, s.depth, s.host_ms, s.device_ms);
              }
              return rows;
            });
          },
          "(name, depth, host_ms, device_ms) per completed call, in completion order.")
      .def(
          "counters",
          [](const profiler& p) {
            return while_idle([&] {
              std::vector<counter_row> rows;
              for (const counter_record& c : p.counters()) {
                rows.emplace_back(c.name, c.value);
              }
              return rows;
            });
          },
          "(name, value) per counter.")
      .def("reset", [](profiler& p) { while_idle([&] { p.reset(); }); })
      .def("to_csv",
           [](const profiler& p) {
             return while_idle([&] {
               std::ostringstream out;
               p.write_csv(out);
               return out.str();
             });
           })
      .def("to_json", [](const profiler& p) {
        return while_idle([&] {
          std::ostringstream out;
          p.write_json(out);
          return out.str();
        });
      });
}

}  // namespace dyng::python
