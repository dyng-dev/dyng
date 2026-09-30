// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file resources.cpp
 * @brief dyng::resources, the core enumerations, backend availability, logging and the build
 *        configuration.
 */
#include "common.hpp"

#include <dyng/config.hpp>
#include <dyng/core/backend.hpp>
#include <dyng/core/logging.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/stream.hpp>
#include <dyng/core/types.hpp>
#include <dyng/version.hpp>

#include <nanobind/stl/string.h>

#include <cstdint>

#if defined(DYNG_PYTHON_HAS_OPENMP) && DYNG_PYTHON_HAS_OPENMP
#include <omp.h>
#endif

#ifndef DYNG_PYTHON_BUILD_TYPE
#define DYNG_PYTHON_BUILD_TYPE "unknown"
#endif
#ifndef DYNG_PYTHON_COMPILER
#define DYNG_PYTHON_COMPILER "unknown"
#endif

namespace dyng::python {

void bind_resources(nb::module_& m) {
  nb::enum_<backend>(m, "Backend", "An execution backend.")
      .value("sequential", backend::sequential)
      .value("openmp", backend::openmp)
      .value("cuda", backend::cuda);
  nb::enum_<memory_space>(m, "MemorySpace", "A memory space.")
      .value("host", memory_space::host)
      .value("pinned_host", memory_space::pinned_host)
      .value("device", memory_space::device)
      .value("managed", memory_space::managed);
  nb::enum_<copy_policy>(m, "CopyPolicy", "What to do before an implicit copy of an input.")
      .value("allow", copy_policy::allow)
      .value("warn", copy_policy::warn)
      .value("error", copy_policy::error);
  nb::enum_<engine>(m, "Engine", "An update engine.")
      .value("automatic", engine::automatic)
      .value("fused", engine::fused)
      .value("operators", engine::operators);
  nb::enum_<determinism>(m, "Determinism", "What two runs agree on.")
      .value("bitwise", determinism::bitwise)
      .value("exact_value", determinism::exact_value)
      .value("tolerance", determinism::tolerance);
  nb::enum_<log_level>(m, "LogLevel", "A log level.")
      .value("off", log_level::off)
      .value("error", log_level::error)
      .value("warn", log_level::warn)
      .value("info", log_level::info)
      .value("debug", log_level::debug)
      .value("trace", log_level::trace);

  m.def("backend_available", &backend_available, nb::arg("backend"),
        "Whether a backend is compiled in (and, for cuda, a device is visible).");
  m.def("default_backend", &default_backend, "The backend of default resources.");
  m.def("set_log_level", &set_log_level, nb::arg("level"), "Set the library's log level.");
  m.def("get_log_level", &get_log_level, "The library's log level.");

  nb::class_<resources>(m, "Resources", "Execution resources (a shared handle).")
      .def(nb::init<>(), "Resources of the default backend.")
      .def_static("sequential", &resources::sequential, "The sequential backend.")
      .def_static("openmp", &resources::openmp, nb::arg("num_threads") = 0,
                  "The OpenMP backend (0 threads: the OpenMP default).")
      .def_static(
          "cuda",
          [](int device, std::uintptr_t stream, int host_threads) {
            const stream_ref s = stream == 0
                                     ? stream_ref()
                                     : stream_ref(reinterpret_cast<cuda_stream_handle>(stream));
            return resources::cuda(device, s, host_threads);
          },
          nb::arg("device") = 0, nb::arg("stream") = 0, nb::arg("host_threads") = 0,
          "The CUDA backend (stream: a cudaStream_t as an integer, 0 = per-thread default).")
      .def_prop_ro("backend", &resources::get_backend)
      .def_prop_ro("device", &resources::device)
      .def_prop_ro(
          "stream",
          [](const resources& r) { return reinterpret_cast<std::uintptr_t>(r.stream().get()); })
      .def_prop_ro("num_threads", &resources::num_threads)
      .def_prop_ro("default_space", &resources::default_space)
      .def_prop_rw("copy_policy", &resources::get_copy_policy, &resources::set_copy_policy)
      .def_prop_ro("workspace_bytes", &resources::workspace_bytes)
      .def("release_workspaces",
           [](const resources& r) { without_gil([&] { r.release_workspaces(); }); })
      .def("warm_up", [](const resources& r) { without_gil([&] { r.warm_up(); }); })
      .def("synchronize", [](const resources& r) { without_gil([&] { r.synchronize(); }); })
      .def(
          "attach_profiler", [](resources& r, profiler* p) { r.attach_profiler(p); },
          nb::arg("profiler").none(), nb::keep_alive<1, 2>(),
          "Attach a profiler to the handle (None detaches); the profiler is kept alive by this "
          "object.")
      .def("has_profiler", [](const resources& r) { return r.get_profiler() != nullptr; });

  m.attr("__version__") = DYNG_VERSION_STRING;
  m.def(
      "library_version", [] { return std::string(library_version().string); },
      "The version of the compiled library.");
  nb::dict config;
  config["openmp"] = static_cast<bool>(DYNG_HAS_OPENMP);
  config["cuda"] = static_cast<bool>(DYNG_HAS_CUDA);
  config["build_type"] = DYNG_PYTHON_BUILD_TYPE;
  config["compiler"] = DYNG_PYTHON_COMPILER;
#if defined(DYNG_PYTHON_HAS_OPENMP) && DYNG_PYTHON_HAS_OPENMP
  config["openmp_version"] = static_cast<int>(_OPENMP);
#else
  config["openmp_version"] = 0;
#endif
  m.attr("build_config") = config;
  m.def("openmp_max_threads", [] {
#if defined(DYNG_PYTHON_HAS_OPENMP) && DYNG_PYTHON_HAS_OPENMP
    return omp_get_max_threads();
#else
    return 1;
#endif
  });
}

}  // namespace dyng::python
