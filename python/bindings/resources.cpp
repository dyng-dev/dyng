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

#include <nanobind/stl/optional.h>
#include <nanobind/stl/string.h>

#include <cstdint>
#include <optional>

#if defined(DYNG_PYTHON_HAS_OPENMP) && DYNG_PYTHON_HAS_OPENMP
#include <omp.h>
#endif

#ifndef DYNG_PYTHON_BUILD_TYPE
#define DYNG_PYTHON_BUILD_TYPE "unknown"
#endif
#ifndef DYNG_PYTHON_COMPILER
#define DYNG_PYTHON_COMPILER "unknown"
#endif
#ifndef DYNG_PYTHON_PLUGIN
#define DYNG_PYTHON_PLUGIN ""
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
          [](int device, std::optional<std::uintptr_t> stream, int host_threads) {
            // None: the per-thread default stream (a default stream_ref). An integer is a
            // cudaStream_t as C++ reads it: 0 is the legacy default stream, as in
            // stream_ref(0), PyTorch's and CuPy's default streams.
            const stream_ref s = stream.has_value()
                                     ? stream_ref(reinterpret_cast<cuda_stream_handle>(*stream))
                                     : stream_ref();
            return resources::cuda(device, s, host_threads);
          },
          nb::arg("device") = 0, nb::arg("stream").none() = nb::none(), nb::arg("host_threads") = 0,
          "The CUDA backend (stream: None = the per-thread default stream, or a cudaStream_t as "
          "an integer: 0 = the legacy default stream).")
      .def_prop_ro("backend", &resources::get_backend)
      .def_prop_ro("device", &resources::device)
      .def_prop_ro(
          "stream",
          [](const resources& r) { return reinterpret_cast<std::uintptr_t>(r.stream().get()); })
      .def_prop_ro("num_threads", &resources::num_threads)
      .def_prop_ro("default_space", &resources::default_space)
      // The copy policy and the attached profiler are plain fields of the shared handle state,
      // read by every native call (C++: "during setup, not while another thread uses the
      // handle"). They are changed while no native call runs (while_idle) and read under the
      // shared native-call lock, so a Python thread entering dyng.profile() or setting the policy
      // never races with another thread's call.
      .def_prop_rw(
          "copy_policy",
          [](const resources& r) { return without_gil([&] { return r.get_copy_policy(); }); },
          [](resources& r, copy_policy p) { while_idle([&] { r.set_copy_policy(p); }); })
      .def_prop_ro("workspace_bytes", &resources::workspace_bytes)
      .def("release_workspaces",
           [](const resources& r) { without_gil([&] { r.release_workspaces(); }); })
      .def("warm_up", [](const resources& r) { without_gil([&] { r.warm_up(); }); })
      .def("synchronize", [](const resources& r) { without_gil([&] { r.synchronize(); }); })
      .def(
          "attach_profiler",
          [](resources& r, profiler* p) { while_idle([&] { r.attach_profiler(p); }); },
          nb::arg("profiler").none(), nb::keep_alive<1, 2>(),
          "Attach a profiler to the handle (None detaches); the profiler is kept alive by this "
          "object.")
      .def("has_profiler", [](const resources& r) {
        return without_gil([&] { return r.get_profiler() != nullptr; });
      });

  m.attr("__version__") = DYNG_VERSION_STRING;
  m.def(
      "library_version", [] { return std::string(library_version().string); },
      "The version of the compiled library.");
  nb::dict config;
  config["openmp"] = static_cast<bool>(DYNG_HAS_OPENMP);
  config["cuda"] = static_cast<bool>(DYNG_HAS_CUDA);
  config["build_type"] = DYNG_PYTHON_BUILD_TYPE;
  config["compiler"] = DYNG_PYTHON_COMPILER;
  // The CUDA plugin this module belongs to ("cu12", "cu13"; empty in dyng._core) and the CUDA
  // part of its build (ADR 0030); empty strings in a build without CUDA.
  config["plugin"] = DYNG_PYTHON_PLUGIN;
#if defined(DYNG_PYTHON_CUDA_TOOLKIT)
  config["cuda_toolkit"] = DYNG_PYTHON_CUDA_TOOLKIT;
  config["cuda_architectures"] = DYNG_PYTHON_CUDA_ARCHITECTURES;
  config["cuda_runtime"] = DYNG_PYTHON_CUDA_RUNTIME;
#else
  config["cuda_toolkit"] = "";
  config["cuda_architectures"] = "";
  config["cuda_runtime"] = "";
#endif
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
