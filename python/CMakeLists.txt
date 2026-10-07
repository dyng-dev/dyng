# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The extension module dyng._core (PLAN Sections 5.4 and 7.7): nanobind, the stable ABI of
# CPython >= 3.12 (one abi3 wheel per platform), nanobind linked statically, and libdyng linked
# statically when the build is a wheel (BUILD_SHARED_LIBS=OFF in the root pyproject.toml), so a
# wheel holds one shared object and no RPATH. The typed Python layer is python/dyng/*.py.
#
# Built by scikit-build-core from the root pyproject.toml:
#   pip install .                                                     # a wheel
#   pip install -e . --no-build-isolation -Ceditable.rebuild=true -Cbuild-dir=build   # development
# or in any CMake build with -DDYNG_BUILD_PYTHON=ON (nanobind is found through the Python
# interpreter CMake finds; `python -m nanobind --cmake_dir`).
#
# DYNG_PYTHON_PLUGIN=cu12|cu13 builds the same module for a CUDA plugin wheel instead (dyng-cu12,
# dyng-cu13; PLAN 5.4 and 7.7, ADR 0030): installed as dyng_cu<N>/_core.abi3.so under the nanobind
# domain dyng_cu<N>, with every backend (DYNG_ENABLE_CUDA=ON, OpenMP), the static CUDA runtime and
# a toolkit of that CUDA major. ci/plugin_pyproject.py renders the pyproject.toml of such a build.
set(DYNG_PYTHON_PLUGIN
    ""
    CACHE STRING "Build the module of a CUDA plugin wheel: empty (dyng._core), cu12 or cu13"
)
set_property(CACHE DYNG_PYTHON_PLUGIN PROPERTY STRINGS "" cu12 cu13)
if(DYNG_PYTHON_PLUGIN STREQUAL "")
  set(_dyng_python_package dyng)
  set(_dyng_nb_domain dyng_cpu)
elseif(DYNG_PYTHON_PLUGIN MATCHES "^cu([0-9]+)$")
  set(_dyng_plugin_major "${CMAKE_MATCH_1}")
  if(NOT DYNG_PYTHON_PLUGIN MATCHES "^cu1[23]$")
    message(FATAL_ERROR "DYNG_PYTHON_PLUGIN=${DYNG_PYTHON_PLUGIN}: the plugins are cu12 and cu13")
  endif()
  if(NOT DYNG_ENABLE_CUDA)
    message(FATAL_ERROR "DYNG_PYTHON_PLUGIN=${DYNG_PYTHON_PLUGIN} needs DYNG_ENABLE_CUDA=ON")
  endif()
  if(NOT DYNG_ENABLE_OPENMP)
    message(FATAL_ERROR "DYNG_PYTHON_PLUGIN=${DYNG_PYTHON_PLUGIN} needs DYNG_ENABLE_OPENMP=ON (a "
                        "plugin module has every backend: it replaces dyng._core)"
    )
  endif()
  string(REGEX MATCH "^[0-9]+" _dyng_toolkit_major "${DYNG_CUDA_COMPILER_RELEASE}")
  if(NOT _dyng_toolkit_major STREQUAL _dyng_plugin_major)
    message(FATAL_ERROR "DYNG_PYTHON_PLUGIN=${DYNG_PYTHON_PLUGIN} needs a CUDA ${_dyng_plugin_major}.x "
                        "toolkit; ${CMAKE_CUDA_COMPILER} is CUDA ${DYNG_CUDA_COMPILER_RELEASE}"
    )
  endif()
  string(TOLOWER "${CMAKE_CUDA_RUNTIME_LIBRARY}" _dyng_cudart_kind)
  if(NOT _dyng_cudart_kind STREQUAL "static")
    message(FATAL_ERROR "DYNG_PYTHON_PLUGIN=${DYNG_PYTHON_PLUGIN} needs "
                        "CMAKE_CUDA_RUNTIME_LIBRARY=Static (a wheel never depends on libcudart.so)"
    )
  endif()
  set(_dyng_python_package dyng_${DYNG_PYTHON_PLUGIN})
  set(_dyng_nb_domain dyng_${DYNG_PYTHON_PLUGIN})
else()
  message(FATAL_ERROR "DYNG_PYTHON_PLUGIN='${DYNG_PYTHON_PLUGIN}': empty, cu12 or cu13")
endif()

# scikit-build-core passes SKBUILD_SABI_COMPONENT (Development.SABIModule) for an abi3 wheel; a
# plain CMake build asks for it optionally (nanobind builds a version-specific module without it).
if(DEFINED SKBUILD_SABI_COMPONENT AND NOT SKBUILD_SABI_COMPONENT STREQUAL "")
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
                                               ${SKBUILD_SABI_COMPONENT}
  )
else()
  find_package(Python 3.12 REQUIRED COMPONENTS Interpreter Development.Module
               OPTIONAL_COMPONENTS Development.SABIModule
  )
endif()

if(NOT nanobind_DIR)
  execute_process(
    COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
    OUTPUT_VARIABLE _dyng_nanobind_dir
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _dyng_nanobind_status
  )
  if(NOT _dyng_nanobind_status EQUAL 0)
    message(FATAL_ERROR "DYNG_BUILD_PYTHON=ON needs nanobind in the Python environment of "
                        "${Python_EXECUTABLE} (pip install nanobind, or conda install nanobind)"
    )
  endif()
  set(nanobind_DIR "${_dyng_nanobind_dir}")
endif()
find_package(nanobind CONFIG REQUIRED)

# NB_DOMAIN keeps the type registrations of the CPU module (dyng_cpu) and the CUDA plugins
# (dyng_cu12, dyng_cu13) apart when more than one is imported into a process (PLAN Section 5.4).
nanobind_add_module(
  _core
  STABLE_ABI
  NB_STATIC
  NB_DOMAIN ${_dyng_nb_domain}
  bindings/module.cpp
  bindings/errors.cpp
  bindings/resources.cpp
  bindings/graph.cpp
  bindings/batch.cpp
  bindings/sssp.cpp
  bindings/cycle_count.cpp
  bindings/mosp.cpp
  bindings/update.cpp
  bindings/io.cpp
  bindings/generators.cpp
  bindings/profiler.cpp
  bindings/registry.cpp
  bindings/testing.cpp
  bindings/arrays.cpp
)
target_compile_features(_core PRIVATE cxx_std_17)
# cpp/src: the private header util/rng.hpp (MOSP's weight stream of `dyng prep widen`).
target_include_directories(_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/bindings"
                                         "${PROJECT_SOURCE_DIR}/cpp/src"
)
target_link_libraries(_core PRIVATE dyng::dyng dyng::testing)
dyng_set_warnings(_core)
if(DYNG_WARNINGS_AS_ERRORS AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  # nanobind's own sources (compiled into nanobind-static) are not held to the project's warnings.
  foreach(_dyng_nb_target nanobind-static-abi3 nanobind-static)
    if(TARGET ${_dyng_nb_target})
      target_compile_options(${_dyng_nb_target} PRIVATE -Wno-error)
    endif()
  endforeach()
endif()
if(DYNG_ENABLE_OPENMP)
  target_link_libraries(_core PRIVATE OpenMP::OpenMP_CXX)
  target_compile_definitions(_core PRIVATE DYNG_PYTHON_HAS_OPENMP=1)
endif()
target_compile_definitions(_core PRIVATE "DYNG_PYTHON_BUILD_TYPE=\"$<CONFIG>\""
                                         "DYNG_PYTHON_COMPILER=\"${CMAKE_CXX_COMPILER_ID} ${CMAKE_CXX_COMPILER_VERSION}\""
                                         "DYNG_PYTHON_PLUGIN=\"${DYNG_PYTHON_PLUGIN}\""
)
# What show_config() reports about the CUDA part of the build (empty in a CPU build).
if(DYNG_ENABLE_CUDA)
  string(TOLOWER "${CMAKE_CUDA_RUNTIME_LIBRARY}" _dyng_cudart_kind)
  string(REPLACE ";" "," _dyng_cuda_archs "${DYNG_CUDA_ARCHITECTURES_RESOLVED}")
  target_compile_definitions(
    _core PRIVATE "DYNG_PYTHON_CUDA_TOOLKIT=\"${DYNG_CUDA_COMPILER_RELEASE}\""
                  "DYNG_PYTHON_CUDA_ARCHITECTURES=\"${_dyng_cuda_archs}\""
                  "DYNG_PYTHON_CUDA_RUNTIME=\"${_dyng_cudart_kind}\""
  )
endif()

# In a wheel the module sits next to the typed layer (dyng/_core.abi3.so) or in the plugin's
# package (dyng_cu13/_core.abi3.so).
install(TARGETS _core LIBRARY DESTINATION ${_dyng_python_package} COMPONENT python)
