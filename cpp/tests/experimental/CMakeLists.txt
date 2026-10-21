# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# experimental::count_local_patterns (M9b, ADR 0058; added when triad_count, whose driver it
# uses, is in the build):
#   - the tutorial's host half (examples/tutorial_algorithms/local_patterns/tutorial_host_test.cpp,
#     label cpu: the Ops' callbacks and the brute force without nvcc, so hosted CPU CI runs it);
#   - the Op concept's compile-failure tests through the host compiler (label cpu) and, in CUDA
#     builds, through nvcc and count_local_patterns itself (label gpu: nvcc is needed, no device);
#   - the extension header without nvcc stops at its #error (label cpu);
#   - in CUDA builds: the header compiled alone by nvcc (self-containment), the API tests
#     (local_patterns_test.cu) and the tutorial's test (tutorial_test.cu) on every backend
#     (label gpu; ci/gpu_local.sh). The CUDA test sources get no --expt-relaxed-constexpr or
#     --extended-lambda: they compile as a user's program would.

set(_dyng_tutorial_dir "${PROJECT_SOURCE_DIR}/examples/tutorial_algorithms/local_patterns")

dyng_add_test(
  NAME dyng_local_patterns_tutorial_host_tests
  SOURCES "${_dyng_tutorial_dir}/tutorial_host_test.cpp"
  LABELS cpu triad_count experimental
)

set(_dyng_lp_cf_includes -I${PROJECT_SOURCE_DIR}/cpp/include -I${DYNG_GENERATED_INCLUDE_DIR})
# name|define|expected message (a regular expression)
set(_dyng_lp_cf_cases
    "control||"
    "no_num_bins|-DDYNG_CF_LP_NO_NUM_BINS|the Op must declare .static constexpr int num_bins"
    "zero_bins|-DDYNG_CF_LP_ZERO_BINS|the Op's num_bins must be in .1, dyng::experimental::max_pattern_bins"
    "too_many_bins|-DDYNG_CF_LP_TOO_MANY_BINS|the Op's num_bins must be in .1, dyng::experimental::max_pattern_bins"
    "no_shape|-DDYNG_CF_LP_NO_SHAPE|the Op must declare .static constexpr dyng::experimental::shape shape"
    "not_trivially_copyable|-DDYNG_CF_LP_NOT_TRIVIAL|the Op must be trivially copyable"
    "not_default_constructible|-DDYNG_CF_LP_NOT_DEFAULT|the Op must be default-constructible"
    "no_begin_wedge|-DDYNG_CF_LP_NO_BEGIN_WEDGE|an Op of shape triangles must have .DYNG_HD void begin_wedge"
    "no_triangle|-DDYNG_CF_LP_NO_TRIANGLE|an Op of shape triangles must have .template <class bins_t> DYNG_HD void triangle"
    "no_pair|-DDYNG_CF_LP_NO_PAIR|an Op of shape pairs must have"
    "no_key|-DDYNG_CF_LP_NO_KEY|an Op of shape keys must have"
    "no_flush|-DDYNG_CF_LP_NO_FLUSH|the Op must have .template <class bins_t> DYNG_HD void flush"
    "unsupported_type|-DDYNG_CF_LP_TYPES|supports hypergraph<int32_t, unweighted or int32_t> only"
)
foreach(_case IN LISTS _dyng_lp_cf_cases)
  string(REPLACE "|" ";" _parts "${_case}")
  list(GET _parts 0 _name)
  list(GET _parts 1 _define)
  list(GET _parts 2 _message)
  add_test(NAME experimental.local_patterns.static_assert.${_name}
           COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_define} ${_dyng_lp_cf_includes}
                   ${PROJECT_SOURCE_DIR}/cpp/tests/compile_fail/local_patterns_op.cpp
  )
  set_tests_properties(experimental.local_patterns.static_assert.${_name}
                       PROPERTIES LABELS "cpu;experimental")
  if(NOT _message STREQUAL "")
    set_tests_properties(experimental.local_patterns.static_assert.${_name}
                         PROPERTIES PASS_REGULAR_EXPRESSION "${_message}")
  endif()
  if(DYNG_ENABLE_CUDA AND NOT _name STREQUAL "unsupported_type")
    add_test(NAME experimental.local_patterns.nvcc_static_assert.${_name}
             COMMAND ${CMAKE_CUDA_COMPILER} -std=c++17 -x cu -c ${_define} ${_dyng_lp_cf_includes}
                     ${PROJECT_SOURCE_DIR}/cpp/tests/compile_fail/local_patterns_op.cpp -o
                     ${CMAKE_CURRENT_BINARY_DIR}/local_patterns_op_${_name}.o
    )
    set_tests_properties(experimental.local_patterns.nvcc_static_assert.${_name}
                         PROPERTIES LABELS "gpu;experimental")
    if(NOT _message STREQUAL "")
      set_tests_properties(experimental.local_patterns.nvcc_static_assert.${_name}
                           PROPERTIES PASS_REGULAR_EXPRESSION "${_message}")
    endif()
  endif()
endforeach()

# The extension header without nvcc: its #error names the requirement.
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/cuh_without_nvcc.cpp"
     "#include <dyng/experimental/local_patterns.cuh>\n"
)
add_test(NAME experimental.local_patterns.cuh_needs_nvcc
         COMMAND ${CMAKE_CXX_COMPILER} -std=c++17 -fsyntax-only ${_dyng_lp_cf_includes}
                 ${CMAKE_CURRENT_BINARY_DIR}/cuh_without_nvcc.cpp
)
set_tests_properties(
  experimental.local_patterns.cuh_needs_nvcc
  PROPERTIES LABELS "cpu;experimental" PASS_REGULAR_EXPRESSION
             "local_patterns.cuh> needs nvcc: include it from a .cu file"
)

if(DYNG_ENABLE_CUDA)
  # Self-containment of the extension headers under nvcc (PLAN 4.1: experimental/*.cuh are the
  # public headers that need nvcc; each compiles alone, twice included).
  set(_dyng_cuh_tus)
  foreach(_header dyng/experimental/local_patterns.cuh
                  dyng/experimental/detail/local_patterns_kernels.cuh
  )
    string(MAKE_C_IDENTIFIER "${_header}" _id)
    set(_tu "${CMAKE_CURRENT_BINARY_DIR}/cuh_self_contained/${_id}.cu")
    set(_content "// Generated: <${_header}> compiles alone under nvcc.\n"
                 "#include <${_header}>\n#include <${_header}>\n"
    )
    string(CONCAT _content ${_content})
    if(EXISTS "${_tu}")
      file(READ "${_tu}" _old)
    else()
      set(_old "")
    endif()
    if(NOT _old STREQUAL _content)
      file(WRITE "${_tu}" "${_content}")
    endif()
    list(APPEND _dyng_cuh_tus "${_tu}")
  endforeach()
  add_library(dyng_cuh_self_contained OBJECT ${_dyng_cuh_tus})
  target_link_libraries(dyng_cuh_self_contained PRIVATE dyng::dyng)
  target_compile_features(dyng_cuh_self_contained PRIVATE cxx_std_17 cuda_std_17)
  dyng_set_warnings(dyng_cuh_self_contained)

  dyng_add_test(
    NAME dyng_local_patterns_tests
    SOURCES local_patterns_test.cu
    LABELS gpu triad_count experimental
  )
  target_include_directories(dyng_local_patterns_tests PRIVATE "${PROJECT_SOURCE_DIR}/cpp/src")
  target_compile_features(dyng_local_patterns_tests PRIVATE cuda_std_17)

  dyng_add_test(
    NAME dyng_local_patterns_tutorial_tests
    SOURCES "${_dyng_tutorial_dir}/tutorial_test.cu"
    LABELS gpu triad_count experimental
  )
  target_compile_features(dyng_local_patterns_tutorial_tests PRIVATE cuda_std_17)
endif()
