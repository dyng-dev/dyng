# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Header self-containment (PLAN Section 4.1): every public header must compile on its own with
# the host compiler, without any CUDA header. One generated translation unit per header, compiled
# into an OBJECT library; the build fails if a header misses an include or includes CUDA.

file(
  GLOB_RECURSE _dyng_public_headers
  RELATIVE "${PROJECT_SOURCE_DIR}/cpp/include"
  CONFIGURE_DEPENDS "${PROJECT_SOURCE_DIR}/cpp/include/dyng/*.hpp"
)
list(APPEND _dyng_public_headers dyng/version.hpp dyng/config.hpp)

set(_dyng_header_tus)
foreach(header IN LISTS _dyng_public_headers)
  string(MAKE_C_IDENTIFIER "${header}" _id)
  set(_tu "${CMAKE_CURRENT_BINARY_DIR}/header_self_contained/${_id}.cpp")
  # Also host-compilable (PLAN Section 4.1; the 0.1 API review, ADR 0023): the header must not
  # pull in a CUDA, CUB, Thrust or libcu++ header, even in a CUDA build where their include
  # directories are on the command line.
  string(
    CONCAT _content
    "// Generated: checks that <${header}> is self-contained and host-only.\n"
    "#include <${header}>\n#include <${header}>\n"
    "#if defined(__CUDA_RUNTIME_H__) || defined(__CUDA_RUNTIME_API_H__) || "
    "defined(__cuda_cuda_h__) || defined(CUB_VERSION) || defined(THRUST_VERSION) || "
    "defined(_LIBCUDACXX_VERSION) || defined(_CCCL_VERSION)\n"
    "#error \"<${header}> includes a CUDA header: public headers must compile without CUDA\"\n"
    "#endif\n"
  )
  if(EXISTS "${_tu}")
    file(READ "${_tu}" _old)
  else()
    set(_old "")
  endif()
  if(NOT _old STREQUAL _content)
    file(WRITE "${_tu}" "${_content}")
  endif()
  list(APPEND _dyng_header_tus "${_tu}")
endforeach()

add_library(dyng_header_self_contained OBJECT ${_dyng_header_tus})
target_link_libraries(dyng_header_self_contained PRIVATE dyng::dyng)
target_compile_features(dyng_header_self_contained PRIVATE cxx_std_17)
dyng_set_warnings(dyng_header_self_contained)
