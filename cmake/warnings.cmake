# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Target-based warning flags (PLAN Section 7.3). Never set global CMAKE_CXX_FLAGS.
#   dyng_set_warnings(<target>)  -Wall -Wextra -Wpedantic (+ -Werror if DYNG_WARNINGS_AS_ERRORS)
#
# CUDA files (nvcc): the host part gets -Wall -Wextra through -Xcompiler (not -Wpedantic: the
# code nvcc generates for the host compiler uses GCC extensions such as line markers), and with
# DYNG_WARNINGS_AS_ERRORS both nvcc's own warnings (--Werror all-warnings) and the host
# compiler's (-Xcompiler=-Werror) are errors.

function(dyng_set_warnings target)
  if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    set(_flags -Wall -Wextra -Wpedantic)
    if(DYNG_WARNINGS_AS_ERRORS)
      list(APPEND _flags -Werror)
    endif()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_flags}>)
    set(_cuda_flags -Xcompiler=-Wall,-Wextra)
    if(DYNG_WARNINGS_AS_ERRORS)
      list(APPEND _cuda_flags --Werror=all-warnings -Xcompiler=-Werror)
    endif()
    target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CUDA>:${_cuda_flags}>")
  elseif(MSVC)
    set(_flags /W4 /permissive-)
    if(DYNG_WARNINGS_AS_ERRORS)
      list(APPEND _flags /WX)
    endif()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${_flags}>)
  endif()
endfunction()
