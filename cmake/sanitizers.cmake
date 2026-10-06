# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Host sanitizers (PLAN Section 7.3): DYNG_SANITIZE is empty, "address;undefined" or "thread"
# (the preset fuzz adds Clang's "fuzzer-no-link": coverage instrumentation for libFuzzer).
# With DYNG_STDLIB_ASSERTIONS (ON in Debug builds) the host C++ code is also compiled with
# _GLIBCXX_ASSERTIONS: libstdc++ checks every std::vector::operator[] and the like, so an
# out-of-bounds access aborts in the dev preset instead of passing unseen (it does not change
# the ABI, so objects built with and without it link together).
#   dyng_enable_sanitizers(<target>)

function(dyng_enable_sanitizers target)
  if(DYNG_STDLIB_ASSERTIONS)
    target_compile_definitions(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:_GLIBCXX_ASSERTIONS>")
  endif()
  if(NOT DYNG_SANITIZE)
    return()
  endif()
  if(NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "DYNG_SANITIZE is only supported with GCC and Clang")
  endif()
  list(JOIN DYNG_SANITIZE "," _list)
  # Host code only: nvcc does not take -fsanitize (compute-sanitizer covers device code).
  set(_flags -fsanitize=${_list} -fno-omit-frame-pointer)
  target_compile_options(${target} PRIVATE "$<$<COMPILE_LANGUAGE:CXX>:${_flags}>")
  target_link_options(${target} PRIVATE -fsanitize=${_list})
endfunction()
