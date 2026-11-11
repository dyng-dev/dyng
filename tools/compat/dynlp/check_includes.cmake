# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng-compat-dynlp includes installed public headers only (M10a, ADR 0073): every #include of
# SOURCE is a standard header (<name> without a slash and without dyng) or a header of the
# installed tree INCLUDE_DIR/dyng (dyng/<name>.hpp, dyng/<module>/<name>.hpp; no detail or
# experimental header), and none is a quoted include.
#   cmake -DSOURCE=<dynlp_compat.cpp> -DINCLUDE_DIR=<cpp/include> -P check_includes.cmake
file(STRINGS "${SOURCE}" _lines REGEX "^[ \t]*#[ \t]*include")
set(_bad)
set(_count 0)
foreach(_line IN LISTS _lines)
  math(EXPR _count "${_count} + 1")
  if(_line MATCHES "#[ \t]*include[ \t]*\"")
    list(APPEND _bad "${_line} (a quoted include)")
  elseif(_line MATCHES "#[ \t]*include[ \t]*<(dyng/[^>]+)>")
    set(_header "${CMAKE_MATCH_1}")
    if(_header MATCHES "/detail/|/experimental/")
      list(APPEND _bad "${_line} (not a public header)")
    elseif(NOT EXISTS "${INCLUDE_DIR}/${_header}")
      list(APPEND _bad "${_line} (not under ${INCLUDE_DIR})")
    endif()
  elseif(_line MATCHES "#[ \t]*include[ \t]*<([^>]+)>")
    if(CMAKE_MATCH_1 MATCHES "/")
      list(APPEND _bad "${_line} (a header with a path that is not dyng's)")
    endif()
  else()
    list(APPEND _bad "${_line} (not understood)")
  endif()
endforeach()
if(_bad)
  list(JOIN _bad "\n  " _list)
  message(FATAL_ERROR "dyng-compat-dynlp includes more than the installed public headers:\n  ${_list}")
endif()
message(STATUS "dyng-compat-dynlp: ${_count} includes, all standard or installed public headers")
