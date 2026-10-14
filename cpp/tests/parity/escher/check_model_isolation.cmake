# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The include-graph and link check of dyng::testing::cbst_host_model (M8a, ADR 0037; PLAN 5.8:
# an oracle shares no code with what it checks), and of dyng::testing::rebuild_line_graph (M8b,
# ADR 0042). Run by CTest (cbst_model_isolation, line_graph_oracle_isolation):
#
#   cmake -DCXX=<compiler> -DNM=<nm> -DREPO=<source dir> -DSOURCE=<cbst_host_model.cpp>
#         -DOBJECT=<its object file> [-DHEADER=<its public header>] -P check_model_isolation.cmake
#
# HEADER defaults to cpp/include/dyng/testing/cbst_host_model.hpp.
# 1. The preprocessor's dependency list of the model's source (-MM) names no project file but
#    the model's own header, so nothing of cpp/src (cpp/src/hypergraph in particular) or of any
#    other dynG header is included.
# 2. The model's object file references (nm --undefined-only) no symbol of namespace dyng: it
#    links nothing of the library, only the C++ standard library.
foreach(_var CXX NM REPO SOURCE OBJECT)
  if(NOT DEFINED ${_var} OR "${${_var}}" STREQUAL "")
    message(FATAL_ERROR "check_model_isolation: -D${_var}= is required")
  endif()
endforeach()

execute_process(
  COMMAND "${CXX}" -std=c++17 -I "${REPO}/cpp/include" -MM "${SOURCE}"
  OUTPUT_VARIABLE _deps
  ERROR_VARIABLE _err
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "check_model_isolation: ${CXX} -MM failed: ${_err}")
endif()
string(REPLACE "\\\n" " " _deps "${_deps}")
string(REGEX REPLACE "^[^:]*:" "" _deps "${_deps}")
separate_arguments(_deps UNIX_COMMAND "${_deps}")
if(NOT DEFINED HEADER OR "${HEADER}" STREQUAL "")
  set(HEADER "${REPO}/cpp/include/dyng/testing/cbst_host_model.hpp")
endif()
get_filename_component(_name "${HEADER}" NAME_WE)
set(_allowed "${HEADER}")
set(_bad "")
foreach(_dep IN LISTS _deps)
  file(REAL_PATH "${_dep}" _real BASE_DIRECTORY "${REPO}")
  if(_real STREQUAL "${SOURCE}")
    continue()
  endif()
  string(FIND "${_real}" "${REPO}/" _in_repo)
  if(_in_repo EQUAL 0 AND NOT _real STREQUAL _allowed)
    list(APPEND _bad "${_real}")
  endif()
endforeach()
if(_bad)
  message(FATAL_ERROR "check_model_isolation: ${_name} includes project files other than "
                      "its header: ${_bad}")
endif()

execute_process(
  COMMAND "${NM}" -C --undefined-only "${OBJECT}"
  OUTPUT_VARIABLE _syms
  ERROR_VARIABLE _err
  RESULT_VARIABLE _rc
)
if(NOT _rc EQUAL 0)
  message(FATAL_ERROR "check_model_isolation: ${NM} failed: ${_err}")
endif()
string(REPLACE "\n" ";" _syms "${_syms}")
set(_bad "")
foreach(_sym IN LISTS _syms)
  if(_sym MATCHES "dyng::")
    list(APPEND _bad "${_sym}")
  endif()
endforeach()
if(_bad)
  message(FATAL_ERROR "check_model_isolation: the object of ${_name} references dynG symbols: "
                      "${_bad}")
endif()
list(LENGTH _deps _num_deps)
message(STATUS "${_name}: ${_num_deps} dependencies, no project file but its header; "
               "no undefined dyng symbol")
