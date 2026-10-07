# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The source rules of the CBST store (M8a, acceptance criterion 9; PLAN 4.6 rule 7, 4.7.3): none
# of ESCHER-GPU's process exits, prints, verbose() / ESCHER_DEBUG switches, device-wide
# synchronizations, raw cudaMalloc, Thrust calls without the resources' par_nosync policy or
# kernel launches on the default stream is left in cpp/src/hypergraph; and no public header but
# the documented exception dyng/testing/cbst_host_model.hpp exposes the CBST (its keys, offsets,
# +1 shifts or sentinels: no header names a cbst_ identifier or includes the store).
#
#   cmake -DSOURCE_DIR=<repo>/cpp -P check_source_rules.cmake
cmake_minimum_required(VERSION 3.20)

if(NOT SOURCE_DIR)
  message(FATAL_ERROR "SOURCE_DIR is required")
endif()

set(_violations "")
file(GLOB_RECURSE _sources "${SOURCE_DIR}/src/hypergraph/*.cu" "${SOURCE_DIR}/src/hypergraph/*.cuh"
     "${SOURCE_DIR}/src/hypergraph/*.cpp" "${SOURCE_DIR}/src/hypergraph/*.hpp"
)
if(NOT _sources)
  message(FATAL_ERROR "no sources under ${SOURCE_DIR}/src/hypergraph")
endif()

# name|regular expression (matched against every line that is not a // comment)
set(_rules
    "exit()|(^|[^_A-Za-z0-9])(std::)?(exit|abort|_Exit|quick_exit)[ ]*\\("
    "printf|(^|[^_A-Za-z0-9])(std::)?(printf|fprintf|puts)[ ]*\\("
    "iostream output|std::(cout|cerr|clog)"
    "verbose()|(^|[^_A-Za-z0-9])verbose[ ]*\\("
    "ESCHER_DEBUG|ESCHER_DEBUG"
    "cudaDeviceSynchronize|cudaDeviceSynchronize"
    "raw cudaMalloc|(^|[^_A-Za-z0-9])cudaMalloc(Async|Managed|Host)?[ ]*\\("
    "raw cudaFree|(^|[^_A-Za-z0-9])cudaFree(Async|Host)?[ ]*\\("
    "synchronous cudaMemcpy or cudaMemset|(^|[^_A-Za-z0-9])cuda(Memcpy|Memset)[ ]*\\("
)
foreach(_file IN LISTS _sources)
  file(STRINGS "${_file}" _lines)
  file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_file}")
  set(_number 0)
  set(_in_launch FALSE)
  set(_launch "")
  foreach(_line IN LISTS _lines)
    math(EXPR _number "${_number} + 1")
    string(STRIP "${_line}" _stripped)
    if(_stripped MATCHES "^//" OR _stripped MATCHES "^\\*" OR _stripped MATCHES "^/\\*")
      continue()
    endif()
    string(REGEX REPLACE "//.*$" "" _code "${_line}")
    foreach(_rule IN LISTS _rules)
      string(FIND "${_rule}" "|" _bar)
      string(SUBSTRING "${_rule}" 0 ${_bar} _name)
      math(EXPR _start "${_bar} + 1")
      string(SUBSTRING "${_rule}" ${_start} -1 _regex)
      if(_code MATCHES "${_regex}")
        string(APPEND _violations "${_rel}:${_number}: ${_name}: ${_stripped}\n")
      endif()
    endforeach()
    # Thrust: every algorithm takes the resources' policy (thrust_policy(res, alloc)) first.
    if(_code MATCHES "thrust::(sequence|sort|sort_by_key|stable_sort|stable_sort_by_key|inclusive_scan|exclusive_scan|count_if|count|copy_if|copy|transform|reduce|fill|unique|lower_bound|upper_bound|for_each|gather|scatter)[ ]*\\(")
      if(NOT _code MATCHES "thrust::[a-z_]+[ ]*\\([ ]*thrust_policy[ ]*\\(")
        string(APPEND _violations "${_rel}:${_number}: Thrust without par_nosync.on(stream): ${_stripped}\n")
      endif()
    endif()
    # Kernel launches: the launch configuration names a stream (four arguments).
    if(_code MATCHES "<<<")
      set(_in_launch TRUE)
      set(_launch "")
      set(_launch_line ${_number})
    endif()
    if(_in_launch)
      string(APPEND _launch "${_code}")
      if(_launch MATCHES ">>>")
        string(REGEX REPLACE ".*<<<(.*)>>>.*" "\\1" _config "${_launch}")
        # Count the top-level commas of the configuration.
        string(REGEX REPLACE "\\([^()]*\\)" "" _flat "${_config}")
        string(REGEX REPLACE "\\([^()]*\\)" "" _flat "${_flat}")
        string(REGEX MATCHALL "," _commas "${_flat}")
        list(LENGTH _commas _count)
        if(_count LESS 3)
          string(APPEND _violations "${_rel}:${_launch_line}: launch on the default stream: ${_config}\n")
        endif()
        set(_in_launch FALSE)
      endif()
    endif()
  endforeach()
endforeach()

# The public headers: nothing of the store.
file(GLOB_RECURSE _headers "${SOURCE_DIR}/include/dyng/*.hpp" "${SOURCE_DIR}/include/dyng/*.cuh")
foreach(_header IN LISTS _headers)
  file(RELATIVE_PATH _rel "${SOURCE_DIR}" "${_header}")
  if(_rel STREQUAL "include/dyng/testing/cbst_host_model.hpp")
    continue()  # the documented exception (ADR 0037): its subject is the CBST layout
  endif()
  file(STRINGS "${_header}" _lines REGEX "cbst_|hypergraph/escher|CBST")
  foreach(_line IN LISTS _lines)
    string(STRIP "${_line}" _stripped)
    string(APPEND _violations "${_rel}: a public header exposes the CBST store: ${_stripped}\n")
  endforeach()
endforeach()

if(_violations)
  message(FATAL_ERROR "CBST source rules violated:\n${_violations}")
endif()
list(LENGTH _sources _n)
message(STATUS "CBST source rules: ${_n} files of cpp/src/hypergraph and the public headers are clean")
