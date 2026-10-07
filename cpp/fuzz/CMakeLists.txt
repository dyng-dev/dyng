# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The reader fuzzers (PLAN Sections 5.7 and 8.1; cpp/fuzz/README.md). Each source fuzz_<name>.cpp
# defines LLVMFuzzerTestOneInput() for one reader and builds two ways:
#
#   * DYNG_BUILD_FUZZERS=ON (Clang only; the preset `fuzz`, which also instruments libdyng with
#     -fsanitize=fuzzer-no-link,address,undefined): the libFuzzer executable fuzz_<name>, and the
#     CTest fuzz.corpus.<name> that runs it once over the seed corpus and the crash reproducers;
#     ci/fuzz.sh runs the fuzzers for a bounded time.
#   * DYNG_BUILD_TESTS=ON (any compiler): dyng_fuzz_replay_<name>, the same source linked with
#     replay_main.cpp, and the CTest fuzz.replay.<name> (labels cpu;io;fuzz) over the same inputs,
#     so every pull request replays the corpus and every fixed crash with the regular compilers.
#     For a target with a writer, also dyng_fuzz_selftest_<name> (DYNG_FUZZ_CORRUPT_WRITES=1: every
#     file the writer produces gets a line of garbage) and the CTest fuzz.selftest.<name>, which
#     passes only if the replay reports the broken round trip: the check can fail.

set(DYNG_FUZZ_TARGETS
    matrix_market
    edge_list
    csr_triplet
    legacy_batch
    batch_text
    result_io
)
# The targets whose reader has a writer (the write -> read round trip).
set(DYNG_FUZZ_ROUND_TRIP_TARGETS
    edge_list
    csr_triplet
    legacy_batch
    batch_text
    result_io
)

if(DYNG_BUILD_FUZZERS AND NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  message(FATAL_ERROR "DYNG_BUILD_FUZZERS=ON needs Clang (libFuzzer); use the preset fuzz with "
                      "CXX=clang++ (ci/fuzz.sh)"
  )
endif()

foreach(_name IN LISTS DYNG_FUZZ_TARGETS)
  # The seed corpus and, when a crash was fixed, its reproducers.
  set(_inputs "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${_name}")
  if(IS_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/regressions/${_name}")
    list(APPEND _inputs "${CMAKE_CURRENT_SOURCE_DIR}/regressions/${_name}")
  endif()

  if(DYNG_BUILD_FUZZERS)
    add_executable(fuzz_${_name} fuzz_${_name}.cpp)
    target_link_libraries(fuzz_${_name} PRIVATE dyng::dyng)
    target_compile_features(fuzz_${_name} PRIVATE cxx_std_17)
    dyng_set_warnings(fuzz_${_name})
    dyng_enable_sanitizers(fuzz_${_name})
    target_compile_options(fuzz_${_name} PRIVATE -fsanitize=fuzzer,address,undefined
                                                 -fno-sanitize-recover=undefined)
    target_link_options(fuzz_${_name} PRIVATE -fsanitize=fuzzer,address,undefined)
    # Without a directory of its own, libFuzzer runs each named file once (no new corpus).
    set(_files "")
    foreach(_dir IN LISTS _inputs)
      file(GLOB _found LIST_DIRECTORIES false "${_dir}/*")
      list(SORT _found)
      list(APPEND _files ${_found})
    endforeach()
    add_test(NAME fuzz.corpus.${_name} COMMAND fuzz_${_name} ${_files})
    set_tests_properties(
      fuzz.corpus.${_name}
      PROPERTIES LABELS "cpu;io;fuzz" ENVIRONMENT
                 "UBSAN_OPTIONS=print_stacktrace=1:halt_on_error=1;ASAN_OPTIONS=detect_leaks=1"
    )
  endif()

  if(DYNG_BUILD_TESTS)
    add_executable(dyng_fuzz_replay_${_name} fuzz_${_name}.cpp replay_main.cpp)
    target_link_libraries(dyng_fuzz_replay_${_name} PRIVATE dyng::dyng)
    target_compile_features(dyng_fuzz_replay_${_name} PRIVATE cxx_std_17)
    dyng_set_warnings(dyng_fuzz_replay_${_name})
    dyng_enable_sanitizers(dyng_fuzz_replay_${_name})
    add_test(NAME fuzz.replay.${_name} COMMAND dyng_fuzz_replay_${_name} ${_inputs})
    set_tests_properties(fuzz.replay.${_name} PROPERTIES LABELS "cpu;io;fuzz")
    if(_name IN_LIST DYNG_FUZZ_ROUND_TRIP_TARGETS)
      add_executable(dyng_fuzz_selftest_${_name} fuzz_${_name}.cpp replay_main.cpp)
      target_link_libraries(dyng_fuzz_selftest_${_name} PRIVATE dyng::dyng)
      target_compile_features(dyng_fuzz_selftest_${_name} PRIVATE cxx_std_17)
      target_compile_definitions(dyng_fuzz_selftest_${_name} PRIVATE DYNG_FUZZ_CORRUPT_WRITES=1)
      dyng_set_warnings(dyng_fuzz_selftest_${_name})
      dyng_enable_sanitizers(dyng_fuzz_selftest_${_name})
      add_test(NAME fuzz.selftest.${_name} COMMAND dyng_fuzz_selftest_${_name}
                                                    "${CMAKE_CURRENT_SOURCE_DIR}/corpus/${_name}")
      set_tests_properties(
        fuzz.selftest.${_name} PROPERTIES LABELS "cpu;io;fuzz" PASS_REGULAR_EXPRESSION
                                          "fuzz check failed"
      )
    endif()
  endif()
endforeach()
