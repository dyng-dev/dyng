# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# dyng_extract_doc_snippet(<file> <name> <code-out> <expected-out>)
#
# Extract the code snippet <name> of a Markdown file (a fenced block after the comment
# `<!-- snippet: <name> -->`) into <code-out>, and its output block (after
# `<!-- snippet-output: <name> -->`) into <expected-out>, at configure time; an edit of the file
# re-runs the configure step. The format is that of ci/doc_snippets.py (PLAN Section 9.6: the
# README quickstarts are compiled and run, so they cannot drift).

function(_dyng_snippet_block text marker file out_var)
  string(FIND "${text}" "${marker}" _at)
  if(_at EQUAL -1)
    message(FATAL_ERROR "${file}: no '${marker}' (ci/doc_snippets.py describes the format)")
  endif()
  string(SUBSTRING "${text}" ${_at} -1 _rest)
  string(FIND "${_rest}" "```" _open)
  string(SUBSTRING "${_rest}" ${_open} -1 _rest)
  string(FIND "${_rest}" "\n" _eol)  # the end of the opening fence line (```cpp)
  math(EXPR _eol "${_eol} + 1")
  string(SUBSTRING "${_rest}" ${_eol} -1 _rest)
  string(FIND "${_rest}" "\n```" _close)
  if(_close EQUAL -1)
    message(FATAL_ERROR "${file}: unterminated code block after '${marker}'")
  endif()
  math(EXPR _close "${_close} + 1")
  string(SUBSTRING "${_rest}" 0 ${_close} _block)
  set(${out_var} "${_block}" PARENT_SCOPE)
endfunction()

function(dyng_extract_doc_snippet file name code_out expected_out)
  set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${file}")
  file(READ "${file}" _text)
  _dyng_snippet_block("${_text}" "<!-- snippet: ${name} -->" "${file}" _code)
  _dyng_snippet_block("${_text}" "<!-- snippet-output: ${name} -->" "${file}" _expected)
  file(CONFIGURE OUTPUT "${code_out}" CONTENT "${_code}" @ONLY)
  file(CONFIGURE OUTPUT "${expected_out}" CONTENT "${_expected}" @ONLY)
endfunction()
