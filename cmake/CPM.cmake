# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Bootstrap for CPM.cmake (https://github.com/cpm-cmake/CPM.cmake, MIT). CPM itself is not
# vendored (PLAN Section 3.4): this file downloads the pinned release once, verifies its SHA-256,
# and includes it. Set CPM_SOURCE_CACHE (scripts/dev_env.sh does) to share downloads between
# build trees; offline builds can point CPM_DOWNLOAD_LOCATION at a pre-downloaded copy.

set(DYNG_CPM_VERSION 0.43.2)
set(DYNG_CPM_SHA256 49a3bef91ceb65bb66d57255e12d1ffd22abc2f6408fa9fe4534c544a2f232aa)

if(CPM_SOURCE_CACHE)
  set(CPM_DOWNLOAD_LOCATION "${CPM_SOURCE_CACHE}/cpm/CPM_${DYNG_CPM_VERSION}.cmake")
elseif(DEFINED ENV{CPM_SOURCE_CACHE})
  set(CPM_DOWNLOAD_LOCATION "$ENV{CPM_SOURCE_CACHE}/cpm/CPM_${DYNG_CPM_VERSION}.cmake")
else()
  set(CPM_DOWNLOAD_LOCATION "${CMAKE_BINARY_DIR}/cmake/CPM_${DYNG_CPM_VERSION}.cmake")
endif()
get_filename_component(CPM_DOWNLOAD_LOCATION "${CPM_DOWNLOAD_LOCATION}" ABSOLUTE)

file(
  DOWNLOAD
  "https://github.com/cpm-cmake/CPM.cmake/releases/download/v${DYNG_CPM_VERSION}/CPM.cmake"
  "${CPM_DOWNLOAD_LOCATION}"
  EXPECTED_HASH SHA256=${DYNG_CPM_SHA256}
)

include("${CPM_DOWNLOAD_LOCATION}")
