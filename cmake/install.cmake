# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# Install and export rules (PLAN Section 7.6): the dyng::dyng target (and dyng::testing once it
# exists), the public headers, the generated version.hpp / config.hpp, and the package config
# for find_package(dyng).

include(CMakePackageConfigHelpers)

set(DYNG_INSTALL_CMAKEDIR "${CMAKE_INSTALL_LIBDIR}/cmake/dyng")

set(_dyng_install_targets dyng)
if(TARGET dyng_testing)
  list(APPEND _dyng_install_targets dyng_testing)
endif()

install(
  TARGETS ${_dyng_install_targets}
  EXPORT dyng-targets
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)

install(
  DIRECTORY "${PROJECT_SOURCE_DIR}/cpp/include/dyng"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  FILES_MATCHING
  PATTERN "*.hpp"
)
install(
  FILES "${DYNG_GENERATED_INCLUDE_DIR}/dyng/version.hpp" "${DYNG_GENERATED_INCLUDE_DIR}/dyng/config.hpp"
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/dyng
)

install(
  EXPORT dyng-targets
  NAMESPACE dyng::
  FILE dyng-targets.cmake
  DESTINATION ${DYNG_INSTALL_CMAKEDIR}
)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/dyng-config.cmake.in"
  "${PROJECT_BINARY_DIR}/dyng-config.cmake"
  INSTALL_DESTINATION ${DYNG_INSTALL_CMAKEDIR}
)

# Before 1.0 a minor release may break the API (SemVer 0.x), hence SameMinorVersion.
if(PROJECT_VERSION_MAJOR EQUAL 0)
  set(_dyng_compat SameMinorVersion)
else()
  set(_dyng_compat SameMajorVersion)
endif()
write_basic_package_version_file(
  "${PROJECT_BINARY_DIR}/dyng-config-version.cmake"
  VERSION ${PROJECT_VERSION}
  COMPATIBILITY ${_dyng_compat}
)

install(
  FILES "${PROJECT_BINARY_DIR}/dyng-config.cmake" "${PROJECT_BINARY_DIR}/dyng-config-version.cmake"
  DESTINATION ${DYNG_INSTALL_CMAKEDIR}
)
