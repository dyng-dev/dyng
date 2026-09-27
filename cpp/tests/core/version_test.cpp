// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/version.hpp>

#include <gtest/gtest.h>

#include <string>

TEST(Version, HeaderAndLibraryAgree) {
  const auto h = dyng::header_version();
  const auto l = dyng::library_version();
  EXPECT_EQ(h.major_version, l.major_version);
  EXPECT_EQ(h.minor_version, l.minor_version);
  EXPECT_EQ(h.patch_version, l.patch_version);
  EXPECT_STREQ(h.string, l.string);
}

TEST(Version, StringStartsWithNumericVersion) {
  const std::string numeric = std::to_string(DYNG_VERSION_MAJOR) + "." +
                              std::to_string(DYNG_VERSION_MINOR) + "." +
                              std::to_string(DYNG_VERSION_PATCH);
  EXPECT_EQ(std::string(DYNG_VERSION_STRING).rfind(numeric, 0), 0U) << DYNG_VERSION_STRING;
  EXPECT_EQ(DYNG_VERSION,
            DYNG_VERSION_MAJOR * 10000 + DYNG_VERSION_MINOR * 100 + DYNG_VERSION_PATCH);
}
