// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <stdexcept>
#include <string>
#include <type_traits>

namespace {

int checked_positive(int n) {
  DYNG_EXPECTS(n > 0, "n must be positive, got ", n);
  return n;
}

[[noreturn]] void always_fail() {
  DYNG_FAIL("broken invariant ", 42);
}

}  // namespace

TEST(Error, HierarchyDerivesFromBase) {
  static_assert(std::is_base_of_v<std::runtime_error, dyng::error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::invalid_argument_error>);
  static_assert(std::is_base_of_v<dyng::invalid_argument_error, dyng::stale_result_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::io_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::capacity_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::not_supported_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::convergence_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::cuda_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::out_of_memory_error>);
  static_assert(std::is_base_of_v<dyng::error, dyng::internal_error>);
  static_assert(!std::is_base_of_v<std::invalid_argument, dyng::invalid_argument_error>);
  SUCCEED();
}

TEST(Error, ExpectsPassesWhenConditionHolds) {
  EXPECT_EQ(checked_positive(3), 3);
}

TEST(Error, ExpectsThrowsInvalidArgumentWithMessageAndLocation) {
  try {
    checked_positive(-2);
    FAIL() << "expected invalid_argument_error";
  } catch (const dyng::invalid_argument_error& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("n must be positive, got -2"), std::string::npos) << what;
    EXPECT_NE(what.find("[expected: n > 0]"), std::string::npos) << what;
    EXPECT_NE(what.find("error_test.cpp:"), std::string::npos) << what;
    EXPECT_EQ(what.find("/"), std::string::npos) << "only the file name is reported: " << what;
  }
}

TEST(Error, FailThrowsInternalError) {
  try {
    always_fail();
  } catch (const dyng::internal_error& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("internal error: broken invariant 42"), std::string::npos) << what;
    return;
  }
  FAIL() << "expected internal_error";
}

TEST(Error, IoErrorCarriesLocation) {
  const dyng::io_error e("bad token", "graph.mtx", 12, 7);
  EXPECT_STREQ(e.what(), "bad token");
  EXPECT_EQ(e.path(), "graph.mtx");
  EXPECT_EQ(e.line(), 12);
  EXPECT_EQ(e.column(), 7);
  const dyng::io_error unknown("missing");
  EXPECT_TRUE(unknown.path().empty());
  EXPECT_EQ(unknown.line(), 0);
}

TEST(Error, CudaErrorCarriesCode) {
  const dyng::cuda_error e("cudaMalloc failed", 2);
  EXPECT_EQ(e.code(), 2);
  EXPECT_THROW(throw e, dyng::error);
}

TEST(Error, SourceBasename) {
  EXPECT_STREQ(dyng::detail::source_basename("/a/b/c.cpp"), "c.cpp");
  EXPECT_STREQ(dyng::detail::source_basename("c.cpp"), "c.cpp");
  EXPECT_STREQ(dyng::detail::source_basename("a\\b.cpp"), "b.cpp");
  EXPECT_STREQ(dyng::detail::source_basename(nullptr), "");
}
