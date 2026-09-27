// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>

#include <gtest/gtest.h>

#include <chrono>
#include <future>
#include <string>
#include <utility>
#include <vector>

namespace {

/// Restores the global log level and sink after each test.
class Logging : public ::testing::Test {
 protected:
  void SetUp() override {
    saved_level_ = dyng::get_log_level();
    dyng::set_log_sink([this](dyng::log_level level, std::string_view message) {
      captured_.emplace_back(level, std::string(message));
    });
  }
  void TearDown() override {
    dyng::set_log_sink({});
    dyng::set_log_level(saved_level_);
  }

  std::vector<std::pair<dyng::log_level, std::string>> captured_;

 private:
  dyng::log_level saved_level_ = dyng::log_level::warn;
};

}  // namespace

TEST_F(Logging, FiltersByLevel) {
  dyng::set_log_level(dyng::log_level::warn);
  dyng::log_message(dyng::log_level::error, "e");
  dyng::log_message(dyng::log_level::warn, "w");
  dyng::log_message(dyng::log_level::info, "i");
  dyng::log_message(dyng::log_level::debug, "d");
  ASSERT_EQ(captured_.size(), 2U);
  EXPECT_EQ(captured_[0].first, dyng::log_level::error);
  EXPECT_EQ(captured_[0].second, "e");
  EXPECT_EQ(captured_[1].second, "w");

  dyng::set_log_level(dyng::log_level::trace);
  dyng::log_message(dyng::log_level::trace, "t");
  EXPECT_EQ(captured_.size(), 3U);

  dyng::set_log_level(dyng::log_level::off);
  dyng::log_message(dyng::log_level::error, "nothing");
  EXPECT_EQ(captured_.size(), 3U);
}

TEST_F(Logging, OffIsNeverEnabled) {
  dyng::set_log_level(dyng::log_level::trace);
  EXPECT_FALSE(dyng::log_enabled(dyng::log_level::off));
  EXPECT_TRUE(dyng::log_enabled(dyng::log_level::trace));
}

TEST_F(Logging, ParseAndNameLevels) {
  EXPECT_EQ(dyng::parse_log_level("DEBUG"), dyng::log_level::debug);
  EXPECT_EQ(dyng::parse_log_level("warning"), dyng::log_level::warn);
  EXPECT_EQ(dyng::parse_log_level("off"), dyng::log_level::off);
  EXPECT_THROW((void)dyng::parse_log_level("loud"), dyng::invalid_argument_error);
  for (auto level : {dyng::log_level::off, dyng::log_level::error, dyng::log_level::warn,
                     dyng::log_level::info, dyng::log_level::debug, dyng::log_level::trace}) {
    EXPECT_EQ(dyng::parse_log_level(dyng::to_string(level)), level);
  }
}

TEST_F(Logging, EmptySinkRestoresDefault) {
  dyng::set_log_sink({});
  dyng::set_log_level(dyng::log_level::off);
  EXPECT_NO_THROW(dyng::log_message(dyng::log_level::error, "not printed"));
}

TEST_F(Logging, SinkMayLogAgain) {
  // A sink that logs (for example a forwarding sink whose handler calls back into dynG) must not
  // deadlock: the sink runs without the library's lock.
  dyng::set_log_level(dyng::log_level::info);
  int depth = 0;
  dyng::set_log_sink([this, &depth](dyng::log_level level, std::string_view message) {
    captured_.emplace_back(level, std::string(message));
    if (depth++ == 0) {
      dyng::log_message(dyng::log_level::info, "nested");
    }
  });
  auto logged =
      std::async(std::launch::async, [] { dyng::log_message(dyng::log_level::warn, "outer"); });
  ASSERT_EQ(logged.wait_for(std::chrono::seconds(10)), std::future_status::ready)
      << "a sink that logs deadlocked";
  logged.get();
  ASSERT_EQ(captured_.size(), 2U);
  EXPECT_EQ(captured_[0].second, "outer");
  EXPECT_EQ(captured_[1].second, "nested");
}

TEST_F(Logging, SinkMayReplaceItself) {
  dyng::set_log_level(dyng::log_level::info);
  dyng::set_log_sink([this](dyng::log_level level, std::string_view message) {
    captured_.emplace_back(level, std::string(message));
    dyng::set_log_sink([this](dyng::log_level, std::string_view) {
      captured_.emplace_back(dyng::log_level::error, "second sink");
    });
  });
  dyng::log_message(dyng::log_level::info, "first");
  dyng::log_message(dyng::log_level::info, "again");
  ASSERT_EQ(captured_.size(), 2U);
  EXPECT_EQ(captured_[0].second, "first");
  EXPECT_EQ(captured_[1].second, "second sink");
}
