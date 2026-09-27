// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file util_test.cpp
 * @brief The private text scanner, text writer and legacy random-number reproduction.
 */
#include "support/data_paths.hpp"
#include "util/parser.hpp"
#include "util/rng.hpp"
#include "util/text_writer.hpp"

#include <dyng/core/error.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <limits>
#include <random>
#include <string>
#include <vector>

namespace {

TEST(TextScanner, TokensLinesAndColumns) {
  const std::string text = "a  bb\n\n\tccc d\r\ne";
  dyng::detail::text_scanner scanner(text, "f.txt");
  dyng::detail::token tok;
  std::vector<std::string> seen;
  std::vector<std::int64_t> lines;
  std::vector<std::int64_t> columns;
  while (scanner.next_token(tok)) {
    seen.emplace_back(tok.text);
    lines.push_back(tok.line);
    columns.push_back(tok.column);
  }
  EXPECT_EQ(seen, (std::vector<std::string>{"a", "bb", "ccc", "d", "e"}));
  EXPECT_EQ(lines, (std::vector<std::int64_t>{1, 1, 3, 3, 4}));
  EXPECT_EQ(columns, (std::vector<std::int64_t>{1, 4, 2, 6, 1}));

  dyng::detail::text_scanner by_line(text, "f.txt");
  std::vector<int> per_line;
  while (by_line.next_line()) {
    int count = 0;
    while (by_line.next_in_line(tok)) {
      ++count;
    }
    per_line.push_back(count);
  }
  EXPECT_EQ(per_line, (std::vector<int>{2, 0, 2, 1}));
}

TEST(TextScanner, StrictIntegers) {
  std::int64_t value = 0;
  EXPECT_TRUE(dyng::detail::parse_int64("-42", value));
  EXPECT_EQ(value, -42);
  EXPECT_TRUE(dyng::detail::parse_int64("007", value));
  EXPECT_EQ(value, 7);
  for (const char* bad : {"", "+1", "1.0", "12abc", "0x10", "9223372036854775808", "-", " 1"}) {
    EXPECT_FALSE(dyng::detail::parse_int64(bad, value)) << bad;
  }
  const std::string text = "5 70000";
  dyng::detail::text_scanner scanner(text, "in.txt");
  dyng::detail::token tok;
  ASSERT_TRUE(scanner.next_token(tok));
  EXPECT_EQ(dyng::detail::parse_integer<std::int16_t>(scanner, tok, 0, 32767, "x"), 5);
  ASSERT_TRUE(scanner.next_token(tok));
  try {
    (void)dyng::detail::parse_integer<std::int16_t>(scanner, tok, 0, 32767, "x");
    FAIL() << "expected io_error";
  } catch (const dyng::io_error& e) {
    EXPECT_EQ(e.path(), "in.txt");
    EXPECT_EQ(e.line(), 1);
    EXPECT_EQ(e.column(), 3);
    EXPECT_NE(std::string(e.what()).find("in.txt:1:3"), std::string::npos) << e.what();
  }
}

TEST(TextWriter, WritesAndReportsErrors) {
  dyng::test::temp_dir tmp;
  {
    dyng::detail::text_writer out(tmp.path("a/b/c.txt"));
    out.put(std::int64_t{-12});
    out.put_char(' ');
    out.put("INF");
    out.put_char('\n');
    out.close();
    out.close();  // idempotent
  }
  EXPECT_EQ(dyng::test::read_text(tmp.path("a/b/c.txt")), "-12 INF\n");
  EXPECT_THROW(dyng::detail::text_writer(tmp.path("a/b/c.txt/cannot")), dyng::io_error);
  EXPECT_THROW((void)dyng::detail::read_file(tmp.path("missing.txt")), dyng::io_error);
}

TEST(LegacyUniformInt, DeterministicAndInRange) {
  std::mt19937 a(12345);
  std::mt19937 b(12345);
  for (int i = 0; i < 1000; ++i) {
    const auto x = dyng::detail::legacy_uniform_int(a, 1, 100);
    EXPECT_EQ(x, dyng::detail::legacy_uniform_int(b, 1, 100));
    EXPECT_GE(x, 1);
    EXPECT_LE(x, 100);
  }
  EXPECT_THROW((void)dyng::detail::legacy_uniform_int(a, 2, 1), dyng::invalid_argument_error);
  EXPECT_THROW((void)dyng::detail::legacy_uniform_int(a, 0, std::int64_t{1} << 33),
               dyng::invalid_argument_error);
}

#if defined(__GLIBCXX__) && defined(_GLIBCXX_RELEASE) && _GLIBCXX_RELEASE >= 11
// libstdc++ 11+ (Lemire's method) is what the pinned originals were built with; the library
// reproduction must agree with it draw for draw.
TEST(LegacyUniformInt, MatchesLibstdcxxUniformIntDistribution) {
  struct range {
    std::int64_t a;
    std::int64_t b;
  };
  const range ranges[] = {
      {1, 100},
      {1, 30},
      {0, 0},
      {-5, 5},
      {1, std::numeric_limits<std::int32_t>::max()},
      {std::numeric_limits<std::int32_t>::min(), std::numeric_limits<std::int32_t>::max()},
      {0, (std::int64_t{1} << 31) + 12345},
      {0, 0xffffffffLL},
      {7, 3000000000LL}};
  for (const auto& r : ranges) {
    SCOPED_TRACE(std::to_string(r.a) + ".." + std::to_string(r.b));
    for (const std::uint32_t seed : {1u, 7u, 12345u, 777u}) {
      std::mt19937 ours(seed);
      std::mt19937 theirs(seed);
      std::uniform_int_distribution<long long> dist(r.a, r.b);
      for (int i = 0; i < 2000; ++i) {
        ASSERT_EQ(dyng::detail::legacy_uniform_int(ours, r.a, r.b), dist(theirs));
      }
      if (r.a >= std::numeric_limits<int>::min() && r.b <= std::numeric_limits<int>::max()) {
        std::mt19937 ours_int(seed);
        std::mt19937 theirs_int(seed);
        std::uniform_int_distribution<int> dist_int(static_cast<int>(r.a), static_cast<int>(r.b));
        for (int i = 0; i < 2000; ++i) {
          ASSERT_EQ(dyng::detail::legacy_uniform_int(ours_int, r.a, r.b), dist_int(theirs_int));
        }
      }
    }
  }
}
#endif

}  // namespace
