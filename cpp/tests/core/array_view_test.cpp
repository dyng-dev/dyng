// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/array_view.hpp>

#include <gtest/gtest.h>

#include <numeric>
#include <type_traits>
#include <vector>

TEST(ArrayView, IsTriviallyCopyable) {
  static_assert(std::is_trivially_copyable_v<dyng::array_view<int>>);
  static_assert(std::is_trivially_copyable_v<dyng::array_view<const double>>);
  SUCCEED();
}

TEST(ArrayView, DefaultIsEmptyHost) {
  const dyng::array_view<int> v;
  EXPECT_TRUE(v.empty());
  EXPECT_EQ(v.size(), 0U);
  EXPECT_EQ(v.data(), nullptr);
  EXPECT_EQ(v.space(), dyng::memory_space::host);
  EXPECT_EQ(v.device(), -1);
}

TEST(ArrayView, ViewsVector) {
  std::vector<int> data(5);
  std::iota(data.begin(), data.end(), 10);
  auto v = dyng::host_view(data);
  static_assert(std::is_same_v<decltype(v), dyng::array_view<int>>);
  EXPECT_EQ(v.size(), 5U);
  EXPECT_EQ(v.size_bytes(), 5 * sizeof(int));
  EXPECT_EQ(v[2], 12);
  v[2] = 99;
  EXPECT_EQ(data[2], 99);
  EXPECT_EQ(std::accumulate(v.begin(), v.end(), 0), 10 + 11 + 99 + 13 + 14);

  const std::vector<int>& cdata = data;
  auto cv = dyng::host_view(cdata);
  static_assert(std::is_same_v<decltype(cv), dyng::array_view<const int>>);
  EXPECT_EQ(cv.data(), data.data());
}

TEST(ArrayView, ConvertsToConst) {
  std::vector<float> data{1.0F, 2.0F};
  const dyng::array_view<float> v = dyng::host_view(data);
  const dyng::array_view<const float> cv = v;
  EXPECT_EQ(cv.data(), v.data());
  EXPECT_EQ(cv.size(), v.size());
  static_assert(!std::is_convertible_v<dyng::array_view<const float>, dyng::array_view<float>>);
  static_assert(!std::is_convertible_v<dyng::array_view<int>, dyng::array_view<const float>>);
}

TEST(ArrayView, CheckedAccess) {
  std::vector<int> data{1, 2, 3};
  auto v = dyng::host_view(data);
  EXPECT_EQ(v.at(1), 2);
  EXPECT_THROW((void)v.at(3), dyng::invalid_argument_error);
  auto d = dyng::device_view(data.data(), data.size(), 1);
  EXPECT_EQ(d.space(), dyng::memory_space::device);
  EXPECT_EQ(d.device(), 1);
  EXPECT_THROW((void)d.at(0), dyng::invalid_argument_error);
}

TEST(ArrayView, Subview) {
  std::vector<int> data{0, 1, 2, 3, 4, 5};
  auto v = dyng::host_view(data);
  auto s = v.subview(2, 3);
  EXPECT_EQ(s.size(), 3U);
  EXPECT_EQ(s[0], 2);
  EXPECT_EQ(s[2], 4);
  EXPECT_TRUE(v.subview(6, 0).empty());
  EXPECT_THROW((void)v.subview(4, 3), dyng::invalid_argument_error);
  EXPECT_THROW((void)v.subview(7, 0), dyng::invalid_argument_error);
}

TEST(ArrayView, HostAccessibility) {
  EXPECT_TRUE(dyng::is_host_accessible(dyng::memory_space::host));
  EXPECT_TRUE(dyng::is_host_accessible(dyng::memory_space::pinned_host));
  EXPECT_TRUE(dyng::is_host_accessible(dyng::memory_space::managed));
  EXPECT_FALSE(dyng::is_host_accessible(dyng::memory_space::device));
}
