// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/buffer.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

TEST(Buffer, IsMoveOnly) {
  static_assert(!std::is_copy_constructible_v<dyng::buffer<int>>);
  static_assert(!std::is_copy_assignable_v<dyng::buffer<int>>);
  static_assert(std::is_nothrow_move_constructible_v<dyng::buffer<int>>);
  static_assert(std::is_nothrow_move_assignable_v<dyng::buffer<int>>);
  SUCCEED();
}

TEST(Buffer, DefaultIsEmpty) {
  const dyng::buffer<double> b;
  EXPECT_TRUE(b.empty());
  EXPECT_EQ(b.data(), nullptr);
  EXPECT_EQ(b.space(), dyng::memory_space::host);
}

TEST(Buffer, AllocatesFromResources) {
  const auto res = dyng::resources::sequential();
  dyng::buffer<std::int64_t> b(res, 100);
  EXPECT_EQ(b.size(), 100U);
  ASSERT_NE(b.data(), nullptr);
  EXPECT_EQ(b.space(), dyng::memory_space::host);
  EXPECT_EQ(b.device(), -1);
  for (std::size_t i = 0; i < b.size(); ++i) {
    b[i] = static_cast<std::int64_t>(i * i);
  }
  auto v = b.view();
  EXPECT_EQ(v.size(), 100U);
  EXPECT_EQ(v[9], 81);
  const auto& cb = b;
  dyng::array_view<const std::int64_t> cv = cb.view();
  EXPECT_EQ(cv[10], 100);
}

TEST(Buffer, MoveTransfersOwnership) {
  const auto res = dyng::resources::sequential();
  dyng::buffer<int> a(res, 10);
  int* p = a.data();
  dyng::buffer<int> b(std::move(a));
  EXPECT_EQ(b.data(), p);
  EXPECT_EQ(b.size(), 10U);
  EXPECT_TRUE(a.empty());  // NOLINT(bugprone-use-after-move): moved-from state is specified
  dyng::buffer<int> c(res, 3);
  c = std::move(b);
  EXPECT_EQ(c.data(), p);
  EXPECT_EQ(c.size(), 10U);
}

TEST(Buffer, ResizeKeepsPrefix) {
  const auto res = dyng::resources::sequential();
  dyng::buffer<int> b(res, 4);
  std::iota(b.data(), b.data() + 4, 1);
  b.resize(8);
  ASSERT_EQ(b.size(), 8U);
  EXPECT_EQ(b[0], 1);
  EXPECT_EQ(b[3], 4);
  b.resize(2);
  ASSERT_EQ(b.size(), 2U);
  EXPECT_EQ(b[1], 2);
  b.resize(0);
  EXPECT_TRUE(b.empty());
}

TEST(Copy, ToVectorAndCopyOnHost) {
  const auto res = dyng::resources::sequential();
  std::vector<int> src{5, 6, 7};
  auto out = dyng::to_vector(res, dyng::host_view(src));
  EXPECT_EQ(out, src);
  const std::vector<int>& csrc = src;
  auto out2 = dyng::to_vector(res, dyng::host_view(csrc));
  EXPECT_EQ(out2, src);

  std::vector<int> dst(3, 0);
  dyng::copy(res, dyng::host_view(csrc), dyng::host_view(dst));
  EXPECT_EQ(dst, src);
  std::vector<int> wrong(2);
  EXPECT_THROW(dyng::copy(res, dyng::host_view(csrc), dyng::host_view(wrong)),
               dyng::invalid_argument_error);

  auto buf = dyng::to_space(res, dyng::host_view(csrc), dyng::memory_space::host);
  EXPECT_EQ(dyng::to_vector(res, buf.view()), src);
  EXPECT_THROW((void)dyng::to_space(res, dyng::host_view(csrc), dyng::memory_space::device),
               dyng::invalid_argument_error);
}

TEST(Copy, DeviceCopiesAreNotSupportedWithoutCuda) {
  const auto res = dyng::resources::sequential();
  std::vector<int> fake(2);
  auto dev = dyng::device_view(fake.data(), fake.size());
  EXPECT_THROW((void)dyng::to_vector(res, dev), dyng::not_supported_error);
}
