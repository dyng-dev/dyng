// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include "support/gtest_helpers.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/memory.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <string>
#include <utility>

#if DYNG_HAS_OPENMP
#include <omp.h>
#endif

namespace {

/// A host resource that reports device memory, to check the space validation.
class fake_device_resource {
 public:
  void* allocate(dyng::stream_ref, std::size_t, std::size_t) {
    return nullptr;
  }
  void deallocate(dyng::stream_ref, void*, std::size_t, std::size_t) noexcept {}
  void* allocate_sync(std::size_t, std::size_t) {
    return nullptr;
  }
  void deallocate_sync(void*, std::size_t, std::size_t) noexcept {}
  [[nodiscard]] dyng::memory_space space() const noexcept {
    return dyng::memory_space::device;
  }
};

}  // namespace

TEST(Resources, Sequential) {
  const auto res = dyng::resources::sequential();
  EXPECT_EQ(res.get_backend(), dyng::backend::sequential);
  EXPECT_EQ(res.num_threads(), 1);
  EXPECT_EQ(res.device(), -1);
  EXPECT_EQ(res.default_space(), dyng::memory_space::host);
  EXPECT_EQ(res.memory().space(), dyng::memory_space::host);
  EXPECT_TRUE(res.stream().is_per_thread_default());
  EXPECT_EQ(res.get_copy_policy(), dyng::copy_policy::allow);
  EXPECT_EQ(res.get_profiler(), nullptr);
  EXPECT_NO_THROW(res.warm_up());
  EXPECT_NO_THROW(res.synchronize());
}

TEST(Resources, DefaultUsesDefaultBackend) {
  const dyng::resources res;
  EXPECT_EQ(res.get_backend(), dyng::default_backend());
}

TEST(Resources, OpenmpThreadCount) {
  DYNG_SKIP_IF_NO_OPENMP();
#if DYNG_HAS_OPENMP
  const auto res = dyng::resources::openmp(3);
  EXPECT_EQ(res.get_backend(), dyng::backend::openmp);
  EXPECT_EQ(res.num_threads(), 3);
  EXPECT_EQ(res.default_space(), dyng::memory_space::host);
  const auto automatic = dyng::resources::openmp();
  EXPECT_EQ(automatic.num_threads(), omp_get_max_threads());
  EXPECT_THROW((void)dyng::resources::openmp(-1), dyng::invalid_argument_error);

  // The thread count drives a parallel region without touching the global OpenMP setting.
  const int global_before = omp_get_max_threads();
  std::atomic<int> team{0};
#pragma omp parallel num_threads(res.num_threads())
  {
#pragma omp single
    team = omp_get_num_threads();
  }
  EXPECT_EQ(team.load(), 3);
  EXPECT_EQ(omp_get_max_threads(), global_before);
#endif
}

TEST(Resources, OpenmpNotBuiltThrows) {
  if (dyng::backend_available(dyng::backend::openmp)) {
    GTEST_SKIP() << "OpenMP is built";
  }
  EXPECT_THROW((void)dyng::resources::openmp(), dyng::not_supported_error);
}

TEST(Resources, CudaNotBuiltThrowsNotSupported) {
  if (DYNG_HAS_CUDA) {
    GTEST_SKIP() << "CUDA is built (covered by the gpu tests)";
  }
  EXPECT_THROW((void)dyng::resources::cuda(), dyng::not_supported_error);
  try {
    (void)dyng::resources::cuda(0);
  } catch (const dyng::not_supported_error& e) {
    EXPECT_NE(std::string(e.what()).find("available: sequential"), std::string::npos);
  }
}

TEST(Resources, CopiesShareOneHandle) {
  auto a = dyng::resources::sequential();
  auto b = a;  // NOLINT(performance-unnecessary-copy-initialization): the copy is the point
  dyng::profiler prof;
  a.attach_profiler(&prof);
  a.set_copy_policy(dyng::copy_policy::error);
  EXPECT_EQ(b.get_profiler(), &prof);
  EXPECT_EQ(b.get_copy_policy(), dyng::copy_policy::error);
  b.attach_profiler(nullptr);
  EXPECT_EQ(a.get_profiler(), nullptr);

  // A new object is independent.
  const auto c = dyng::resources::sequential();
  EXPECT_EQ(c.get_copy_policy(), dyng::copy_policy::allow);
}

TEST(Resources, MovedFromStaysValid) {
  auto res = dyng::resources::sequential();
  dyng::profiler prof;
  res.attach_profiler(&prof);
  const dyng::resources moved(std::move(res));
  // A move shares the handle like a copy: the source keeps working (no null handle).
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): the point of the test
  EXPECT_EQ(res.get_backend(), dyng::backend::sequential);
  EXPECT_EQ(res.num_threads(), 1);
  EXPECT_EQ(res.get_profiler(), &prof);
  EXPECT_EQ(moved.get_profiler(), &prof);

  auto target = dyng::resources::sequential();
  target = std::move(moved);  // NOLINT(performance-move-const-arg): const, so this copies
  auto other = dyng::resources::sequential();
  dyng::resources assigned = dyng::resources::sequential();
  assigned = std::move(other);
  // NOLINTNEXTLINE(bugprone-use-after-move,clang-analyzer-cplusplus.Move): the point of the test
  EXPECT_EQ(other.get_backend(), dyng::backend::sequential);
  EXPECT_EQ(other.memory().space(), dyng::memory_space::host);
  EXPECT_EQ(target.get_profiler(), &prof);
  res.attach_profiler(nullptr);
}

TEST(Resources, MemoryResourceMustSuitBackend) {
  auto res = dyng::resources::sequential();
  fake_device_resource device_mr;
  EXPECT_THROW(res.set_memory_resource(device_mr), dyng::invalid_argument_error);
  dyng::host_memory_resource host_mr;
  res.set_memory_resource(host_mr);
  EXPECT_EQ(res.memory(), dyng::memory_resource_ref(host_mr));
}

class ResourcesByBackend : public ::testing::TestWithParam<dyng::backend> {};

TEST_P(ResourcesByBackend, HostBackendsUseHostMemory) {
  const auto res = dyng::test::make_resources(GetParam());
  EXPECT_EQ(res.get_backend(), GetParam());
  EXPECT_EQ(res.default_space(), dyng::memory_space::host);
  EXPECT_GE(res.num_threads(), 1);
}

INSTANTIATE_TEST_SUITE_P(Host, ResourcesByBackend, ::testing::ValuesIn(dyng::test::host_backends()),
                         dyng::test::backend_name{});
