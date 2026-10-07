// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cuda_error_test.cpp
 * @brief CUDA error reporting: DYNG_CUDA_TRY, DYNG_CUDA_TRY_NO_THROW, DYNG_CHECK_KERNEL and the
 *        sticky device error word (PLAN Section 4.7.3).
 *
 * Tests that make CUDA API calls fail on purpose belong to the suite CudaApiErrors: ci/gpu_local.sh
 * runs them under compute-sanitizer without API-error reporting, and every other test with it.
 * CudaApiErrors.AForkedChildIsToldAboutFork forks after CUDA is initialized: the child cannot use
 * CUDA, and the error must name fork (the Python plugin's selection never initializes CUDA in the
 * parent, ADR 0031; a parent that used CUDA itself still forks children that cannot).
 */
#include "core/resources_access.hpp"
#include "cuda/cuda_test_kernels.hpp"
#include "support/gtest_helpers.hpp"
#include "util/cuda_check.hpp"
#include "util/device_error_flags.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/logging.hpp>
#include <dyng/core/resources.hpp>

#include <cuda_runtime_api.h>

#include <gtest/gtest.h>

#include <sys/types.h>
#include <sys/wait.h>
#include <unistd.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

using dyng::detail::bits_of;
using dyng::detail::device_error;

TEST(CudaApiErrors, TryThrowsCudaErrorWithNameCallAndLocation) {
  DYNG_SKIP_IF_NO_CUDA();
  try {
    DYNG_CUDA_TRY(cudaSetDevice(-1));
    ADD_FAILURE() << "expected cuda_error";
  } catch (const dyng::cuda_error& e) {
    EXPECT_EQ(e.code(), static_cast<int>(cudaErrorInvalidDevice));
    const std::string what = e.what();
    EXPECT_NE(what.find("cudaErrorInvalidDevice"), std::string::npos) << what;
    EXPECT_NE(what.find("cudaSetDevice(-1)"), std::string::npos) << what;
    EXPECT_NE(what.find("cuda_error_test.cpp:"), std::string::npos) << what;
  }
  // The error was cleared: it does not resurface at the next check.
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaApiErrors, TryReportsAllocationFailureAsOutOfMemory) {
  DYNG_SKIP_IF_NO_CUDA();
  void* p = nullptr;
  EXPECT_THROW(DYNG_CUDA_TRY(cudaMalloc(&p, std::size_t{1} << 46)), dyng::out_of_memory_error);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaApiErrors, TryNoThrowLogs) {
  DYNG_SKIP_IF_NO_CUDA();
  std::vector<std::string> lines;
  const auto previous = dyng::get_log_level();
  dyng::set_log_level(dyng::log_level::error);
  dyng::set_log_sink([&](dyng::log_level, std::string_view m) { lines.emplace_back(m); });
  DYNG_CUDA_TRY_NO_THROW(cudaSetDevice(-1));
  dyng::set_log_sink(nullptr);
  dyng::set_log_level(previous);
  ASSERT_EQ(lines.size(), 1u);
  EXPECT_NE(lines[0].find("cudaErrorInvalidDevice"), std::string::npos);
  EXPECT_EQ(cudaGetLastError(), cudaSuccess);
}

TEST(CudaApiErrors, CheckKernelReportsALaunchFailure) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  try {
    dyng::test::launch_invalid_configuration(res.stream());
    ADD_FAILURE() << "expected cuda_error";
  } catch (const dyng::cuda_error& e) {
    // CUDA 12 reports cudaErrorInvalidConfiguration, CUDA 13 cudaErrorInvalidValue.
    EXPECT_TRUE(e.code() == static_cast<int>(cudaErrorInvalidConfiguration) ||
                e.code() == static_cast<int>(cudaErrorInvalidValue))
        << e.what();
    EXPECT_NE(std::string(e.what()).find("cudaGetLastError()"), std::string::npos) << e.what();
  }
  EXPECT_NO_THROW(res.synchronize());  // not sticky
}

TEST(CudaApiErrors, AForkedChildIsToldAboutFork) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();  // initializes CUDA in this process
  res.synchronize();
  const pid_t pid = fork();
  ASSERT_GE(pid, 0);
  if (pid == 0) {
    // The child: only async-signal-safe exits after the attempt (no gtest assertions here).
    int code = 1;
    try {
      (void)dyng::resources::cuda();
      code = 2;  // CUDA worked in a forked child: the test's premise does not hold
    } catch (const dyng::error& e) {
      const std::string what = e.what();
      code = what.find("fork") != std::string::npos ? 0 : 3;
    } catch (...) {
      code = 4;
    }
    _exit(code);
  }
  int status = 0;
  ASSERT_EQ(waitpid(pid, &status, 0), pid);
  ASSERT_TRUE(WIFEXITED(status));
  EXPECT_EQ(WEXITSTATUS(status), 0)
      << "1: no exception, 2: CUDA worked after fork, 3: the message does not name fork, "
         "4: another exception";
}

TEST(DeviceErrorFlags, KernelsRaiseAndTheHostThrows) {
  DYNG_SKIP_IF_NO_CUDA();
  const auto res = dyng::resources::cuda();
  dyng::detail::device_error_flags flags(res);
  EXPECT_EQ(flags.read(res.stream()), 0u);

  // One kind from many threads.
  dyng::test::launch_raise_device_errors(res.stream(), flags.data(),
                                         bits_of(device_error::capacity), 256);
  flags.enqueue_readback(res.stream());
  res.synchronize();
  EXPECT_EQ(flags.mirrored(), bits_of(device_error::capacity));
  try {
    flags.throw_if_raised("test::op", "row 3 overflowed");
    ADD_FAILURE() << "expected capacity_error";
  } catch (const dyng::capacity_error& e) {
    const std::string what = e.what();
    EXPECT_NE(what.find("test::op"), std::string::npos) << what;
    EXPECT_NE(what.find("capacity"), std::string::npos) << what;
    EXPECT_NE(what.find("row 3 overflowed"), std::string::npos) << what;
  }

  // Reset clears; the bits are sticky until then.
  dyng::test::launch_raise_device_errors(res.stream(), flags.data(),
                                         bits_of(device_error::parent_cycle), 1);
  EXPECT_EQ(flags.read(res.stream()),
            bits_of(device_error::capacity) | bits_of(device_error::parent_cycle));
  flags.reset(res.stream());
  EXPECT_EQ(flags.read(res.stream()), 0u);
  EXPECT_NO_THROW(flags.throw_if_raised("test::op"));

  dyng::test::launch_raise_device_errors(res.stream(), flags.data(),
                                         bits_of(device_error::parent_cycle), 32);
  (void)flags.read(res.stream());
  EXPECT_THROW(flags.throw_if_raised("test::op"), dyng::invalid_argument_error);

  flags.reset(res.stream());
  dyng::test::launch_raise_device_errors(res.stream(), flags.data(),
                                         bits_of(device_error::internal), 1);
  (void)flags.read(res.stream());
  EXPECT_THROW(flags.throw_if_raised("test::op"), dyng::internal_error);
}

TEST(DeviceErrorFlags, ExceptionMapping) {
  using dyng::detail::describe_device_errors;
  using dyng::detail::throw_device_errors;
  EXPECT_EQ(describe_device_errors(0), "none");
  EXPECT_EQ(
      describe_device_errors(bits_of(device_error::capacity) | bits_of(device_error::parent_cycle)),
      "capacity, parent_cycle");
  EXPECT_EQ(describe_device_errors(0x10), "unknown bits 0x10");
  EXPECT_THROW(throw_device_errors(bits_of(device_error::invalid_input), "x"),
               dyng::invalid_argument_error);
  // The most specific kind wins; unknown bits are library bugs.
  EXPECT_THROW(throw_device_errors(
                   bits_of(device_error::invalid_input) | bits_of(device_error::capacity), "x"),
               dyng::capacity_error);
  EXPECT_THROW(throw_device_errors(0x10, "x"), dyng::internal_error);
}

}  // namespace
