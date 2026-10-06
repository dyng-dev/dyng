// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file fuzz_regression_test.cpp
 * @brief Regression tests of the reader fuzzers' findings (cpp/fuzz): each test is a minimized
 *        input of one finding, with the error the reader must report and the memory it may take.
 *
 * The findings were allocations sized by a count in the file that the rest of the file cannot
 * hold: a few bytes made a reader reserve gigabytes before it reported the short file. This
 * executable replaces the global operator new with one that records the largest single request,
 * so the tests check the bound with the regular compilers too (not under the sanitizers, which
 * replace operator new themselves; the size checks are then skipped, and the libFuzzer build
 * checks them instead: its CTest fuzz.corpus.<target> replays the reproducers of
 * cpp/fuzz/regressions with a 2 GB allocation limit).
 */
#include "support/data_paths.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/error.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/matrix_market.hpp>

#include <gtest/gtest.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <new>
#include <string>

namespace {

/// The largest single allocation request since the last reset (every thread).
std::atomic<std::size_t> largest_request{0};

}  // namespace

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  std::size_t seen = largest_request.load(std::memory_order_relaxed);
  while (size > seen && !largest_request.compare_exchange_weak(seen, size)) {
  }
  void* p = std::malloc(size == 0 ? 1 : size);
  if (p == nullptr) {
    throw std::bad_alloc();
  }
  return p;
}
void operator delete(void* p) noexcept {
  std::free(p);
}
void operator delete(void* p, std::size_t /*size*/) noexcept {
  std::free(p);
}
// NOLINTEND(cppcoreguidelines-no-malloc)
#endif

namespace {

using dyng::test::temp_dir;
using dyng::test::write_text;

/// Far more than any of these inputs needs, far less than the findings asked for.
constexpr std::size_t allocation_bound = std::size_t{64} << 20;

/// Runs `body`, expects an io_error whose message contains `expected`, and checks that no single
/// allocation exceeded allocation_bound.
template <typename body_t>
void expect_bounded_io_error(const body_t& body, const std::string& expected) {
  largest_request.store(0);
  try {
    body();
    ADD_FAILURE() << "expected dyng::io_error";
  } catch (const dyng::io_error& e) {
    EXPECT_NE(std::string(e.what()).find(expected), std::string::npos) << e.what();
  }
  if (!DYNG_TEST_SANITIZED) {
    EXPECT_LT(largest_request.load(), allocation_bound);
  }
}

// fuzz_matrix_market, 2026-10-06 (cpp/fuzz/regressions/matrix_market/oom_announced_entries): the
// size line announced 3,000,000,000,003 entries of a symmetric matrix; the reader reserved
// min(entries, 2^26) entries twice over (3 GB) before it found the file empty.
TEST(FuzzRegression, MatrixMarketAnnouncedEntriesDoNotSizeTheReservation) {
  temp_dir tmp;
  const std::string path = tmp.path("announced.mtx");
  write_text(path, "%%MatrixMarket matrix coordinate pattern symmetric\n3 3 3000000000003\n");
  expect_bounded_io_error(
      [&] {
        dyng::io::matrix_market_options options;
        options.drop_self_loops = true;
        (void)dyng::io::read_matrix_market<std::int32_t, std::int32_t>(path, options);
      },
      "the file ends after 0 of 3000000000003 entries");
}

// fuzz_csr_triplet, 2026-10-06: RowPtr announced a billion edges and ColInd held one; the strict
// reader of ColInd reserved the announced count (4 GB) before counting the indices.
TEST(FuzzRegression, CsrTripletAnnouncedEdgesDoNotSizeTheColumnReservation) {
  temp_dir tmp;
  const std::string prefix = tmp.path("g");
  write_text(prefix + "RowPtr.txt", "0\n1000000000\n");
  write_text(prefix + "ColInd.txt", "0\n");
  write_text(prefix + "Values.txt", "1\n");
  expect_bounded_io_error(
      [&] { (void)dyng::io::read_csr_triplet<std::int32_t, std::int64_t, std::int32_t>(prefix); },
      "1 column indices, but the row offsets announce 1000000000 edges");
}

// fuzz_csr_triplet, 2026-10-06: with m edges in RowPtr and ColInd, a Values file of ONE line with
// K weights made the strict reader allocate m * K weights (here 20,000 x 20,000, 1.6 GB) before it
// counted the lines.
TEST(FuzzRegression, CsrTripletWeightColumnsAreSizedByTheLinesTheFileHolds) {
  temp_dir tmp;
  const std::string prefix = tmp.path("g");
  constexpr int m = 20000;
  std::string col_ind;
  std::string line;
  for (int e = 0; e < m; ++e) {
    col_ind += "0\n";
    line += e == 0 ? "1" : " 1";
  }
  write_text(prefix + "RowPtr.txt", "0\n" + std::to_string(m) + "\n");
  write_text(prefix + "ColInd.txt", col_ind);
  write_text(prefix + "Values.txt", line + "\n");
  expect_bounded_io_error(
      [&] { (void)dyng::io::read_csr_triplet<std::int32_t, std::int64_t, std::int32_t>(prefix); },
      "1 weight lines, but the graph has 20000 edges");
}

}  // namespace
