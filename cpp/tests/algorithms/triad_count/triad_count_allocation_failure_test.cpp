// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_count_allocation_failure_test.cpp
 * @brief A host allocation failure inside triad_count::compute(), update() or result::clone() is
 *        an out_of_memory_error, as the header documents (the triad_count API review, ADR 0060):
 *        no std::bad_alloc leaves compute() or clone().
 *
 * This executable replaces the global operator new with one that can be told to fail the n-th
 * allocation on the calling thread (not compiled under the sanitizers, which replace operator new
 * themselves; the tests then skip). Every allocation of each call is failed once, in turn, on the
 * sequential backend (the calling thread does all the host work).
 */
#include "support/gtest_helpers.hpp"

#include <dyng/core/backend.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/triad_count.hpp>

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <new>
#include <string>
#include <vector>

namespace {

/// Allocations on this thread until one fails (0: none fails).
thread_local long long fail_countdown = 0;

}  // namespace

#if !DYNG_TEST_SANITIZED
// NOLINTBEGIN(cppcoreguidelines-no-malloc): the replaceable global allocation functions
void* operator new(std::size_t size) {
  if (fail_countdown > 0 && --fail_countdown == 0) {
    throw std::bad_alloc();
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

namespace tc = dyng::triad_count;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;

hgraph make(const dyng::resources& res) {
  const std::vector<std::vector<std::int32_t>> rows = {
      {0, 1, 2}, {1, 2, 3}, {2, 3, 4}, {0, 2, 4}, {4, 5, 6}, {1, 5, 6, 7}, {0, 7}};
  dyng::hyperedge_list<std::int32_t, dyng::unweighted> list;
  list.num_vertices = 8;
  for (const auto& r : rows) list.add_hyperedge(dyng::host_view(r));
  dyng::hypergraph_properties props;
  props.line_graph = dyng::line_graph_policy::sorted_merge;  // built here, not inside compute()
  return hgraph::from_hyperedges(res, list.view(), props);
}

/// Fails the k-th allocation of `call` for k = 1, 2, ... until the call allocates fewer than k
/// times. The call either succeeds (it handled the failure itself) or throws out_of_memory_error;
/// a std::bad_alloc is a failure. Returns the number of out_of_memory_errors.
int fail_every_allocation(const std::function<void()>& prepare, const std::function<void()>& call) {
  int failures = 0;
  for (long long k = 1; k < 100000; ++k) {
    prepare();
    bool oom = false;
    fail_countdown = k;
    try {
      call();
    } catch (const dyng::out_of_memory_error&) {
      oom = true;
    } catch (const std::bad_alloc&) {
      fail_countdown = 0;
      ADD_FAILURE() << "allocation " << k << ": std::bad_alloc left the library";
      return failures;
    }
    const bool reached = fail_countdown == 0;
    fail_countdown = 0;
    if (!reached) break;  // the call made fewer than k allocations: every one was failed once
    failures += oom ? 1 : 0;
  }
  return failures;
}

TEST(TriadCountAllocationFailure, ComputeReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  std::vector<dyng::resources> backends{dyng::resources::sequential()};
  if (dyng::backend_available(dyng::backend::openmp))
    backends.push_back(dyng::resources::openmp(1));
  for (const dyng::resources& res : backends) {
    for (const tc::pattern p : {tc::pattern::h_motif, tc::pattern::stathyper_type1,
                                tc::pattern::stathyper_type2, tc::pattern::stathyper_type3}) {
      SCOPED_TRACE(std::string(dyng::to_string(res.get_backend())) + " " +
                   std::string(tc::to_string(p)));
      const hgraph h = make(res);
      tc::options o;
      o.kind = p;
      EXPECT_GT(fail_every_allocation([] {}, [&] { (void)tc::compute(res, h, o); }), 0);
    }
  }
}

TEST(TriadCountAllocationFailure, CloneReportsOutOfMemory) {
  if (DYNG_TEST_SANITIZED) GTEST_SKIP() << "operator new is the sanitizer's";
  const dyng::resources res = dyng::resources::sequential();
  const hgraph h = make(res);
  const tc::result r = tc::compute(res, h);
  EXPECT_GT(fail_every_allocation([] {}, [&] { (void)r.clone(res); }), 0);
}

// update(): run_update() translates the allocation failures that leave the update
// (hyper_update.cpp). A failure inside the container's OpenMP regions on the host backends
// (incidence_model::v2h_append, the line graph's row build) ends the process instead
// (std::terminate): a defect of the hypergraph container found by this review and left to the
// author (ADR 0060, finding 8; cpp/src/hypergraph is outside M9b's scope), so update() is not
// swept here.

}  // namespace
