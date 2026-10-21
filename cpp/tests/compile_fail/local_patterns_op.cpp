// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file local_patterns_op.cpp
 * @brief Compile-failure tests of the Op concept of experimental::count_local_patterns (M9b,
 *        ADR 0058): one ill-formed Op per define, each stopped by its own static_assert message
 *        (cpp/tests/experimental/CMakeLists.txt checks the message). Compiled twice: by the host
 *        compiler with -fsyntax-only (the checks of <dyng/experimental/detail/
 *        local_patterns_engine.hpp>, hosted CPU CI) and, in CUDA builds, by nvcc as a CUDA source
 *        through count_local_patterns<op_t>::compute() itself (<dyng/experimental/
 *        local_patterns.cuh>). Without a define it compiles.
 *
 *   DYNG_CF_LP_NO_NUM_BINS        no num_bins
 *   DYNG_CF_LP_ZERO_BINS          num_bins = 0
 *   DYNG_CF_LP_TOO_MANY_BINS      num_bins = max_pattern_bins + 1
 *   DYNG_CF_LP_NO_SHAPE           no shape
 *   DYNG_CF_LP_NOT_TRIVIAL        a member that owns memory (not trivially copyable)
 *   DYNG_CF_LP_NOT_DEFAULT        no default constructor
 *   DYNG_CF_LP_NO_BEGIN_WEDGE     shape triangles without begin_wedge
 *   DYNG_CF_LP_NO_TRIANGLE        shape triangles without triangle
 *   DYNG_CF_LP_NO_PAIR            shape pairs without pair
 *   DYNG_CF_LP_NO_KEY             shape keys without key
 *   DYNG_CF_LP_NO_FLUSH           no flush
 *   DYNG_CF_LP_TYPES              access_rows() of an unsupported hypergraph type
 */
#if defined(__CUDACC__)
#include <dyng/experimental/local_patterns.cuh>
#endif
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/experimental/detail/local_patterns_engine.hpp>
#include <dyng/experimental/local_patterns.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <cstdint>
#include <vector>

namespace dx = ::dyng::experimental;

namespace {

struct op {
#if defined(DYNG_CF_LP_ZERO_BINS)
  static constexpr int num_bins = 0;
#elif defined(DYNG_CF_LP_TOO_MANY_BINS)
  static constexpr int num_bins = dx::max_pattern_bins + 1;
#elif !defined(DYNG_CF_LP_NO_NUM_BINS)
  static constexpr int num_bins = 1;
#endif
#if !defined(DYNG_CF_LP_NO_SHAPE)
#if defined(DYNG_CF_LP_NO_PAIR)
  static constexpr dx::shape shape = dx::shape::pairs;
#elif defined(DYNG_CF_LP_NO_KEY)
  static constexpr dx::shape shape = dx::shape::keys;
#else
  static constexpr dx::shape shape = dx::shape::triangles;
#endif
#endif
#if defined(DYNG_CF_LP_NOT_TRIVIAL)
  std::vector<int> owned;
#endif
#if defined(DYNG_CF_LP_NOT_DEFAULT)
  explicit op(int k) : threshold(k) {}
  int threshold;
#endif
  std::uint64_t count = 0;
#if !defined(DYNG_CF_LP_NO_BEGIN_WEDGE)
  DYNG_HD void begin_wedge(const dx::hyperedge_rows&, int, int) {}
#endif
#if !defined(DYNG_CF_LP_NO_TRIANGLE)
  template <class bins_t>
  DYNG_HD void triangle(const dx::hyperedge_rows&, int, bins_t&) {
    ++count;
  }
#endif
#if defined(DYNG_CF_LP_NO_KEY)
  template <class bins_t>
  DYNG_HD void pair(const dx::hyperedge_rows&, int, int, bins_t&) {}
#endif
#if !defined(DYNG_CF_LP_NO_FLUSH)
  template <class bins_t>
  DYNG_HD void flush(bins_t& bins) {
    bins.add(0, count);
  }
#endif
};

#if defined(__CUDACC__)
[[maybe_unused]] void use(const dyng::resources& res,
                          const dyng::hypergraph<std::int32_t, dyng::unweighted>& h) {
  (void)dx::count_local_patterns<op>::compute(res, h);
}
#else
[[maybe_unused]] constexpr bool checked = ::dyng::detail::local_patterns::check_op<op>();
#endif

#if defined(DYNG_CF_LP_TYPES)
[[maybe_unused]] dx::row_access rows(const dyng::hypergraph<std::int64_t, dyng::unweighted>& h) {
  return dx::access_rows(h);
}
#endif

}  // namespace
