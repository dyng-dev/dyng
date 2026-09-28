// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file sssp_fixture_test.cpp
 * @brief Byte parity of sssp::compute() and sssp::update() with MOSP-OpenMP@c352151.
 *
 * The fixtures in cpp/tests/data/mosp_sssp were produced by the pinned original
 * (parity/fixtures/sssp/make_sssp_fixtures.sh): `mospPrep init` (Dijkstra with lowest-id ties) for
 * the initial trees and the `mosp` driver (mospUpdate -> sospUpdateCpu) for the updated trees,
 * cross-checked there against sequentialSOSPUpdate, parallelSOSPUpdate and `mospPrep expected`.
 * The cases: the 10 generateTestCases cases and the 17 other graph/io cases (parallel edges,
 * self-loops, unsorted rows, delete-all, empty batch, K = 1..5 and 32), the three count-to-infinity
 * regressions of mospTest, the three MOSP_ESCHER test_mosp_update cases and a K = 2 ties case.
 * For every case, objective, backend and index type, the files written by dynG must be
 * byte-identical to the original's, and `invalidated` must equal the original's counter.
 */
#include "support/data_paths.hpp"
#include "support/gtest_helpers.hpp"

#include <dyng/core/resources.hpp>
#include <dyng/graph/graph.hpp>
#include <dyng/graph/graph_properties.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/csr_triplet.hpp>
#include <dyng/io/result_io.hpp>
#include <dyng/sssp.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dyng::test::data_path;
using dyng::test::read_lines;
using dyng::test::read_text;

struct fixture_case {
  std::string name;
  std::string input;  // relative to cpp/tests/data
  int num_weights = 0;
};

std::vector<fixture_case> load_cases() {
  std::vector<fixture_case> cases;
  for (const std::string& line : read_lines(data_path("mosp_sssp/cases.txt"))) {
    std::istringstream in(line);
    fixture_case c;
    in >> c.name >> c.input >> c.num_weights;
    cases.push_back(c);
  }
  return cases;
}

/// The expected `invalidated` counter of objective k (stats.txt: "obj<k> invalidated <n>").
std::int64_t expected_invalidated(const std::string& name, int k) {
  for (const std::string& line : read_lines(data_path("mosp_sssp/" + name + "/stats.txt"))) {
    std::istringstream in(line);
    std::string obj;
    std::string word;
    std::int64_t value = -1;
    in >> obj >> word >> value;
    if (obj == "obj" + std::to_string(k)) {
      return value;
    }
  }
  return -1;
}

/// The backend configurations every case runs on: sequential, and OpenMP with 1, 2 and 4 threads;
/// in the CUDA test executable (DYNG_TEST_CUDA) the cuda backend (none without a device).
std::vector<dyng::resources> configurations() {
#if defined(DYNG_TEST_CUDA) && DYNG_TEST_CUDA
  if (dyng::backend_available(dyng::backend::cuda)) {
    return {dyng::resources::cuda()};
  }
  return {};
#endif
  std::vector<dyng::resources> out{dyng::resources::sequential()};
  if (dyng::backend_available(dyng::backend::openmp)) {
    for (int threads : {1, 2, 4}) {
      out.push_back(dyng::resources::openmp(threads));
    }
  }
  return out;
}

template <typename graph_t>
class SsspMospFixture : public ::testing::Test {};

using graph_types = ::testing::Types<dyng::graph<std::int32_t, std::int32_t, std::int32_t>,
                                     dyng::graph<std::int32_t, std::int64_t, std::int32_t>,
                                     dyng::graph<std::int64_t, std::int64_t, std::int32_t>>;
TYPED_TEST_SUITE(SsspMospFixture, graph_types, dyng::test::type_index_name);

TYPED_TEST(SsspMospFixture, ComputeAndUpdateAreByteEqualToTheOriginal) {
  using graph_t = TypeParam;
  using vertex_t = typename graph_t::vertex_type;
  using edge_t = typename graph_t::edge_type;
  using weight_t = typename graph_t::weight_type;
  const std::vector<fixture_case> cases = load_cases();
  ASSERT_EQ(cases.size(), 34u);
  dyng::test::temp_dir tmp;
  std::int64_t comparisons = 0;
  for (const fixture_case& c : cases) {
    SCOPED_TRACE(c.name);
    const std::string input = data_path(c.input) + "/";
    const std::string expected = data_path("mosp_sssp/" + c.name) + "/";
    dyng::io::csr_triplet_options csr_options;
    csr_options.num_weights = c.num_weights;
    const auto original =
        dyng::io::read_csr_triplet<vertex_t, edge_t, weight_t>(input + "graphCsr", csr_options);
    dyng::io::legacy_batch_options batch_options;
    batch_options.num_weights = c.num_weights;
    batch_options.num_vertices = original.num_vertices();
    const auto batch = dyng::io::read_legacy_batch<vertex_t, weight_t>(
        input + "insert.txt", input + "delete.txt", batch_options);
    for (const dyng::resources& res : configurations()) {
      SCOPED_TRACE(std::string(dyng::to_string(res.get_backend())) + " x" +
                   std::to_string(res.num_threads()));
      for (int k = 0; k < c.num_weights; ++k) {
        SCOPED_TRACE("objective " + std::to_string(k));
        const std::string obj = "obj" + std::to_string(k) + "/";
        const std::string init_dist = expected + "init/" + obj + "distancesOriginal.txt";
        const std::string init_tree = expected + "init/" + obj + "SSSPTreeOriginal.txt";
        dyng::sssp::options opt;
        opt.objective = k;

        // compute() == mospPrep init (Dijkstra).
        auto g = graph_t::from_csr(res, original.view(), dyng::graph_properties::mosp_compatible());
        const auto computed = dyng::sssp::compute(res, g, vertex_t{0}, opt);
        const std::string out = tmp.path(c.name + "/" + obj);
        dyng::io::write_distances(out + "computed_d.txt",
                                  dyng::host_view(dyng::test::host_copy(computed.distances())));
        dyng::io::write_parents(out + "computed_t.txt",
                                dyng::host_view(dyng::test::host_copy(computed.parents())));
        EXPECT_TRUE(read_text(out + "computed_d.txt") == read_text(init_dist));
        EXPECT_TRUE(read_text(out + "computed_t.txt") == read_text(init_tree));

        // update() from the original's initial files == mosp.
        const auto dist = dyng::io::read_distances<std::int64_t>(init_dist, g.num_vertices());
        const auto tree = dyng::io::read_parents<vertex_t>(init_tree, g.num_vertices());
        auto r = dyng::sssp::result<vertex_t>::from_arrays(
            res, g, vertex_t{0}, dyng::host_view(dist), dyng::host_view(tree),
            /*canonicalize=*/false, opt);
        const dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), r);
        dyng::io::write_distances(out + "updated_d.txt",
                                  dyng::host_view(dyng::test::host_copy(r.distances())));
        dyng::io::write_parents(out + "updated_t.txt",
                                dyng::host_view(dyng::test::host_copy(r.parents())));
        EXPECT_TRUE(read_text(out + "updated_d.txt") ==
                    read_text(expected + "updated/" + obj + "distancesUpdated.txt"));
        EXPECT_TRUE(read_text(out + "updated_t.txt") ==
                    read_text(expected + "updated/" + obj + "SSSPTreeUpdated.txt"));
        EXPECT_EQ(st.invalidated, expected_invalidated(c.name, k));
        EXPECT_EQ(r.graph_version(), g.version());
        comparisons += 4;
      }
    }
  }
  if (configurations().empty()) {
    GTEST_SKIP() << "no CUDA device visible";
  }
  EXPECT_GT(comparisons, 0);
}

// The regressions spelled out (they are also in the byte comparison above).
TEST(SsspMospRegressions, CountToInfinityAndDisconnect) {
  using graph_t = dyng::graph<std::int32_t, std::int32_t, std::int32_t>;
  for (const dyng::resources& res : configurations()) {
    SCOPED_TRACE(std::string(dyng::to_string(res.get_backend())));
    struct expectation {
      const char* name;
      int num_weights;
      std::vector<std::int64_t> distances;  // objective 0
    };
    // c2i_0: mospTest runRegressions, n = 6, seeds 621705 / 250813: d(1) = 90, not 60.
    // escher_disconnect: MOSP_ESCHER test_mosp_update: d(1) = 100, d(2) = 101.
    const std::vector<expectation> cases = {{"c2i_0", 2, {0, 90, 4, 37, 44, 86}},
                                            {"escher_disconnect", 1, {0, 100, 101, 50}}};
    for (const expectation& e : cases) {
      SCOPED_TRACE(e.name);
      const std::string input = data_path(std::string("mosp_sssp/") + e.name + "/input/");
      const auto original =
          dyng::io::read_csr_triplet<std::int32_t, std::int32_t, std::int32_t>(input + "graphCsr");
      auto g = graph_t::from_csr(res, original.view(), dyng::graph_properties::mosp_compatible());
      dyng::io::legacy_batch_options options;
      options.num_weights = e.num_weights;
      options.num_vertices = g.num_vertices();
      const auto batch = dyng::io::read_legacy_batch<std::int32_t, std::int32_t>(
          input + "insert.txt", input + "delete.txt", options);
      auto r = dyng::sssp::compute(res, g, 0);
      (void)dyng::sssp::update(res, g, batch.view(), r);
      EXPECT_EQ(dyng::test::host_copy(r.distances()), e.distances);
    }
  }
}

}  // namespace
