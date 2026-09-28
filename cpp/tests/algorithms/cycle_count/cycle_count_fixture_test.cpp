// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cycle_count_fixture_test.cpp
 * @brief Parity of cycle_count with CycleEnumeration-GPU@0a976ad on the committed fixtures: the
 *        histograms of its sequential Johnson and OpenMP counter and of
 *        update_static_histogram[_openmp], bit for bit, on both host backends; and of its CUDA
 *        backend (every scheduler and kind of work item, update_static_histogram_cuda) on cuda.
 *
 * The expectations come from the pinned original (parity/fixtures/cycle_enum/
 * make_cycle_enum_fixtures.sh, the exporter's `counts`, `count-file` and `update-file` commands):
 *   - cases/case_NNN.counts: 80 random graphs with arbitrary batches, k = 2..7 and no bound: the
 *     static histograms before and after the batch and the updated histogram;
 *   - counts/cNN.expected: static histograms and generated-batch updates on the parser and
 *     generator fixture graphs (read with io::read_edge_list, batches from
 *     generators::legacy::cycle_enum_batch).
 */
#include "graph/graph_impl.hpp"
#include "support/cycle_count_support.hpp"
#include "support/cycle_enum_text.hpp"
#include "support/data_paths.hpp"

#include <dyng/core/error.hpp>
#include <dyng/cycle_count.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/io/edge_list_io.hpp>

#include <gtest/gtest.h>

#include <cstdint>
#include <iterator>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace {

using dyng::backend;
using dyng::resources;
using dyng::unweighted;
using dyng::test::cc_counts;
using dyng::test::cc_edge;
using dyng::test::cc_text;
using dyng::test::data_path;
namespace cycle_count = dyng::cycle_count;
using graph_u = dyng::graph<std::int32_t, std::int64_t, unweighted>;
using graph_u32 = dyng::graph<std::int32_t, std::int32_t, unweighted>;

cycle_count::options bound(int k) {
  cycle_count::options opt;
  opt.max_length = k;
  return opt;
}

/// Options with a bound and a CUDA scheduler / kind of work items.
cycle_count::options scheduled(int k, cycle_count::cuda_scheduler scheduler,
                               cycle_count::cuda_work_items items) {
  cycle_count::options opt = bound(k);
  opt.scheduler = scheduler;
  opt.work_items = items;
  return opt;
}

/// The CUDA schedulers of the original, by the suffix of the exporter's line names.
struct cuda_variant {
  const char* name;
  cycle_count::cuda_scheduler scheduler;
  cycle_count::cuda_work_items items;
};
constexpr cuda_variant cuda_variants[] = {
    {"", cycle_count::cuda_scheduler::work_queue, cycle_count::cuda_work_items::automatic},
    {"_naive", cycle_count::cuda_scheduler::naive, cycle_count::cuda_work_items::automatic},
    {"_roots", cycle_count::cuda_scheduler::work_queue, cycle_count::cuda_work_items::roots},
    {"_edges", cycle_count::cuda_scheduler::work_queue, cycle_count::cuda_work_items::edges},
    {"_two_hop", cycle_count::cuda_scheduler::work_queue, cycle_count::cuda_work_items::two_hop},
};

/// The exporter's line "<name> <hist>" ("<name>" alone for an empty histogram).
std::string line(const std::string& name, const std::vector<std::uint64_t>& counts) {
  const std::string h = cc_text(counts);
  return h.empty() ? name : name + " " + h;
}

/// "<name> <hist>" lines of an expectation file, keyed by "<section>/<name>".
std::map<std::string, std::string> read_expectations(const std::string& path) {
  std::map<std::string, std::string> out;
  std::istringstream in(dyng::test::read_text(path));
  std::string section;
  for (std::string text; std::getline(in, text);) {
    const std::size_t space = text.find(' ');
    const std::string name = text.substr(0, space);
    const std::string rest = space == std::string::npos ? "" : text.substr(space + 1);
    if (name == "k") {
      section = rest;
      continue;
    }
    out[section + "/" + name] = rest;
  }
  return out;
}

class CycleCountFixtures : public ::testing::TestWithParam<backend> {
 protected:
  resources res_ = dyng::test::cc_resources(GetParam());
  /// The original's name of the static counter / update of this backend.
  [[nodiscard]] std::string prefix() const {
    switch (GetParam()) {
      case backend::openmp:
        return "omp_";
      case backend::cuda:
        return "cuda_";
      case backend::sequential:
        break;
    }
    return "seq_";
  }
  /// The expectation file of this backend: the CPU backends' `.counts` / `.expected`, the CUDA
  /// backend's `.cuda`.
  [[nodiscard]] std::string suffix(const char* cpu) const {
    return GetParam() == backend::cuda ? ".cuda" : cpu;
  }
};

TEST_P(CycleCountFixtures, RandomCasesEqualTheOriginal) {
  const std::vector<std::string> names = dyng::test::cycle_enum_case_names();
  ASSERT_EQ(names.size(), 80U);
  for (const std::string& name : names) {
    SCOPED_TRACE(name);
    const dyng::test::cycle_enum_case c =
        dyng::test::read_cycle_enum_case(data_path("cycle_enum/cases/" + name + ".txt"));
    const auto expected =
        read_expectations(data_path("cycle_enum/cases/" + name + suffix(".counts")));
    std::vector<cc_edge> edges;
    for (std::size_t i = 0; i < c.src.size(); ++i) {
      edges.emplace_back(static_cast<std::int32_t>(c.src[i]), static_cast<std::int32_t>(c.dst[i]));
    }
    dyng::edge_batch<std::int32_t, unweighted> batch;
    for (std::size_t i = 0; i < c.del_src.size(); ++i) {
      batch.delete_edge(static_cast<std::int32_t>(c.del_src[i]),
                        static_cast<std::int32_t>(c.del_dst[i]));
    }
    for (std::size_t i = 0; i < c.ins_src.size(); ++i) {
      batch.insert_edge(static_cast<std::int32_t>(c.ins_src[i]),
                        static_cast<std::int32_t>(c.ins_dst[i]));
    }
    for (const int k : {2, 3, 4, 5, 6, 7, -1}) {
      SCOPED_TRACE(::testing::Message() << "k=" << k);
      const std::string key = std::to_string(k) + "/" + prefix();
      graph_u g = dyng::test::cc_graph<graph_u>(res_, c.n, edges);
      cycle_count::result r = cycle_count::compute(res_, g, bound(k));
      EXPECT_EQ(cc_text(cc_counts(r)), expected.at(key + "before"));
      if (GetParam() == backend::cuda) {
        // Every scheduler of the original's CUDA backend (the naive one, and the work queue with
        // every kind of work item).
        const graph_u32 g32 = dyng::test::cc_graph<graph_u32>(res_, c.n, edges);
        for (const cuda_variant& v : cuda_variants) {
          SCOPED_TRACE(v.name);
          const std::string want = expected.at(std::to_string(k) + "/cuda" + v.name + "_before");
          EXPECT_EQ(
              cc_text(cc_counts(cycle_count::compute(res_, g, scheduled(k, v.scheduler, v.items)))),
              want);
          EXPECT_EQ(cc_text(cc_counts(
                        cycle_count::compute(res_, g32, scheduled(k, v.scheduler, v.items)))),
                    want);
        }
      }
      (void)cycle_count::update(res_, g, batch.view(), r);
      // The original updates bounded histograms only; an unbounded dynG update must equal the
      // original's static count after the batch.
      EXPECT_EQ(cc_text(cc_counts(r)), expected.at(key + (k > 0 ? "update" : "after")));
      EXPECT_EQ(cc_text(cc_counts(cycle_count::compute(res_, g, bound(k)))),
                expected.at(key + "after"));
    }
  }
}

TEST_P(CycleCountFixtures, FixtureGraphsEqualTheOriginal) {
  const std::vector<std::string> jobs =
      dyng::test::read_lines(data_path("cycle_enum/counts/cases.txt"));
  ASSERT_EQ(jobs.size(), 23U);
  for (std::size_t i = 0; i < jobs.size(); ++i) {
    SCOPED_TRACE(jobs[i]);
    std::istringstream fields(jobs[i]);
    std::string what;
    std::string file;
    int k = 0;
    fields >> what >> file >> k;
    const std::string name = (i < 10 ? "c0" : "c") + std::to_string(i);
    const std::vector<std::string> expected =
        dyng::test::read_lines(data_path("cycle_enum/counts/" + name + suffix(".expected")));
    const auto edges =
        dyng::io::read_edge_list<std::int32_t, unweighted>(data_path("cycle_enum/" + file));
    graph_u32 g =
        graph_u32::from_edges(res_, edges.view(), dyng::graph_properties::cycle_enum_compatible());
    const bool cuda = GetParam() == backend::cuda;
    const std::string mine = GetParam() == backend::openmp ? "omp" : "seq";
    if (what == "count" && cuda) {
      ASSERT_EQ(expected.size(), 5U);
      for (std::size_t v = 0; v < std::size(cuda_variants); ++v) {
        const cuda_variant& variant = cuda_variants[v];
        EXPECT_EQ(line(std::string("cuda") + variant.name,
                       cc_counts(cycle_count::compute(
                           res_, g, scheduled(k, variant.scheduler, variant.items)))),
                  expected[v]);
      }
      continue;
    }
    if (what == "count") {
      ASSERT_EQ(expected.size(), 2U);
      EXPECT_EQ(line(mine, cc_counts(cycle_count::compute(res_, g, bound(k)))),
                expected[GetParam() == backend::openmp ? 1 : 0]);
      continue;
    }
    ASSERT_EQ(expected.size(), cuda ? 3U : 4U);
    dyng::generators::legacy::cycle_enum_batch_options opt;
    std::int64_t window = -1;
    fields >> opt.num_deletions >> opt.num_insertions >> opt.seed;
    if (fields >> window) {
      opt.locality_window = window;
    }
    cycle_count::result r = cycle_count::compute(res_, g, bound(k));
    EXPECT_EQ(line("prior", cc_counts(r)), expected[1]);
    const auto batch =
        dyng::generators::legacy::cycle_enum_batch(dyng::detail::graph_access::out_view(g), opt);
    const cycle_count::stats st = cycle_count::update(res_, g, batch.view(), r);
    EXPECT_EQ("deletions " + std::to_string(st.deletions) + " insertions " +
                  std::to_string(st.insertions),
              expected[0]);
    if (cuda) {
      EXPECT_EQ(line("cuda_update", cc_counts(r)), expected[2]);
      continue;
    }
    EXPECT_EQ(line("seq_update", cc_counts(r)), expected[2]);
    EXPECT_EQ(line("omp_update", cc_counts(r)), expected[3]);
  }
}

INSTANTIATE_TEST_SUITE_P(Backends, CycleCountFixtures,
                         ::testing::ValuesIn(dyng::test::suite_backends()),
                         [](const ::testing::TestParamInfo<backend>& info) {
                           return dyng::test::cc_name(info.param);
                         });
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(CycleCountFixtures);

}  // namespace
