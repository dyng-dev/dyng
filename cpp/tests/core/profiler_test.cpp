// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/core/resources.hpp>

#include <gtest/gtest.h>

#include <sstream>
#include <string>

TEST(Profiler, NameScheme) {
  EXPECT_TRUE(dyng::profiler::is_valid_name("sssp.loop"));
  EXPECT_TRUE(dyng::profiler::is_valid_name("graph.apply"));
  EXPECT_TRUE(dyng::profiler::is_valid_name("label_propagation.identify_affected.components"));
  EXPECT_TRUE(dyng::profiler::is_valid_name("cycle_count.count_minus"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("sssp"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("Sssp.loop"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("sssp..loop"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("sssp.loop."));
  EXPECT_FALSE(dyng::profiler::is_valid_name(".loop"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("sssp.1loop"));
  EXPECT_FALSE(dyng::profiler::is_valid_name("sssp/loop"));
  EXPECT_FALSE(dyng::profiler::is_valid_name(""));
}

TEST(Profiler, RecordsNestedStagesAndAggregates) {
  dyng::profiler prof;
  for (int i = 0; i < 3; ++i) {
    dyng::scoped_stage outer(&prof, "sssp.update");
    {
      dyng::scoped_stage inner(&prof, "sssp.loop");
    }
  }
  ASSERT_EQ(prof.stages().size(), 2U);
  EXPECT_EQ(prof.stages()[0].name, "sssp.update");
  EXPECT_EQ(prof.stages()[0].depth, 0);
  EXPECT_EQ(prof.stages()[0].calls, 3);
  EXPECT_EQ(prof.stages()[1].name, "sssp.loop");
  EXPECT_EQ(prof.stages()[1].depth, 1);
  EXPECT_EQ(prof.stages()[1].calls, 3);
  EXPECT_GE(prof.stages()[0].host_ms, prof.stages()[1].host_ms);
  EXPECT_EQ(prof.stages()[0].device_ms, 0.0);
  ASSERT_EQ(prof.samples().size(), 6U);
  EXPECT_EQ(prof.samples()[0].name, "sssp.loop");  // inner completes first
  EXPECT_EQ(prof.samples()[1].name, "sssp.update");
  EXPECT_DOUBLE_EQ(prof.total_host_ms("sssp.update"), prof.stages()[0].host_ms);
  EXPECT_EQ(prof.total_host_ms("sssp.missing"), 0.0);
}

TEST(Profiler, Counters) {
  dyng::profiler prof;
  prof.add_counter("sssp.affected", 5);
  prof.add_counter("sssp.affected", 7);
  prof.add_counter("graph.inserted_edges", 1);
  EXPECT_EQ(prof.counter("sssp.affected"), 12);
  EXPECT_EQ(prof.counter("graph.inserted_edges"), 1);
  EXPECT_EQ(prof.counter("graph.none"), 0);
  EXPECT_THROW(prof.add_counter("affected", 1), dyng::invalid_argument_error);
}

TEST(Profiler, RejectsBadUse) {
  dyng::profiler prof;
  EXPECT_THROW(prof.begin_stage("Loop"), dyng::invalid_argument_error);
  EXPECT_THROW(prof.end_stage(), dyng::invalid_argument_error);
  prof.begin_stage("sssp.loop");
  EXPECT_THROW(prof.reset(), dyng::invalid_argument_error);
  prof.end_stage(1.5);
  EXPECT_EQ(prof.stages()[0].device_ms, 1.5);
  prof.reset();
  EXPECT_TRUE(prof.stages().empty());
  EXPECT_TRUE(prof.samples().empty());
}

TEST(Profiler, CsvMatchesMospTimingFormat) {
  dyng::profiler prof;
  prof.begin_stage("sssp.loop");
  prof.end_stage();
  prof.add_counter("sssp.affected", 42);
  std::ostringstream out;
  prof.write_csv(out);
  std::istringstream in(out.str());
  std::string line;
  std::getline(in, line);
  EXPECT_EQ(line, "kind,name,value");
  std::getline(in, line);
  EXPECT_EQ(line.rfind("stage,sssp.loop,", 0), 0U) << line;
  // six decimals, as MOSP's "%.6f"
  EXPECT_EQ(line.size() - line.find('.', std::string("stage,sssp.loop,").size()) - 1, 6U) << line;
  std::getline(in, line);
  EXPECT_EQ(line, "counter,sssp.affected,42");
}

TEST(Profiler, Json) {
  dyng::profiler prof;
  prof.begin_stage("graph.apply");
  prof.end_stage();
  prof.add_counter("graph.inserted_edges", 3);
  std::ostringstream out;
  prof.write_json(out);
  const std::string json = out.str();
  EXPECT_EQ(json.rfind("{\"stages\": [{\"name\": \"graph.apply\", \"depth\": 0, \"calls\": 1, ", 0),
            0U)
      << json;
  EXPECT_NE(json.find("\"counters\": [{\"name\": \"graph.inserted_edges\", \"value\": 3}]"),
            std::string::npos)
      << json;
}

TEST(Profiler, ScopedStageThroughResources) {
  auto res = dyng::resources::sequential();
  {
    dyng::scoped_stage none(res, "sssp.loop");  // no profiler attached: nothing happens
  }
  dyng::profiler prof(dyng::profiler_options{/*sync_stages=*/true, false, false});
  res.attach_profiler(&prof);
  {
    dyng::scoped_stage stage(res, "sssp.seed");
    stage.stop();
    stage.stop();  // idempotent
  }
  ASSERT_EQ(prof.stages().size(), 1U);
  EXPECT_EQ(prof.stages()[0].calls, 1);
  EXPECT_THROW(dyng::scoped_stage(res, "BAD"), dyng::invalid_argument_error);
  res.attach_profiler(nullptr);
}

// profiler_options::nvtx adds an NVTX range per stage in builds with DYNG_WITH_NVTX (a no-op
// without an attached NVTX tool) and is ignored otherwise; the recording is the same either way.
TEST(Profiler, NvtxRangesDoNotChangeTheRecord) {
  dyng::profiler_options options;
  options.nvtx = true;
  dyng::profiler prof(options);
  auto res = dyng::resources::sequential();
  res.attach_profiler(&prof);
  {
    dyng::scoped_stage outer(res, "test.outer");
    dyng::scoped_stage inner(res, "test.inner");
  }
  res.attach_profiler(nullptr);
  ASSERT_EQ(prof.stages().size(), 2U);
  EXPECT_EQ(prof.stages()[0].name, "test.outer");
  EXPECT_EQ(prof.stages()[1].depth, 1);
  EXPECT_EQ(prof.samples().size(), 2U);
}
