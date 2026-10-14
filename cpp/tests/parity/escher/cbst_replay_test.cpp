// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_replay_test.cpp
 * @brief The host model replays the committed CBST traces of ESCHER-GPU@abcf9b2 and
 *        MOSP_ESCHER@4b86159 (cpp/tests/data/goldens/cbst_trace) with 100% agreement, under each
 *        original's preset and every field set; perturbed traces are reported as mismatches
 *        (ADR 0037).
 */
#include "parity/escher/cbst_compare.hpp"
#include "parity/escher/cbst_trace.hpp"
#include "support/data_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::test::cbst_compare {
namespace {

std::vector<std::string> committed_traces() {
  std::vector<std::string> out;
  const std::filesystem::path root = data_path("goldens/cbst_trace");
  if (!std::filesystem::is_directory(root)) return out;
  for (const auto& e : std::filesystem::recursive_directory_iterator(root))
    if (e.is_regular_file() && e.path().extension() == ".cbt") out.push_back(e.path().string());
  std::sort(out.begin(), out.end());
  return out;
}

std::string describe(const replay_report& r) {
  std::ostringstream s;
  s << r.mismatch_count << " mismatches";
  for (const auto& m : r.mismatches)
    s << "\n  op " << m.op_index << " (tree " << m.tree << " " << m.dataset << ", its op "
      << m.tree_op << ", " << m.op << ") " << m.field << " key " << m.key << ": trace "
      << m.expected << ", model " << m.actual;
  return s.str();
}

replay_report replay_file(const std::string& path, field_set fields) {
  auto in = cbst_trace::reader::open(path);
  options opt;
  opt.fields = fields;
  return replay(*in, host_model_factory(""), opt, path);
}

class CommittedTrace : public ::testing::TestWithParam<field_set> {};

TEST_P(CommittedTrace, TheHostModelAgreesWithEveryOperation) {
  const auto traces = committed_traces();
  ASSERT_FALSE(traces.empty()) << "no committed traces under " << data_path("goldens/cbst_trace");
  std::int64_t ops = 0, mappings = 0;
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    const replay_report r = replay_file(path, GetParam());
    EXPECT_TRUE(r.ok()) << describe(r);
    EXPECT_EQ(r.unreplayable, 0);
    EXPECT_EQ(r.compared, r.ops);
    EXPECT_GT(r.ops, 0);
    ops += r.ops;
    mappings += r.mappings;
  }
  EXPECT_GT(ops, 100);
  EXPECT_GT(mappings, 10);
}

INSTANTIATE_TEST_SUITE_P(FieldSets, CommittedTrace,
                         ::testing::Values(field_set::semantic, field_set::offsets,
                                           field_set::exact),
                         [](const ::testing::TestParamInfo<field_set>& p) {
                           return std::string(to_string(p.param));
                         });

// --- Perturbed traces -------------------------------------------------------------------------

// A trace re-serialized from its records (format 1, every array inline), so a test can change one
// input or output and replay the result.
class trace_writer {
 public:
  void put(const cbst_trace::record& r) {
    std::string p;
    u32(p, static_cast<std::uint32_t>(r.fields.size()));
    for (const auto& [name, f] : r.fields) {
      switch (f.kind) {
        case cbst_trace::field::type::integer:
          p += '\x01';
          name_(p, name);
          u64(p, static_cast<std::uint64_t>(f.integer));
          break;
        case cbst_trace::field::type::text:
          p += '\x02';
          name_(p, name);
          u32(p, static_cast<std::uint32_t>(f.text.size()));
          p += f.text;
          break;
        case cbst_trace::field::type::texts:
          p += '\x04';
          name_(p, name);
          u32(p, static_cast<std::uint32_t>(f.texts.size()));
          for (const auto& t : f.texts) {
            u32(p, static_cast<std::uint32_t>(t.size()));
            p += t;
          }
          break;
        case cbst_trace::field::type::array: {
          p += '\x03';
          name_(p, name);
          const auto& a = f.array;
          using st = cbst_trace::array_field::storage;
          if (a.kind == st::side_file)
            throw cbst_trace::trace_error("side-file array in a test trace");
          p += static_cast<char>(a.kind == st::inline_data ? 0 : (a.kind == st::hash_only ? 2 : 3));
          p += static_cast<char>(a.elem_bytes);
          u64(p, a.count);
          if (a.kind == st::inline_data) {
            if (a.elem_bytes == 4)
              p.append(reinterpret_cast<const char*>(a.i32.data()), a.i32.size() * 4);
            else
              p.append(reinterpret_cast<const char*>(a.i64.data()), a.i64.size() * 8);
          } else if (a.kind == st::hash_only) {
            u64(p, a.hash);
          }
          break;
        }
      }
    }
    u32(out_, static_cast<std::uint32_t>(r.kind));
    u64(out_, p.size());
    out_ += p;
  }
  [[nodiscard]] std::string bytes() const {
    return std::string("DYNGCBT1\x01\x00\x00\x00\x00\x00\x00\x00", 16) + out_;
  }

 private:
  static void u32(std::string& s, std::uint32_t v) {
    s.append(reinterpret_cast<const char*>(&v), 4);
  }
  static void u64(std::string& s, std::uint64_t v) {
    s.append(reinterpret_cast<const char*>(&v), 8);
  }
  static void name_(std::string& s, const std::string& n) {
    const auto len = static_cast<std::uint16_t>(n.size());
    s.append(reinterpret_cast<const char*>(&len), 2);
    s += n;
  }
  std::string out_;
};

std::vector<cbst_trace::record> records_of(const std::string& path) {
  auto in = cbst_trace::reader::open(path);
  std::vector<cbst_trace::record> out;
  cbst_trace::record r;
  while (in->next(r)) out.push_back(r);
  return out;
}

cbst_trace::field* field_of(cbst_trace::record& r, const std::string& name) {
  for (auto& f : r.fields)
    if (f.first == name) return &f.second;
  return nullptr;
}

replay_report replay_records(const std::vector<cbst_trace::record>& recs, field_set fields) {
  trace_writer w;
  for (const auto& r : recs) w.put(r);
  std::istringstream in(w.bytes());
  cbst_trace::reader reader(in);
  options opt;
  opt.fields = fields;
  return replay(reader, host_model_factory(""), opt, "perturbed");
}

// The op index (count of op records before it) of a record.
std::int64_t op_index_of(const std::vector<cbst_trace::record>& recs, std::size_t at) {
  std::int64_t n = 0;
  for (std::size_t i = 0; i < at; ++i) n += recs[i].kind == cbst_trace::record_kind::op;
  return n;
}

std::string trace_path(const std::string& relative) {
  return data_path("goldens/cbst_trace/" + relative);
}

TEST(PerturbedTrace, TheUnchangedRecordsReplayClean) {
  for (const char* t :
       {"escher_gpu_targeted/targeted_small.cbt", "mosp_escher_targeted/targeted_small.cbt"}) {
    const replay_report r = replay_records(records_of(trace_path(t)), field_set::exact);
    EXPECT_TRUE(r.ok()) << t << ": " << describe(r);
  }
}

// One item size changed: an insert item loses its last value to the next item (the payload and
// the item count stay); the model's result differs from the recorded one at that op.
TEST(PerturbedTrace, OneItemSizeChangedIsAMismatch) {
  for (const char* t :
       {"escher_gpu_targeted/targeted_small.cbt", "mosp_escher_targeted/targeted_small.cbt",
        "escher_gpu_test/001-main.cbt"}) {
    SCOPED_TRACE(t);
    std::vector<cbst_trace::record> recs = records_of(trace_path(t));
    std::size_t at = recs.size();
    for (std::size_t i = 0; i < recs.size() && at == recs.size(); ++i) {
      if (recs[i].kind != cbst_trace::record_kind::op || recs[i].text("op") != "insert") continue;
      auto& prefix = field_of(recs[i], "in.prefix")->array.i32;
      for (std::size_t k = 0; k + 1 < prefix.size(); ++k) {
        const std::int32_t size = prefix[k] - (k == 0 ? 0 : prefix[k - 1]);
        if (size >= 2) {
          --prefix[k];
          at = i;
          break;
        }
      }
    }
    ASSERT_LT(at, recs.size()) << "no insert with an item of two values";
    for (field_set f : {field_set::semantic, field_set::exact}) {
      const replay_report r = replay_records(recs, f);
      ASSERT_FALSE(r.ok());
      EXPECT_EQ(r.mismatches.front().op_index, op_index_of(recs, at)) << describe(r);
    }
  }
}

// One tie reordered: an insert whose items took two deleted slots of equal capacity, with the
// recorded mapping of the two items swapped (the original breaking the tie the other way).
TEST(PerturbedTrace, OneTieReorderedIsAMismatch) {
  for (const char* t :
       {"escher_gpu_targeted/targeted_small.cbt", "mosp_escher_targeted/targeted_small.cbt"}) {
    SCOPED_TRACE(t);
    std::vector<cbst_trace::record> recs = records_of(trace_path(t));
    // The state before every op, to find two reused slots of equal length.
    auto in = cbst_trace::reader::open(trace_path(t));
    cbst_trace::state_tracker tracker;
    std::size_t at = recs.size();
    for (std::size_t i = 0; i < recs.size() && at == recs.size(); ++i) {
      cbst_trace::record& r = recs[i];
      if (r.kind == cbst_trace::record_kind::op && r.text("op") == "insert" &&
          r.has("out.mapping")) {
        const cbst_trace::tree_state* before =
            tracker.tree(static_cast<std::uint32_t>(r.integer("tree")));
        auto& mapping = field_of(r, "out.mapping")->array.i32;
        for (std::size_t a = 0; before && a < mapping.size() && at == recs.size(); ++a)
          for (std::size_t b = a + 1; b < mapping.size(); ++b) {
            const auto ka = before->by_key.find(mapping[a]);
            const auto kb = before->by_key.find(mapping[b]);
            if (ka == before->by_key.end() || kb == before->by_key.end()) continue;
            if (ka->second.avail == 1 && kb->second.avail == 1 &&
                ka->second.length == kb->second.length) {
              std::swap(mapping[a], mapping[b]);
              at = i;
              break;
            }
          }
      }
      (void)tracker.apply(recs[i], *in);
    }
    ASSERT_LT(at, recs.size()) << "no insert into two deleted slots of equal capacity";
    const replay_report r = replay_records(recs, field_set::semantic);
    ASSERT_FALSE(r.ok());
    EXPECT_EQ(r.mismatches.front().op_index, op_index_of(recs, at)) << describe(r);
    EXPECT_EQ(r.mismatches.front().field, "out.mapping");
  }
}

// An op missing on the store's side (a policy without compact()) fails, and so does a trace that
// ends inside an operation.
TEST(PerturbedTrace, AnOpTheStoreCannotRunIsAMismatch) {
  auto in = cbst_trace::reader::open(trace_path("mosp_escher_targeted/targeted_small.cbt"));
  options opt;
  const replay_report r = replay(*in, host_model_factory("escher_gpu"), opt, "b-under-a");
  ASSERT_FALSE(r.ok());
  bool compact = false;
  for (const auto& m : r.mismatches) compact |= m.op == "compact" && m.field == "status";
  EXPECT_TRUE(compact) << describe(r);
  std::vector<cbst_trace::record> recs = records_of(trace_path("escher_gpu_test/001-main.cbt"));
  ASSERT_EQ(recs.back().kind, cbst_trace::record_kind::end);
  field_of(recs.back(), "exit")->text = "exit-during-op";
  const replay_report e = replay_records(recs, field_set::semantic);
  ASSERT_FALSE(e.ok());
  EXPECT_EQ(e.mismatches.back().field, "trace.exit");
}

// A hash-only input is reported as unreplayable from that op on, never skipped silently.
TEST(PerturbedTrace, AHashOnlyInputMakesTheTreeUnreplayable) {
  std::vector<cbst_trace::record> recs = records_of(trace_path("escher_gpu_test/001-main.cbt"));
  std::size_t at = recs.size();
  for (std::size_t i = 0; i < recs.size(); ++i)
    if (recs[i].kind == cbst_trace::record_kind::op && recs[i].text("op") == "fill") {
      at = i;
      break;
    }
  ASSERT_LT(at, recs.size());
  auto& a = field_of(recs[at], "in.payload")->array;
  a.kind = cbst_trace::array_field::storage::hash_only;
  a.hash = cbst_trace::fnv1a64(a.i32.data(), a.i32.size() * 4);
  a.i32.clear();
  const replay_report r = replay_records(recs, field_set::semantic);
  EXPECT_TRUE(r.ok()) << describe(r);
  EXPECT_GT(r.unreplayable, 0);
  const auto& tree = r.trees.at(static_cast<std::uint32_t>(recs[at].integer("tree")));
  EXPECT_NE(tree.unreplayable_reason.find("hash-only"), std::string::npos)
      << tree.unreplayable_reason;
}

TEST(ReplayReport, JsonHasTheCounters) {
  const replay_report r = replay_file(trace_path("escher_gpu_test/001-main.cbt"), field_set::exact);
  const std::string j = r.to_json();
  for (const char* key : {"\"ok\": true", "\"ops\": 8", "\"mismatch_count\": 0", "\"trees\": [",
                          "\"fields\": \"exact\""})
    EXPECT_NE(j.find(key), std::string::npos) << key << " in " << j.substr(0, 400);
}

}  // namespace
}  // namespace dyng::test::cbst_compare
