// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_trace_test.cpp
 * @brief The CBST trace reader on the committed traces of ESCHER-GPU@abcf9b2 and
 *        MOSP_ESCHER@4b86159 (cpp/tests/data/goldens/cbst_trace; ADR 0036): every state digest
 *        rebuilds to the hashes the recorder wrote, and a changed byte is detected.
 */
#include "parity/escher/cbst_trace.hpp"

#include "support/data_paths.hpp"

#include <gtest/gtest.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <sstream>
#include <string>
#include <vector>

namespace dyng::test::cbst_trace {
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

struct replay_result {
  std::size_t ops = 0;
  std::vector<std::string> problems;
  std::string exit;
  std::string repo;
};

replay_result replay(reader& r) {
  replay_result out;
  state_tracker tracker;
  record rec;
  while (r.next(rec)) {
    if (rec.kind == record_kind::header) out.repo = rec.text("repo");
    if (rec.kind == record_kind::end) out.exit = rec.text("exit");
    if (rec.kind == record_kind::op) ++out.ops;
    for (const auto& p : tracker.apply(rec, r))
      out.problems.push_back("seq " + std::to_string(rec.integer("seq")) + ": " + p);
  }
  return out;
}

TEST(CbstTrace, CommittedTracesRebuildToTheRecordedHashes) {
  const auto traces = committed_traces();
  ASSERT_FALSE(traces.empty()) << "no committed traces under " << data_path("goldens/cbst_trace");
  std::size_t ops = 0;
  for (const auto& path : traces) {
    SCOPED_TRACE(path);
    auto r = reader::open(path);
    const replay_result res = replay(*r);
    EXPECT_TRUE(res.problems.empty()) << res.problems.front();
    EXPECT_EQ(res.exit, "normal");
    EXPECT_TRUE(res.repo == "ESCHER-GPU" || res.repo == "MOSP_ESCHER") << res.repo;
    EXPECT_GT(res.ops, 0U);
    ops += res.ops;
  }
  EXPECT_GT(ops, 100U);
}

TEST(CbstTrace, BothOriginalsAreCommitted) {
  bool a = false, b = false;
  for (const auto& path : committed_traces()) {
    auto r = reader::open(path);
    record rec;
    ASSERT_TRUE(r->next(rec));
    ASSERT_EQ(rec.kind, record_kind::header);
    a |= rec.text("repo") == "ESCHER-GPU";
    b |= rec.text("repo") == "MOSP_ESCHER";
    EXPECT_EQ(rec.integer("format_version"), static_cast<std::int64_t>(format_version));
  }
  EXPECT_TRUE(a);
  EXPECT_TRUE(b);
}

// A changed row value is reported by the state tracker (the row hash and the state hash).
TEST(CbstTrace, AChangedRowValueIsDetected) {
  const auto traces = committed_traces();
  ASSERT_FALSE(traces.empty());
  const std::string bytes = read_text(traces.front());
  // Find the first op record with a non-empty inline key.rows and change one value.
  std::istringstream probe(bytes);
  reader r(probe);
  record rec;
  std::size_t offset = 16;
  bool changed = false;
  std::string mutated = bytes;
  while (!changed && r.next(rec)) {
    std::uint64_t len = 0;
    std::memcpy(&len, bytes.data() + offset + 4, 8);
    if (rec.kind == record_kind::op && rec.has("key.rows") &&
        rec.array("key.rows").kind == array_field::storage::inline_data &&
        !rec.array("key.rows").i32.empty()) {
      // the name "key.rows" is followed by: kind, elem, count (8 bytes), values
      const std::string name = "key.rows";
      const auto at = bytes.find(name, offset + 12);
      ASSERT_NE(at, std::string::npos);
      const std::size_t first_value = at + name.size() + 1 + 1 + 8;
      mutated[first_value] = static_cast<char>(mutated[first_value] ^ 0x01);
      changed = true;
    }
    offset += 12 + static_cast<std::size_t>(len);
  }
  ASSERT_TRUE(changed);
  std::istringstream in(mutated);
  reader m(in);
  const replay_result res = replay(m);
  ASSERT_FALSE(res.problems.empty());
}

TEST(CbstTrace, RejectsWhatIsNotATrace) {
  std::istringstream not_a_trace("DYNGCBT0\x01\x00\x00\x00\x00\x00\x00\x00");
  EXPECT_THROW(reader r(not_a_trace), trace_error);
  std::string v2("DYNGCBT1\x02\x00\x00\x00\x00\x00\x00\x00", 16);
  std::istringstream wrong_version(v2);
  EXPECT_THROW(reader r(wrong_version), trace_error);
  // A record header that announces more bytes than there are.
  std::string truncated("DYNGCBT1\x01\x00\x00\x00\x00\x00\x00\x00", 16);
  truncated += std::string("\x03\x00\x00\x00\x40\x00\x00\x00\x00\x00\x00\x00", 12);
  truncated += "abc";
  std::istringstream t(truncated);
  reader r(t);
  record rec;
  EXPECT_THROW(r.next(rec), trace_error);
}

TEST(CbstTrace, ReadsTheFieldTypes) {
  // One note record: an integer, a text, an inline int32 array and a hash-only array.
  std::string b("DYNGCBT1\x01\x00\x00\x00\x00\x00\x00\x00", 16);
  std::string p;
  auto u16 = [&](std::uint16_t v) { p.append(reinterpret_cast<const char*>(&v), 2); };
  auto u32 = [&](std::uint32_t v) { p.append(reinterpret_cast<const char*>(&v), 4); };
  auto u64 = [&](std::uint64_t v) { p.append(reinterpret_cast<const char*>(&v), 8); };
  u32(4);
  p += '\x01';
  u16(3);
  p += "seq";
  u64(7);
  p += '\x02';
  u16(4);
  p += "text";
  u32(2);
  p += "hi";
  p += '\x03';
  u16(1);
  p += "a";
  p += '\x00';
  p += '\x04';
  u64(2);
  u32(5);
  u32(static_cast<std::uint32_t>(-6));
  p += '\x03';
  u16(1);
  p += "h";
  p += '\x02';
  p += '\x04';
  u64(3);
  u64(99);
  const std::uint32_t tag = 6;
  const std::uint64_t len = p.size();
  b.append(reinterpret_cast<const char*>(&tag), 4);
  b.append(reinterpret_cast<const char*>(&len), 8);
  b += p;
  std::istringstream in(b);
  reader r(in);
  record rec;
  ASSERT_TRUE(r.next(rec));
  EXPECT_EQ(rec.kind, record_kind::note);
  EXPECT_EQ(rec.integer("seq"), 7);
  EXPECT_EQ(rec.text("text"), "hi");
  EXPECT_EQ(r.ints(rec.array("a")), (std::vector<std::int32_t>{5, -6}));
  EXPECT_EQ(rec.array("h").kind, array_field::storage::hash_only);
  EXPECT_EQ(rec.array("h").hash, 99U);
  EXPECT_THROW((void)r.ints(rec.array("h")), trace_error);
  EXPECT_FALSE(r.next(rec));
}

TEST(CbstTrace, Fnv1a64MatchesTheReferenceValues) {
  EXPECT_EQ(fnv1a64("", 0), 0xcbf29ce484222325ULL);
  EXPECT_EQ(fnv1a64("a", 1), 0xaf63dc4c8601ec8cULL);
  EXPECT_EQ(fnv1a64("foobar", 6), 0x85944171f73967e8ULL);
}

}  // namespace
}  // namespace dyng::test::cbst_trace
