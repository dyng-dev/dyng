// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hsosp_dump.cpp
 * @brief The H-SOSP dump reader and the oracle check of a dump (see hsosp_dump.hpp).
 */
#include "parity/hsosp/hsosp_dump.hpp"

#include <dyng/testing/hyper_sssp_reference.hpp>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hsosp_dump {

namespace {

const char* const tag_names[] = {"",           "header",      "session",     "load",
                                 "device_build", "begin_batch", "apply_batch", "sssp",
                                 "compare",    "end"};

std::string quote_for_shell(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

struct cursor {
  const std::string& b;
  std::size_t pos = 0;
  std::size_t end = 0;

  void need(std::size_t n) const {
    if (pos + n > end) throw dump_error("H-SOSP dump: truncated record");
  }
  std::uint64_t u(int bytes) {
    need(static_cast<std::size_t>(bytes));
    std::uint64_t v = 0;
    for (int i = 0; i < bytes; ++i)
      v |= static_cast<std::uint64_t>(static_cast<unsigned char>(b[pos + static_cast<std::size_t>(i)]))
           << (8 * i);
    pos += static_cast<std::size_t>(bytes);
    return v;
  }
  std::string text(std::size_t n) {
    need(n);
    std::string s = b.substr(pos, n);
    pos += n;
    return s;
  }
};

}  // namespace

std::int64_t record::i64(const std::string& name) const {
  auto it = fields.find(name);
  if (it == fields.end() || it->second.type != field::kind::i64)
    throw dump_error("H-SOSP dump: " + tag + " record has no int64 field " + name);
  return it->second.value;
}

const std::string& record::str(const std::string& name) const {
  auto it = fields.find(name);
  if (it == fields.end() || it->second.type != field::kind::str)
    throw dump_error("H-SOSP dump: " + tag + " record has no string field " + name);
  return it->second.text;
}

const std::vector<std::int32_t>& record::ints(const std::string& name) const {
  auto it = fields.find(name);
  if (it == fields.end() || it->second.type != field::kind::array32)
    throw dump_error("H-SOSP dump: " + tag + " record has no int32 array " + name);
  return it->second.ints;
}

const std::vector<std::int64_t>& record::longs(const std::string& name) const {
  auto it = fields.find(name);
  if (it == fields.end() || it->second.type != field::kind::array64)
    throw dump_error("H-SOSP dump: " + tag + " record has no int64 array " + name);
  return it->second.longs;
}

std::vector<record> parse_dump(const std::string& bytes) {
  if (bytes.size() < 16 || bytes.compare(0, 8, "DYNGHSD1") != 0)
    throw dump_error("H-SOSP dump: not a DYNGHSD1 file");
  cursor head{bytes, 8, bytes.size()};
  const auto version = static_cast<std::uint32_t>(head.u(4));
  if (version != format_version)
    throw dump_error("H-SOSP dump: format " + std::to_string(version) + ", this reader reads " +
                     std::to_string(format_version));
  std::vector<record> out;
  std::size_t pos = 16;
  while (pos < bytes.size()) {
    cursor c{bytes, pos, bytes.size()};
    const auto tag = static_cast<std::uint32_t>(c.u(4));
    const std::uint64_t len = c.u(8);
    if (c.pos + len > bytes.size()) throw dump_error("H-SOSP dump: truncated record");
    record r;
    r.tag = tag < std::size(tag_names) ? tag_names[tag] : "tag" + std::to_string(tag);
    c.end = c.pos + static_cast<std::size_t>(len);
    const auto count = static_cast<std::uint32_t>(c.u(4));
    for (std::uint32_t k = 0; k < count; ++k) {
      const auto type = static_cast<int>(c.u(1));
      const auto name_len = static_cast<std::size_t>(c.u(2));
      const std::string name = c.text(name_len);
      field f;
      if (type == 1) {
        f.type = field::kind::i64;
        f.value = static_cast<std::int64_t>(c.u(8));
      } else if (type == 2) {
        f.type = field::kind::str;
        f.text = c.text(static_cast<std::size_t>(c.u(4)));
      } else if (type == 3) {
        const auto storage = static_cast<int>(c.u(1));
        const auto elem = static_cast<int>(c.u(1));
        const std::uint64_t n = c.u(8);
        if (storage != 0 || (elem != 4 && elem != 8))
          throw dump_error("H-SOSP dump: array " + name + " is not inline int32 / int64");
        c.need(static_cast<std::size_t>(n) * static_cast<std::size_t>(elem));
        if (elem == 4) {
          f.type = field::kind::array32;
          f.ints.resize(static_cast<std::size_t>(n));
          if (n) std::memcpy(f.ints.data(), bytes.data() + c.pos, static_cast<std::size_t>(n) * 4);
        } else {
          f.type = field::kind::array64;
          f.longs.resize(static_cast<std::size_t>(n));
          if (n) std::memcpy(f.longs.data(), bytes.data() + c.pos, static_cast<std::size_t>(n) * 8);
        }
        c.pos += static_cast<std::size_t>(n) * static_cast<std::size_t>(elem);
      } else if (type == 4) {
        f.type = field::kind::strings;
        const auto items = static_cast<std::uint32_t>(c.u(4));
        for (std::uint32_t i = 0; i < items; ++i)
          f.list.push_back(c.text(static_cast<std::size_t>(c.u(4))));
      } else {
        throw dump_error("H-SOSP dump: unknown field type of " + name);
      }
      r.fields.emplace(name, std::move(f));
    }
    out.push_back(std::move(r));
    pos = c.end;
  }
  return out;
}

std::vector<record> read_dump(const std::string& path) {
  std::string bytes;
  if (path.size() > 4 && path.compare(path.size() - 4, 4, ".zst") == 0) {
    FILE* f = popen(("zstd -q -dc --long=31 " + quote_for_shell(path)).c_str(), "r");
    if (f == nullptr) throw dump_error("H-SOSP dump: cannot run zstd for " + path);
    char buf[1 << 16];
    std::size_t n = 0;
    while ((n = std::fread(buf, 1, sizeof buf, f)) > 0) bytes.append(buf, n);
    if (pclose(f) != 0) throw dump_error("H-SOSP dump: zstd failed on " + path);
  } else {
    std::ifstream in(path, std::ios::binary);
    if (!in) throw dump_error("H-SOSP dump: cannot open " + path);
    bytes.assign(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
  }
  return parse_dump(bytes);
}

// --- The oracle check -------------------------------------------------------------------------

namespace {

// The hypergraph of a session as its records give it (0-based ids = B id - 1).
struct state {
  bool known = false;
  std::string why = "no bulk load recorded";
  std::int32_t num_vertices = 0;
  std::vector<std::vector<std::int32_t>> rows;
  std::vector<std::int64_t> weights;
  std::int64_t version = 0;  // bumped by every change

  void csr(std::vector<std::int64_t>& off, std::vector<std::int32_t>& mem) const {
    off.assign(1, 0);
    mem.clear();
    for (const auto& r : rows) {
      mem.insert(mem.end(), r.begin(), r.end());
      off.push_back(static_cast<std::int64_t>(mem.size()));
    }
  }
};

std::vector<std::vector<std::int32_t>> rows_of(const record& r, const std::string& name) {
  const auto& off = r.longs(name + ".offsets");
  const auto& val = r.ints(name + ".values");
  std::vector<std::vector<std::int32_t>> out;
  out.reserve(off.empty() ? 0 : off.size() - 1);
  for (std::size_t i = 0; i + 1 < off.size(); ++i)
    out.emplace_back(val.begin() + off[i], val.begin() + off[i + 1]);
  return out;
}

std::vector<std::pair<std::int64_t, std::int64_t>> pairs_of(const record& r, const char* kind) {
  const std::string base = std::string("pairs.") + kind;
  const auto& a = r.ints(base + ".a");
  const auto& b = r.ints(base + ".b");
  std::vector<std::pair<std::int64_t, std::int64_t>> out;
  for (std::size_t i = 0; i < a.size() && i < b.size(); ++i) out.emplace_back(a[i], b[i]);
  std::sort(out.begin(), out.end());
  return out;
}

void note(oracle_report& rep, std::size_t max_problems, const std::string& text) {
  if (rep.problems.size() < max_problems) rep.problems.push_back(text);
}

}  // namespace

oracle_report check_with_oracle(const std::vector<record>& records, std::size_t max_problems) {
  oracle_report rep;
  std::map<std::int64_t, state> sessions;
  // The reference of the last state of a session (version -> distances and parents).
  std::map<std::int64_t, std::pair<std::int64_t, std::pair<std::vector<std::int64_t>,
                                                         std::vector<std::int32_t>>>>
      cache;
  for (const record& r : records) {
    if (!r.has("session")) continue;
    const std::int64_t sid = r.i64("session");
    if (sid <= 0) continue;
    state& s = sessions[sid];
    const std::string where = "session " + std::to_string(sid);
    if (r.tag == "session") {
      ++rep.sessions;
      s.num_vertices = static_cast<std::int32_t>(r.i64("num_vertices"));
    } else if (r.tag == "load") {
      if (r.str("phase") == "enter") {
        s.rows = rows_of(r, "rows");
        s.weights = r.longs("weights");
        s.known = false;
        s.why = "the bulk load did not finish";
      } else if (r.str("exception").empty()) {
        s.known = true;
        ++s.version;
      }
    } else if (r.tag == "begin_batch" && r.str("phase") == "exit" && r.str("exception").empty()) {
      // The host model changed; the next apply_batch record (the same batch) gives the rows.
      s.known = false;
      s.why = "a batch began without applyBatch (no post-batch rows recorded)";
    } else if (r.tag == "apply_batch") {
      if (!r.str("exception").empty()) {
        s.known = false;
        s.why = "applyBatch threw";
        continue;
      }
      // The state before the batch: the last known one (the begin_batch inside applyBatch only
      // marked it stale).
      std::vector<std::int64_t> pre_off, post_off;
      std::vector<std::int32_t> pre_mem, post_mem;
      s.csr(pre_off, pre_mem);
      const auto ids = r.ints("rows.ids");
      const auto rows = rows_of(r, "rows");
      const auto& w = r.longs("rows.weight");
      const auto& alive = r.ints("rows.alive");
      const auto max_id = static_cast<std::size_t>(r.i64("max_id"));
      if (r.str("rows.kind") == "all") {
        s.rows.assign(max_id, {});
        s.weights.assign(max_id, 0);
      } else {
        s.rows.resize(std::max(s.rows.size(), max_id));
        s.weights.resize(std::max(s.weights.size(), max_id), 0);
      }
      for (std::size_t k = 0; k < ids.size(); ++k) {
        const auto id = static_cast<std::size_t>(ids[k] - 1);
        s.rows[id] = alive[k] ? rows[k] : std::vector<std::int32_t>{};
        s.weights[id] = w[k];
      }
      s.known = true;
      ++s.version;
      s.csr(post_off, post_mem);
      const auto delta =
          dyng::testing::line_graph_delta(pre_off, pre_mem, post_off, post_mem, s.num_vertices);
      const auto del = pairs_of(r, "deleted");
      const auto ins = pairs_of(r, "inserted");
      ++rep.batches_checked;
      if (delta.first != del || delta.second != ins) {
        ++rep.pair_mismatches;
        note(rep, max_problems,
             where + " batch " + std::to_string(r.i64("index")) + ": pairs differ (oracle " +
                 std::to_string(delta.first.size()) + " deleted / " +
                 std::to_string(delta.second.size()) + " inserted, B " +
                 std::to_string(del.size()) + " / " + std::to_string(ins.size()) + ")");
      }
      if (static_cast<std::int64_t>(del.size()) != r.i64("delta.num_del") ||
          static_cast<std::int64_t>(ins.size()) != r.i64("delta.num_ins"))
        note(rep, max_problems, where + ": the sorted keys disagree with numDel / numIns");
    } else if (r.tag == "sssp") {
      if (!r.str("exception").empty()) continue;
      if (!s.known) {
        ++rep.not_checked;
        continue;
      }
      auto& entry = cache[sid];
      if (entry.first != s.version || entry.second.first.empty()) {
        std::vector<std::int64_t> off;
        std::vector<std::int32_t> mem;
        s.csr(off, mem);
        entry.first = s.version;
        entry.second = dyng::testing::hyper_sssp_reference(off, mem, s.num_vertices, s.weights,
                                                           r.i64("source") - 1);
      }
      const auto& want_d = entry.second.first;
      const auto& want_p = entry.second.second;
      const auto& got_d = r.longs("distances");
      const auto& got_p = r.ints("parents");
      ++rep.sssp_checked;
      std::int64_t dm = 0, pm = 0, first_d = -1, first_p = -1;
      const std::size_t n = std::max({want_d.size(), got_d.size(), got_p.size()});
      constexpr std::int64_t inf = INT64_MAX / 4;
      for (std::size_t id = 0; id < n; ++id) {
        const std::int64_t wd = id < want_d.size() ? want_d[id] : inf;
        const std::int64_t gd = id < got_d.size() ? got_d[id] : inf;
        const std::int32_t wp = id < want_p.size() ? want_p[id] : -1;
        const std::int32_t gp = id < got_p.size() ? got_p[id] : -1;
        if (wd != gd && dm++ == 0) first_d = static_cast<std::int64_t>(id);
        if (wp != gp && pm++ == 0) first_p = static_cast<std::int64_t>(id);
      }
      rep.distance_mismatches += dm;
      rep.parent_mismatches += pm;
      if (dm || pm) {
        std::ostringstream o;
        o << where << " " << r.str("kind") << " after batch " << r.i64("batches") << ": " << dm
          << " distance and " << pm << " parent differences";
        if (first_d >= 0)
          o << "; id " << first_d << ": B "
            << (static_cast<std::size_t>(first_d) < got_d.size() ? got_d[static_cast<std::size_t>(first_d)] : inf)
            << ", oracle "
            << (static_cast<std::size_t>(first_d) < want_d.size() ? want_d[static_cast<std::size_t>(first_d)] : inf);
        if (first_p >= 0)
          o << "; id " << first_p << ": parent B "
            << (static_cast<std::size_t>(first_p) < got_p.size() ? got_p[static_cast<std::size_t>(first_p)] : -1)
            << ", oracle "
            << (static_cast<std::size_t>(first_p) < want_p.size() ? want_p[static_cast<std::size_t>(first_p)] : -1);
        note(rep, max_problems, o.str());
      }
    }
  }
  return rep;
}

}  // namespace dyng::test::hsosp_dump
