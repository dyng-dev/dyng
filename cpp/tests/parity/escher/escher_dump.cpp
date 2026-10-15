// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/options.cpp (parseOptions: the batch and graph options)
/**
 * @file escher_dump.cpp
 * @brief ESCHER-GPU's batch dumps: the record decoder, the command line and the by-content
 *        resolver.
 */
#include "parity/escher/escher_dump.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdlib>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::escher_dump {

namespace tr = dyng::test::cbst_trace;

namespace {

long long parse_int(const std::string& s, long long lo, long long hi, const std::string& what) {
  char* end = nullptr;
  errno = 0;
  const long long v = std::strtoll(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0' || errno == ERANGE || v < lo || v > hi) {
    throw tr::trace_error("ESCHER-GPU command line: " + what + " '" + s + "' out of range");
  }
  return v;
}

}  // namespace

escher_options escher_options::parse(const std::vector<std::string>& argv) {
  escher_options o;
  std::vector<std::string> positional;
  bool mode_set = false;
  for (std::size_t i = 1; i < argv.size(); ++i) {
    const std::string& a = argv[i];
    if (a.rfind("--", 0) != 0) {
      positional.push_back(a);
      continue;
    }
    const auto value = [&]() -> const std::string& {
      if (i + 1 >= argv.size())
        throw tr::trace_error("ESCHER-GPU command line: " + a + " needs a value");
      return argv[++i];
    };
    if (a == "--input") {
      o.from_file = true;
      (void)value();
    } else if (a == "--batches") {
      o.batches = parse_int(value(), 0, 2147483647LL, a);
    } else if (a == "--batch-size") {
      const long long v = parse_int(value(), 0, 2147483647LL, a);
      o.deletions = v / 2;
      o.insertions = v - v / 2;
      o.composition_set = true;
    } else if (a == "--deletions") {
      o.deletions = parse_int(value(), 0, 2147483647LL, a);
      o.composition_set = true;
    } else if (a == "--insertions") {
      o.insertions = parse_int(value(), 0, 2147483647LL, a);
      o.composition_set = true;
    } else if (a == "--seed") {
      const std::string& s = value();
      char* end = nullptr;
      errno = 0;
      o.seed = std::strtoull(s.c_str(), &end, 10);
      if (s.empty() || s[0] == '-' || *end != '\0' || errno == ERANGE) {
        throw tr::trace_error("ESCHER-GPU command line: --seed '" + s + "'");
      }
    } else if (a == "--insert-mode") {
      const std::string& m = value();
      if (m != "random" && m != "sample") throw tr::trace_error("ESCHER-GPU: --insert-mode " + m);
      o.sample_inserts = m == "sample";
      mode_set = true;
    } else if (a == "--timing") {
      (void)value();
    } else if (a == "--check" || a == "--reference" || a == "--verbose") {
      continue;
    } else {
      throw tr::trace_error("ESCHER-GPU command line: unknown option " + a);
    }
  }
  const std::size_t fixed = o.from_file ? 0 : 4;
  if (positional.size() < fixed || positional.size() > fixed + 2) {
    throw tr::trace_error("ESCHER-GPU command line: wrong number of positional values");
  }
  if (!o.from_file) {
    o.num_hyperedges = parse_int(positional[0], 1, 2147483647LL, "num_hyperedges");
    o.max_vertices_per_edge =
        static_cast<std::int32_t>(parse_int(positional[1], 1, 2147483647LL, "max_vertices"));
    o.min_vertex_id =
        static_cast<std::int32_t>(parse_int(positional[2], 1, 2147483646LL, "min_vertex_id"));
    o.max_vertex_id =
        static_cast<std::int32_t>(parse_int(positional[3], 1, 2147483646LL, "max_vertex_id"));
  }
  if (positional.size() > fixed) {
    o.payload_capacity = parse_int(positional[fixed], 0, 2147483647LL, "payload_capacity");
  }
  if (!mode_set) o.sample_inserts = o.from_file;
  return o;
}

rows read_rows(tr::reader& in, const tr::record& rec, const std::string& name, int shift) {
  const std::vector<std::int32_t> values = in.ints(rec.array(name + ".values"));
  const std::vector<std::int64_t> prefix = in.longs(rec.array(name + ".prefix"));
  rows out(prefix.size());
  std::int64_t begin = 0;
  for (std::size_t i = 0; i < prefix.size(); ++i) {
    if (prefix[i] < begin || prefix[i] > static_cast<std::int64_t>(values.size())) {
      throw tr::trace_error("batch record: malformed " + name + ".prefix");
    }
    out[i].reserve(static_cast<std::size_t>(prefix[i] - begin));
    for (std::int64_t k = begin; k < prefix[i]; ++k) {
      out[i].push_back(values[static_cast<std::size_t>(k)] - shift);
    }
    begin = prefix[i];
  }
  return out;
}

decoder::event decoder::feed(const tr::record& rec, tr::reader& in) {
  if (rec.kind == tr::record_kind::header) {
    repo_ = rec.text("repo");
    const tr::field* f = rec.find("argv");
    argv_ = f != nullptr ? f->texts : std::vector<std::string>{};
    return event::header;
  }
  if (rec.kind != tr::record_kind::batch) return event::none;
  const std::string hook = rec.text("hook");
  const std::string phase = rec.text("phase");
  if (rec.has("exception.type")) {
    exception_ = rec.text("exception.type");
    return event::exception;
  }
  if (hook == "escher_gpu.build" && phase == "exit") {
    build_ = build_record{};
    build_.loaded = read_rows(in, rec, "loaded_rows", 1);
    build_.num_vertices = static_cast<std::int32_t>(rec.integer("max_vertex_id"));
    build_.num_hyperedge_keys = rec.integer("num_hyperedge_keys");
    build_.device_bytes = rec.integer("device_bytes", -1);
    return event::build;
  }
  if (hook == "escher_gpu.update" && phase == "enter") {
    batch_ = batch_record{};
    batch_.index = rec.integer("batch");
    for (const std::int32_t k : in.ints(rec.array("deleted_keys"))) {
      batch_.deleted_ids.push_back(static_cast<std::int64_t>(k) - 1);
    }
    batch_.deleted_rows = read_rows(in, rec, "deleted_rows", 1);
    batch_.inserted_rows = read_rows(in, rec, "inserted_rows", 1);
    return event::none;
  }
  if (hook == "escher_gpu.update" && phase == "exit") {
    for (const std::int32_t k : in.ints(rec.array("inserted_keys"))) {
      batch_.inserted_ids.push_back(static_cast<std::int64_t>(k) - 1);
    }
    batch_.num_hyperedge_keys = rec.integer("num_hyperedge_keys");
    batch_.device_bytes = rec.integer("device_bytes", -1);
    return event::batch;
  }
  return event::none;
}

std::vector<std::int32_t> normalized(std::vector<std::int32_t> row) {
  std::sort(row.begin(), row.end());
  row.erase(std::unique(row.begin(), row.end()), row.end());
  return row;
}

void by_content_resolver::reset(const rows& by_id) {
  ids_of_.clear();
  content_of_.clear();
  for (std::size_t id = 0; id < by_id.size(); ++id) {
    if (by_id[id].empty()) continue;
    std::vector<std::int32_t> key = normalized(by_id[id]);
    ids_of_[key].insert(static_cast<std::int64_t>(id));
    content_of_.emplace(static_cast<std::int64_t>(id), std::move(key));
  }
}

std::vector<std::int64_t> by_content_resolver::resolve(const rows& deleted_rows,
                                                       std::vector<bool>* ambiguous) const {
  std::vector<std::int64_t> out;
  out.reserve(deleted_rows.size());
  if (ambiguous != nullptr) ambiguous->assign(deleted_rows.size(), false);
  std::set<std::int64_t> taken;
  for (std::size_t i = 0; i < deleted_rows.size(); ++i) {
    std::int64_t chosen = -1;
    const auto it = ids_of_.find(normalized(deleted_rows[i]));
    if (it != ids_of_.end()) {
      for (const std::int64_t id : it->second) {
        if (taken.count(id) == 0) {
          chosen = id;
          break;
        }
      }
      if (ambiguous != nullptr) (*ambiguous)[i] = it->second.size() > 1;
    }
    if (chosen >= 0) taken.insert(chosen);
    out.push_back(chosen);
  }
  return out;
}

void by_content_resolver::commit(const std::vector<std::int64_t>& deleted_ids, const rows& inserted,
                                 const std::vector<std::int64_t>& inserted_ids) {
  for (const std::int64_t id : deleted_ids) {
    const auto c = content_of_.find(id);
    if (c == content_of_.end()) continue;
    auto it = ids_of_.find(c->second);
    if (it != ids_of_.end()) {
      it->second.erase(id);
      if (it->second.empty()) ids_of_.erase(it);
    }
    content_of_.erase(c);
  }
  for (std::size_t i = 0; i < inserted.size() && i < inserted_ids.size(); ++i) {
    std::vector<std::int32_t> key = normalized(inserted[i]);
    ids_of_[key].insert(inserted_ids[i]);
    content_of_[inserted_ids[i]] = std::move(key);
  }
}

}  // namespace dyng::test::escher_dump
