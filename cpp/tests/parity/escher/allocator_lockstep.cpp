// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file allocator_lockstep.cpp
 * @brief The host id allocator in lockstep with MOSP_ESCHER@4b86159's recorded CBST trees and the
 *        host model (M11a criterion 8; ADR 0085).
 */
#include "parity/escher/allocator_lockstep.hpp"

#include "hypergraph/capacity.hpp"
#include "hypergraph/id_allocator.hpp"
#include "parity/escher/cbst_compare.hpp"

#include <dyng/core/error.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>
#include <dyng/testing/cbst_host_model.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::allocator_lockstep {

namespace tr = dyng::test::cbst_trace;
namespace cc = dyng::test::cbst_compare;
using dyng::detail::hyper::id_allocator;

namespace {

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char c : s) {
    if (c == '"' || c == '\\') {
      out += '\\';
      out += c;
    } else if (static_cast<unsigned char>(c) < 0x20) {
      out += ' ';
    } else {
      out += c;
    }
  }
  return out + "\"";
}

/// One tree run in lockstep.
struct tree_run {
  std::string name;
  std::unique_ptr<id_allocator> alloc;
  std::unique_ptr<testing::cbst_host_model> model;
  std::map<std::int32_t, std::vector<std::int32_t>> rows;  // the rows as the operations wrote them
  std::int64_t capacity = 0, alignment = 4, ops = 0;
  bool stopped = false;  // undefined after a failed operation, or no longer sets
};

class runner {
 public:
  runner(tr::reader& in, const options& opt, report& rep) : in_(in), opt_(opt), rep_(rep) {}

  void on(const tr::record& rec) {
    const auto problems = tracker_.apply(rec, in_);
    for (const auto& p : problems) mismatch("trace: " + p);
    if (rec.kind == tr::record_kind::header) rep_.repo = rec.text("repo");
    if (rec.kind == tr::record_kind::tree) {
      const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
      tree_run& t = trees_[id];
      t = tree_run{};
      t.name = rec.text("dataset_name");
      t.capacity = rec.integer("payload_capacity");
      t.alignment = rec.integer("alignment", 4);
      return;
    }
    if (rec.kind == tr::record_kind::op) on_op(rec);
  }

 private:
  bool selected(const std::string& name) const {
    return opt_.trees == "all" || name == opt_.trees;
  }

  void mismatch(const std::string& what) {
    ++rep_.mismatch_count;
    if (rep_.mismatches.size() < opt_.max_mismatches) rep_.mismatches.push_back(what);
  }

  void on_op(const tr::record& rec) {
    const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
    auto found = trees_.find(id);
    if (found == trees_.end() || !selected(found->second.name)) return;
    tree_run& t = found->second;
    const std::string op = rec.text("op");
    if (op == "construct") start(t, rec);
    if (!t.alloc || t.stopped) return;
    if (op == "construct") {
      compare(t, id, "construct");
      return;
    }
    cc::op_request req;
    std::string missing;
    if (!cc::decode_op(rec, in_, req, missing)) {
      ++rep_.unreplayable;
      t.stopped = true;
      return;
    }
    ++rep_.ops;
    ++t.ops;
    const bool trace_threw = rec.integer("status") != 0;
    const std::string where =
        t.name + " (tree " + std::to_string(id) + ") op " + std::to_string(t.ops) + " " + op;
    bool threw = false, model_threw = false;
    std::vector<std::int32_t> mapping;
    try {
      if (op == "insert") {
        std::vector<std::int32_t> counts;
        for (std::size_t i = 0; i < req.keys.size(); ++i)
          counts.push_back(req.prefix[i] - (i == 0 ? 0 : req.prefix[i - 1]));
        mapping = t.alloc->insert(counts);
      } else if (op == "erase") {
        t.alloc->erase(req.keys);
      } else if (op == "fill") {
        if (!sets_after_fill(t, req)) {
          ++rep_.not_sets;
          t.stopped = true;
          return;
        }
        for (std::size_t i = 0; i < req.keys.size(); ++i) {
          const std::int32_t k = req.prefix[i] - (i == 0 ? 0 : req.prefix[i - 1]);
          const std::int32_t room =
              t.alloc->tail_capacity(req.keys[i]) -
              std::min(t.alloc->occupancy(req.keys[i]), t.alloc->tail_capacity(req.keys[i]));
          rep_.overflow_fills += k > room ? 1 : 0;
        }
        ++rep_.fills;
        t.alloc->append(req.keys, req.prefix, req.payload);
      } else if (op == "unfill") {
        ++rep_.unfills;
        // Only the members of each row (an absent value changes nothing in the original either).
        std::vector<std::int32_t> prefix, values;
        for (std::size_t i = 0; i < req.keys.size(); ++i) {
          const std::int32_t begin = i == 0 ? 0 : req.prefix[i - 1];
          const auto& row = t.rows[req.keys[i]];
          std::set<std::int32_t> taken;
          for (std::int32_t k = begin; k < req.prefix[i]; ++k) {
            const std::int32_t v = req.payload[static_cast<std::size_t>(k)];
            if (std::find(row.begin(), row.end(), v) != row.end() && taken.insert(v).second)
              values.push_back(v);
          }
          prefix.push_back(static_cast<std::int32_t>(values.size()));
          rep_.multi_segment_unfills += t.alloc->multi_segment(req.keys[i]) ? 1 : 0;
        }
        t.alloc->remove_values(req.keys, prefix, values);
      } else if (op == "compact") {
        (void)t.alloc->compact();
      } else {
        mismatch(where + ": an operation the allocator does not model");
        t.stopped = true;
        return;
      }
    } catch (const dyng::capacity_error&) {
      threw = true;
    } catch (const dyng::invalid_argument_error&) {
      threw = true;
    }
    testing::cbst_op_result mr;
    if (op == "insert") {
      mr = t.model->insert(req.keys, req.payload, req.prefix);
    } else if (op == "erase") {
      mr = t.model->erase(req.keys);
    } else if (op == "fill") {
      mr = t.model->append(req.keys, req.payload, req.prefix);
    } else if (op == "unfill") {
      mr = t.model->remove_values(req.keys, req.payload, req.prefix);
    } else {
      mr = t.model->compact();
    }
    model_threw = mr.status != testing::cbst_op_status::ok;
    if (threw != trace_threw)
      mismatch(where + ": the allocator " + (threw ? "threw" : "did not throw") +
               ", the original " +
               (trace_threw ? "threw " + rec.text("exception.type") : "did not"));
    if (model_threw != trace_threw) mismatch(where + ": the model and the original disagree");
    if (trace_threw) {
      ++rep_.exceptions;
      t.stopped = true;  // both are undefined after a failure (the container poisons)
      return;
    }
    if (op == "insert") {
      ++rep_.inserts;
      const std::vector<std::int32_t> want =
          rec.has("out.mapping") ? in_.ints(rec.array("out.mapping")) : mr.mapping;
      if (mapping != want) mismatch(where + ": the InsertMapping differs from the original's");
      if (mapping != mr.mapping) mismatch(where + ": the InsertMapping differs from the model's");
    }
    track_rows(t, op, req, mapping);
    compare(t, id, where);
  }

  void start(tree_run& t, const tr::record& rec) {
    cc::op_request req;
    std::string missing;
    if (!cc::decode_op(rec, in_, req, missing)) {
      ++rep_.unreplayable;
      return;
    }
    ++rep_.trees;
    // MOSP_ESCHER's store: a fixed capacity, compaction at its trigger points (the mosp_escher
    // policy the container maps capacity_policy::mosp_escher() to).
    escher_policy_ = dyng::detail::hyper::to_store_policy(capacity_policy::mosp_escher());
    escher_policy_.alignment = static_cast<std::int32_t>(req.alignment);
    t.alloc = std::make_unique<id_allocator>(escher_policy_, req.capacity, t.name);
    std::vector<std::int32_t> counts;
    t.rows.clear();
    for (std::size_t i = 0; i < req.keys.size(); ++i) {
      std::vector<std::int32_t> row;
      for (std::int64_t p = req.starts[i];; ++p) {
        const std::int32_t v = req.flat[static_cast<std::size_t>(p)];
        if (v == INT_MIN || v == 0 || v < 0) break;
        row.push_back(v);
      }
      counts.push_back(req.has_row_counts ? req.row_counts[i]
                                          : static_cast<std::int32_t>(row.size()));
      t.rows[req.keys[i]] = row;
    }
    try {
      t.alloc->construct(counts);
    } catch (const dyng::invalid_argument_error&) {
      t.stopped = true;
    }
    t.model = std::make_unique<testing::cbst_host_model>(testing::cbst_host_policy::mosp_escher(),
                                                         req.capacity, req.alignment, t.name);
    const auto r = t.model->construct(req.keys, req.starts, req.flat,
                                      req.has_row_counts ? &req.row_counts : nullptr);
    if (r.status != testing::cbst_op_status::ok || rec.integer("status") != 0) t.stopped = true;
    for (const auto& [key, row] : t.rows)
      if (std::set<std::int32_t>(row.begin(), row.end()).size() != row.size()) {
        ++rep_.not_sets;
        t.stopped = true;
        break;
      }
  }

  // Whether the rows stay sets after a fill (a value added to a row that holds it, or twice).
  bool sets_after_fill(tree_run& t, const cc::op_request& req) {
    for (std::size_t i = 0; i < req.keys.size(); ++i) {
      std::set<std::int32_t> row(t.rows[req.keys[i]].begin(), t.rows[req.keys[i]].end());
      for (std::int32_t k = i == 0 ? 0 : req.prefix[i - 1]; k < req.prefix[i]; ++k)
        if (!row.insert(req.payload[static_cast<std::size_t>(k)]).second) return false;
    }
    return true;
  }

  void track_rows(tree_run& t, const std::string& op, const cc::op_request& req,
                  const std::vector<std::int32_t>& mapping) {
    if (op == "insert") {
      for (std::size_t i = 0; i < mapping.size(); ++i) {
        const std::int32_t begin = i == 0 ? 0 : req.prefix[i - 1];
        t.rows[mapping[i]].assign(req.payload.begin() + begin, req.payload.begin() + req.prefix[i]);
      }
    } else if (op == "fill") {
      for (std::size_t i = 0; i < req.keys.size(); ++i) {
        const std::int32_t begin = i == 0 ? 0 : req.prefix[i - 1];
        auto& row = t.rows[req.keys[i]];
        row.insert(row.end(), req.payload.begin() + begin, req.payload.begin() + req.prefix[i]);
      }
    } else if (op == "unfill") {
      for (std::size_t i = 0; i < req.keys.size(); ++i) {
        const std::int32_t begin = i == 0 ? 0 : req.prefix[i - 1];
        const std::set<std::int32_t> drop(req.payload.begin() + begin,
                                          req.payload.begin() + req.prefix[i]);
        auto& row = t.rows[req.keys[i]];
        row.erase(std::remove_if(row.begin(), row.end(),
                                 [&](std::int32_t v) { return drop.count(v) != 0; }),
                  row.end());
      }
    }
  }

  // The allocator against the trace's state and the model's records.
  void compare(tree_run& t, std::uint32_t id, const std::string& where) {
    const tr::tree_state* st = tracker_.tree(id);
    if (st == nullptr) return;
    ++rep_.states;
    const id_allocator& a = *t.alloc;
    std::ostringstream os;
    std::int64_t live = 0, dead = 0;
    for (const auto& [key, ks] : st->by_key) {
      ++rep_.keys_compared;
      const auto want =
          ks.avail != 0 ? id_allocator::key_state::dead : id_allocator::key_state::live;
      if (a.state_of(key) != want) os << "key " << key << " state; ";
      if (a.length(key) != ks.length)
        os << "key " << key << " length " << a.length(key) << " vs " << ks.length << "; ";
      (ks.avail != 0 ? dead : live) += 1;
      if (ks.avail != 0) continue;
      if (a.count(key) != ks.row_len)
        os << "key " << key << " count " << a.count(key) << " vs " << ks.row_len << "; ";
      if (a.occupancy(key) != ks.occupancy)
        os << "key " << key << " occupancy " << a.occupancy(key) << " vs " << ks.occupancy << "; ";
      if (a.tail_capacity(key) != ks.tail_capacity)
        os << "key " << key << " tail capacity " << a.tail_capacity(key) << " vs "
           << ks.tail_capacity << "; ";
      if (a.multi_segment(key) != (ks.segments > 1))
        os << "key " << key << " segments " << ks.segments << "; ";
    }
    if (live != a.live() || dead != a.dead()) os << "key counts; ";
    if (a.bump() != st->initial_payload_size)
      os << "bump " << a.bump() << " vs " << st->initial_payload_size << "; ";
    if (a.capacity() != st->fixed_size)
      os << "capacity " << a.capacity() << " vs " << st->fixed_size << "; ";
    // The model (an independent implementation of the same store): the same fields.
    std::int64_t model_compactions = 0;
    for (const auto& e : t.model->events())
      model_compactions += e.what == testing::cbst_model_event::kind::compaction ? 1 : 0;
    if (model_compactions != a.compactions())
      os << "compactions " << a.compactions() << " vs the model's " << model_compactions << "; ";
    for (const auto& rec : t.model->records()) {
      if (rec.avail != 0) continue;
      if (a.occupancy(rec.key) != rec.occupancy || a.count(rec.key) != rec.row_len) {
        os << "key " << rec.key << " differs from the model; ";
        break;
      }
    }
    if (a.bump() != t.model->bump()) os << "bump differs from the model; ";
    for (const auto& p : a.check()) os << "allocator: " << p << "; ";
    const std::string diff = os.str();
    if (!diff.empty()) {
      mismatch(where + ": " + diff);
      t.stopped = true;  // report the first divergence of a tree
    }
  }

 public:
  void finish() {
    for (const auto& [id, t] : trees_)
      if (t.alloc) rep_.compactions += t.alloc->compactions();
  }

 private:
  tr::reader& in_;
  options opt_;
  report& rep_;
  tr::state_tracker tracker_;
  std::map<std::uint32_t, tree_run> trees_;
  dyng::detail::escher::store_policy escher_policy_;
};

}  // namespace

std::string report::to_json() const {
  std::ostringstream os;
  os << "{\"trace\": " << json_string(trace) << ", \"repo\": " << json_string(repo)
     << ", \"ok\": " << (ok() ? "true" : "false") << ", \"trees\": " << trees
     << ", \"ops\": " << ops << ", \"states\": " << states
     << ", \"keys_compared\": " << keys_compared << ", \"inserts\": " << inserts
     << ", \"fills\": " << fills << ", \"unfills\": " << unfills
     << ", \"multi_segment_unfills\": " << multi_segment_unfills
     << ", \"overflow_fills\": " << overflow_fills << ", \"compactions\": " << compactions
     << ", \"exceptions\": " << exceptions << ", \"not_sets\": " << not_sets
     << ", \"unreplayable\": " << unreplayable << ", \"seconds\": " << seconds
     << ", \"mismatch_count\": " << mismatch_count << ", \"mismatches\": [";
  for (std::size_t i = 0; i < mismatches.size(); ++i)
    os << (i ? ", " : "") << json_string(mismatches[i]);
  os << "]}";
  return os.str();
}

report run(tr::reader& in, const options& opt, const std::string& name) {
  const auto start = std::chrono::steady_clock::now();
  report rep;
  rep.trace = name;
  runner r(in, opt, rep);
  tr::record rec;
  while (in.next(rec)) r.on(rec);
  r.finish();
  if (!rep.repo.empty() && rep.repo != "MOSP_ESCHER") {
    ++rep.mismatch_count;
    rep.mismatches.push_back("the allocator lockstep reads MOSP_ESCHER's traces, not " + rep.repo +
                             "'s");
  }
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return rep;
}

}  // namespace dyng::test::allocator_lockstep
