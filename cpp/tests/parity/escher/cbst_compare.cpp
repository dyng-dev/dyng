// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file cbst_compare.cpp
 * @brief The CBST trace replay comparator and the host model's adapter (ADR 0037).
 */
#include "parity/escher/cbst_compare.hpp"

#include <unistd.h>

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <sstream>
#include <streambuf>
#include <utility>

namespace dyng::test::cbst_compare {

namespace tr = dyng::test::cbst_trace;
using dyng::testing::cbst_host_model;
using dyng::testing::cbst_host_policy;
using dyng::testing::cbst_op_status;

const char* to_string(field_set f) {
  switch (f) {
    case field_set::semantic:
      return "semantic";
    case field_set::offsets:
      return "offsets";
    case field_set::exact:
      return "exact";
  }
  return "?";
}

bool parse_field_set(const std::string& name, field_set& out) {
  if (name == "semantic") {
    out = field_set::semantic;
  } else if (name == "offsets") {
    out = field_set::offsets;
  } else if (name == "exact") {
    out = field_set::exact;
  } else {
    return false;
  }
  return true;
}

bool order_sensitive(const std::string& dataset) {
  std::string up = dataset;
  for (char& c : up) c = static_cast<char>(std::toupper(static_cast<unsigned char>(c)));
  return up == "H2V" || up == "H2H" || up == "V2V";
}

std::string default_policy(const std::string& repo) {
  if (repo == "ESCHER-GPU") return "escher_gpu";
  if (repo == "MOSP_ESCHER") return "mosp_escher";
  return "";
}

// --- The host model's adapter --------------------------------------------------------------------

host_model_adapter::host_model_adapter(cbst_host_policy policy) : policy_(std::move(policy)) {}
host_model_adapter::~host_model_adapter() = default;

std::string host_model_adapter::name() const {
  return "host_model:" + policy_.name;
}

void host_model_adapter::create_tree(std::uint32_t tree, const std::string& dataset,
                                     std::int64_t capacity, std::int64_t alignment) {
  trees_[tree] = std::make_unique<cbst_host_model>(policy_, capacity,
                                                   static_cast<std::int32_t>(alignment), dataset);
}

void host_model_adapter::destroy_tree(std::uint32_t tree) {
  trees_.erase(tree);
}

const cbst_host_model* host_model_adapter::model(std::uint32_t tree) const {
  const auto it = trees_.find(tree);
  return it == trees_.end() ? nullptr : it->second.get();
}

op_outcome host_model_adapter::run(std::uint32_t tree, const op_request& op) {
  op_outcome out;
  const auto it = trees_.find(tree);
  if (it == trees_.end()) {
    out.status = op_status::unsupported;
    out.message = "no such tree in the model";
    return out;
  }
  cbst_host_model& m = *it->second;
  dyng::testing::cbst_op_result r;
  if (op.op == "construct") {
    r = m.construct(op.keys, op.starts, op.flat, op.has_row_counts ? &op.row_counts : nullptr);
  } else if (op.op == "rebuild") {
    r = m.rebuild(op.starts, op.flat, op.flat_size, op.capacity);
  } else if (op.op == "insert") {
    r = m.insert(op.keys, op.payload, op.prefix);
  } else if (op.op == "fill") {
    r = m.append(op.keys, op.payload, op.prefix);
  } else if (op.op == "erase") {
    r = m.erase(op.keys);
  } else if (op.op == "unfill") {
    r = m.remove_values(op.keys, op.payload, op.prefix);
  } else if (op.op == "compact") {
    r = m.compact();
  } else {
    out.status = op_status::unsupported;
    out.message = "unknown op " + op.op;
    return out;
  }
  switch (r.status) {
    case cbst_op_status::ok:
      out.status = op_status::ok;
      break;
    case cbst_op_status::error:
      out.status = op_status::error;
      break;
    case cbst_op_status::process_exit:
      out.status = op_status::process_exit;
      break;
    case cbst_op_status::undefined:
      out.status = op_status::undefined;
      break;
  }
  out.error_type = r.error_type;
  out.message = r.message;
  out.mapping = std::move(r.mapping);
  out.compact_return = r.compact_return;
  out.has_bins = m.policy().counts_bin_launches;
  out.bins = r.bin_launches;
  return out;
}

tree_digest host_model_adapter::scalars(std::uint32_t tree) {
  tree_digest d;
  const cbst_host_model* m = model(tree);
  if (m == nullptr) return d;
  d.num_records = m->num_records();
  d.fixed_size = m->payload_capacity();
  d.bump = m->bump();
  d.alignment = m->alignment();
  d.scratch_items = m->scratch_items();
  d.scratch_payload = m->scratch_payload();
  d.device_bytes = m->device_bytes();
  d.d = m->reusable_slots();
  return d;
}

tree_digest host_model_adapter::digest(std::uint32_t tree) {
  tree_digest d = scalars(tree);
  const cbst_host_model* m = model(tree);
  if (m == nullptr) return d;
  d.keys = m->keys();
  d.subtree_avail = m->subtree_avail();
  std::vector<dyng::testing::cbst_key_record> recs = m->records();
  d.records.resize(recs.size());
  for (std::size_t i = 0; i < recs.size(); ++i) {
    auto& r = recs[i];
    key_digest& k = d.records[i];
    k.key = r.key;
    k.avail = r.avail;
    k.length = r.length;
    k.size = r.size;
    k.occupancy = r.occupancy;
    k.tail_capacity = r.tail_capacity;
    k.value = r.value;
    k.tail_base = r.tail_base;
    k.segments = r.segments;
    k.row_len = r.row_len;
    k.terminated = r.terminated ? 1 : 0;
    k.row_hash = tr::fnv1a64(r.row.data(), r.row.size() * sizeof(std::int32_t));
    k.row = std::move(r.row);
  }
  return d;
}

std::vector<std::pair<std::int64_t, char>> host_model_adapter::events(std::uint32_t tree) {
  std::vector<std::pair<std::int64_t, char>> out;
  const cbst_host_model* m = model(tree);
  if (m == nullptr) return out;
  for (const auto& e : m->events())
    out.emplace_back(e.op_index,
                     e.what == dyng::testing::cbst_model_event::kind::growth ? 'g' : 'c');
  return out;
}

// --- The replay -----------------------------------------------------------------------------------

namespace {

std::string json_escape(const std::string& s) {
  std::string out;
  out.reserve(s.size() + 2);
  for (char c : s) {
    switch (c) {
      case '"':
        out += "\\\"";
        break;
      case '\\':
        out += "\\\\";
        break;
      case '\n':
        out += "\\n";
        break;
      case '\t':
        out += "\\t";
        break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          char buf[8];
          std::snprintf(buf, sizeof buf, "\\u%04x", static_cast<unsigned char>(c));
          out += buf;
        } else {
          out += c;
        }
    }
  }
  return out;
}

std::string brief(const std::vector<std::int32_t>& v) {
  std::ostringstream s;
  s << "[";
  const std::size_t shown = std::min<std::size_t>(v.size(), 16);
  for (std::size_t i = 0; i < shown; ++i) s << (i ? "," : "") << v[i];
  if (v.size() > shown) s << ",... (" << v.size() << " values)";
  s << "]";
  return s.str();
}

/// The first index where two vectors differ (or the shorter size).
std::size_t first_difference(const std::vector<std::int32_t>& a,
                             const std::vector<std::int32_t>& b) {
  const std::size_t n = std::min(a.size(), b.size());
  for (std::size_t i = 0; i < n; ++i)
    if (a[i] != b[i]) return i;
  return n;
}

/// Per key: (value, tail_base, tail_capacity, segments), the fields a compaction rewrites in rows
/// an operation does not name.
struct layout_entry {
  std::int32_t key, value, tail_base, tail_capacity, segments;
};
struct layout_snapshot {
  std::int64_t bump = 0, fixed_size = 0;
  std::vector<layout_entry> entries;  // sorted by key
  bool known = false;
};

layout_snapshot snapshot_of(const tr::tree_state& t) {
  layout_snapshot s;
  s.known = true;
  s.bump = t.initial_payload_size;
  s.fixed_size = t.fixed_size;
  s.entries.reserve(t.by_key.size());
  for (const auto& [key, k] : t.by_key)
    s.entries.push_back({key, k.value, k.tail_base, k.tail_capacity, k.segments});
  return s;
}

layout_snapshot snapshot_of(const tree_digest& d) {
  layout_snapshot s;
  s.known = true;
  s.bump = d.bump;
  s.fixed_size = d.fixed_size;
  s.entries.reserve(d.records.size());
  for (const auto& k : d.records)
    s.entries.push_back({k.key, k.value, k.tail_base, k.tail_capacity, k.segments});
  std::sort(s.entries.begin(), s.entries.end(),
            [](const layout_entry& a, const layout_entry& b) { return a.key < b.key; });
  return s;
}

/// Whether the change from `before` to `after` shows a compaction: a row the operation does not
/// name changed its offset, tail or segments, or the bump pointer fell (escher_trace.py's rule).
bool shows_compaction(const layout_snapshot& before, const layout_snapshot& after,
                      const std::vector<std::int32_t>& named) {
  if (!before.known || !after.known) return false;
  if (after.bump < before.bump) return true;
  const std::set<std::int32_t> named_set(named.begin(), named.end());
  auto it = before.entries.begin();
  for (const layout_entry& a : after.entries) {
    while (it != before.entries.end() && it->key < a.key) ++it;
    if (it == before.entries.end()) break;
    if (it->key != a.key || named_set.count(a.key)) continue;
    if (it->value != a.value || it->tail_base != a.tail_base ||
        it->tail_capacity != a.tail_capacity || it->segments != a.segments)
      return true;
  }
  return false;
}

struct tree_run {
  std::string dataset;
  bool created = false;
  bool diverged = false;
  bool unreplayable = false;
  std::int64_t ops = 0;
  layout_snapshot trace_before, store_before;
};

class replayer {
 public:
  replayer(tr::reader& in, store_adapter* store, const store_factory* make_store,
           const options& opt, const std::string& name)
      : in_(in), store_(store), make_store_(make_store), opt_(opt) {
    report_.trace = name;
    if (store_) report_.store = store_->name();
    report_.fields = to_string(opt.fields);
  }

  replay_report run() {
    tr::record rec;
    while (in_.next(rec)) {
      if (store_ == nullptr && rec.kind != tr::record_kind::header)
        throw tr::trace_error("CBST trace: a record before the header");
      switch (rec.kind) {
        case tr::record_kind::header:
          report_.repo = rec.text("repo");
          if (store_ == nullptr && make_store_ != nullptr) {
            owned_ = (*make_store_)(report_.repo);
            store_ = owned_.get();
            report_.store = store_->name();
          }
          break;
        case tr::record_kind::tree:
          on_tree(rec);
          break;
        case tr::record_kind::event:
          on_event(rec);
          break;
        case tr::record_kind::op:
          on_op(rec);
          break;
        case tr::record_kind::end:
          report_.exit = rec.text("exit");
          break;
        default:
          break;
      }
    }
    if (report_.exit != "normal") {
      mismatch m;
      m.field = "trace.exit";
      m.expected = "normal";
      m.actual = report_.exit.empty() ? "(no end record)" : report_.exit;
      m.op = "(end)";
      add(m, nullptr);
    }
    for (auto& [id, t] : report_.trees) {
      report_.growths += t.growths;
      report_.compactions += t.compactions;
      report_.exceptions += t.exceptions;
      report_.hash_only_rows += t.hash_only_rows;
    }
    return std::move(report_);
  }

 private:
  void add(mismatch m, tree_summary* t) {
    ++report_.mismatch_count;
    if (t) ++t->mismatches;
    if (report_.mismatches.size() < opt_.max_mismatches) report_.mismatches.push_back(std::move(m));
  }

  void on_tree(const tr::record& rec) {
    const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
    tracker_.apply(rec, in_);
    tree_run& t = runs_[id];
    t.dataset = rec.text("dataset_name");
    tree_summary& s = report_.trees[id];
    s.id = id;
    s.dataset = t.dataset;
    s.order_sensitive = order_sensitive(t.dataset);
    store_->create_tree(id, t.dataset, rec.integer("payload_capacity"),
                        rec.integer("alignment", 4));
    t.created = true;
  }

  void on_event(const tr::record& rec) {
    if (rec.text("event") == "destroy_object") {
      const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
      store_->destroy_tree(id);
      runs_[id].created = false;
    }
  }

  // The values of an input array (false: hash only).
  bool input(const tr::record& rec, const std::string& name, std::vector<std::int32_t>& out,
             std::string& missing) {
    const tr::field* f = rec.find(name);
    if (f == nullptr || f->kind != tr::field::type::array) {
      out.clear();
      return true;
    }
    if (f->array.kind == tr::array_field::storage::hash_only) {
      missing += (missing.empty() ? "" : ", ") + name;
      return false;
    }
    out = in_.ints(f->array);
    return true;
  }

  void stop_tree(tree_run& t, tree_summary& s, const std::string& why) {
    t.unreplayable = true;
    s.unreplayable_reason = why;
  }

  void on_op(const tr::record& rec) {
    const auto id = static_cast<std::uint32_t>(rec.integer("tree"));
    const std::int64_t op_index = report_.ops++;
    tree_run& t = runs_[id];
    tree_summary& s = report_.trees[id];
    s.id = id;
    const std::int64_t tree_op = t.ops++;
    ++s.ops;
    const std::string op = rec.text("op");
    // The trace side always advances (its digests are deltas).
    const tr::tree_state* before_state = tracker_.tree(id);
    layout_snapshot trace_before =
        before_state && rec.has("digest.mode") ? snapshot_of(*before_state) : layout_snapshot{};
    if (t.diverged || t.unreplayable) {
      apply_trace(rec, s, op_index, tree_op, op);
      if (t.unreplayable) {
        ++s.unreplayable;
        ++report_.unreplayable;
      }
      return;
    }
    mismatch base;
    base.seq = rec.integer("seq");
    base.op_index = op_index;
    base.tree_op = tree_op;
    base.tree = id;
    base.dataset = t.dataset;
    base.op = op;
    auto differ = [&](const std::string& field, std::int64_t key, const std::string& expected,
                      const std::string& actual) {
      mismatch m = base;
      m.field = field;
      m.key = key;
      m.expected = expected;
      m.actual = actual;
      add(std::move(m), &s);
      t.diverged = true;
    };
    if (!rec.has("digest.mode")) {
      apply_trace(rec, s, op_index, tree_op, op);
      stop_tree(t, s, "the trace has no state digests (batches mode)");
      ++s.unreplayable;
      ++report_.unreplayable;
      return;
    }
    if (rec.integer("pre.matches", 2) == 0) {
      apply_trace(rec, s, op_index, tree_op, op);
      differ("pre.matches", -1, "the state recorded after the tree's last op",
             "the original's state changed between recorded ops");
      return;
    }
    // Decode the inputs.
    op_request req;
    req.op = op;
    std::string missing;
    bool have = true;
    if (op == "construct" || op == "rebuild") {
      have &= input(rec, "in.keys", req.keys, missing);
      have &= input(rec, "in.start_offsets", req.starts, missing);
      have &= input(rec, "in.flat_payload", req.flat, missing);
      req.num_records = rec.integer("in.num_records");
      req.flat_size = rec.integer("in.flat_payload_size");
      req.capacity = rec.integer("in.capacity");
      req.alignment = rec.integer("in.alignment", 4);
      if (rec.has("in.row_occupancy_null") && rec.integer("in.row_occupancy_null") == 0) {
        req.has_row_counts = true;
        have &= input(rec, "in.row_occupancy", req.row_counts, missing);
      }
      if (have && op == "construct" &&
          (static_cast<std::int64_t>(req.keys.size()) != req.num_records ||
           static_cast<std::int64_t>(req.flat.size()) != req.flat_size)) {
        apply_trace(rec, s, op_index, tree_op, op);
        stop_tree(t, s, "the construct's arrays do not have the recorded sizes");
        ++s.unreplayable;
        ++report_.unreplayable;
        return;
      }
    } else if (op != "compact") {
      have &= input(rec, "in.keys", req.keys, missing);
      have &= input(rec, "in.payload", req.payload, missing);
      have &= input(rec, "in.prefix", req.prefix, missing);
    }
    if (!have) {
      apply_trace(rec, s, op_index, tree_op, op);
      stop_tree(t, s,
                "op " + std::to_string(tree_op) + " (" + op + ") has hash-only inputs: " + missing);
      ++s.unreplayable;
      ++report_.unreplayable;
      return;
    }
    const bool exact = opt_.fields == field_set::exact;
    const bool offsets = opt_.fields != field_set::semantic;
    // The state before the operation (exact): the store must start where the original did.
    const tree_digest pre = store_->scalars(id);
    if (exact) {
      const std::pair<const char*, std::int64_t> before[] = {
          {"before.fixed_size", pre.fixed_size},
          {"before.num_records", pre.num_records},
          {"before.initial_payload_size", pre.bump},
          {"before.alignment", pre.alignment},
          {"before.scratch_items", pre.scratch_items},
          {"before.scratch_payload", pre.scratch_payload},
          {"before.d", pre.d}};
      for (const auto& [name, have_value] : before) {
        if (!rec.has(name)) continue;
        if (rec.integer(name) != have_value)
          differ(name, -1, std::to_string(rec.integer(name)), std::to_string(have_value));
      }
      if (rec.integer("before.device_bytes", -1) >= 0 &&
          rec.integer("before.device_bytes") != pre.device_bytes)
        differ("before.device_bytes", -1, std::to_string(rec.integer("before.device_bytes")),
               std::to_string(pre.device_bytes));
    }
    const op_outcome out = store_->run(id, req);
    ++s.compared;
    ++report_.compared;
    // The outcome.
    const bool threw = rec.integer("status") != 0;
    if (threw) ++s.exceptions;
    if (out.status == op_status::undefined || out.status == op_status::unsupported) {
      differ("status", -1, threw ? "exception " + rec.text("exception.type") : "ok",
             std::string(out.status == op_status::undefined ? "undefined: " : "unsupported: ") +
                 out.message);
    } else if (out.status == op_status::process_exit) {
      differ("status", -1, threw ? "exception " + rec.text("exception.type") : "ok",
             "the process exits: " + out.message);
    } else if (threw != (out.status == op_status::error)) {
      differ("status", -1, threw ? "exception " + rec.text("exception.type") : "ok",
             out.status == op_status::error ? "exception " + out.error_type + ": " + out.message
                                            : "ok");
    } else if (threw && rec.text("exception.type") != out.error_type) {
      differ("exception.type", -1, rec.text("exception.type"), out.error_type);
    }
    // The InsertMapping.
    std::vector<std::int32_t> trace_mapping;
    if (rec.has("out.mapping")) {
      trace_mapping = in_.ints(rec.array("out.mapping"));
      ++report_.mappings;
      if (trace_mapping != out.mapping) {
        const std::size_t i = first_difference(trace_mapping, out.mapping);
        differ("out.mapping", -1,
               "item " + std::to_string(i) + ": " +
                   (i < trace_mapping.size() ? std::to_string(trace_mapping[i]) : "-") + " of " +
                   brief(trace_mapping),
               (i < out.mapping.size() ? std::to_string(out.mapping[i]) : "-") + " of " +
                   brief(out.mapping));
      }
    }
    if (rec.has("out.compact_return") && !threw &&
        rec.integer("out.compact_return") != out.compact_return)
      differ("out.compact_return", -1, std::to_string(rec.integer("out.compact_return")),
             std::to_string(out.compact_return));
    if (exact && out.has_bins && rec.has("bins.fill")) {
      static const char* const names[3] = {"bins.fill", "bins.reuse", "bins.unfill"};
      for (int b = 0; b < 3; ++b) {
        const std::vector<std::int64_t> want = in_.longs(rec.array(names[b]));
        for (int l = 0; l < 3 && l < static_cast<int>(want.size()); ++l)
          if (want[static_cast<std::size_t>(l)] != out.bins[static_cast<std::size_t>(3 * b + l)])
            differ(std::string(names[b]) + "[" + std::to_string(l) + "]", -1,
                   std::to_string(want[static_cast<std::size_t>(l)]),
                   std::to_string(out.bins[static_cast<std::size_t>(3 * b + l)]));
      }
    }
    // The trace's state after the operation, and the store's.
    apply_trace(rec, s, op_index, tree_op, op);
    const tr::tree_state* after = tracker_.tree(id);
    const tree_digest got = store_->digest(id);
    if (after != nullptr) compare_state(rec, *after, got, s, differ, offsets, exact);
    // Growth and compaction, inferred by the same rule on both sides.
    if (after != nullptr && op != "construct" && op != "rebuild") {
      const layout_snapshot trace_after = snapshot_of(*after);
      const layout_snapshot store_after = snapshot_of(got);
      layout_snapshot store_before = t.store_before;
      if (!store_before.known) {
        store_before.known = true;
        store_before.fixed_size = pre.fixed_size;
        store_before.bump = pre.bump;
      }
      const bool trace_grew = rec.integer("after.fixed_size") > rec.integer("before.fixed_size");
      const bool store_grew = got.fixed_size > pre.fixed_size;
      if (trace_grew) ++s.growths;
      if (trace_grew != store_grew)
        differ("event.growth", -1, trace_grew ? "the payload grew" : "no growth",
               store_grew ? "the payload grew" : "no growth");
      std::vector<std::int32_t> named_trace = req.keys, named_store = req.keys;
      if (op == "insert") {
        named_trace = trace_mapping;
        named_store = out.mapping;
      }
      const bool explicit_compact = op == "compact" && !threw;
      const bool trace_compacted =
          explicit_compact || shows_compaction(trace_before, trace_after, named_trace);
      const bool store_compacted =
          explicit_compact || shows_compaction(store_before, store_after, named_store);
      if (trace_compacted) ++s.compactions;
      if (trace_compacted != store_compacted)
        differ("event.compaction", -1, trace_compacted ? "a compaction" : "no compaction",
               store_compacted ? "a compaction" : "no compaction");
      t.store_before = store_after;
    } else {
      t.store_before = snapshot_of(got);
    }
  }

  void apply_trace(const tr::record& rec, tree_summary& s, std::int64_t op_index,
                   std::int64_t tree_op, const std::string& op) {
    for (const std::string& p : tracker_.apply(rec, in_)) {
      mismatch m;
      m.seq = rec.integer("seq");
      m.op_index = op_index;
      m.tree_op = tree_op;
      m.tree = s.id;
      m.dataset = s.dataset;
      m.op = op;
      m.field = "trace";
      m.expected = "the recorded hashes";
      m.actual = p;
      add(std::move(m), &s);
    }
  }

  template <class differ_t>
  void compare_state(const tr::record& rec, const tr::tree_state& want, const tree_digest& got,
                     tree_summary& s, differ_t& differ, bool offsets, bool exact) {
    auto scalar = [&](const char* field, std::int64_t expected, std::int64_t actual) {
      if (expected != actual) differ(field, -1, std::to_string(expected), std::to_string(actual));
    };
    scalar("after.fixed_size", want.fixed_size, got.fixed_size);
    scalar("after.initial_payload_size", want.initial_payload_size, got.bump);
    if (exact) {
      scalar("after.num_records", want.num_records, got.num_records);
      scalar("after.alignment", want.alignment, got.alignment);
      if (rec.has("after.scratch_items"))
        scalar("after.scratch_items", rec.integer("after.scratch_items"), got.scratch_items);
      if (rec.has("after.scratch_payload"))
        scalar("after.scratch_payload", rec.integer("after.scratch_payload"), got.scratch_payload);
      if (rec.has("after.d")) scalar("after.d", rec.integer("after.d"), got.d);
      if (rec.integer("after.device_bytes", -1) >= 0)
        scalar("after.device_bytes", rec.integer("after.device_bytes"), got.device_bytes);
    }
    if (want.keys != got.keys) {
      const std::size_t i = first_difference(want.keys, got.keys);
      differ("state.keys", i < want.keys.size() ? want.keys[i] : -1,
             "rank " + std::to_string(i) + " of " + brief(want.keys), brief(got.keys));
      return;
    }
    if (want.subtree_avail != got.subtree_avail) {
      const std::size_t i = first_difference(want.subtree_avail, got.subtree_avail);
      differ("state.subtree_avail", -1,
             "heap position " + std::to_string(i) + " of " + brief(want.subtree_avail),
             brief(got.subtree_avail));
    }
    if (got.records.size() != want.keys.size()) {
      differ("state.records", -1, std::to_string(want.keys.size()) + " records",
             std::to_string(got.records.size()) + " records");
      return;
    }
    std::vector<std::int32_t> a, b;
    for (std::size_t i = 0; i < want.keys.size(); ++i) {
      const auto it = want.by_key.find(want.keys[i]);
      if (it == want.by_key.end()) {
        differ("trace", want.keys[i], "a record of every key", "the trace has none");
        return;
      }
      const tr::key_state& w = it->second;
      const key_digest& g = got.records[i];
      ++report_.keys_compared;
      auto field = [&](const char* name, std::int64_t expected, std::int64_t actual) {
        if (expected != actual)
          differ(std::string("key.") + name, w.key, std::to_string(expected),
                 std::to_string(actual));
      };
      field("avail", w.avail, g.avail);
      field("length", w.length, g.length);
      if (offsets) {
        field("value", w.value, g.value);
        field("tail_base", w.tail_base, g.tail_base);
      }
      if (exact) {
        field("size", w.size, g.size);
        field("occupancy", w.occupancy, g.occupancy);
        field("tail_capacity", w.tail_capacity, g.tail_capacity);
        field("segments", w.segments, g.segments);
        field("row_len", w.row_len, g.row_len);
        field("terminated", w.terminated, g.terminated);
      }
      // The row: by its hash when the trace has only the hash; in order (H2V, H2H, V2V, and
      // always in exact mode) or as a multiset.
      if (!w.row_known) {
        ++s.hash_only_rows;
        if (w.row_hash != g.row_hash)
          differ("key.row (hash)", w.key, std::to_string(w.row_hash), std::to_string(g.row_hash));
        continue;
      }
      if (w.row == g.row) continue;
      if (exact || s.order_sensitive) {
        const std::size_t j = first_difference(w.row, g.row);
        differ("key.row", w.key, "position " + std::to_string(j) + " of " + brief(w.row),
               brief(g.row));
        continue;
      }
      a = w.row;
      b = g.row;
      std::sort(a.begin(), a.end());
      std::sort(b.begin(), b.end());
      if (a != b) differ("key.row (multiset)", w.key, brief(w.row), brief(g.row));
    }
  }

  tr::reader& in_;
  store_adapter* store_;
  const store_factory* make_store_;
  std::unique_ptr<store_adapter> owned_;
  options opt_;
  replay_report report_;
  tr::state_tracker tracker_;
  std::map<std::uint32_t, tree_run> runs_;
};

}  // namespace

replay_report replay(tr::reader& in, store_adapter& store, const options& opt,
                     const std::string& name) {
  replayer r(in, &store, nullptr, opt, name);
  return r.run();
}

replay_report replay(tr::reader& in, const store_factory& make_store, const options& opt,
                     const std::string& name) {
  replayer r(in, nullptr, &make_store, opt, name);
  return r.run();
}

store_factory host_model_factory(const std::string& policy) {
  return [policy](const std::string& repo) -> std::unique_ptr<store_adapter> {
    const std::string name = policy.empty() ? default_policy(repo) : policy;
    cbst_host_policy p;
    if (!cbst_host_policy::from_name(name, p))
      throw tr::trace_error("no host-model policy '" + name + "' (original '" + repo + "')");
    return std::make_unique<host_model_adapter>(std::move(p));
  };
}

// --- Trace sources --------------------------------------------------------------------------------

namespace {

std::string shell_quote(const std::string& s) {
  std::string out = "'";
  for (char c : s) {
    if (c == '\'')
      out += "'\\''";
    else
      out += c;
  }
  return out + "'";
}

bool ends_with(const std::string& s, const std::string& suffix) {
  return s.size() >= suffix.size() &&
         s.compare(s.size() - suffix.size(), suffix.size(), suffix) == 0;
}

}  // namespace

/// An input stream buffer over a pipe (popen).
struct trace_source::pipe : std::streambuf {
  explicit pipe(FILE* f) : file(f) {}
  ~pipe() override {
    if (file) pclose(file);
  }
  pipe(const pipe&) = delete;
  pipe& operator=(const pipe&) = delete;
  int_type underflow() override {
    if (gptr() < egptr()) return traits_type::to_int_type(*gptr());
    if (file == nullptr) return traits_type::eof();
    const std::size_t n = std::fread(buf, 1, sizeof buf, file);
    if (n == 0) return traits_type::eof();
    setg(buf, buf, buf + n);
    return traits_type::to_int_type(*gptr());
  }
  FILE* file;
  char buf[1 << 16];
};

trace_source::trace_source(const std::string& path, const std::string& tmp_dir) {
  namespace fs = std::filesystem;
  if (!ends_with(path, ".zst")) {
    reader_ = tr::reader::open(path);
    return;
  }
  if (!fs::is_regular_file(path)) throw tr::trace_error("CBST trace: cannot open " + path);
  FILE* f = popen(("zstd -q -dc --long=31 " + shell_quote(path)).c_str(), "r");
  if (f == nullptr) throw tr::trace_error("CBST trace: cannot run zstd for " + path);
  pipe_ = std::make_unique<pipe>(f);
  stream_ = std::make_unique<std::istream>(pipe_.get());
  reader_ = std::make_unique<tr::reader>(*stream_);
  const fs::path blob = fs::path(path).parent_path() / "trace.blob.zst";
  if (fs::exists(blob)) {
    static int counter = 0;
    blob_tmp_ = (fs::path(tmp_dir) / ("cbst_replay_" + std::to_string(getpid()) + "_" +
                                      std::to_string(++counter) + ".blob"))
                    .string();
    const std::string cmd =
        "zstd -q -d --long=31 -f " + shell_quote(blob.string()) + " -o " + shell_quote(blob_tmp_);
    if (std::system(cmd.c_str()) != 0)
      throw tr::trace_error("CBST trace: cannot decompress " + blob.string());
    reader_->set_side_file(blob_tmp_);
  }
}

trace_source::~trace_source() {
  reader_.reset();
  stream_.reset();
  pipe_.reset();
  if (!blob_tmp_.empty()) {
    std::error_code ec;
    std::filesystem::remove(blob_tmp_, ec);
  }
}

std::string replay_report::to_json() const {
  std::ostringstream o;
  o << "{\"trace\": \"" << json_escape(trace) << "\", \"repo\": \"" << json_escape(repo)
    << "\", \"store\": \"" << json_escape(store) << "\", \"fields\": \"" << json_escape(fields)
    << "\", \"exit\": \"" << json_escape(exit) << "\", \"ok\": " << (ok() ? "true" : "false")
    << ", \"ops\": " << ops << ", \"compared\": " << compared
    << ", \"unreplayable\": " << unreplayable << ", \"keys_compared\": " << keys_compared
    << ", \"hash_only_rows\": " << hash_only_rows << ", \"mappings\": " << mappings
    << ", \"exceptions\": " << exceptions << ", \"growths\": " << growths
    << ", \"compactions\": " << compactions << ", \"mismatch_count\": " << mismatch_count
    << ", \"mismatches\": [";
  for (std::size_t i = 0; i < mismatches.size(); ++i) {
    const mismatch& m = mismatches[i];
    o << (i ? ", " : "") << "{\"seq\": " << m.seq << ", \"op_index\": " << m.op_index
      << ", \"tree_op\": " << m.tree_op << ", \"tree\": " << m.tree << ", \"dataset\": \""
      << json_escape(m.dataset) << "\", \"op\": \"" << json_escape(m.op) << "\", \"field\": \""
      << json_escape(m.field) << "\", \"key\": " << m.key << ", \"expected\": \""
      << json_escape(m.expected) << "\", \"actual\": \"" << json_escape(m.actual) << "\"}";
  }
  o << "], \"trees\": [";
  bool first = true;
  for (const auto& [id, t] : trees) {
    o << (first ? "" : ", ") << "{\"tree\": " << id << ", \"dataset\": \"" << json_escape(t.dataset)
      << "\", \"ops\": " << t.ops << ", \"compared\": " << t.compared
      << ", \"mismatches\": " << t.mismatches << ", \"unreplayable\": " << t.unreplayable
      << ", \"growths\": " << t.growths << ", \"compactions\": " << t.compactions
      << ", \"exceptions\": " << t.exceptions << ", \"hash_only_rows\": " << t.hash_only_rows
      << ", \"order_sensitive\": " << (t.order_sensitive ? "true" : "false")
      << ", \"unreplayable_reason\": \"" << json_escape(t.unreplayable_reason) << "\"}";
    first = false;
  }
  o << "], \"notes\": [";
  for (std::size_t i = 0; i < notes.size(); ++i)
    o << (i ? ", " : "") << "\"" << json_escape(notes[i]) << "\"";
  o << "]}";
  return o.str();
}

}  // namespace dyng::test::cbst_compare
