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
    case field_set::policy:
      return "policy";
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
  } else if (name == "policy") {
    out = field_set::policy;
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

bool equivalent_exception(const std::string& op, const std::string& trace_type,
                          const std::string& store_type) {
  if (trace_type == store_type) return true;
  // MOSP_ESCHER@4b86159 throws escher::EscherError at every site; the dynG store throws the
  // exception of PLAN 4.7.3 for the site (M8a step 5, ADR 0039): the construct validation
  // (escher/structure/operations.cu:196-211) invalid_argument_error (or capacity_error for a
  // record count beyond the int32 scratch), the payload overflow of a fill (:436-444), of the
  // surplus rows of an insert (:713-721) and of a compaction (compactCBST) capacity_error.
  if (trace_type != "escher::EscherError") return false;
  if (op == "construct")
    return store_type == "dyng::invalid_argument_error" || store_type == "dyng::capacity_error";
  if (op == "fill" || op == "insert" || op == "compact")
    return store_type == "dyng::capacity_error";
  return false;
}

std::string digest_difference(const tree_digest& a, const tree_digest& b) {
  std::ostringstream o;
  auto scalar = [&](const char* name, std::int64_t x, std::int64_t y) {
    if (o.tellp() == 0 && x != y && x >= 0 && y >= 0)
      o << name << ": store " << x << ", model " << y;
  };
  scalar("num_records", a.num_records, b.num_records);
  scalar("fixed_size", a.fixed_size, b.fixed_size);
  scalar("bump", a.bump, b.bump);
  scalar("alignment", a.alignment, b.alignment);
  scalar("scratch_items", a.scratch_items, b.scratch_items);
  scalar("scratch_payload", a.scratch_payload, b.scratch_payload);
  scalar("device_bytes", a.device_bytes, b.device_bytes);
  if (o.tellp() != 0) return o.str();
  if (a.keys != b.keys) return "keys";
  if (a.subtree_avail != b.subtree_avail) return "subtree_avail";
  if (a.records.size() != b.records.size()) return "record count";
  for (std::size_t i = 0; i < a.records.size(); ++i) {
    const key_digest &x = a.records[i], &y = b.records[i];
    const std::int64_t fx[] = {x.avail, x.length,    x.size,     x.occupancy, x.tail_capacity,
                               x.value, x.tail_base, x.segments, x.row_len,   x.terminated};
    const std::int64_t fy[] = {y.avail, y.length,    y.size,     y.occupancy, y.tail_capacity,
                               y.value, y.tail_base, y.segments, y.row_len,   y.terminated};
    static const char* const names[] = {"avail",         "length",    "size",      "occupancy",
                                        "tail_capacity", "value",     "tail_base", "segments",
                                        "row_len",       "terminated"};
    for (std::size_t f = 0; f < 10; ++f)
      if (fx[f] != fy[f])
        return "key " + std::to_string(x.key) + " " + names[f] + ": store " +
               std::to_string(fx[f]) + ", model " + std::to_string(fy[f]);
    if (x.row != y.row) return "key " + std::to_string(x.key) + " row";
  }
  return "";
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
  /// The keys of a MOSP_ESCHER construct without row counts whose trace occupancy is still that
  /// construct's 0, and the store's (scanned) occupancy of each (construct_without_row_counts).
  std::map<std::int32_t, std::int64_t> legacy_occupancy;
  /// field_set::policy: the tree's first operation at which the payload policies differ (-1:
  /// none yet), and whether a per-key length has differed since.
  std::int64_t policy_point = -1;
  bool length_differs = false;
};

// The values of an input array (false: hash only).
bool input(const tr::record& rec, tr::reader& in, const std::string& name,
           std::vector<std::int32_t>& out, std::string& missing) {
  const tr::field* f = rec.find(name);
  if (f == nullptr || f->kind != tr::field::type::array) {
    out.clear();
    return true;
  }
  if (f->array.kind == tr::array_field::storage::hash_only) {
    missing += (missing.empty() ? "" : ", ") + name;
    return false;
  }
  out = in.ints(f->array);
  return true;
}

}  // namespace

bool decode_op(const tr::record& rec, tr::reader& in, op_request& req, std::string& missing) {
  req = op_request{};
  req.op = rec.text("op");
  const std::string& op = req.op;
  bool have = true;
  if (op == "construct" || op == "rebuild") {
    have &= input(rec, in, "in.keys", req.keys, missing);
    have &= input(rec, in, "in.start_offsets", req.starts, missing);
    have &= input(rec, in, "in.flat_payload", req.flat, missing);
    req.num_records = rec.integer("in.num_records");
    req.flat_size = rec.integer("in.flat_payload_size");
    req.capacity = rec.integer("in.capacity");
    req.alignment = rec.integer("in.alignment", 4);
    if (rec.has("in.row_occupancy_null") && rec.integer("in.row_occupancy_null") == 0) {
      req.has_row_counts = true;
      have &= input(rec, in, "in.row_occupancy", req.row_counts, missing);
    }
  } else if (op != "compact") {
    have &= input(rec, in, "in.keys", req.keys, missing);
    have &= input(rec, in, "in.payload", req.payload, missing);
    have &= input(rec, in, "in.prefix", req.prefix, missing);
  }
  return have;
}

namespace {

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
  // Reports the pending mismatches of an op when it has been compared.
  struct pending_flush {
    replayer* r;
    tree_run* t;
    tree_summary* s;
    pending_flush(replayer* replay, tree_run* run, tree_summary* summary)
        : r(replay), t(run), s(summary) {}
    pending_flush(const pending_flush&) = delete;
    pending_flush& operator=(const pending_flush&) = delete;
    ~pending_flush() {
      if (r->pending_.empty()) return;
      for (mismatch& m : r->pending_) r->add(std::move(m), s);
      r->pending_.clear();
      t->diverged = true;
    }
  };

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
      pending_.push_back(std::move(m));
    };
    // The mismatches of the op are reported when it has been compared (a documented divergence
    // can still replace them, below); any mismatch ends the comparison of the tree.
    pending_flush flush{this, &t, &s};
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
    std::string missing;
    const bool have = decode_op(rec, in_, req, missing);
    if (have && op == "construct" &&
        (static_cast<std::int64_t>(req.keys.size()) != req.num_records ||
         static_cast<std::int64_t>(req.flat.size()) != req.flat_size)) {
      apply_trace(rec, s, op_index, tree_op, op);
      stop_tree(t, s, "the construct's arrays do not have the recorded sizes");
      ++s.unreplayable;
      ++report_.unreplayable;
      return;
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
    const bool offsets = opt_.fields == field_set::offsets || opt_.fields == field_set::exact;
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
    // The documented divergences of the port (ADR 0038). construct_below_payload: a construct
    // whose capacity is below its payload; ESCHER-GPU printed "Overflow" and kept a tree without
    // nodes (and MOSP_ESCHER threw EscherError), dynG throws invalid_argument_error before it
    // changes or allocates anything.
    if (op == "construct" && req.num_records > 0 && req.capacity < req.flat_size &&
        out.status == op_status::error && out.error_type == "dyng::invalid_argument_error") {
      documented_divergence d;
      d.seq = base.seq;
      d.op_index = op_index;
      d.tree_op = tree_op;
      d.tree = id;
      d.dataset = t.dataset;
      d.op = op;
      d.rule = "construct_below_payload";
      d.detail = std::string("trace: ") +
                 (threw ? "exception " + rec.text("exception.type") : "no exception") +
                 "; store: " + out.error_type + ": " + out.message;
      report_.documented.push_back(std::move(d));
      apply_trace(rec, s, op_index, tree_op, op);
      t.diverged = true;
      return;
    }
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
    } else if (threw && !equivalent_exception(op, rec.text("exception.type"), out.error_type)) {
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
    // The documented divergence construct_without_row_counts (M8a step 5, ADR 0039; acceptance
    // criterion 11 of M8a: MOSP_ESCHER's nullptr path is never ported). A construct of
    // MOSP_ESCHER without row counts (rowOccupancy = nullptr) gave every row the tail occupancy
    // 0, where the dynG store scans the rows (ADR 0039 (c)). Only that difference is exempted:
    // the occupancy of the keys built that way, while the trace still has 0 and the store still
    // has the scanned count, and MOSP_ESCHER's own integrity verdict, which counts one metadata
    // violation per live such key (escher/structure/integrity.cu@4b86159:160-186), when it is
    // exactly that count and the store's verdict is clean. Every other field of the op, and every
    // later op of the tree, is still compared.
    if (op == "construct") {
      t.legacy_occupancy.clear();
      if (report_.repo == "MOSP_ESCHER" && !req.has_row_counts && out.status == op_status::ok) {
        for (const mismatch& m : pending_)
          if (m.field == "key.occupancy" && m.expected == "0")
            t.legacy_occupancy[static_cast<std::int32_t>(m.key)] = std::stoll(m.actual);
      }
    }
    if (!t.legacy_occupancy.empty() && after != nullptr)
      exempt_legacy_occupancy(rec, *after, t, base, differ);
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
    if (opt_.fields == field_set::policy && after != nullptr)
      judge_policy_differences(rec, *after, t, base, threw);
  }

  // field_set::policy (above): the differences of an op that the payload policies explain are
  // documented instead of reported; everything else stays a mismatch.
  void judge_policy_differences(const tr::record& rec, const tr::tree_state& after, tree_run& t,
                                const mismatch& base, bool threw) {
    if (pending_.empty()) return;
    auto document = [&](const std::string& rule, const std::string& detail) {
      documented_divergence d;
      d.seq = base.seq;
      d.op_index = base.op_index;
      d.tree_op = base.tree_op;
      d.tree = base.tree;
      d.dataset = base.dataset;
      d.op = base.op;
      d.rule = rule;
      d.detail = detail;
      report_.documented.push_back(std::move(d));
    };
    const std::string& op = base.op;
    // The original's fixed capacity threw where the store's policy grew the payload: the
    // original's state after its exception is not the store's; the tree ends here.
    if (threw && (op == "fill" || op == "insert" || op == "compact") &&
        rec.text("exception.type") == "escher::EscherError") {
      for (const mismatch& m : pending_) {
        if (m.field == "status" && m.actual == "ok") {
          document("policy_exception", "the original threw " + rec.text("exception.type") +
                                           " (its fixed payload capacity); the store's policy "
                                           "grew the payload (capacity " +
                                           std::to_string(rec.integer("after.fixed_size")) +
                                           " in the trace)");
          pending_.clear();
          t.diverged = true;
          return;
        }
      }
    }
    static const std::set<std::string> layout_fields = {
        "after.fixed_size", "after.initial_payload_size", "event.growth", "event.compaction"};
    const bool point_here = std::any_of(pending_.begin(), pending_.end(), [](const mismatch& m) {
      return layout_fields.count(m.field) != 0;
    });
    const bool earlier_point = t.policy_point >= 0;
    if (!point_here && !earlier_point) return;  // not explained by the policies: mismatches
    // An InsertMapping that differs after an earlier policy difference changed a length.
    if (earlier_point && t.length_differs &&
        std::any_of(pending_.begin(), pending_.end(),
                    [](const mismatch& m) { return m.field == "out.mapping"; })) {
      for (const mismatch& m : pending_) {
        if (m.field != "out.mapping") continue;
        document("policy_mapping", "the InsertMapping differs after the policy difference at op " +
                                       std::to_string(t.policy_point) + " of the tree (trace " +
                                       m.expected + ", store " + m.actual + ")");
        break;
      }
      pending_.clear();
      t.diverged = true;
      return;
    }
    std::vector<std::string> fields;
    std::vector<mismatch> keep;
    for (mismatch& m : pending_) {
      bool explained = layout_fields.count(m.field) != 0;
      if (m.field == "key.length") {
        explained = true;
        t.length_differs = true;
      }
      if (m.field.rfind("key.row", 0) == 0) {
        const auto k = after.by_key.find(static_cast<std::int32_t>(m.key));
        explained = k != after.by_key.end() && k->second.avail != 0;  // a deleted key's row
      }
      if (explained) {
        if (std::find(fields.begin(), fields.end(), m.field) == fields.end())
          fields.push_back(m.field);
      } else {
        keep.push_back(std::move(m));
      }
    }
    pending_ = std::move(keep);
    if (point_here && !earlier_point) t.policy_point = base.tree_op;
    if (!fields.empty()) {
      std::string list;
      for (const auto& f : fields) list += (list.empty() ? "" : ", ") + f;
      document("policy_difference", list + (point_here ? "" : " (after the policy difference)"));
    }
  }

  // The exemptions of construct_without_row_counts (above) at one op; then the keys whose trace
  // occupancy is no longer the construct's 0 leave the set.
  template <class differ_t>
  void exempt_legacy_occupancy(const tr::record& rec, const tr::tree_state& after, tree_run& t,
                               const mismatch& base, differ_t& differ) {
    std::int64_t exempted = 0, live = 0;
    std::vector<mismatch> keep;
    keep.reserve(pending_.size());
    for (mismatch& m : pending_) {
      const auto it = t.legacy_occupancy.find(static_cast<std::int32_t>(m.key));
      if (m.field == "key.occupancy" && it != t.legacy_occupancy.end() && m.expected == "0" &&
          m.actual == std::to_string(it->second)) {
        ++exempted;
        const auto k = after.by_key.find(it->first);
        if (k != after.by_key.end() && k->second.avail == 0) ++live;
        continue;
      }
      keep.push_back(std::move(m));
    }
    const std::int64_t trace_rows = rec.integer("check.rows", -1);
    std::int64_t rows_exempted = -1;
    for (auto it = keep.begin(); it != keep.end(); ++it) {
      if (it->field == "check.rows" && it->actual == "0" && trace_rows == live) {
        rows_exempted = trace_rows;
        keep.erase(it);
        break;
      }
    }
    pending_ = std::move(keep);
    // MOSP_ESCHER's checker reports every live such key; a trace whose verdict does not is not
    // explained by the rule.
    if (live > 0 && trace_rows >= 0 && trace_rows != live && rows_exempted < 0)
      differ("check.rows (construct_without_row_counts)", -1,
             std::to_string(live) + " metadata violations (the live keys built without row counts)",
             std::to_string(trace_rows) + " in the trace");
    if (exempted > 0) {
      documented_divergence d;
      d.seq = base.seq;
      d.op_index = base.op_index;
      d.tree_op = base.tree_op;
      d.tree = base.tree;
      d.dataset = base.dataset;
      d.op = base.op;
      d.rule = "construct_without_row_counts";
      d.detail = std::to_string(exempted) +
                 " key.occupancy fields exempted (the trace's 0 of MOSP_ESCHER's rowOccupancy = "
                 "nullptr against the store's scanned count; " +
                 std::to_string(live) + " live)" +
                 (rows_exempted >= 0 ? "; check.rows " + std::to_string(rows_exempted) +
                                           " (its metadata violations of those keys) against 0"
                                     : std::string());
      report_.documented.push_back(std::move(d));
    }
    for (auto it = t.legacy_occupancy.begin(); it != t.legacy_occupancy.end();) {
      const auto k = after.by_key.find(it->first);
      if (k == after.by_key.end() || k->second.occupancy != 0)
        it = t.legacy_occupancy.erase(it);
      else
        ++it;
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
    // The integrity verdict, after the state: the original's checkTreeRows on the rows as read,
    // and the store's own check where it has one (the device store; the host model has none).
    if (opt_.fields == field_set::policy) {
      // The store's own checks must be clean (the original's verdict is about its own layout).
      if (got.integrity >= 0) scalar("check.rows", 0, got.integrity);
    } else if (got.integrity >= 0 && rec.integer("check.rows", -1) >= 0) {
      scalar("check.rows", rec.integer("check.rows"), got.integrity);
    }
    // The store's availability check against the original's (MOSP_ESCHER records it; the states
    // of ESCHER-GPU, which has no such check, must have none either), and dynG's checks of the
    // index arrays, which no state of an original fails (M8a step 4).
    if (got.integrity_subtree >= 0)
      scalar("check.subtree_avail", rec.integer("check.subtree_avail", 0), got.integrity_subtree);
    if (got.integrity_index >= 0) scalar("check.index", 0, got.integrity_index);
  }

  tr::reader& in_;
  store_adapter* store_;
  const store_factory* make_store_;
  std::unique_ptr<store_adapter> owned_;
  options opt_;
  replay_report report_;
  std::vector<mismatch> pending_;  // the mismatches of the op being compared
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

cbst_host_policy store_equivalent_host_policy(const std::string& preset) {
  // The host-model policy that behaves like the dynG store under one of its presets (M8a step 5,
  // ADR 0039): the original's behaviour, with the rules the store adds under every preset
  // (exceptions, the construct validation, the fill clamps, the caller's row counts as a fast
  // path and the scan without them, never MOSP_ESCHER's legacy occupancy 0; rebuild_from_device,
  // compact() and the degree-bin counters exist). dynG's default preset is the model's own
  // cbst_host_policy::dyng_default(), written independently of the store.
  cbst_host_policy p;
  if (preset == "escher_gpu") {
    p = cbst_host_policy::escher_gpu();
    p.name = "store:escher_gpu";
  } else if (preset == "mosp_escher") {
    p = cbst_host_policy::mosp_escher();
    p.name = "store:mosp_escher";
    p.has_rebuild = true;
  } else {
    return p;
  }
  p.failure = cbst_host_policy::failure_rule::exception;
  p.error_type = "dyng::capacity_error";
  p.validate_construct = true;
  p.clamp_fill_tail = true;
  p.occupancy = cbst_host_policy::occupancy_rule::caller_counts_or_scan;
  p.has_compact = true;
  p.counts_bin_launches = true;
  return p;
}

bool resolve_host_policy(const std::string& name, cbst_host_policy& out) {
  const std::string prefix = "store:";
  if (name.rfind(prefix, 0) == 0) {
    out = store_equivalent_host_policy(name.substr(prefix.size()));
    return !out.name.empty();
  }
  return cbst_host_policy::from_name(name, out);
}

store_factory host_model_factory(const std::string& policy) {
  return [policy](const std::string& repo) -> std::unique_ptr<store_adapter> {
    const std::string name = policy.empty() ? default_policy(repo) : policy;
    cbst_host_policy p;
    if (!resolve_host_policy(name, p))
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
  o << "], \"documented\": [";
  for (std::size_t i = 0; i < documented.size(); ++i) {
    const documented_divergence& d = documented[i];
    o << (i ? ", " : "") << "{\"seq\": " << d.seq << ", \"op_index\": " << d.op_index
      << ", \"tree_op\": " << d.tree_op << ", \"tree\": " << d.tree << ", \"dataset\": \""
      << json_escape(d.dataset) << "\", \"op\": \"" << json_escape(d.op) << "\", \"rule\": \""
      << json_escape(d.rule) << "\", \"detail\": \"" << json_escape(d.detail) << "\"}";
  }
  o << "], \"notes\": [";
  for (std::size_t i = 0; i < notes.size(); ++i)
    o << (i ? ", " : "") << "\"" << json_escape(notes[i]) << "\"";
  o << "]}";
  return o.str();
}

}  // namespace dyng::test::cbst_compare
