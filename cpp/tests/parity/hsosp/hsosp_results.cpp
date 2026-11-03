// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (applyBatch's skip rule for
// the virtual hyperedges) and hsosp/src/hsospBench.cu (the state A update, state B recompute
// order of a batch)
/**
 * @file hsosp_results.cpp
 * @brief The result-side replay of MOSP_ESCHER's H-SOSP dumps through hyper_sssp (see
 *        hsosp_results.hpp).
 */
#include "parity/hsosp/hsosp_results.hpp"

#include "algorithms/hyper_sssp/virtual_endpoints.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "support/sha256.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/hyper_sssp.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>
#include <dyng/update.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <exception>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hsosp_results {

const char* to_string(mode m) noexcept {
  switch (m) {
    case mode::by_id:
      return "by_id";
    case mode::by_id_public:
      return "by_id_public";
    case mode::by_key:
      return "by_key";
  }
  return "?";
}

namespace {

namespace hd = dyng::test::hsosp_dump;
namespace hv = dyng::detail::hyper_sssp_virtual;
using whgraph = hypergraph<std::int32_t, std::int32_t>;
using wbatch = hyperedge_batch<std::int32_t, std::int32_t>;
using result_t = hyper_sssp::result<std::int32_t>;
using access = dyng::detail::hypergraph_access;
constexpr std::int64_t inf = INT64_MAX / 4;

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

double as_double(const hd::record& r, const std::string& name, double fallback) {
  if (!r.has(name)) return fallback;
  return std::strtod(r.str(name).c_str(), nullptr);
}

/// The SHA-256 of distances as int64 little-endian over `count` ids (padded with INF).
std::string distances_digest(const std::vector<std::int64_t>& d, std::size_t count) {
  std::vector<std::int64_t> c(count, inf);
  for (std::size_t i = 0; i < count && i < d.size(); ++i) c[i] = d[i];
  // int64 little-endian (the canonical encoding, ADR 0080)
  test::sha256 h;
  for (const std::int64_t v : c) {
    const auto u = static_cast<std::uint64_t>(v);
    unsigned char b[8];
    for (int k = 0; k < 8; ++k) b[k] = static_cast<unsigned char>(u >> (8 * k));
    h.update(b, 8);
  }
  return h.hex();
}

/// The SHA-256 of parents as int32 little-endian over `count` ids (padded with -1).
std::string parents_digest(const std::vector<std::int32_t>& p, std::size_t count) {
  std::vector<std::int32_t> c(count, -1);
  for (std::size_t i = 0; i < count && i < p.size(); ++i) c[i] = p[i];
  test::sha256 h;
  h.update_le(c);
  return h.hex();
}

/// One B call of a session (an sssp record), with its index in the dump.
struct b_call {
  const hd::record* rec = nullptr;
  std::int64_t state = 0;
};

class session_results {
 public:
  session_results(const resources& res, const options& opt, report& rep)
      : res_(res), opt_(opt), rep_(rep) {}

  void run(const std::vector<hd::record>& records) {
    for (std::size_t i = 0; i < records.size(); ++i) {
      const hd::record& r = records[i];
      if (r.tag == "session") {
        finish();
        begin_session(r);
      } else if (r.tag == "load") {
        if (r.str("phase") == "enter") load_enter(r);
        if (r.str("phase") == "exit") load_threw_ = !r.str("exception").empty();
      } else if (r.tag == "device_build") {
        device_build(r);
      } else if (r.tag == "begin_batch") {
        if (r.str("phase") == "enter") begin_enter(r);
        if (r.str("phase") == "exit") begin_exit(r);
      } else if (r.tag == "apply_batch") {
        apply_batch(r, records, i);
      } else if (r.tag == "sssp") {
        sssp(r);
      }
    }
    finish();
  }

 private:
  void mismatch(const std::string& what) {
    ++rep_.mismatch_count;
    if (rep_.mismatches.size() < opt_.max_mismatches) rep_.mismatches.push_back(what);
  }

  void finish() {
    if (active_) ++rep_.sessions;
    active_ = false;
    h_.reset();
    results_.clear();
  }

  void begin_session(const hd::record& r) {
    session_ = r.i64("session");
    num_vertices_ = static_cast<std::int32_t>(r.i64("num_vertices"));
    factor_ = as_double(r, "caps.headroom_factor", 1.3);
    extra_ = r.i64("caps.extra_payload_ints");
    load_threw_ = false;
    built_ = false;
    batch_index_ = 0;
    pending_ = false;
    rejected_ = false;
    b_to_dyng_.clear();
  }

  void load_enter(const hd::record& r) {
    rows_offsets_ = r.longs("rows.offsets");
    rows_values_ = r.ints("rows.values");
    weights_ = r.longs("weights");
    source_ = r.i64("source_he");
    target_ = r.i64("target_he");
    // B's source vertex: the member of its source row (the public modes' source).
    const auto s = static_cast<std::size_t>(source_ - 1);
    source_vertex_ = s + 1 < rows_offsets_.size() && rows_offsets_[s] < rows_offsets_[s + 1]
                         ? rows_values_[static_cast<std::size_t>(rows_offsets_[s])]
                         : -1;
  }

  void device_build(const hd::record& r) {
    if (load_threw_ || !r.str("exception").empty()) return;
    detail::hyper::slack::knobs knobs;
    knobs.entry_headroom = as_double(r, "entry_headroom", 1.6);
    knobs.node_capacity = r.i64("max_nodes");
    build(knobs);
  }

  bool is_virtual(std::int64_t b_id) const {
    return b_id == source_ || b_id == target_;
  }

  /// The container of the mode (by_id and by_id_public: B's rows and virtual rows by id; by_key:
  /// the real rows only, as a user loads them).
  void build(const detail::hyper::slack::knobs& knobs) {
    built_ = true;
    active_ = true;
    hypergraph_properties props;
    props.storage = hypergraph_storage::slack_csr;
    props.line_graph = line_graph_policy::slack_csr;
    hyperedge_list<std::int32_t, std::int32_t> list;
    list.num_vertices = num_vertices_;
    b_to_dyng_.clear();
    for (std::size_t i = 0; i + 1 < rows_offsets_.size(); ++i) {
      const auto b_id = static_cast<std::int64_t>(i) + 1;
      const long long w = weights_[i];
      if (opt_.mode == mode::by_key && is_virtual(b_id)) continue;
      if (opt_.mode == mode::by_key) {
        b_to_dyng_[b_id] = static_cast<std::int64_t>(list.weights.size());
      }
      list.add_hyperedge(
          host_view(std::vector<std::int32_t>(rows_values_.begin() + rows_offsets_[i],
                                              rows_values_.begin() + rows_offsets_[i + 1])),
          is_virtual(b_id) && w == 0 ? 1 : static_cast<std::int32_t>(w));
    }
    try {
      if (opt_.mode == mode::by_key) {
        props.reuse = id_reuse::erase_first;  // the documented default (O16)
        h_ = std::make_unique<whgraph>(whgraph::from_hyperedges(res_, list.view(), props));
      } else {
        props.reuse = id_reuse::insert_first;
        props.capacity = capacity_policy::mosp_escher(extra_);
        props.capacity.initial_headroom = factor_ == 1.3 ? 0.3 : factor_ - 1.0;
        h_ = std::make_unique<whgraph>(access::build(res_, list.view(), props, knobs));
      }
    } catch (const dyng::error& e) {
      // A load MOSP_ESCHER accepted and dynG refuses (a weight above 2^28 is refused by
      // hyper_sssp, not by the container: compute() below).
      mismatch("session " + std::to_string(session_) + ": dynG refused the load: " + e.what());
      ++rep_.stopped_sessions;
      active_ = false;
    }
  }

  void begin_enter(const hd::record& r) {
    where_ = "session " + std::to_string(session_) + " batch " + std::to_string(batch_index_++);
    if (!built_ && !load_threw_) build(detail::hyper::slack::knobs{});
    pending_ = true;
    rejected_ = false;
    batch_ = wbatch{};
    const auto id_of = [&](std::int64_t b_id) -> std::int64_t {
      if (opt_.mode != mode::by_key) return b_id - 1;
      const auto it = b_to_dyng_.find(b_id);
      return it == b_to_dyng_.end() ? -1 - b_id : it->second;  // an unknown id stays invalid
    };
    for (const std::int32_t id : r.ints("he_delete")) {
      if (is_virtual(id) || id < 1) continue;
      batch_.delete_hyperedge(id_of(id));
    }
    const auto& dh = r.ints("vtx_delete.he");
    const auto& dv = r.ints("vtx_delete.vertex");
    for (std::size_t i = 0; i < dh.size(); ++i) {
      if (is_virtual(dh[i]) || dh[i] < 1 || dv[i] < 0) continue;
      batch_.delete_incidence(id_of(dh[i]), dv[i]);
    }
    const auto& ih = r.ints("vtx_insert.he");
    const auto& iv = r.ints("vtx_insert.vertex");
    for (std::size_t i = 0; i < ih.size(); ++i) {
      if (is_virtual(ih[i]) || ih[i] < 1 || iv[i] < 0) continue;
      batch_.insert_incidence(id_of(ih[i]), iv[i]);
    }
    const auto& offsets = r.longs("he_insert.offsets");
    const auto& values = r.ints("he_insert.values");
    const auto& w = r.longs("he_insert.weight");
    for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
      const std::vector<std::int32_t> members(values.begin() + offsets[i],
                                              values.begin() + offsets[i + 1]);
      const long long x = w[i];
      batch_.insert_hyperedge(host_view(members),
                              x < 1 || x > INT_MAX ? 0 : static_cast<std::int32_t>(x));
    }
  }

  void begin_exit(const hd::record& r) {
    if (!r.str("exception").empty()) {
      pending_ = false;  // B rejected the batch: nothing changes (the delta replay checks dynG's)
      rejected_ = true;
      return;
    }
    final_ids_ = r.ints("final_ids");
    dead_he_ = r.has("dead_he") ? r.ints("dead_he") : std::vector<std::int32_t>{};
  }

  /// The B calls of this batch's results: the sssp records after the apply_batch, up to the next
  /// batch or session.
  std::vector<b_call> calls_after(const std::vector<hd::record>& records, std::size_t i) const {
    std::vector<b_call> out;
    for (std::size_t k = i + 1; k < records.size(); ++k) {
      const hd::record& r = records[k];
      if (r.tag == "begin_batch" || r.tag == "apply_batch" || r.tag == "session" ||
          r.tag == "load" || r.tag == "device_build")
        break;
      if (r.tag == "sssp" && r.i64("session") == session_) out.push_back({&r, r.i64("state")});
    }
    return out;
  }

  /// B's UpdateConfig of a call on a result.
  void configure(result_t& r, const hd::record& rec) {
    hyper_sssp::options o = r.get_options();
    o.max_iterations = rec.i64("cfg.max_iterations");
    o.work_budget = as_double(rec, "cfg.work_budget", 1.0);
    r.set_options(o);
    hv::set_block_size(r, static_cast<int>(rec.i64("cfg.block_size")));
  }

  void apply_batch(const hd::record& r, const std::vector<hd::record>& records, std::size_t i) {
    if (!active_ || !h_) return;  // a stopped session (counted where it stopped)
    if (!pending_) {
      if (!rejected_) {
        // An applyBatch without a begin_batch of its own (a lost or reordered record): its batch
        // and the results after it cannot be checked, so the session stops, never a silent skip.
        mismatch("session " + std::to_string(session_) +
                 ": an apply_batch without its begin_batch");
        ++rep_.stopped_sessions;
        active_ = false;
      }
      return;
    }
    pending_ = false;
    ++rep_.batches;
    if (opt_.mode != mode::by_key) {
      if (auto* slack = access::slack(*h_))
        slack->set_entry_headroom(as_double(r, "entry_headroom", 1.6));
    }
    // The results B updates after this batch (hsospUpdate records): one update of them all.
    std::vector<std::int64_t> updated;
    for (const b_call& c : calls_after(records, i)) {
      if (c.rec->str("kind") != "update" || !c.rec->str("exception").empty()) continue;
      const auto it = results_.find(c.state);
      if (it == results_.end() || it->second.version != h_->version()) continue;
      if (std::find(updated.begin(), updated.end(), c.state) != updated.end()) continue;
      configure(*it->second.r, *c.rec);
      updated.push_back(c.state);
    }
    hyper_apply_summary summary;
    try {
      if (updated.empty()) {
        summary = h_->apply(res_, batch_.view());
      } else if (updated.size() == 1) {
        tracked& t = results_.at(updated[0]);
        t.stats = hyper_sssp::update(res_, *h_, batch_.view(), *t.r);
        summary = t.stats.batch;
      } else if (updated.size() == 2) {
        tracked& a = results_.at(updated[0]);
        tracked& b = results_.at(updated[1]);
        std::tie(a.stats, b.stats) = dyng::update(res_, *h_, batch_.view(), *a.r, *b.r);
        summary = a.stats.batch;
      } else {
        mismatch(where_ + ": more than two results updated by one batch (not replayed)");
        ++rep_.stopped_sessions;
        active_ = false;
        return;
      }
    } catch (const dyng::error& e) {
      mismatch(where_ + ": dynG rejected a batch MOSP_ESCHER applied: " + e.what());
      ++rep_.stopped_sessions;
      active_ = false;
      return;
    }
    for (const std::int64_t s : updated) {
      tracked& t = results_.at(s);
      t.version = h_->version();
      t.updated_by_batch = true;
      t.update_compared = false;
    }
    // B's ids of the inserted hyperedges -> dynG's (by_key; by id they are B id - 1).
    if (opt_.mode == mode::by_key) {
      // B's dead ids leave the map first: dynG (erase_first) may give their dynG ids to this
      // batch's insertions.
      for (const std::int32_t id : dead_he_) b_to_dyng_.erase(id);
      if (!map_by_content(r, summary)) return;
    }
    // Results not updated are now stale (B's states it did not update).
    for (auto& [s, t] : results_) {
      if (std::find(updated.begin(), updated.end(), s) == updated.end()) t.updated_by_batch = false;
    }
  }

  void sssp(const hd::record& rec) {
    if (rec.i64("session") != session_ || !active_ || !h_) return;
    const std::int64_t state = rec.i64("state");
    const std::string kind = rec.str("kind");
    const bool b_threw = !rec.str("exception").empty();
    const std::string at = "session " + std::to_string(session_) + " " + kind + " of state " +
                           std::to_string(state) + " after batch " +
                           std::to_string(rec.i64("batches"));
    auto it = results_.find(state);
    if (kind == "recompute") {
      ++rep_.recomputes;
      bool threw = false;
      try {
        if (it == results_.end() || it->second.version != h_->version()) {
          tracked t;
          t.r = std::make_unique<result_t>(compute_mode(rec));
          t.version = h_->version();
          it = results_.insert_or_assign(state, std::move(t)).first;
        } else {
          configure(*it->second.r, rec);
          hv::recompute(res_, *h_, *it->second.r);
          it->second.updated_by_batch = false;
        }
      } catch (const dyng::invalid_argument_error&) {
        threw = true;
      }
      if (threw != b_threw)
        mismatch(at + ": MOSP_ESCHER " + (b_threw ? "threw" : "ran") + ", dynG " +
                 (threw ? "threw" : "ran"));
      if (b_threw) ++rep_.original_exceptions;
      if (threw || b_threw) return;
      compare(rec, *it->second.r, at, kind, state, nullptr);
      // update == compute (bitwise): another state updated by this batch equals this recompute.
      for (const auto& [s, t] : results_) {
        if (s == state || !t.updated_by_batch || t.version != h_->version()) continue;
        ++rep_.compute_checked;
        if (host_of(*t.r) != host_of(*it->second.r))
          mismatch(at + ": the update of state " + std::to_string(s) +
                   " differs from this recompute (update != compute)");
      }
      return;
    }
    // An update B recorded: the result was updated with its batch (apply_batch above).
    ++rep_.updates;
    if (b_threw) {
      ++rep_.original_exceptions;
      return;
    }
    if (it == results_.end() || !it->second.updated_by_batch || it->second.update_compared ||
        it->second.version != h_->version()) {
      ++rep_.not_checked;
      return;
    }
    it->second.update_compared = true;
    compare(rec, *it->second.r, at, kind, state, &it->second.stats);
    if (opt_.check_compute) {
      ++rep_.compute_checked;
      const result_t fresh = compute_mode(rec);
      if (host_of(fresh) != host_of(*it->second.r)) mismatch(at + ": update != compute()");
    }
  }

  /// compute() of the mode on the current hypergraph.
  result_t compute_mode(const hd::record& rec) {
    hyper_sssp::options o;
    o.max_iterations = rec.i64("cfg.max_iterations");
    o.work_budget = as_double(rec, "cfg.work_budget", 1.0);
    const int block = static_cast<int>(rec.i64("cfg.block_size"));
    if (opt_.mode == mode::by_id) {
      detail::hyper_sssp_endpoints ends;
      ends.source_hyperedge = rec.i64("source") - 1;
      ends.zero_weight_a = source_ - 1;
      ends.zero_weight_b = target_ - 1;
      return hv::compute(res_, *h_, ends, o, block);
    }
    result_t r = hyper_sssp::compute(res_, *h_, source_vertex_, o);
    hv::set_block_size(r, block);
    return r;
  }

  std::pair<std::vector<std::int64_t>, std::vector<std::int32_t>> host_of(const result_t& r) const {
    return {to_vector(res_, r.hyperedge_distances()), to_vector(res_, r.hyperedge_parents())};
  }

  /// One B call against dynG's result (by the mode's rule).
  void compare(const hd::record& rec, const result_t& r, const std::string& at,
               const std::string& kind, std::int64_t state, const hyper_sssp::stats* stats) {
    const auto [d, p] = host_of(r);
    call_digest cd;
    cd.session = session_;
    cd.state = state;
    cd.batches = rec.i64("batches");
    cd.kind = kind;
    cd.distances = distances_digest(d, d.size());
    cd.parents = parents_digest(p, p.size());
    if (stats != nullptr) {
      cd.fallback = stats->fallback_used;
      cd.reason = std::string(hyper_sssp::to_string(stats->fallback_reason));
      cd.b_fallback = rec.has("stats.fallback_recompute") ? rec.i64("stats.fallback_recompute") != 0
                      : rec.has("fallback")               ? rec.i64("fallback") != 0
                                                          : false;
      (cd.fallback == cd.b_fallback ? rep_.fallback_equal : rep_.fallback_differ) += 1;
      rep_.dyng_fallbacks += cd.fallback ? 1 : 0;
      rep_.b_fallbacks += cd.b_fallback ? 1 : 0;
    }
    rep_.calls.push_back(cd);
    const bool digest = rec.has("distances.sha256");
    if (digest) {
      ++rep_.digest_calls;
      if (opt_.mode != mode::by_id) {
        ++rep_.digest_skipped;  // the public modes compare full arrays (T and P50)
        return;
      }
      // The canonical encodings span max(bound_B, bound_dynG) ids (ADR 0080): B's digest covers
      // its count, and every id of dynG's at or past it must be dead or unreachable (INF, -1).
      const auto count = static_cast<std::size_t>(rec.i64("distances.count"));
      const auto pcount = static_cast<std::size_t>(rec.i64("parents.count"));
      rep_.ids_compared += static_cast<std::int64_t>(std::max(count, d.size()));
      if (distances_digest(d, count) != rec.str("distances.sha256"))
        mismatch(at + ": the distances' digest differs");
      if (parents_digest(p, pcount) != rec.str("parents.sha256"))
        mismatch(at + ": the parents' digest differs");
      std::int64_t past = 0;
      for (std::size_t id = count; id < d.size(); ++id) past += d[id] != inf ? 1 : 0;
      for (std::size_t id = pcount; id < p.size(); ++id) past += p[id] != -1 ? 1 : 0;
      if (past != 0)
        mismatch(at + ": " + std::to_string(past) +
                 " distances or parents past B's bound are not INF / -1");
      return;
    }
    const auto& bd = rec.longs("distances");
    const auto& bp = rec.ints("parents");
    if (opt_.mode == mode::by_id) {
      const std::size_t n = std::max({bd.size(), d.size()});
      std::int64_t dm = 0, pm = 0, first = -1;
      for (std::size_t id = 0; id < n; ++id) {
        const std::int64_t want_d = id < bd.size() ? bd[id] : inf;
        const std::int64_t got_d = id < d.size() ? d[id] : inf;
        const std::int32_t want_p = id < bp.size() ? bp[id] : -1;
        const std::int32_t got_p = id < p.size() ? p[id] : -1;
        if (want_d != got_d && dm++ == 0 && first < 0) first = static_cast<std::int64_t>(id);
        if (want_p != got_p && pm++ == 0 && first < 0) first = static_cast<std::int64_t>(id);
      }
      rep_.ids_compared += static_cast<std::int64_t>(n);
      if (dm || pm)
        mismatch(at + ": " + std::to_string(dm) + " distance and " + std::to_string(pm) +
                 " parent differences by id (first at id " + std::to_string(first) + ")");
      return;
    }
    // The public modes: distances on the real hyperedges, documented parent differences, and the
    // canonical rule of every parent.
    std::int64_t dm = 0, pm = 0;
    std::string first;
    for (std::size_t i = 0; i < bd.size(); ++i) {
      const auto b_id = static_cast<std::int64_t>(i) + 1;
      if (is_virtual(b_id)) continue;
      std::int64_t id = static_cast<std::int64_t>(i);
      if (opt_.mode == mode::by_key) {
        const auto m = b_to_dyng_.find(b_id);
        if (m == b_to_dyng_.end()) {
          if (bd[i] < inf) ++dm;  // a live B hyperedge dynG does not have
          continue;
        }
        id = m->second;
      }
      const auto u = static_cast<std::size_t>(id);
      const std::int64_t got_d = u < d.size() ? d[u] : inf;
      ++rep_.ids_compared;
      if (got_d != bd[i]) {
        if (dm++ == 0) first = "distance of B id " + std::to_string(b_id);
        continue;
      }
      if (opt_.mode == mode::by_key) continue;  // parents: the canonical check below
      const std::int32_t got_p = u < p.size() ? p[u] : -1;
      const std::int32_t want = bp[i];
      if (want == static_cast<std::int32_t>(source_ - 1)) {
        ++rep_.source_row_parents;
        if (got_p != -1 && pm++ == 0) first = "parent of B id " + std::to_string(b_id);
      } else if (want == static_cast<std::int32_t>(target_ - 1)) {
        ++rep_.target_row_parents;  // dynG's: checked by the canonical rule
      } else if (got_p != want && pm++ == 0) {
        first = "parent of B id " + std::to_string(b_id);
      }
    }
    const auto [od, op] = oracle_errors(d, p, at);
    if (dm || pm || od || op)
      mismatch(at + ": " + std::to_string(dm) + " distance and " + std::to_string(pm) +
               " parent differences from MOSP_ESCHER, " + std::to_string(od) + " distance and " +
               std::to_string(op) + " parent differences from the oracle" +
               (first.empty() ? "" : " (first: " + first + ")"));
  }

  /// The oracle's check of the public modes (dyng::testing::check_hyper_sssp_tree, ADR 0082):
  /// dynG's distances and parents against hyper_sssp_reference from the source vertex, on the
  /// container's public rows (h2v() and weights()) only, so nothing of the slack_csr store's
  /// internals or of hyper_sssp is shared with the rule that checks them. @return {distance
  /// mismatches, parent errors} over every id.
  std::pair<std::int64_t, std::int64_t> oracle_errors(const std::vector<std::int64_t>& d,
                                                      const std::vector<std::int32_t>& p,
                                                      const std::string& at) {
    const auto v = h_->h2v();
    const std::vector<std::int64_t> offsets = to_vector(res_, v.row_ptr);
    const std::vector<std::int32_t> members = to_vector(res_, v.col_ind);
    const std::vector<std::int32_t> w32 = to_vector(res_, h_->weights());
    std::vector<std::int64_t> weights(w32.begin(), w32.end());
    weights.resize(offsets.empty() ? 0 : offsets.size() - 1, 0);
    rep_.canonical_checked +=
        static_cast<std::int64_t>(std::max({weights.size(), d.size(), p.size()}));
    try {
      return dyng::testing::check_hyper_sssp_tree(offsets, members, num_vertices_, weights,
                                                  source_vertex_, true, d, p);
    } catch (const std::exception& e) {
      mismatch(at + ": the oracle refused the hypergraph: " + e.what());
      return {0, 0};
    }
  }

  /// by_key (HD2: hyperedges by (sorted vertex set, weight)): B's ids of this batch's insertions
  /// are mapped to dynG's, and each mapped dynG row must hold the sorted vertex set and the
  /// weight of B's row in the apply_batch record (its touched rows), so a mapping that pairs two
  /// different hyperedges is a mismatch, never a silent misattribution. @return Whether the
  /// session goes on.
  bool map_by_content(const hd::record& r, const hyper_apply_summary& summary) {
    if (final_ids_.size() != summary.inserted_ids.size()) {
      mismatch(where_ + ": MOSP_ESCHER inserted " + std::to_string(final_ids_.size()) +
               " hyperedges, dynG " + std::to_string(summary.inserted_ids.size()));
      ++rep_.stopped_sessions;
      active_ = false;
      return false;
    }
    const auto& ids = r.ints("rows.ids");
    const auto& offsets = r.longs("rows.offsets");
    const auto& values = r.ints("rows.values");
    const auto& alive = r.ints("rows.alive");
    const auto& row_weights = r.longs("rows.weight");
    std::map<std::int32_t, std::size_t> b_row;
    for (std::size_t k = 0; k < ids.size(); ++k) b_row[ids[k]] = k;
    const auto v = h_->h2v();
    const std::vector<std::int64_t> ptr = to_vector(res_, v.row_ptr);
    const std::vector<std::int32_t> cols = to_vector(res_, v.col_ind);
    const std::vector<std::int32_t> w = to_vector(res_, h_->weights());
    std::int64_t differ = 0;
    std::string first;
    for (std::size_t k = 0; k < final_ids_.size(); ++k) {
      const std::int64_t mine = summary.inserted_ids[k];
      b_to_dyng_[final_ids_[k]] = mine;
      ++rep_.keys_compared;
      const auto it = b_row.find(final_ids_[k]);
      const auto u = static_cast<std::size_t>(mine);
      bool same = it != b_row.end() && alive[it->second] != 0 && mine >= 0 && u + 1 < ptr.size();
      if (same) {
        std::vector<std::int32_t> got(cols.begin() + ptr[u], cols.begin() + ptr[u + 1]);
        std::vector<std::int32_t> want(values.begin() + offsets[it->second],
                                       values.begin() + offsets[it->second + 1]);
        std::sort(got.begin(), got.end());
        std::sort(want.begin(), want.end());
        same = got == want && u < w.size() && w[u] == row_weights[it->second];
      }
      if (!same && differ++ == 0)
        first = "B id " + std::to_string(final_ids_[k]) + " -> dynG id " + std::to_string(mine);
    }
    if (differ != 0)
      mismatch(
          where_ + ": " + std::to_string(differ) +
          " inserted hyperedges differ in (sorted vertex set, weight) from MOSP_ESCHER's (first: " +
          first + ")");
    return true;
  }

  struct tracked {
    std::unique_ptr<result_t> r;
    std::uint64_t version = 0;      ///< the hypergraph version it matches
    bool updated_by_batch = false;  ///< updated by the last batch
    bool update_compared = false;   ///< its B update record was compared
    hyper_sssp::stats stats;        ///< its last update's stats
  };

  resources res_;
  options opt_;
  report& rep_;
  std::unique_ptr<whgraph> h_;
  std::map<std::int64_t, tracked> results_;
  std::map<std::int64_t, std::int64_t> b_to_dyng_;  ///< by_key: B id -> dynG id
  std::int64_t session_ = 0, extra_ = 0, source_ = 0, target_ = 0, batch_index_ = 0;
  std::int32_t num_vertices_ = 0, source_vertex_ = -1;
  double factor_ = 1.3;
  bool load_threw_ = false, built_ = false, pending_ = false, rejected_ = false, active_ = false;
  std::string where_;
  std::vector<std::int64_t> rows_offsets_, weights_;
  std::vector<std::int32_t> rows_values_, final_ids_, dead_he_;
  wbatch batch_;
};

}  // namespace

std::string report::to_json() const {
  std::ostringstream os;
  os << "{\"file\": " << json_string(file) << ", \"backend\": " << json_string(backend)
     << ", \"mode\": " << json_string(mode) << ", \"ok\": " << (ok() ? "true" : "false")
     << ", \"sessions\": " << sessions << ", \"stopped_sessions\": " << stopped_sessions
     << ", \"batches\": " << batches << ", \"updates\": " << updates
     << ", \"recomputes\": " << recomputes << ", \"digest_calls\": " << digest_calls
     << ", \"ids_compared\": " << ids_compared
     << ", \"original_exceptions\": " << original_exceptions
     << ", \"source_row_parents\": " << source_row_parents
     << ", \"target_row_parents\": " << target_row_parents
     << ", \"canonical_checked\": " << canonical_checked << ", \"keys_compared\": " << keys_compared
     << ", \"digest_skipped\": " << digest_skipped << ", \"compute_checked\": " << compute_checked
     << ", \"fallback_equal\": " << fallback_equal << ", \"fallback_differ\": " << fallback_differ
     << ", \"dyng_fallbacks\": " << dyng_fallbacks << ", \"b_fallbacks\": " << b_fallbacks
     << ", \"not_checked\": " << not_checked << ", \"seconds\": " << seconds
     << ", \"mismatch_count\": " << mismatch_count << ", \"mismatches\": [";
  for (std::size_t i = 0; i < mismatches.size(); ++i)
    os << (i ? ", " : "") << json_string(mismatches[i]);
  os << "], \"calls\": [";
  for (std::size_t i = 0; i < calls.size(); ++i) {
    const call_digest& c = calls[i];
    os << (i ? ", " : "") << "[" << c.session << ", " << c.state << ", " << json_string(c.kind)
       << ", " << c.batches << ", " << json_string(c.distances) << ", " << json_string(c.parents)
       << ", " << (c.b_fallback ? 1 : 0) << ", " << (c.fallback ? 1 : 0) << ", "
       << json_string(c.reason) << "]";
  }
  os << "]}";
  return os.str();
}

report replay(const std::vector<hsosp_dump::record>& records, const resources& res,
              const options& opt, const std::string& name) {
  const auto start = std::chrono::steady_clock::now();
  report rep;
  rep.file = name;
  rep.backend = std::string(dyng::to_string(res.get_backend()));
  rep.mode = to_string(opt.mode);
  session_results s(res, opt, rep);
  s.run(records);
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return rep;
}

}  // namespace dyng::test::hsosp_results
