// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (applyBatch's skippedOps and
// the virtual hyperedges' skip rule)
/**
 * @file hsosp_replay.cpp
 * @brief The by-id replay of MOSP_ESCHER's H-SOSP dumps through dyng::hypergraph (see
 *        hsosp_replay.hpp).
 */
#include "parity/hsosp/hsosp_replay.hpp"

#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "support/sha256.hpp"

#include <dyng/core/error.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/hyper_sssp_reference.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hsosp_replay {

namespace {

namespace hd = dyng::test::hsosp_dump;
using whgraph = hypergraph<std::int32_t, std::int32_t>;
using access = dyng::detail::hypergraph_access;

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

class session_replay {
 public:
  session_replay(const resources& res, const options& opt, report& rep)
      : res_(res), opt_(opt), rep_(rep) {}

  void on(const hd::record& r) {
    if (r.tag == "session") {
      begin_session(r);
    } else if (r.tag == "load") {
      if (r.str("phase") == "enter") load_enter(r);
      if (r.str("phase") == "exit") load_exit(r);
    } else if (r.tag == "device_build") {
      device_build(r);
    } else if (r.tag == "begin_batch") {
      if (r.str("phase") == "enter") begin_enter(r);
      if (r.str("phase") == "exit") begin_exit(r);
    } else if (r.tag == "apply_batch") {
      apply_batch(r);
    }
  }

  void finish() {
    if (pending_) {
      // A begin_batch without its apply_batch: B ran beginBatch alone (its tests' refusals throw
      // before they change anything; anything else would stop the session).
      ++rep_.stopped_sessions;
      mismatch(where_ + ": a batch MOSP_ESCHER began and never applied");
      pending_ = false;
    }
  }

 private:
  void mismatch(const std::string& what) {
    ++rep_.mismatch_count;
    if (rep_.mismatches.size() < opt_.max_mismatches) rep_.mismatches.push_back(what);
  }

  void begin_session(const hd::record& r) {
    finish();
    h_.reset();
    session_ = r.i64("session");
    num_vertices_ = static_cast<std::int32_t>(r.i64("num_vertices"));
    factor_ = as_double(r, "caps.headroom_factor", 1.3);
    extra_ = r.i64("caps.extra_payload_ints");
    loaded_ = false;
    built_ = false;
    load_threw_ = false;
    batch_index_ = 0;
  }

  void load_enter(const hd::record& r) {
    rows_offsets_ = r.longs("rows.offsets");
    rows_values_ = r.ints("rows.values");
    weights_ = r.longs("weights");
    source_ = r.i64("source_he");
    target_ = r.i64("target_he");
  }

  void load_exit(const hd::record& r) {
    load_threw_ = !r.str("exception").empty();
    b_pairs_ = r.has("h2h_pair_count") ? r.i64("h2h_pair_count") : 0;
    if (load_threw_) {
      // B's bulkLoad rejected the rows (a weight): dynG must reject them too.
      build(detail::hyper::slack::knobs{}, true, "session " + std::to_string(session_) + " load");
    }
  }

  void device_build(const hd::record& r) {
    if (load_threw_) return;
    if (batch_index_ != 0) {
      mismatch("session " + std::to_string(session_) + ": a device build after its batches");
      ++rep_.stopped_sessions;
      return;
    }
    detail::hyper::slack::knobs knobs;
    knobs.entry_headroom = as_double(r, "entry_headroom", 1.6);
    knobs.node_capacity = r.i64("max_nodes");
    device_record_ = &r;
    build(knobs, !r.str("exception").empty(), "session " + std::to_string(session_) + " build");
    device_record_ = nullptr;
  }

  // The container, with B's knobs (its entry headroom and node capacity: the device build); a
  // session whose batches come without a device build (B's beginBatch refusal tests) gets the
  // defaults.
  void build(const detail::hyper::slack::knobs& knobs, bool b_threw, const std::string& where) {
    if (!built_ && !b_threw) ++rep_.sessions;  // a session counts once (B may build twice)
    built_ = true;
    hypergraph_properties props;
    props.storage = hypergraph_storage::slack_csr;
    props.line_graph = line_graph_policy::slack_csr;
    props.reuse = id_reuse::insert_first;
    props.capacity = capacity_policy::mosp_escher(extra_);
    // floor(size * factor) == size + floor(size * (factor - 1)) for B's factors; 1.3 - 1.0 is not
    // 0.3 in binary (the preset's 0.3 is the value proven equal, M8b deviation 2).
    props.capacity.initial_headroom = factor_ == 1.3 ? 0.3 : factor_ - 1.0;
    // B's virtual rows weigh 0; the container takes [1, 2^31 - 1]: they are loaded with weight 1.
    hyperedge_list<std::int32_t, std::int32_t> list;
    list.num_vertices = num_vertices_;
    list.offsets.assign(rows_offsets_.begin(), rows_offsets_.end());
    list.members.assign(rows_values_.begin(), rows_values_.end());
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      const auto b_id = static_cast<std::int64_t>(i) + 1;
      const long long w = weights_[i];
      list.weights.push_back((b_id == source_ || b_id == target_) && w == 0
                                 ? 1
                                 : (w < INT_MIN || w > INT_MAX ? 0 : static_cast<std::int32_t>(w)));
    }
    bool threw = false;
    try {
      h_ = std::make_unique<whgraph>(access::build(res_, list.view(), props, knobs));
    } catch (const dyng::invalid_argument_error&) {
      threw = true;
    }
    // A weight in (2^28, 2^31 - 1]: MOSP_ESCHER's bulkLoad refuses it, the container keeps
    // [1, 2^31 - 1] (frozen since 0.3) and hyper_sssp refuses it (HD11; M11a step 4): recorded, not
    // a difference of the container.
    bool weight_limit = false;
    for (std::size_t i = 0; i < weights_.size(); ++i)
      weight_limit = weight_limit || (weights_[i] > (1LL << 28) && weights_[i] <= INT_MAX);
    if (b_threw && !threw && weight_limit) {
      ++rep_.hyper_sssp_weight_limit;
    } else if (b_threw != threw) {
      mismatch(where + ": MOSP_ESCHER " + (b_threw ? "rejected" : "accepted") + " it, dynG " +
               (threw ? "rejected" : "accepted") + " it");
    }
    if (b_threw) {
      ++rep_.original_exceptions;
      h_.reset();
      return;
    }
    if (!h_) return;
    loaded_ = true;
    const auto* slack = access::slack(*h_);
    if (slack->rows().pair_count != b_pairs_)
      mismatch(where + ": the pair count " + std::to_string(slack->rows().pair_count) +
               " is not h2hPairCount " + std::to_string(b_pairs_));
    if (device_record_ != nullptr) compare_device(*device_record_, where, "dev.");
    check(where);
  }

  // On the CUDA store, MOSP_ESCHER's device graph is ported as is (ADR 0089): its node capacity,
  // entries, tail cursor and live entries, the incidence mirror's capacities and tails, and its
  // graph_mb scope (DeviceH2H::deviceBytes()) equal the dump's after the build and every batch.
  void compare_device(const hd::record& r, const std::string& where, const std::string& prefix) {
    if (res_.get_backend() != backend::cuda || !h_) return;
    const auto* slack = access::slack(*h_);
    const detail::hyper::slack_row_view v = slack->rows();
    const std::pair<const char*, std::int64_t> fields[] = {
        {"max_nodes", v.node_capacity},    {"num_nodes", v.num_nodes},
        {"cap_entries", v.cap_entries},    {"used_entries", v.used_entries},
        {"num_entries", v.num_entries},    {"inc.he_capacity", v.mirror_he_capacity},
        {"inc.he_tail", v.mirror_he_tail}, {"inc.v_capacity", v.mirror_v_capacity},
        {"inc.v_tail", v.mirror_v_tail},   {"device_bytes", slack->graph_scope_bytes()},
    };
    ++rep_.device_layouts_compared;
    for (const auto& [name, mine] : fields) {
      if (!r.has(prefix + name)) continue;
      if (r.i64(prefix + name) != mine) {
        mismatch(where + ": the device " + name + " " + std::to_string(mine) + " is not B's " +
                 std::to_string(r.i64(prefix + name)));
      }
    }
  }

  void begin_enter(const hd::record& r) {
    where_ = "session " + std::to_string(session_) + " batch " + std::to_string(batch_index_++);
    if (!built_ && !load_threw_) build(detail::hyper::slack::knobs{}, false, where_);
    pending_ = true;
    batch_ = wbatch{};
    filtered_ = 0;
    dropped_ = 0;
    const auto is_virtual = [&](std::int64_t b_id) { return b_id == source_ || b_id == target_; };
    for (const std::int32_t id : r.ints("he_delete")) {
      if (is_virtual(id)) {
        ++filtered_;  // B skips ops on its virtual hyperedges
        continue;
      }
      batch_.delete_hyperedge(static_cast<std::int64_t>(id) - 1);
    }
    const auto& dh = r.ints("vtx_delete.he");
    const auto& dv = r.ints("vtx_delete.vertex");
    for (std::size_t i = 0; i < dh.size(); ++i) {
      if (is_virtual(dh[i])) {
        ++filtered_;
        continue;
      }
      batch_.delete_incidence(static_cast<std::int64_t>(dh[i]) - 1, dv[i]);
    }
    const auto& ih = r.ints("vtx_insert.he");
    const auto& iv = r.ints("vtx_insert.vertex");
    for (std::size_t i = 0; i < ih.size(); ++i) {
      if (is_virtual(ih[i])) {
        ++filtered_;
        continue;
      }
      batch_.insert_incidence(static_cast<std::int64_t>(ih[i]) - 1, iv[i]);
    }
    vertex_kind_ = !dh.empty() || !ih.empty();
    const auto& offsets = r.longs("he_insert.offsets");
    const auto& values = r.ints("he_insert.values");
    const auto& w = r.longs("he_insert.weight");
    for (std::size_t i = 0; i + 1 < offsets.size(); ++i) {
      std::vector<std::int32_t> members(values.begin() + offsets[i],
                                        values.begin() + offsets[i + 1]);
      std::vector<std::int32_t> unique = members;
      std::sort(unique.begin(), unique.end());
      unique.erase(std::unique(unique.begin(), unique.end()), unique.end());
      dropped_ += static_cast<std::int64_t>(members.size() - unique.size());  // B's sort + unique
      // A weight outside int32 cannot be given to dynG: it becomes one the batch rejects (as B
      // rejects every weight outside [1, 2^28]).
      const long long x = w[i];
      batch_.insert_hyperedge(host_view(members),
                              x < 1 || x > INT_MAX ? 0 : static_cast<std::int32_t>(x));
    }
  }

  void begin_exit(const hd::record& r) {
    b_threw_ = !r.str("exception").empty();
    if (!b_threw_) {
      final_ids_ = r.ints("final_ids");
      final_weights_ = r.longs("final_weights");
      dead_he_ = r.ints("dead_he");
      b_skipped_ = r.i64("skipped_ops");
    }
    if (b_threw_) {
      // B's validation rejected the batch before anything changed (no apply_batch follows).
      pending_ = false;
      ++rep_.original_exceptions;
      if (!h_) return;
      const std::uint64_t version = h_->version();
      bool threw = false;
      try {
        (void)h_->apply(res_, batch_.view());
      } catch (const dyng::invalid_argument_error&) {
        threw = true;
      }
      if (!threw)
        mismatch(where_ + ": MOSP_ESCHER rejected the batch (" + r.str("exception") +
                 "), dynG did not");
      if (h_->version() != version) mismatch(where_ + ": a rejected batch changed the hypergraph");
    }
  }

  void apply_batch(const hd::record& r) {
    if (!pending_) return;
    pending_ = false;
    if (!h_) {
      ++rep_.stopped_sessions;
      mismatch(where_ + ": no container to apply the batch to");
      return;
    }
    ++rep_.batches;
    if (vertex_kind_) ++rep_.vertex_batches;
    if (auto* slack = access::slack(*h_))
      slack->set_entry_headroom(as_double(r, "entry_headroom", 1.6));
    std::vector<std::int64_t> pre_offsets;
    std::vector<std::int32_t> pre_members;
    if (opt_.oracle) {
      const auto v = h_->h2v();
      pre_offsets = dyng::test::hyper::host_vector(v.row_ptr);
      pre_members = dyng::test::hyper::host_vector(v.col_ind);
    }
    const auto before = access::slack(*h_)->counters();
    hyper_apply_summary s;
    try {
      s = h_->apply(res_, batch_.view());
    } catch (const dyng::error& e) {
      mismatch(where_ + ": dynG rejected a batch MOSP_ESCHER applied: " + e.what());
      h_.reset();
      ++rep_.stopped_sessions;
      return;
    }
    const auto& a = access::applied(*h_);
    const auto& pairs = access::pairs(*h_);
    // Ids.
    std::vector<std::int64_t> want;
    for (const std::int32_t id : final_ids_) want.push_back(static_cast<std::int64_t>(id) - 1);
    rep_.ids_compared += static_cast<std::int64_t>(want.size());
    if (s.inserted_ids != want) mismatch(where_ + ": inserted_ids differ from finalIds - 1");
    // Skips: ignored deletions + skipped incidence edits + filtered == skippedOps; the dropped
    // members are B's deduplicated ones (B does not count them).
    rep_.skipped_ops += b_skipped_;
    rep_.filtered_virtual += filtered_;
    const std::int64_t skipped_edits = s.skipped_ops - s.ignored_deletions - s.dropped_members;
    rep_.skipped_edits += skipped_edits;
    rep_.removals += s.deleted_incidences;
    rep_.additions += s.inserted_incidences;
    if ((s.skipped_ops - s.dropped_members) + filtered_ != b_skipped_)
      mismatch(where_ + ": ignored deletions " + std::to_string(s.ignored_deletions) +
               " + skipped edits " + std::to_string(skipped_edits) + " + filtered " +
               std::to_string(filtered_) + " != skippedOps " + std::to_string(b_skipped_));
    if (s.dropped_members != dropped_)
      mismatch(where_ + ": dropped members " + std::to_string(s.dropped_members) +
               " != B's deduplicated " + std::to_string(dropped_));
    // Dead and new ids, counters.
    std::vector<std::int64_t> dead;
    for (const std::int32_t id : dead_he_) dead.push_back(static_cast<std::int64_t>(id) - 1);
    std::sort(dead.begin(), dead.end());
    std::vector<std::int64_t> got_dead = a.dead_ids;
    std::sort(got_dead.begin(), got_dead.end());
    if (got_dead != dead) mismatch(where_ + ": the dead ids differ from deadHe - 1");
    if (r.i64("bt.new_he") != s.inserted_hyperedges ||
        r.i64("bt.dead_he") != static_cast<std::int64_t>(a.dead_ids.size()))
      mismatch(where_ + ": newHe / deadHe counts differ");
    // The weights of the new ids (B's final_weights).
    const auto weights = dyng::test::hyper::host_vector(h_->weights());
    for (std::size_t i = 0; i < want.size() && i < final_weights_.size(); ++i)
      if (want[i] < 0 || static_cast<std::size_t>(want[i]) >= weights.size() ||
          weights[static_cast<std::size_t>(want[i])] != final_weights_[i])
        mismatch(where_ + ": the weight of id " + std::to_string(want[i]) + " differs");
    // The pairs.
    const auto same = [](const std::vector<std::int64_t>& mine,
                         const std::vector<std::int32_t>& b) {
      return mine.size() == b.size() &&
             std::equal(mine.begin(), mine.end(), b.begin(), [](std::int64_t x, std::int32_t y) {
               return x == static_cast<std::int64_t>(y);
             });
    };
    ++rep_.pair_batches;
    if (!a.has_line_graph_delta) mismatch(where_ + ": the batch has no pair delta");
    if (r.has("pairs.sha256")) {
      // A digest case (ADR 0081): the canonical pairs (each list sorted, int32 little-endian,
      // the deleted list first) by their SHA-256 and counts.
      ++rep_.digest_batches;
      rep_.deleted_pairs += r.i64("pairs.deleted.a.count");
      rep_.inserted_pairs += r.i64("pairs.inserted.a.count");
      test::sha256 h;
      for (int list = 0; list < 2; ++list) {
        const auto& x = list == 0 ? pairs.deleted_a : pairs.inserted_a;
        const auto& y = list == 0 ? pairs.deleted_b : pairs.inserted_b;
        std::vector<std::pair<std::int64_t, std::int64_t>> sorted;
        for (std::size_t i = 0; i < x.size(); ++i) sorted.emplace_back(x[i], y[i]);
        std::sort(sorted.begin(), sorted.end());
        std::vector<std::int32_t> flat;
        for (const auto& [u, w] : sorted) {
          flat.push_back(static_cast<std::int32_t>(u));
          flat.push_back(static_cast<std::int32_t>(w));
        }
        h.update_le(flat);
      }
      if (static_cast<std::int64_t>(pairs.deleted_a.size()) != r.i64("pairs.deleted.a.count") ||
          static_cast<std::int64_t>(pairs.inserted_a.size()) != r.i64("pairs.inserted.a.count") ||
          h.hex() != r.str("pairs.sha256"))
        mismatch(where_ + ": the pairs' digest differs (" + std::to_string(pairs.deleted_a.size()) +
                 " + " + std::to_string(pairs.inserted_a.size()) + " pairs)");
    } else {
      const auto& da = r.ints("pairs.deleted.a");
      const auto& db = r.ints("pairs.deleted.b");
      const auto& ia = r.ints("pairs.inserted.a");
      const auto& ib = r.ints("pairs.inserted.b");
      rep_.deleted_pairs += static_cast<std::int64_t>(da.size());
      rep_.inserted_pairs += static_cast<std::int64_t>(ia.size());
      if (!same(pairs.deleted_a, da) || !same(pairs.deleted_b, db))
        mismatch(where_ + ": the deleted pairs differ (" + std::to_string(pairs.deleted_a.size()) +
                 " vs " + std::to_string(da.size()) + ")");
      if (!same(pairs.inserted_a, ia) || !same(pairs.inserted_b, ib))
        mismatch(where_ + ": the inserted pairs differ (" +
                 std::to_string(pairs.inserted_a.size()) + " vs " + std::to_string(ia.size()) +
                 ")");
    }
    const auto* slack = access::slack(*h_);
    if (slack->rows().pair_count != r.i64("h2h_pair_count"))
      mismatch(where_ + ": the pair count " + std::to_string(slack->rows().pair_count) +
               " is not h2hPairCount " + std::to_string(r.i64("h2h_pair_count")));
    const auto after = slack->counters();
    (after.candidates == r.i64("bt.candidates") ? rep_.candidates_equal : rep_.candidates_differ) +=
        1;
    rep_.csr_rebuilds += after.csr_rebuilds - before.csr_rebuilds;
    rep_.b_csr_rebuilds += r.i64("bt.csr_rebuilt");
    rep_.mirror_reuploads += after.mirror_reuploads - before.mirror_reuploads;
    rep_.b_mirror_reuploads += r.i64("bt.mirror_reuploads");
    rep_.node_growths += after.node_growths - before.node_growths;
    if (res_.get_backend() == backend::cuda) {
      // The CUDA store is B's device graph: its candidates, rebuilds and mirror re-uploads are
      // B's on every batch (M11a criterion 10).
      if (after.candidates != r.i64("bt.candidates"))
        mismatch(where_ + ": candidates " + std::to_string(after.candidates) + " != B's " +
                 std::to_string(r.i64("bt.candidates")));
      if (after.csr_rebuilds - before.csr_rebuilds != r.i64("bt.csr_rebuilt"))
        mismatch(where_ + ": csrRebuilt differs");
      if (after.mirror_reuploads - before.mirror_reuploads != r.i64("bt.mirror_reuploads"))
        mismatch(where_ + ": mirrorReuploads " +
                 std::to_string(after.mirror_reuploads - before.mirror_reuploads) + " != B's " +
                 std::to_string(r.i64("bt.mirror_reuploads")));
      compare_device(r, where_, "dev.");
    }
    // The rows B recorded (every row, or the touched ones), dead ones empty, and their weights.
    const auto& ids = r.ints("rows.ids");
    const auto& offsets = r.longs("rows.offsets");
    const auto& values = r.ints("rows.values");
    const auto& alive = r.ints("rows.alive");
    const auto& row_weights = r.longs("rows.weight");
    const auto v = h_->h2v();
    const auto ptr = dyng::test::hyper::host_vector(v.row_ptr);
    const auto cols = dyng::test::hyper::host_vector(v.col_ind);
    const std::int64_t bound = h_->hyperedge_id_bound();
    for (std::size_t k = 0; k < ids.size(); ++k) {
      ++rep_.rows_compared;
      const std::int64_t id = static_cast<std::int64_t>(ids[k]) - 1;
      std::vector<std::int32_t> mine;
      if (id < bound)
        mine.assign(cols.begin() + ptr[static_cast<std::size_t>(id)],
                    cols.begin() + ptr[static_cast<std::size_t>(id) + 1]);
      std::vector<std::int32_t> theirs;
      if (alive[k] != 0)
        theirs.assign(values.begin() + offsets[k], values.begin() + offsets[k + 1]);
      if (mine != theirs) {
        mismatch(where_ + ": the row of id " + std::to_string(id) + " differs (" +
                 std::to_string(mine.size()) + " vs " + std::to_string(theirs.size()) +
                 " members)");
        break;
      }
      if (alive[k] != 0) {
        const bool is_virtual = ids[k] == source_ || ids[k] == target_;
        const long long w = weights[static_cast<std::size_t>(id)];
        if (is_virtual ? w != 1 : w != row_weights[k]) {
          mismatch(where_ + ": the weight of id " + std::to_string(id) + " differs");
          break;
        }
      }
    }
    if (opt_.oracle) {
      // Against the oracle: the pair delta of the two states, and the whole line graph.
      ++rep_.oracle_states;
      const std::vector<std::int64_t> post_offsets(ptr.begin(), ptr.end());
      const std::vector<std::int32_t> post_members(cols.begin(), cols.end());
      const auto want_delta = dyng::testing::line_graph_delta(
          pre_offsets, pre_members, post_offsets, post_members, h_->num_vertices());
      std::vector<std::pair<std::int64_t, std::int64_t>> got_deleted, got_inserted;
      for (std::size_t i = 0; i < pairs.deleted_a.size(); ++i)
        got_deleted.emplace_back(pairs.deleted_a[i], pairs.deleted_b[i]);
      for (std::size_t i = 0; i < pairs.inserted_a.size(); ++i)
        got_inserted.emplace_back(pairs.inserted_a[i], pairs.inserted_b[i]);
      if (got_deleted != want_delta.first || got_inserted != want_delta.second)
        mismatch(where_ + ": the pairs differ from testing::line_graph_delta");
      const std::string problems = dyng::test::hyper::check_line_graph(*h_);
      if (!problems.empty()) mismatch(where_ + ": the line-graph oracle: " + problems);
    }
    check(where_);
  }

  void check(const std::string& where) {
    if (!opt_.integrity || !h_) return;
    try {
      h_->check_integrity(res_);
    } catch (const dyng::error& e) {
      mismatch(where + ": check_integrity: " + e.what());
    }
  }

  using wbatch = hyperedge_batch<std::int32_t, std::int32_t>;
  resources res_;
  options opt_;
  report& rep_;
  std::unique_ptr<whgraph> h_;
  std::int64_t session_ = 0, extra_ = 0, source_ = 0, target_ = 0, b_pairs_ = 0;
  std::int32_t num_vertices_ = 0;
  double factor_ = 1.3;
  bool loaded_ = false, built_ = false, load_threw_ = false, pending_ = false, b_threw_ = false;
  bool vertex_kind_ = false;
  std::int64_t batch_index_ = 0, filtered_ = 0, dropped_ = 0, b_skipped_ = 0;
  std::string where_;
  std::vector<std::int64_t> rows_offsets_, weights_, final_weights_;
  std::vector<std::int32_t> rows_values_, final_ids_, dead_he_;
  wbatch batch_;
  const hd::record* device_record_ = nullptr;  // the device_build record of build()
};

}  // namespace

std::string report::to_json() const {
  std::ostringstream os;
  os << "{\"file\": " << json_string(file) << ", \"backend\": " << json_string(backend)
     << ", \"threads\": " << threads << ", \"ok\": " << (ok() ? "true" : "false")
     << ", \"sessions\": " << sessions << ", \"stopped_sessions\": " << stopped_sessions
     << ", \"batches\": " << batches << ", \"vertex_batches\": " << vertex_batches
     << ", \"ids_compared\": " << ids_compared << ", \"rows_compared\": " << rows_compared
     << ", \"pair_batches\": " << pair_batches << ", \"deleted_pairs\": " << deleted_pairs
     << ", \"inserted_pairs\": " << inserted_pairs << ", \"filtered_virtual\": " << filtered_virtual
     << ", \"skipped_ops\": " << skipped_ops << ", \"skipped_edits\": " << skipped_edits
     << ", \"removals\": " << removals << ", \"additions\": " << additions
     << ", \"original_exceptions\": " << original_exceptions
     << ", \"hyper_sssp_weight_limit\": " << hyper_sssp_weight_limit
     << ", \"oracle_states\": " << oracle_states << ", \"candidates_equal\": " << candidates_equal
     << ", \"candidates_differ\": " << candidates_differ << ", \"csr_rebuilds\": " << csr_rebuilds
     << ", \"b_csr_rebuilds\": " << b_csr_rebuilds << ", \"mirror_reuploads\": " << mirror_reuploads
     << ", \"b_mirror_reuploads\": " << b_mirror_reuploads
     << ", \"device_layouts_compared\": " << device_layouts_compared
     << ", \"digest_batches\": " << digest_batches << ", \"node_growths\": " << node_growths
     << ", \"seconds\": " << seconds << ", \"mismatch_count\": " << mismatch_count
     << ", \"mismatches\": [";
  for (std::size_t i = 0; i < mismatches.size(); ++i)
    os << (i ? ", " : "") << json_string(mismatches[i]);
  os << "]}";
  return os.str();
}

report replay(const std::vector<hsosp_dump::record>& records, const resources& res,
              const options& opt, const std::string& name) {
  const auto start = std::chrono::steady_clock::now();
  report rep;
  rep.file = name;
  rep.backend = std::string(to_string(res.get_backend()));
  rep.threads = res.num_threads();
  session_replay replay(res, opt, rep);
  for (const auto& r : records) replay.on(r);
  replay.finish();
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return rep;
}

}  // namespace dyng::test::hsosp_replay
