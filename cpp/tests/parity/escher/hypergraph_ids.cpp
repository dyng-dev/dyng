// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (payloadCapacity; the host model)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (constructFromRows)
/**
 * @file hypergraph_ids.cpp
 * @brief Container-level id parity against both originals' batch records (M8b, criterion 4).
 */
#include "parity/escher/hypergraph_ids.hpp"

#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/hypergraph_support.hpp"
#include "parity/escher/escher_dump.hpp"

#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/generators/legacy.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/hypergraph_io.hpp>

#include <algorithm>
#include <chrono>
#include <climits>
#include <cstdint>
#include <map>
#include <memory>
#include <random>
#include <set>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::hypergraph_ids {

namespace tr = dyng::test::cbst_trace;
namespace ed = dyng::test::escher_dump;
namespace hs = dyng::test::hyper;
using rows = hs::rows;

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

using ed::read_rows;

std::vector<std::int32_t> sorted_unique(std::vector<std::int32_t> r) {
  std::sort(r.begin(), r.end());
  r.erase(std::unique(r.begin(), r.end()), r.end());
  return r;
}

// The construct layout size of rows (padding rule 1).
std::int64_t flat_size(const rows& r) {
  std::int64_t total = 0;
  for (const auto& row : r) total += (static_cast<std::int64_t>(row.size()) + 4) / 4 * 4;
  return total;
}

// The rows of the live ids, sorted: the content multiset of a state.
rows content_multiset(const rows& by_id) {
  rows out;
  for (const auto& r : by_id)
    if (!r.empty()) out.push_back(r);
  std::sort(out.begin(), out.end());
  return out;
}

// The transpose of rows by id: per vertex, the sorted ids.
std::vector<std::vector<std::int64_t>> transpose(const rows& by_id, std::int32_t n) {
  std::vector<std::vector<std::int64_t>> out(static_cast<std::size_t>(n));
  for (std::size_t id = 0; id < by_id.size(); ++id)
    for (const std::int32_t v : by_id[id])
      out[static_cast<std::size_t>(v)].push_back(static_cast<std::int64_t>(id));
  return out;
}

class recorder {
 public:
  explicit recorder(report& r, std::size_t max) : r_(r), max_(max) {}
  void mismatch(const std::string& what) {
    ++r_.mismatch_count;
    if (r_.mismatches.size() < max_) r_.mismatches.push_back(what);
  }

 private:
  report& r_;
  std::size_t max_;
};

// Compares a container's h2v (by id, padded to the longer side with empty rows) and v2h with the
// rows by id of the original's model.
template <typename weight_t>
void compare_state(const hypergraph<std::int32_t, weight_t>& h, const rows& model, std::int32_t n,
                   const std::string& where, report& rep, recorder& out) {
  rows got = hs::h2v_rows(h);
  rows want = model;
  const std::size_t size = std::max(got.size(), want.size());
  got.resize(size);
  want.resize(size);
  rep.rows_compared += static_cast<std::int64_t>(size);
  for (std::size_t i = 0; i < size; ++i) {
    if (got[i] != want[i]) {
      out.mismatch(where + ": h2v row of id " + std::to_string(i) + " differs (" +
                   std::to_string(got[i].size()) + " vs " + std::to_string(want[i].size()) +
                   " members)");
      break;
    }
  }
  if (h.properties().store_v2h && hs::v2h_rows(h) != transpose(want, n))
    out.mismatch(where + ": v2h differs from the transpose of the original's rows");
}

// The container's h2h against testing::rebuild_line_graph of its own h2v (M8b criterion 6; the
// oracle shares no code with the container).
template <typename weight_t>
void check_oracle(const hypergraph<std::int32_t, weight_t>& h, const std::string& where,
                  report& rep, recorder& out) {
  ++rep.line_graph_oracle_states;
  const std::string problems = hs::check_line_graph(h);
  if (!problems.empty()) out.mismatch(where + ": the line-graph oracle: " + problems);
}

// The kernel-path counters of a container.
void record_paths(const storage_stats& st, report& rep) {
  const storage_path_counters* families[3] = {&st.fill, &st.reuse, &st.unfill};
  for (int f = 0; f < 3; ++f) {
    rep.paths[3 * f] = families[f]->thread;
    rep.paths[3 * f + 1] = families[f]->warp;
    rep.paths[3 * f + 2] = families[f]->block;
  }
}

// ---- ESCHER-GPU ---------------------------------------------------------------------------

class escher_gpu_replay {
 public:
  escher_gpu_replay(tr::reader& in, const resources& res, const options& opt, report& rep)
      : in_(in), res_(res), opt_(opt), rep_(rep), out_(rep, opt.max_mismatches) {}

  void on(const tr::record& rec) {
    if (rec.kind == tr::record_kind::tree) {
      capacity_[rec.text("dataset_name")] = rec.integer("payload_capacity");
      tree_name_[static_cast<std::uint32_t>(rec.integer("tree"))] = rec.text("dataset_name");
      if (opt_.line_graph && rec.text("dataset_name") == "H2H") {
        h2h_tree_ = static_cast<std::uint32_t>(rec.integer("tree"));
        has_h2h_tree_ = true;
        (void)tracker_.apply(rec, in_);
      }
      return;
    }
    if (rec.kind == tr::record_kind::op) {
      // ESCHER-GPU's deviceBytes() of the tree after the operation (its op records).
      const auto tree = static_cast<std::uint32_t>(rec.integer("tree"));
      const std::string& name = tree_name_[tree];
      if (rec.has("after.device_bytes")) tree_bytes_[name] = rec.integer("after.device_bytes");
      if (opt_.line_graph && has_h2h_tree_ && tree == h2h_tree_) on_h2h_op(rec);
      return;
    }
    if (rec.kind == tr::record_kind::header) {
      (void)decoder_.feed(rec, in_);
      if (opt_.legacy_generator) {
        try {
          argv_options_ = ed::escher_options::parse(decoder_.argv());
          argv_ok_ = true;
        } catch (const tr::trace_error& e) {
          out_.mismatch(std::string("the legacy generator: ") + e.what());
        }
      }
      return;
    }
    if (rec.kind != tr::record_kind::batch) return;
    const ed::decoder::event ev = decoder_.feed(rec, in_);
    if (ev == ed::decoder::event::exception) {
      ++rep_.original_exceptions;
      out_.mismatch(rec.text("hook") + " " + rec.text("phase") + ": the original threw " +
                    decoder_.exception() + " (not replayed)");
      stopped_ = true;
      return;
    }
    if (stopped_) return;
    if (ev == ed::decoder::event::build) build(rec);
    if (ev == ed::decoder::event::batch) update(rec);
  }

 private:
  void build(const tr::record& rec) {
    ++rep_.sessions;
    model_ = decoder_.build().loaded;
    n_ = decoder_.build().num_vertices;
    hg_list_.reset();
    if (!opt_.hg_file.empty()) check_hg_file();
    if (opt_.legacy_generator && argv_ok_) start_generator();
    // ESCHER-GPU's payloadCapacity: max(requested, size + size / 8 + 1024) for H2V and V2H; the
    // requested capacity (--payload-capacity) is recovered from the trace's H2V tree.
    const std::int64_t h2v_size = flat_size(model_);
    const std::int64_t formula = h2v_size + h2v_size / 8 + 1024;
    const std::int64_t traced = capacity_.count("H2V") != 0 ? capacity_["H2V"] : formula;
    const std::int64_t requested = traced > formula ? traced : 0;
    props_.line_graph = line_graph_policy::none;
    props_.reuse = id_reuse::erase_first;
    props_.capacity = capacity_policy::escher_gpu(requested);
    const auto list = hs::list_of(n_, model_);
    using hgraph = hypergraph<std::int32_t, unweighted>;
    if (opt_.mode != "by_content") {
      hypergraph_properties by_id_props = props_;
      if (opt_.line_graph) by_id_props.line_graph = line_graph_policy::sorted_merge;
      const std::size_t mark = profiled_samples();
      // With --hg the by-id container is built from the file through the public reader (its
      // rows equal A's loaded rows, checked above), as a user loads coauth-DBLP (M8b step 5:
      // the memory gate through read_hypergraph).
      const auto view = hg_list_ ? hg_list_->view() : list.view();
      by_id_ = std::make_unique<hgraph>(hgraph::from_hyperedges(res_, view, by_id_props));
      record_stages(mark, 0);
      if (opt_.model) model_lockstep_ = std::make_unique<hs::lockstep>(n_, model_, props_);
      rep_.h2v_capacity_trace = traced;
      rep_.h2v_capacity_dyng = by_id_->store_stats().payload_capacity;
      if (rep_.h2v_capacity_dyng != traced)
        out_.mismatch("build: the H2V payload capacity " + std::to_string(rep_.h2v_capacity_dyng) +
                      " is not the trace's " + std::to_string(traced));
      record_bytes(rec, "build");
      if (opt_.compare) check(*by_id_, "build");
      if (opt_.line_graph) compare_h2h(*by_id_, "build");
    }
    if (opt_.mode != "by_id" && opt_.compare) {
      by_content_ = std::make_unique<hgraph>(hgraph::from_hyperedges(res_, list.view(), props_));
      resolver_.reset(model_);
      content_ids_in_sync_ = true;
    }
  }

  // The .hg file of the run read with ESCHER-GPU's option set: A's loaded rows (criterion 13).
  void check_hg_file() {
    auto list = io::read_hypergraph<std::int32_t, unweighted>(
        opt_.hg_file, io::hypergraph_read_options::escher_gpu());
    rep_.hg_hyperedges += list.num_hyperedges();
    if (static_cast<std::int32_t>(list.num_vertices) != n_)
      out_.mismatch("the .hg file: " + std::to_string(list.num_vertices) + " vertices, A " +
                    std::to_string(n_));
    if (hs::rows_of_list(list) != model_)
      out_.mismatch("the .hg file: the rows read with escher_gpu() differ from A's loaded rows (" +
                    std::to_string(list.num_hyperedges()) + " vs " + std::to_string(model_.size()) +
                    " hyperedges)");
    hg_list_ = std::make_unique<hyperedge_list<std::int32_t, unweighted>>(std::move(list));
  }

  // The legacy generators (criterion 16): the synthetic hypergraph from one
  // std::mt19937_64(--seed), and the generator that draws every batch from it.
  void start_generator() {
    const ed::escher_options& a = argv_options_;
    rng_ = std::mt19937_64(a.seed);
    namespace lg = generators::legacy;
    lg::escher_hypergraph_options g;
    if (!a.from_file) {
      g.num_hyperedges = a.num_hyperedges;
      g.max_cardinality = a.max_vertices_per_edge;
      g.min_vertex = a.min_vertex_id - 1;
      g.max_vertex = a.max_vertex_id - 1;
      const auto list = lg::escher_hypergraph<std::int32_t, unweighted>(rng_, g);
      ++rep_.generated_graphs;
      if (hs::rows_of_list(list) != model_ || static_cast<std::int32_t>(list.num_vertices) != n_)
        out_.mismatch("the legacy generator: escher_hypergraph differs from A's loaded rows");
    } else {
      // A's driver fills its graph parameters from the loaded file.
      g.max_cardinality = 1;
      for (const auto& r : model_)
        g.max_cardinality =
            std::max<std::int32_t>(g.max_cardinality, static_cast<std::int32_t>(r.size()));
      g.min_vertex = 0;
      g.max_vertex = n_ - 1;
    }
    g.num_hyperedges = 0;
    lg::escher_batch_options b;
    b.num_deletions = a.composition_set ? a.deletions : -1;
    b.num_insertions = a.composition_set ? a.insertions : -1;
    b.insert_mode =
        a.sample_inserts ? lg::escher_insert_mode::sample : lg::escher_insert_mode::random;
    b.graph = g;
    generator_ = std::make_unique<lg::escher_batch_generator>(rng_, b);
  }

  // The batch the legacy generator draws from the by-id container before A's batch is applied.
  void check_generated(const ed::batch_record& b, const std::string& where) {
    if (!generator_ || !by_id_) return;
    const auto gen = generator_->next(res_, *by_id_);
    ++rep_.generated_batches;
    rows inserted;
    for (std::size_t i = 0; i < gen.num_insertions(); ++i)
      inserted.emplace_back(gen.insert_members().begin() + gen.insert_offsets()[i],
                            gen.insert_members().begin() + gen.insert_offsets()[i + 1]);
    const rows got_rows = hs::h2v_rows(*by_id_);
    rows deleted_rows;
    for (const std::int64_t id : gen.delete_hyperedges())
      deleted_rows.push_back(static_cast<std::size_t>(id) < got_rows.size()
                                 ? got_rows[static_cast<std::size_t>(id)]
                                 : std::vector<std::int32_t>{});
    if (gen.delete_hyperedges() != b.deleted_ids || deleted_rows != b.deleted_rows ||
        inserted != b.inserted_rows) {
      out_.mismatch(where + ": the legacy generator's batch differs from A's (" +
                    std::to_string(gen.num_deletions()) + "/" +
                    std::to_string(gen.num_insertions()) + " vs " +
                    std::to_string(b.deleted_ids.size()) + "/" +
                    std::to_string(b.inserted_rows.size()) + ")");
    } else {
      ++rep_.generated_batches_equal;
    }
  }

  // The profiler's samples so far (0 without a profiler).
  std::size_t profiled_samples() const {
    const profiler* prof = res_.get_profiler();
    return prof != nullptr ? prof->samples().size() : 0;
  }

  // The stages completed since `mark`, as the samples of `batch` (the timing file of the maintain
  // reading: the same records as ESCHER-GPU's --timing).
  void record_stages(std::size_t mark, std::int64_t batch) {
    const profiler* prof = res_.get_profiler();
    if (prof == nullptr) return;
    for (std::size_t i = mark; i < prof->samples().size(); ++i) {
      const stage_sample& sample = prof->samples()[i];
      rep_.stage_times.push_back({sample.name, batch, sample.host_ms});
    }
  }

  // The container's device bytes next to A's DynamicHypergraph::deviceBytes() at the same point;
  // with the line graph both are in A's whole scope, and the ratio is the memory gate (<= 1.05x,
  // PLAN 8.6; criterion 8).
  void record_bytes(const tr::record& rec, const std::string& where) {
    const std::int64_t mine = by_id_->store_stats().device_bytes;
    const std::int64_t theirs = rec.integer("device_bytes", -1);
    rep_.device_bytes.emplace_back(mine, theirs);
    rep_.original_store_bytes.push_back(original_store_bytes(rec));
    if (opt_.line_graph && theirs > 0) {
      const double ratio = static_cast<double>(mine) / static_cast<double>(theirs);
      rep_.memory_ratio_max = std::max(rep_.memory_ratio_max, ratio);
      if (ratio > 1.05) {
        out_.mismatch(where + ": device bytes " + std::to_string(mine) + " are " +
                      std::to_string(ratio) + "x ESCHER-GPU's " + std::to_string(theirs) +
                      " (the memory gate is 1.05x)");
      }
    }
  }

  // An operation of A's H2H tree: the hashes of the layout its construct or rebuild received (A's
  // flatHyperedgeAdjacency or its delta merge), and its state digest.
  void on_h2h_op(const tr::record& rec) {
    const std::string op = rec.text("op");
    if (op == "construct" || op == "rebuild") {
      pending_ = layout_hashes{};
      pending_.valid = true;
      if (!rec.has("in.start_offsets") || !rec.has("in.flat_payload")) {
        // A batch extract (parity/escher_trace.py extract-batches) keeps no operation inputs.
        pending_.unrecorded = true;
        inputs_missing_ = true;
        return;
      }
      pending_.op = op;
      pending_.starts_count = static_cast<std::int64_t>(rec.array("in.start_offsets").count);
      pending_.starts_hash = array_hash(rec.array("in.start_offsets"));
      pending_.payload_count = static_cast<std::int64_t>(rec.array("in.flat_payload").count);
      pending_.payload_hash = array_hash(rec.array("in.flat_payload"));
    }
    if (rec.has("digest.mode")) {
      digest_fresh_ = true;
      for (const auto& problem : tracker_.apply(rec, in_))
        out_.mismatch("ESCHER-GPU's H2H digest after its " + op + ": " + problem);
    }
  }

  // The FNV-1a 64 hash of an int32 array field (inline values are hashed here; side-file and
  // hash-only arrays carry the recorded hash).
  static std::uint64_t array_hash(const tr::array_field& a) {
    if (a.kind == tr::array_field::storage::inline_data)
      return tr::fnv1a64(a.i32.data(), a.i32.size() * sizeof(std::int32_t));
    return a.hash;
  }

  // The container's line graph against A's after the same point: the layout (row offsets and
  // padded payload, ESCHER-GPU's construct layout) against the hashes of A's last H2H construct
  // or rebuild input, then, with a state digest, every key's row hash and the digest's rows.
  void compare_h2h(const hypergraph<std::int32_t, unweighted>& h, const std::string& where) {
    ++rep_.h2h_states;
    const detail::hyper::line_graph_layout layout =
        detail::hypergraph_access::store(h).line_graph();
    if (!pending_.valid && !inputs_missing_) {
      out_.mismatch(where + ": h2h: ESCHER-GPU built or rebuilt no H2H before this point");
      return;
    }
    if (!pending_.valid || pending_.unrecorded) {
      // An extract keeps neither the H2H inputs nor (some of) its operations.
      ++rep_.h2h_unrecorded;
      pending_.valid = false;
      return;
    }
    std::vector<std::int32_t> starts = layout.starts, counts = layout.counts,
                              payload = layout.payload;
    // A key count below A's (not expected: both are the largest id bound so far) would be empty
    // rows of 4 slots at the end.
    while (static_cast<std::int64_t>(starts.size()) < pending_.starts_count) {
      starts.push_back(static_cast<std::int32_t>(payload.size()));
      counts.push_back(0);
      payload.insert(payload.end(), {0, 0, 0, INT_MIN});
    }
    const bool equal = static_cast<std::int64_t>(starts.size()) == pending_.starts_count &&
                       static_cast<std::int64_t>(payload.size()) == pending_.payload_count &&
                       tr::fnv1a64(starts.data(), starts.size() * 4) == pending_.starts_hash &&
                       tr::fnv1a64(payload.data(), payload.size() * 4) == pending_.payload_hash;
    if (equal) {
      ++rep_.h2h_layouts_equal;
      rep_.h2h_payload_entries += static_cast<std::int64_t>(payload.size());
    } else {
      out_.mismatch(where + ": h2h: the layout differs from ESCHER-GPU's H2H " + pending_.op +
                    " input (" + std::to_string(starts.size()) + " keys and " +
                    std::to_string(payload.size()) +
                    " payload entries; A: " + std::to_string(pending_.starts_count) + " and " +
                    std::to_string(pending_.payload_count) + ")");
    }
    pending_.valid = false;
    const tr::tree_state* t = tracker_.tree(h2h_tree_);
    if (!digest_fresh_ || t == nullptr) return;
    digest_fresh_ = false;
    bool reported = false;
    for (const auto& [key, ks] : t->by_key) {
      std::vector<std::int32_t> row;
      if (key >= 1 && static_cast<std::size_t>(key) <= starts.size()) {
        const auto k = static_cast<std::size_t>(key) - 1;
        row.assign(payload.begin() + starts[k], payload.begin() + starts[k] + counts[k]);
      }
      ++rep_.h2h_key_hashes;
      bool same = tr::fnv1a64(row.data(), row.size() * 4) == ks.row_hash;
      if (ks.row_known) {
        ++rep_.h2h_full_rows;
        same = same && row == ks.row;
      }
      if (!same && !reported) {
        out_.mismatch(where + ": h2h: the row of id " + std::to_string(key - 1) + " differs (" +
                      std::to_string(row.size()) + " vs " + std::to_string(ks.row_len) +
                      " neighbours)");
        reported = true;
      }
    }
  }

  // A's bytes in the container's scope: the H2V and V2H trees and the H2V row view (its
  // RowLocator over keys 0..numHyperedgeKeys, 3 ints per key).
  std::int64_t original_store_bytes(const tr::record& rec) const {
    const auto h = tree_bytes_.find("H2V"), v = tree_bytes_.find("V2H");
    if (h == tree_bytes_.end() || v == tree_bytes_.end() || !rec.has("num_hyperedge_keys"))
      return -1;
    return h->second + v->second + (rec.integer("num_hyperedge_keys") + 1) * 3 * 4;
  }

  void check(const hypergraph<std::int32_t, unweighted>& h, const std::string& where) {
    compare_state(h, model_, n_, where, rep_, out_);
    if (opt_.line_graph && opt_.line_graph_oracle) check_oracle(h, where, rep_, out_);
    if (opt_.integrity) {
      try {
        h.check_integrity(res_);
      } catch (const dyng::error& e) {
        out_.mismatch(where + ": check_integrity: " + e.what());
      }
    }
    if (model_lockstep_) {
      const std::string diff = model_lockstep_->compare(h);
      if (!diff.empty()) out_.mismatch(where + ": the host model: " + diff);
    }
  }

  void update(const tr::record& rec) {
    ++rep_.batches;
    const ed::batch_record& b = decoder_.batch();
    const std::string where = "batch " + std::to_string(b.index);
    const std::vector<std::int64_t>& want_ids = b.inserted_ids;
    hs::test_batch tb;
    tb.deletions = b.deleted_ids;
    tb.insertions = b.inserted_rows;
    // A's model (its host vector h2v_): deleted keys cleared, inserted rows at their keys.
    for (const std::int64_t id : b.deleted_ids)
      if (id >= 0 && static_cast<std::size_t>(id) < model_.size())
        model_[static_cast<std::size_t>(id)].clear();
    for (std::size_t i = 0; i < want_ids.size(); ++i) {
      if (want_ids[i] < 0) continue;
      const auto id = static_cast<std::size_t>(want_ids[i]);
      if (id >= model_.size()) model_.resize(id + 1);
      model_[id] = sorted_unique(b.inserted_rows[i]);
    }
    if (by_id_) {
      if (generator_) check_generated(b, where);
      const auto batch = hs::batch_of(tb);
      const std::size_t mark = profiled_samples();
      const auto s = by_id_->apply(res_, batch.view());
      record_stages(mark, b.index);
      rep_.ids_compared += static_cast<std::int64_t>(want_ids.size());
      if (s.inserted_ids != want_ids)
        out_.mismatch(where + ": inserted_ids differ from insertedKeys - 1");
      if (s.ignored_deletions != 0 || s.dropped_members != 0)
        out_.mismatch(where + ": ESCHER-GPU's batch had skipped ops");
      if (model_lockstep_) {
        const auto expected = model_lockstep_->apply(tb);
        if (expected.ids != s.inserted_ids)
          out_.mismatch(where + ": the host model gives other ids");
      }
      record_bytes(rec, where);
      if (opt_.compare) check(*by_id_, where);
      if (opt_.line_graph) compare_h2h(*by_id_, where);
    }
    if (by_content_) content_update(b, where);
    if (by_id_) record_paths(by_id_->store_stats(), rep_);
  }

  // Deletion by content (escher_dump::by_content_resolver): every recorded deleted row resolved
  // to the smallest live id with that member set that the batch has not taken yet.
  void content_update(const ed::batch_record& b, const std::string& where) {
    hs::test_batch tb;
    tb.insertions = b.inserted_rows;
    std::vector<bool> ambiguous;
    const std::vector<std::int64_t> resolved = resolver_.resolve(b.deleted_rows, &ambiguous);
    bool ambiguous_here = false;
    for (std::size_t i = 0; i < resolved.size(); ++i) {
      if (resolved[i] < 0) {
        out_.mismatch(where + ": by content: deleted row " + std::to_string(i) +
                      " has no live hyperedge");
        continue;
      }
      if (ambiguous[i] && resolved[i] != b.deleted_ids[i]) {
        ++rep_.ambiguous;
        ambiguous_here = true;
      }
      tb.deletions.push_back(resolved[i]);
    }
    const auto batch = hs::batch_of(tb);
    const auto s = by_content_->apply(res_, batch.view());
    resolver_.commit(tb.deletions, tb.insertions, s.inserted_ids);
    if (ambiguous_here && rep_.first_ambiguous_batch < 0) rep_.first_ambiguous_batch = b.index;
    if (content_ids_in_sync_) {
      // Until the first ambiguous resolution the by-content run must give A's ids (M8b review:
      // the check sat behind the flag it cleared, so it never fired).
      if (!ambiguous_here && s.inserted_ids != b.inserted_ids)
        out_.mismatch(where + ": by content: the ids differ before any ambiguous resolution");
      if (ambiguous_here || s.inserted_ids != b.inserted_ids) {
        content_ids_in_sync_ = false;
        rep_.content_diverged_at = b.index;
      }
    }
    // Both modes give the same state: the same member sets, and while no ambiguous resolution
    // has happened, the same ids.
    const rows got = hs::h2v_rows(*by_content_);
    ++rep_.content_states;
    if (content_multiset(got) != content_multiset(model_)) {
      out_.mismatch(where + ": by content: the live member sets differ from the original's");
    } else {
      ++rep_.content_states_equal;
    }
    if (content_ids_in_sync_) {
      compare_state(*by_content_, model_, n_, where + " (by content)", rep_, out_);
    }
    if (opt_.integrity) {
      try {
        by_content_->check_integrity(res_);
      } catch (const dyng::error& e) {
        out_.mismatch(where + ": by content: check_integrity: " + e.what());
      }
    }
  }

  tr::reader& in_;
  resources res_;
  options opt_;
  report& rep_;
  recorder out_;
  std::map<std::string, std::int64_t> capacity_;
  std::map<std::uint32_t, std::string> tree_name_;
  std::map<std::string, std::int64_t> tree_bytes_;
  // The line graph's comparison: A's H2H tree, its digests and its last construct / rebuild input.
  struct layout_hashes {
    bool valid = false;
    bool unrecorded = false;  // the trace keeps no inputs of the operation
    std::string op;
    std::int64_t starts_count = 0, payload_count = 0;
    std::uint64_t starts_hash = 0, payload_hash = 0;
  };
  std::uint32_t h2h_tree_ = 0;
  bool has_h2h_tree_ = false, digest_fresh_ = false, inputs_missing_ = false;
  layout_hashes pending_;
  tr::state_tracker tracker_;
  hypergraph_properties props_;
  std::int32_t n_ = 0;
  rows model_;
  std::unique_ptr<hypergraph<std::int32_t, unweighted>> by_id_, by_content_;
  std::unique_ptr<hyperedge_list<std::int32_t, unweighted>> hg_list_;  // --hg: the file's list
  std::unique_ptr<hs::lockstep> model_lockstep_;
  ed::decoder decoder_;
  ed::by_content_resolver resolver_;
  ed::escher_options argv_options_;
  bool argv_ok_ = false;
  std::mt19937_64 rng_;
  std::unique_ptr<generators::legacy::escher_batch_generator> generator_;
  bool content_ids_in_sync_ = true;
  bool stopped_ = false;
};

// ---- MOSP_ESCHER --------------------------------------------------------------------------

class mosp_escher_replay {
 public:
  using whgraph = hypergraph<std::int32_t, std::int32_t>;

  mosp_escher_replay(tr::reader& in, const resources& res, const options& opt, report& rep)
      : in_(in), res_(res), opt_(opt), rep_(rep), out_(rep, opt.max_mismatches) {}

  void on(const tr::record& rec) {
    if (rec.kind == tr::record_kind::header) {
      const tr::field* f = rec.find("argv");
      if (f != nullptr) argv_ = f->texts;
      return;
    }
    if (rec.kind == tr::record_kind::tree) {
      tree_name_[static_cast<std::uint32_t>(rec.integer("tree"))] = rec.text("dataset_name");
      tree_capacity_[static_cast<std::uint32_t>(rec.integer("tree"))] =
          rec.integer("payload_capacity");
      return;
    }
    if (rec.kind == tr::record_kind::op) {
      on_op(rec);
      return;
    }
    if (rec.kind != tr::record_kind::batch) return;
    const std::string hook = rec.text("hook"), phase = rec.text("phase");
    if (hook == "mosp_escher.bulk_load" && phase == "enter") bulk_enter(rec);
    if (hook == "mosp_escher.bulk_load" && phase == "exit") bulk_exit(rec);
    if (hook == "mosp_escher.begin_batch" && phase == "enter") begin_enter(rec);
    if (hook == "mosp_escher.begin_batch" && phase == "exit") begin_exit(rec);
  }

 private:
  void on_op(const tr::record& rec) {
    const auto tree = static_cast<std::uint32_t>(rec.integer("tree"));
    const std::string op = rec.text("op");
    if (loading_ && op == "construct") {
      const std::string name = tree_name_[tree];
      if (name == "h2v") {
        h2v_tree_ = tree;
        h2v_flat_ = rec.integer("in.flat_payload_size");
        h2v_capacity_ = rec.integer("in.capacity", tree_capacity_[tree]);
      } else if (name == "v2h") {
        num_vertices_ = static_cast<std::int32_t>(rec.integer("in.num_records"));
      }
      sizing_.emplace_back(rec.integer("in.flat_payload_size"),
                           rec.integer("in.capacity", tree_capacity_[tree]));
    }
    if (awaiting_insert_ && op == "insert" && tree_name_[tree] == "h2v") {
      awaiting_insert_ = false;
      insert_tree_ = tree;
      insert_payload_ = static_cast<std::int64_t>(rec.array("in.payload").count);
    }
  }

  void bulk_enter(const tr::record& rec) {
    ++rep_.sessions;
    session_.reset();
    model_lockstep_.reset();
    stopped_ = false;
    loading_ = true;
    rows_ = read_rows(in_, rec, "rows", 0);
    weights_ = in_.longs(rec.array("weights"));
    source_ = rec.integer("source_he");
    target_ = rec.integer("target_he");
    num_vertices_ = 0;
    sizing_.clear();
  }

  // MOSP_ESCHER's Caps of the session (hypergraph/src/DynamicHypergraph.cpp:79-107: every tree
  // starts at floor(flat * headroomFactor) + extraPayloadInts, at most 2.0e9): the factor its
  // drivers and tests set (1.3 by default; 2.0, 1.25, 1.5 in its tests) for which the trees agree
  // on one extra, and that extra. capacity_policy::mosp_escher(extra) with initial_headroom =
  // factor - 1 expresses it exactly (size + floor(size * (factor - 1)) == floor(size * factor)
  // for these factors; CapacityPolicy tests).
  bool derive_caps(double& factor, std::int64_t& extra) const {
    for (const double f : {1.3, 2.0, 1.25, 1.5}) {
      bool consistent = !sizing_.empty();
      std::int64_t e = -1;
      for (const auto& [flat, capacity] : sizing_) {
        if (capacity >= 2000000000) continue;  // capped: no information
        const std::int64_t x = capacity - static_cast<std::int64_t>(static_cast<double>(flat) * f);
        if (x < 0 || (e >= 0 && x != e)) consistent = false;
        e = x;
      }
      if (consistent && e >= 0) {
        factor = f;
        extra = e;
        return true;
      }
    }
    return false;
  }

  void bulk_exit(const tr::record& rec) {
    loading_ = false;
    if (rec.has("exception.type")) {
      ++rep_.original_exceptions;
      stopped_ = true;
      return;
    }
    // B's virtual source and target are ordinary hyperedges here (weight 0 loaded as 1); every
    // other weight is loaded as B's, which must lie in [1, 2^31 - 1].
    model_.clear();
    for (const auto& r : rows_) model_.push_back(sorted_unique(r));
    std::vector<std::int32_t> w;
    for (std::size_t i = 0; i < weights_.size(); ++i) {
      const long long x = weights_[i];
      const auto b_id = static_cast<std::int64_t>(i) + 1;
      if (b_id == source_ || b_id == target_) {
        w.push_back(x < 1 ? 1 : static_cast<std::int32_t>(x));
      } else if (x < 1 || x > INT_MAX) {
        out_.mismatch("session " + std::to_string(rep_.sessions) + ": B's weight " +
                      std::to_string(x) + " of hyperedge " + std::to_string(b_id) +
                      " is outside [1, 2^31 - 1]");
        w.push_back(1);
      } else {
        w.push_back(static_cast<std::int32_t>(x));
      }
    }
    double factor = 1.3;
    std::int64_t extra = 0;
    if (!derive_caps(factor, extra)) {
      out_.mismatch("session " + std::to_string(rep_.sessions) +
                    ": the trees' capacities fit no Caps of MOSP_ESCHER");
      stopped_ = true;
      return;
    }
    // --line-graph: B's sessions keep the line graph (sorted_merge under insert_first), checked
    // against the oracle after the load and every batch (M8b criterion 6).
    props_.line_graph = opt_.line_graph ? line_graph_policy::sorted_merge : line_graph_policy::none;
    props_.reuse = id_reuse::insert_first;
    props_.capacity = capacity_policy::mosp_escher(extra);
    // 1.3 - 1.0 is not 0.3 in binary; the preset's 0.3 is the value proven equal (the other
    // factors subtract exactly).
    props_.capacity.initial_headroom = factor == 1.3 ? 0.3 : factor - 1.0;
    // The container is built from B's raw rows (repeated members included): dynG's own
    // normalization must give B's (sorted, deduplicated) rows, which check() compares.
    const auto list = hs::list_of<std::int32_t>(num_vertices_, rows_, w);
    if (!opt_.hg_file.empty() && !hg_checked_) check_hg_file();
    session_ = std::make_unique<whgraph>(whgraph::from_hyperedges(res_, list.view(), props_));
    // The loaded weights by id (B id = dynG id + 1) are B's.
    if (hs::host_vector(session_->weights()) != w)
      out_.mismatch("session " + std::to_string(rep_.sessions) +
                    ": the container's weights differ from B's loaded weights");
    if (opt_.model)
      model_lockstep_ =
          std::make_unique<hs::lockstep>(num_vertices_, model_, props_, "mosp_escher");
    rep_.h2v_capacity_trace = h2v_capacity_;
    rep_.h2v_capacity_dyng = session_->store_stats().payload_capacity;
    if (rep_.h2v_capacity_dyng != h2v_capacity_)
      out_.mismatch("session " + std::to_string(rep_.sessions) + ": the h2v payload capacity " +
                    std::to_string(rep_.h2v_capacity_dyng) + " is not the trace's " +
                    std::to_string(h2v_capacity_));
    check("session " + std::to_string(rep_.sessions) + " load");
  }

  // The .hg file of an hsospBench --hg run read with H-SOSP's option set (criterion 13): B's
  // loaded rows without its virtual source (first) and target (last) hyperedges, in B's member
  // order, its first-appearance numbering, its vertex count and its weights, bit for bit.
  void check_hg_file() {
    hg_checked_ = true;
    std::uint64_t seed = 20260725U;  // hsospBench's defaults
    std::int64_t maxcard = 25;
    bool hg_run = false;
    for (std::size_t i = 1; i + 1 < argv_.size(); ++i) {
      if (argv_[i] == "--seed") seed = std::stoull(argv_[i + 1]);
      if (argv_[i] == "--maxcard") maxcard = std::stoll(argv_[i + 1]);
      if (argv_[i] == "--hg") hg_run = true;
    }
    if (!hg_run) {
      out_.mismatch("--hg: the trace's run did not load a .hg file");
      return;
    }
    const auto list = io::read_hypergraph<std::int32_t, std::int32_t>(
        opt_.hg_file, io::hypergraph_read_options::hsosp(seed, maxcard));
    rep_.hg_hyperedges += list.num_hyperedges();
    if (rows_.size() < 2) {
      out_.mismatch("--hg: the bulk load has no virtual hyperedges");
      return;
    }
    const rows want(rows_.begin() + 1, rows_.end() - 1);
    const std::vector<std::int64_t> want_w(weights_.begin() + 1, weights_.end() - 1);
    if (hs::rows_of_list(list) != want)
      out_.mismatch("--hg: the rows read with hsosp() differ from B's loaded rows (" +
                    std::to_string(list.num_hyperedges()) + " vs " + std::to_string(want.size()) +
                    " hyperedges)");
    if (static_cast<std::int32_t>(list.num_vertices) != num_vertices_)
      out_.mismatch("--hg: " + std::to_string(list.num_vertices) + " vertices, B " +
                    std::to_string(num_vertices_));
    if (std::vector<std::int64_t>(list.weights.begin(), list.weights.end()) != want_w)
      out_.mismatch("--hg: the weights drawn by hsosp() differ from B's");
  }

  void begin_enter(const tr::record& rec) {
    pending_ = false;
    counting_after_stop_ = false;
    const bool vertex_kind =
        rec.array("vtx_delete.he").count != 0 || rec.array("vtx_insert.he").count != 0;
    // Every vertex-kind batch is counted, and every hyperedge-kind batch that comes after a
    // session stopped (with its ids, at its exit): what the replay does not compare (M11).
    if (vertex_kind) ++rep_.vertex_batches;
    if (!session_ || stopped_) {
      if (!vertex_kind && stopped_) {
        ++rep_.hyperedge_batches_after_stop;
        counting_after_stop_ = true;
      }
      return;
    }
    if (vertex_kind) {
      // A vertex-kind batch (incidence edits, M11): the session stops here.
      stopped_ = true;
      ++rep_.stopped_sessions;
      return;
    }
    pending_ = true;
    he_delete_ = in_.ints(rec.array("he_delete"));
    he_insert_ = read_rows(in_, rec, "he_insert", 0);
    he_weights_ = in_.longs(rec.array("he_insert.weight"));
    awaiting_insert_ = !he_insert_.empty();
    insert_payload_ = 0;
    insert_tree_ = h2v_tree_;
  }

  void begin_exit(const tr::record& rec) {
    if (counting_after_stop_) {
      counting_after_stop_ = false;
      if (!rec.has("exception.type"))
        rep_.ids_after_stop += static_cast<std::int64_t>(rec.array("final_ids").count);
      return;
    }
    if (!pending_) return;
    pending_ = false;
    ++rep_.batches;
    const std::string where =
        "session " + std::to_string(rep_.sessions) + " batch " + std::to_string(rep_.batches);
    hs::test_batch tb;
    std::int64_t filtered = 0;
    for (const std::int32_t id : he_delete_) {
      if (id == source_ || id == target_) {
        ++filtered;  // B skips ops on its virtual hyperedges
        continue;
      }
      tb.deletions.push_back(static_cast<std::int64_t>(id) - 1);
    }
    rep_.filtered_virtual += filtered;
    tb.insertions = he_insert_;
    for (const long long w : he_weights_) tb.weights.push_back(static_cast<std::int32_t>(w));
    const auto batch = hs::batch_of<std::int32_t>(tb);
    if (rec.has("exception.type")) {
      // B rejected the batch before changing anything (its validation); so must dynG.
      ++rep_.original_exceptions;
      bool threw = false;
      try {
        (void)session_->apply(res_, batch.view());
      } catch (const dyng::invalid_argument_error&) {
        threw = true;
      }
      if (!threw)
        out_.mismatch(where + ": B threw " + rec.text("exception.type") + ", dynG did not");
      return;
    }
    if (insert_tree_ != h2v_tree_) out_.mismatch(where + ": the batch's insert is on another tree");
    const auto s = session_->apply(res_, batch.view());
    const std::vector<std::int32_t> final_ids = in_.ints(rec.array("final_ids"));
    std::vector<std::int64_t> want;
    for (const std::int32_t id : final_ids) want.push_back(static_cast<std::int64_t>(id) - 1);
    rep_.ids_compared += static_cast<std::int64_t>(want.size());
    if (s.inserted_ids != want) out_.mismatch(where + ": inserted_ids differ from finalIds - 1");
    ++rep_.skipped_compared;
    rep_.skipped_ops += rec.integer("skipped_ops");
    if (s.ignored_deletions + filtered != rec.integer("skipped_ops"))
      out_.mismatch(where + ": ignored deletions " + std::to_string(s.ignored_deletions) +
                    " + filtered " + std::to_string(filtered) + " != skippedOps " +
                    std::to_string(rec.integer("skipped_ops")));
    std::int64_t raw = 0;
    for (const auto& r : he_insert_) raw += static_cast<std::int64_t>(r.size());
    rep_.dropped_members += s.dropped_members;
    if (s.dropped_members != raw - insert_payload_)
      out_.mismatch(where + ": dropped members " + std::to_string(s.dropped_members) +
                    " != B's deduplicated " + std::to_string(raw - insert_payload_));
    // B's model: dead_he cleared, the new hyperedges at their final ids.
    for (const std::int32_t id : in_.ints(rec.array("dead_he")))
      if (id >= 1 && static_cast<std::size_t>(id) <= model_.size())
        model_[static_cast<std::size_t>(id) - 1].clear();
    for (std::size_t i = 0; i < final_ids.size(); ++i) {
      const auto id = static_cast<std::size_t>(final_ids[i]);
      if (id > model_.size()) model_.resize(id);
      model_[id - 1] = sorted_unique(he_insert_[i]);
    }
    // The weights of the new ids (B's final_weights).
    const std::vector<std::int64_t> final_weights = in_.longs(rec.array("final_weights"));
    const auto weights = hs::host_vector(session_->weights());
    if (final_weights.size() != want.size())
      out_.mismatch(where + ": B's trace has " + std::to_string(final_weights.size()) +
                    " final weights for " + std::to_string(want.size()) + " final ids");
    for (std::size_t i = 0; i < want.size() && i < final_weights.size(); ++i)
      if (weights[static_cast<std::size_t>(want[i])] != final_weights[i])
        out_.mismatch(where + ": the weight of id " + std::to_string(want[i]) + " differs");
    if (model_lockstep_) {
      const auto expected = model_lockstep_->apply(tb);
      if (expected.ids != s.inserted_ids) out_.mismatch(where + ": the host model gives other ids");
    }
    check(where);
  }

  void check(const std::string& where) {
    compare_state(*session_, model_, num_vertices_, where, rep_, out_);
    if (opt_.line_graph && opt_.line_graph_oracle) check_oracle(*session_, where, rep_, out_);
    if (opt_.integrity) {
      try {
        session_->check_integrity(res_);
      } catch (const dyng::error& e) {
        out_.mismatch(where + ": check_integrity: " + e.what());
      }
    }
    if (model_lockstep_) {
      const std::string diff = model_lockstep_->compare(*session_);
      if (!diff.empty()) out_.mismatch(where + ": the host model: " + diff);
    }
  }

  tr::reader& in_;
  resources res_;
  options opt_;
  report& rep_;
  recorder out_;
  std::map<std::uint32_t, std::string> tree_name_;
  std::map<std::uint32_t, std::int64_t> tree_capacity_;
  bool loading_ = false, stopped_ = false, pending_ = false, awaiting_insert_ = false;
  bool counting_after_stop_ = false;
  std::uint32_t h2v_tree_ = 0, insert_tree_ = 0;
  std::int64_t h2v_flat_ = 0, h2v_capacity_ = 0, insert_payload_ = 0;
  std::int32_t num_vertices_ = 0;
  std::int64_t source_ = 0, target_ = 0;
  rows rows_, model_, he_insert_;
  std::vector<std::pair<std::int64_t, std::int64_t>> sizing_;  // (flat, capacity) per tree
  std::vector<std::int64_t> weights_, he_weights_;
  std::vector<std::int32_t> he_delete_;
  hypergraph_properties props_;
  std::unique_ptr<whgraph> session_;
  std::unique_ptr<hs::lockstep> model_lockstep_;
  std::vector<std::string> argv_;
  bool hg_checked_ = false;
};

}  // namespace

std::string report::to_json() const {
  std::ostringstream os;
  os << "{\"trace\": " << json_string(trace) << ", \"repo\": " << json_string(repo)
     << ", \"backend\": " << json_string(backend) << ", \"threads\": " << threads
     << ", \"mode\": " << json_string(mode) << ", \"ok\": " << (ok() ? "true" : "false")
     << ", \"sessions\": " << sessions << ", \"batches\": " << batches
     << ", \"vertex_batches\": " << vertex_batches << ", \"stopped_sessions\": " << stopped_sessions
     << ", \"hyperedge_batches_after_stop\": " << hyperedge_batches_after_stop
     << ", \"ids_after_stop\": " << ids_after_stop
     << ", \"line_graph_oracle_states\": " << line_graph_oracle_states
     << ", \"first_ambiguous_batch\": " << first_ambiguous_batch
     << ", \"ids_compared\": " << ids_compared << ", \"rows_compared\": " << rows_compared
     << ", \"filtered_virtual\": " << filtered_virtual
     << ", \"skipped_compared\": " << skipped_compared << ", \"skipped_ops\": " << skipped_ops
     << ", \"dropped_members\": " << dropped_members << ", \"ambiguous\": " << ambiguous
     << ", \"content_diverged_at\": " << content_diverged_at
     << ", \"generated_graphs\": " << generated_graphs
     << ", \"generated_batches\": " << generated_batches
     << ", \"generated_batches_equal\": " << generated_batches_equal
     << ", \"hg_hyperedges\": " << hg_hyperedges << ", \"content_states\": " << content_states
     << ", \"content_states_equal\": " << content_states_equal
     << ", \"original_exceptions\": " << original_exceptions
     << ", \"h2v_capacity_trace\": " << h2v_capacity_trace
     << ", \"h2v_capacity_dyng\": " << h2v_capacity_dyng << ", \"h2h_states\": " << h2h_states
     << ", \"h2h_layouts_equal\": " << h2h_layouts_equal
     << ", \"h2h_unrecorded\": " << h2h_unrecorded
     << ", \"h2h_payload_entries\": " << h2h_payload_entries
     << ", \"h2h_key_hashes\": " << h2h_key_hashes << ", \"h2h_full_rows\": " << h2h_full_rows
     << ", \"memory_ratio_max\": " << memory_ratio_max << ", \"seconds\": " << seconds
     << ", \"paths\": [" << paths[0] << ", " << paths[1] << ", " << paths[2] << ", " << paths[3]
     << ", " << paths[4] << ", " << paths[5] << ", " << paths[6] << ", " << paths[7] << ", "
     << paths[8] << "]"
     << ", \"device_bytes\": [";
  for (std::size_t i = 0; i < device_bytes.size(); ++i)
    os << (i ? ", " : "") << "[" << device_bytes[i].first << ", " << device_bytes[i].second << "]";
  os << "], \"original_store_bytes\": [";
  for (std::size_t i = 0; i < original_store_bytes.size(); ++i)
    os << (i ? ", " : "") << original_store_bytes[i];
  os << "], \"mismatch_count\": " << mismatch_count << ", \"mismatches\": [";
  for (std::size_t i = 0; i < mismatches.size(); ++i)
    os << (i ? ", " : "") << json_string(mismatches[i]);
  os << "]}";
  return os.str();
}

report replay(tr::reader& in, const resources& res, const options& opt, const std::string& name) {
  const auto start = std::chrono::steady_clock::now();
  report rep;
  rep.trace = name;
  rep.backend = std::string(to_string(res.get_backend()));
  rep.threads = res.num_threads();
  rep.mode = opt.mode;
  tr::record rec;
  std::unique_ptr<escher_gpu_replay> a;
  std::unique_ptr<mosp_escher_replay> b;
  while (in.next(rec)) {
    if (rec.kind == tr::record_kind::header) {
      rep.repo = rec.text("repo");
      if (rep.repo == "ESCHER-GPU") {
        a = std::make_unique<escher_gpu_replay>(in, res, opt, rep);
        a->on(rec);  // the command line (the generator check, the .hg check)
      }
      if (rep.repo == "MOSP_ESCHER") {
        options b_opt = opt;
        b_opt.mode = "by_id";  // B deletes by id (its own replay rule)
        rep.mode = "by_id";
        b = std::make_unique<mosp_escher_replay>(in, res, b_opt, rep);
        b->on(rec);  // the command line (the .hg check)
      }
      continue;
    }
    if (a) a->on(rec);
    if (b) b->on(rec);
  }
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
  return rep;
}

}  // namespace dyng::test::hypergraph_ids
