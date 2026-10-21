// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay.cpp
 * @brief The triad_count replay of ESCHER-GPU's golden set (triad_replay.hpp; ADR 0050).
 */
#include "parity/escher/triad_replay.hpp"

#include "parity/escher/escher_dump.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/io/batch_io.hpp>
#include <dyng/io/hypergraph_io.hpp>
#if DYNG_TRIAD_REPLAY_ALGORITHM
#include "algorithms/triad_count/problem.hpp"

#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>
#endif

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <memory>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace dyng::test::triad_replay {

namespace {

namespace ed = dyng::test::escher_dump;
using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using host_batch = dyng::hyperedge_batch<std::int32_t, dyng::unweighted>;
using batch_view = dyng::hyperedge_batch_view<std::int32_t, dyng::unweighted>;

/**
 * The counter evaluated after the build and after every batch. apply() applies the batch (a
 * counter that maintains counts applies it through dyng::update) and returns the summary; counts()
 * gives the counts of the current state.
 */
class counter {
 public:
  counter() = default;
  counter(const counter&) = delete;
  counter& operator=(const counter&) = delete;
  counter(counter&&) = delete;
  counter& operator=(counter&&) = delete;
  virtual ~counter() = default;
  virtual void build(const resources& res, hgraph& h) = 0;
  virtual hyper_apply_summary apply(const resources& res, hgraph& h, const batch_view& b) = 0;
  virtual pattern_counts counts(const resources& res, const hgraph& h) = 0;
  /// The launch counters per (mode, shape) recorded so far (the hook of M9a steps 3-4).
  virtual void launches(std::map<std::string, std::int64_t>& out) const {
    (void)out;
  }
  std::int64_t evaluations = 0;
};

/// The last list the oracle counted and its counts, shared by the modes' counters.
struct oracle_cache {
  bool valid = false;
  std::vector<std::int64_t> offsets;
  std::vector<std::int32_t> members;
  pattern_counts counts;
};

/// The oracle on the live hyperedges copied to the host. An evaluation of a list equal to the
/// one counted last (the other mode's container in the same state) reuses its counts: the oracle
/// is a function of the list.
class oracle_counter final : public counter {
 public:
  oracle_counter(const options& opt, std::shared_ptr<oracle_cache> cache)
      : opt_(opt), cache_(std::move(cache)) {}
  void build(const resources& res, hgraph& h) override {
    (void)res;
    (void)h;
  }
  hyper_apply_summary apply(const resources& res, hgraph& h, const batch_view& b) override {
    return h.apply(res, b);
  }
  pattern_counts counts(const resources& res, const hgraph& h) override {
    const auto list = h.to_hyperedge_list(res);
    if (cache_->valid && list.offsets == cache_->offsets && list.members == cache_->members)
      return cache_->counts;
    ++evaluations;
    const auto all =
        testing::triad_reference_all(list.offsets, list.members, list.num_vertices, opt_.patterns,
                                     testing::triad_reference_options{opt_.oracle_threads});
    pattern_counts out;
    for (const auto p : opt_.patterns) {
      std::vector<std::int64_t> v;
      for (const std::uint64_t x : all.of(p)) v.push_back(static_cast<std::int64_t>(x));
      out[testing::triad_pattern_name(p)] = std::move(v);
    }
    cache_->offsets = list.offsets;
    cache_->members = list.members;
    cache_->counts = out;
    cache_->valid = true;
    return out;
  }

 private:
  const options& opt_;
  std::shared_ptr<oracle_cache> cache_;
};

#if DYNG_TRIAD_REPLAY_ALGORITHM
/// The pattern of triad_count for an oracle pattern.
triad_count::pattern kind_of(testing::triad_pattern p) {
  switch (p) {
    case testing::triad_pattern::h_motif:
      return triad_count::pattern::h_motif;
    case testing::triad_pattern::stathyper_type1:
      return triad_count::pattern::stathyper_type1;
    case testing::triad_pattern::stathyper_type2:
      return triad_count::pattern::stathyper_type2;
    case testing::triad_pattern::stathyper_type3:
      return triad_count::pattern::stathyper_type3;
  }
  throw std::invalid_argument("unknown pattern");
}

/// triad_count: one result per requested pattern, computed after the build and kept up to date by
/// one dyng::update() per batch (all results, one commit); after every batch the counts must equal
/// a fresh compute() ("baseline + deltas == recount"; unless opt.recount is off).
class algorithm_counter final : public counter {
 public:
  explicit algorithm_counter(const options& opt) : opt_(opt) {}
  void build(const resources& res, hgraph& h) override {
    results_.clear();
    for (const auto p : opt_.patterns) {
      triad_count::options o;
      o.kind = kind_of(p);
      results_.push_back(triad_count::compute(res, h, o));
    }
  }
  hyper_apply_summary apply(const resources& res, hgraph& h, const batch_view& b) override {
    std::vector<triad_count::stats> st;
    switch (results_.size()) {
      case 1: {
        const auto [s0] = dyng::update(res, h, b, results_[0]);
        st = {s0};
        break;
      }
      case 2: {
        const auto [s0, s1] = dyng::update(res, h, b, results_[0], results_[1]);
        st = {s0, s1};
        break;
      }
      case 3: {
        const auto [s0, s1, s2] = dyng::update(res, h, b, results_[0], results_[1], results_[2]);
        st = {s0, s1, s2};
        break;
      }
      case 4: {
        const auto [s0, s1, s2, s3] =
            dyng::update(res, h, b, results_[0], results_[1], results_[2], results_[3]);
        st = {s0, s1, s2, s3};
        break;
      }
      default:
        throw std::invalid_argument("the algorithm counter takes 1 to 4 patterns");
    }
    for (std::size_t i = 0; i < st.size(); ++i) {
      // The stats agree with the counts: delta == counts after - counts before, the frontier sizes
      // are the batch's, triads added - removed == the net delta.
      std::int64_t net = 0;
      for (const std::int64_t d : st[i].delta) net += d;
      if (static_cast<std::uint64_t>(net) != st[i].triads_added - st[i].triads_removed ||
          st[i].deleted_frontier != st[i].batch.deleted_hyperedges ||
          st[i].inserted_frontier != static_cast<std::int64_t>(st[i].batch.inserted_ids.size()))
        throw std::runtime_error("triad_count::stats disagree with the update (pattern " +
                                 std::string(triad_count::to_string(results_[i].kind())) + ")");
    }
    return st.front().batch;
  }
  pattern_counts counts(const resources& res, const hgraph& h) override {
    pattern_counts out;
    for (std::size_t i = 0; i < results_.size(); ++i) {
      const auto c = results_[i].counts();
      std::vector<std::int64_t> v(c.size());
      for (std::size_t k = 0; k < c.size(); ++k) v[k] = static_cast<std::int64_t>(c[k]);
      if (opt_.recount && h.version() != results_[i].graph_version())
        throw std::runtime_error("triad_count: the result is stale");
      if (opt_.recount && evaluations > 0) {
        triad_count::options o;
        o.kind = results_[i].kind();
        const triad_count::result recount = triad_count::compute(res, h, o);
        const auto fresh = recount.counts();
        for (std::size_t k = 0; k < c.size(); ++k) {
          if (fresh[k] != c[k])
            throw std::runtime_error("triad_count: baseline + deltas != recount (pattern " +
                                     std::string(triad_count::to_string(o.kind)) + ", bin " +
                                     std::to_string(k) + ": " + std::to_string(c[k]) +
                                     " != " + std::to_string(fresh[k]) + ")");
        }
      }
      out[testing::triad_pattern_name(opt_.patterns[i])] = std::move(v);
    }
    ++evaluations;
    return out;
  }
  /// The CUDA engine's launch counters of every result: "<pattern>.<mode>.<shape>.<outcome>"
  /// (mode full or frontier; shape triangles, pairs or keys; outcome kernels, no_sources,
  /// no_wedges), summed over the modes' counters.
  void launches(std::map<std::string, std::int64_t>& out) const override {
    static constexpr const char* modes[2] = {"full", "frontier"};
    static constexpr const char* shapes[3] = {"triangles", "pairs", "keys"};
    for (const auto& r : results_) {
      const detail::triad_count_state* st = detail::triad_count_access::find(r);
      if (st == nullptr) continue;
      const std::string p(triad_count::to_string(r.kind()));
      for (int m = 0; m < 2; ++m) {
        for (int k = 0; k < 3; ++k) {
          const std::string key = p + "." + modes[m] + "." + shapes[k] + ".";
          const detail::triad_launch_counters& c = st->launches;
          if (c.kernels[m][k] != 0) out[key + "kernels"] += c.kernels[m][k];
          if (c.no_sources[m][k] != 0) out[key + "no_sources"] += c.no_sources[m][k];
          if (c.no_wedges[m][k] != 0) out[key + "no_wedges"] += c.no_wedges[m][k];
        }
      }
    }
  }

 private:
  const options& opt_;
  std::vector<triad_count::result> results_;
};
#endif

std::unique_ptr<counter> make_counter(const options& opt,
                                      const std::shared_ptr<oracle_cache>& cache) {
  if (opt.counter == "oracle") return std::make_unique<oracle_counter>(opt, cache);
  if (opt.counter == "algorithm") {
#if DYNG_TRIAD_REPLAY_ALGORITHM
    return std::make_unique<algorithm_counter>(opt);
#else
    throw not_available(
        "--counter algorithm: triad_count is not in this build (DYNG_ALGORITHMS leaves it out)");
#endif
  }
  throw std::invalid_argument("unknown counter " + opt.counter);
}

pattern_counts difference(const pattern_counts& after, const pattern_counts& before) {
  pattern_counts out;
  for (const auto& [name, v] : after) {
    const auto& b = before.at(name);
    std::vector<std::int64_t> d(v.size());
    for (std::size_t i = 0; i < v.size(); ++i) d[i] = v[i] - b[i];
    out[name] = std::move(d);
  }
  return out;
}

void json_counts(std::ostringstream& out, const pattern_counts& c) {
  out << "{";
  bool first = true;
  for (const auto& [name, v] : c) {
    out << (first ? "" : ", ") << "\"" << name << "\": [";
    for (std::size_t i = 0; i < v.size(); ++i) out << (i ? ", " : "") << v[i];
    out << "]";
    first = false;
  }
  out << "}";
}

std::string json_string(const std::string& s) {
  std::string out = "\"";
  for (const char ch : s) {
    if (ch == '"' || ch == '\\') {
      out += '\\';
      out += ch;
    } else if (static_cast<unsigned char>(ch) < 0x20) {
      out += ' ';
    } else {
      out += ch;
    }
  }
  return out + "\"";
}

struct mode_state {
  std::string mode;
  hgraph h;
  std::unique_ptr<counter> count;
  pattern_counts current;
  ed::by_content_resolver resolver;  // by_content only
  mode_report rep;
};

ed::rows rows_of(const std::vector<std::int64_t>& off, const std::vector<std::int32_t>& mem) {
  ed::rows out;
  for (std::size_t j = 0; j + 1 < off.size(); ++j)
    out.emplace_back(mem.begin() + off[j], mem.begin() + off[j + 1]);
  return out;
}

}  // namespace

testing::triad_pattern pattern_of(const std::string& name) {
  using testing::triad_pattern;
  if (name == "h_motif" || name == "main") return triad_pattern::h_motif;
  if (name == "stathyper_type1" || name == "type1") return triad_pattern::stathyper_type1;
  if (name == "stathyper_type2" || name == "type2") return triad_pattern::stathyper_type2;
  if (name == "stathyper_type3" || name == "type3") return triad_pattern::stathyper_type3;
  throw std::invalid_argument("unknown pattern " + name);
}

report replay(const resources& res, const std::string& hg, const std::vector<std::string>& batches,
              const options& opt) {
  const auto t0 = std::chrono::steady_clock::now();
  if (opt.mode != "both" && opt.mode != "by_key" && opt.mode != "by_content")
    throw std::invalid_argument("unknown mode " + opt.mode);
  report rep;
  rep.backend = std::string(to_string(res.get_backend()));
  rep.threads = res.num_threads();
  rep.counter = opt.counter;
  rep.hg = hg;
  io::hypergraph_read_options ro = io::hypergraph_read_options::escher_gpu();
  ro.num_vertices = opt.num_vertices;
  const auto list = io::read_hypergraph<std::int32_t, dyng::unweighted>(hg, ro);
  rep.num_vertices = list.num_vertices;
  rep.num_hyperedges = list.num_hyperedges();
  hypergraph_properties props;
  props.reuse = id_reuse::erase_first;
  props.capacity = capacity_policy::escher_gpu(opt.payload_capacity);
  props.line_graph = line_graph_policy::sorted_merge;

  std::vector<std::unique_ptr<mode_state>> modes;
  const auto cache = std::make_shared<oracle_cache>();
  for (const std::string m : {"by_key", "by_content"}) {
    if (opt.mode != "both" && opt.mode != m) continue;
    auto st = std::make_unique<mode_state>(mode_state{
        m, hgraph::from_hyperedges(res, list.view(), props), make_counter(opt, cache), {}, {}, {}});
    st->rep.mode = m;
    st->count->build(res, st->h);
    st->current = st->count->counts(res, st->h);
    st->rep.full = st->current;
    st->rep.device_bytes_build = st->h.store_stats().device_bytes;
    if (m == "by_content") {
      ed::rows loaded;
      for (std::int64_t i = 0; i < list.num_hyperedges(); ++i) {
        loaded.emplace_back(list.members.begin() + list.offsets[static_cast<std::size_t>(i)],
                            list.members.begin() + list.offsets[static_cast<std::size_t>(i) + 1]);
      }
      st->resolver.reset(loaded);
    }
    modes.push_back(std::move(st));
  }

  const resources host = resources::sequential();  // the batch files are read into host memory
  std::int64_t index = 0;
  for (const std::string& path : batches) {
    io::batch_reader in(path, io::batch_reader_options{list.num_vertices});
    const io::batch_file_header& head = in.header();
    if ((head.flags & io::batch_file_flags::expected_ids) == 0 ||
        (head.flags & io::batch_file_flags::deleted_members) == 0)
      throw std::invalid_argument(path + " lacks the expected ids or the deleted members");
    while (in.next()) {
      ++index;
      const auto batch = in.hyperedge_batch<std::int32_t, dyng::unweighted>();
      const auto ann = in.annotations<std::int32_t>();
      const std::vector<std::int64_t> expected = to_vector(host, ann.expected_ids);
      const ed::rows deleted_sets =
          rows_of(to_vector(host, ann.deleted_offsets), to_vector(host, ann.deleted_members));
      const std::vector<std::int64_t> by_key_ids = to_vector(host, batch.delete_hyperedges);
      const ed::rows inserted =
          rows_of(to_vector(host, batch.insert_offsets), to_vector(host, batch.insert_members));
      for (auto& st : modes) {
        batch_report b;
        b.index = index;
        b.deletions = static_cast<std::int64_t>(by_key_ids.size());
        b.insertions = static_cast<std::int64_t>(inserted.size());
        hyper_apply_summary s;
        if (st->mode == "by_key") {
          s = st->count->apply(res, st->h, batch);
          b.ids_as_expected = s.inserted_ids == expected;
          for (std::size_t i = 0; i < expected.size() || i < s.inserted_ids.size(); ++i) {
            if (i >= expected.size() || i >= s.inserted_ids.size() ||
                expected[i] != s.inserted_ids[i]) {
              b.first_id_mismatch = static_cast<std::int64_t>(i);
              break;
            }
          }
        } else {
          std::vector<bool> amb;
          const std::vector<std::int64_t> resolved = st->resolver.resolve(deleted_sets, &amb);
          host_batch content;
          std::vector<std::int64_t> taken;
          for (std::size_t j = 0; j < resolved.size(); ++j) {
            if (resolved[j] < 0) {
              ++b.unresolved;
              continue;
            }
            b.ambiguous += amb[j] ? 1 : 0;
            content.delete_hyperedge(resolved[j]);
            taken.push_back(resolved[j]);
          }
          for (const auto& r : inserted) content.insert_hyperedge(host_view(r));
          s = st->count->apply(res, st->h, content.view());
          st->resolver.commit(taken, inserted, s.inserted_ids);
        }
        b.deleted = s.deleted_hyperedges;
        b.ignored = s.ignored_deletions;
        b.num_hyperedges_after = s.num_hyperedges_after;
        b.id_bound_after = s.id_bound_after;
        b.device_bytes_after = st->h.store_stats().device_bytes;
        if (opt.integrity) {
          try {
            st->h.check_integrity(res);
          } catch (const std::exception& e) {
            b.integrity_ok = false;
            b.integrity_error = e.what();
          }
        }
        b.counts_after = st->count->counts(res, st->h);
        b.delta = difference(b.counts_after, st->current);
        st->current = b.counts_after;
        st->rep.batches.push_back(std::move(b));
      }
    }
  }
  for (auto& st : modes) {
    st->rep.oracle_evaluations = st->count->evaluations;
    st->count->launches(rep.launches);
    rep.modes.push_back(std::move(st->rep));
  }
  if (!modes.empty()) {
    const storage_stats ss = modes.front()->h.store_stats();
    const storage_path_counters* pc[3] = {&ss.fill, &ss.reuse, &ss.unfill};
    for (int k = 0; k < 3; ++k) {
      rep.paths[3 * k] = pc[k]->thread;
      rep.paths[3 * k + 1] = pc[k]->warp;
      rep.paths[3 * k + 2] = pc[k]->block;
    }
  }
  rep.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return rep;
}

std::string report::to_json() const {
  std::ostringstream out;
  out << "{\"backend\": " << json_string(backend) << ", \"threads\": " << threads
      << ", \"counter\": " << json_string(counter) << ", \"hg\": " << json_string(hg)
      << ", \"num_vertices\": " << num_vertices << ", \"num_hyperedges\": " << num_hyperedges
      << ", \"seconds\": " << seconds << ",\n \"paths\": {\"fill\": [" << paths[0] << ", "
      << paths[1] << ", " << paths[2] << "], \"reuse\": [" << paths[3] << ", " << paths[4] << ", "
      << paths[5] << "], \"unfill\": [" << paths[6] << ", " << paths[7] << ", " << paths[8]
      << "]},\n \"launches\": {";
  bool first = true;
  for (const auto& [k, v] : launches) {
    out << (first ? "" : ", ") << json_string(k) << ": " << v;
    first = false;
  }
  out << "},\n \"modes\": [";
  for (std::size_t m = 0; m < modes.size(); ++m) {
    const mode_report& mr = modes[m];
    out << (m ? ",\n  " : "\n  ") << "{\"mode\": " << json_string(mr.mode)
        << ", \"oracle_evaluations\": " << mr.oracle_evaluations
        << ", \"device_bytes_build\": " << mr.device_bytes_build << ", \"full\": ";
    json_counts(out, mr.full);
    out << ", \"batches\": [";
    for (std::size_t i = 0; i < mr.batches.size(); ++i) {
      const batch_report& b = mr.batches[i];
      out << (i ? ",\n    " : "\n    ") << "{\"index\": " << b.index
          << ", \"deletions\": " << b.deletions << ", \"insertions\": " << b.insertions
          << ", \"deleted\": " << b.deleted << ", \"ignored\": " << b.ignored
          << ", \"unresolved\": " << b.unresolved << ", \"ambiguous\": " << b.ambiguous
          << ", \"ids_as_expected\": " << (b.ids_as_expected ? "true" : "false")
          << ", \"first_id_mismatch\": " << b.first_id_mismatch
          << ", \"integrity_ok\": " << (b.integrity_ok ? "true" : "false")
          << ", \"integrity_error\": " << json_string(b.integrity_error)
          << ", \"num_hyperedges_after\": " << b.num_hyperedges_after
          << ", \"id_bound_after\": " << b.id_bound_after
          << ", \"device_bytes_after\": " << b.device_bytes_after << ", \"counts_after\": ";
      json_counts(out, b.counts_after);
      out << ", \"delta\": ";
      json_counts(out, b.delta);
      out << "}";
    }
    out << "]}";
  }
  out << "\n ]}\n";
  return out.str();
}

}  // namespace dyng::test::triad_replay
