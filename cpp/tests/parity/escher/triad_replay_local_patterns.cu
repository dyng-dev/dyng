// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file triad_replay_local_patterns.cu
 * @brief The replay counter `local_patterns` (M9b criterion 3, ADR 0058): triad_count's four
 *        patterns as Ops of experimental::count_local_patterns (the built-in Ops motif_op,
 *        choose3_op, type2_op and type3_op, instantiated here by nvcc as a user's Op would be),
 *        updated beside triad_count's own results in one commit per batch and compared with them
 *        bit for bit; the counts it reports are count_local_patterns', so parity/compare.py
 *        triad_count --counter local_patterns compares them with ESCHER-GPU's goldens.
 */
#include "algorithms/triad_count/engine/count_ops.hpp"
#include "algorithms/triad_count/engine/motif_ops.hpp"
#include "algorithms/triad_count/engine/stathyper_ops.hpp"
#include "parity/escher/triad_replay_counter.hpp"

#include <dyng/experimental/local_patterns.cuh>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/testing/triad_reference.hpp>
#include <dyng/triad_count.hpp>
#include <dyng/update.hpp>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

namespace dyng::test::triad_replay {

namespace {

namespace dx = dyng::experimental;
namespace tc = dyng::triad_count;
namespace triad = dyng::detail::triad;
using participant = dyng::detail::hyper_update_participant<std::int32_t, dyng::unweighted>;

std::vector<std::uint64_t> to_vector(array_view<const std::uint64_t> c) {
  return {c.begin(), c.end()};
}

/// One pattern: its triad_count result and its count_local_patterns result.
class slot {
 public:
  slot() = default;
  slot(const slot&) = delete;
  slot& operator=(const slot&) = delete;
  slot(slot&&) = delete;
  slot& operator=(slot&&) = delete;
  virtual ~slot() = default;
  virtual void compute(const resources& res, const hgraph& h) = 0;
  /// The participants of both results (their stats written when the update succeeds).
  virtual void participants(std::vector<std::unique_ptr<participant>>& out) = 0;
  /// Both results and both stats equal, bit for bit.
  virtual void check(const std::string& what) const = 0;
  [[nodiscard]] virtual std::vector<std::uint64_t> counts() const = 0;
  [[nodiscard]] virtual std::vector<std::uint64_t> recount(const resources& res,
                                                           const hgraph& h) const = 0;
  [[nodiscard]] virtual std::uint64_t graph_version() const = 0;
};

template <class op_t>
class op_slot final : public slot {
 public:
  explicit op_slot(tc::pattern kind) : kind_(kind) {}

  void compute(const resources& res, const hgraph& h) override {
    tc::options o;
    o.kind = kind_;
    triad_.emplace(tc::compute(res, h, o));
    local_.emplace(dx::count_local_patterns<op_t>::compute(res, h));
    if (to_vector(local_->counts()) != to_vector(triad_->counts()))
      throw std::runtime_error(name() + ": count_local_patterns' full count != triad_count's");
  }

  void participants(std::vector<std::unique_ptr<participant>>& out) override {
    out.push_back(detail::update_traits<tc::result>::template make_participant<hgraph>(
        *triad_, triad_stats_));
    out.push_back(
        detail::update_traits<dx::local_patterns_result<op_t>>::template make_participant<hgraph>(
            *local_, local_stats_));
  }

  void check(const std::string& what) const override {
    const std::string at = name() + " (" + what + "): ";
    if (to_vector(local_->counts()) != to_vector(triad_->counts()))
      throw std::runtime_error(at + "the counts differ from triad_count's");
    const int n = local_->num_bins();
    if (local_stats_.delta.size() != static_cast<std::size_t>(n))
      throw std::runtime_error(at + "stats::delta has the wrong size");
    for (int k = 0; k < n; ++k)
      if (local_stats_.delta[static_cast<std::size_t>(k)] !=
          triad_stats_.delta[static_cast<std::size_t>(k)])
        throw std::runtime_error(at + "the delta of bin " + std::to_string(k) +
                                 " differs from triad_count's");
    if (local_stats_.removed != triad_stats_.triads_removed ||
        local_stats_.added != triad_stats_.triads_added ||
        local_stats_.deleted_frontier != triad_stats_.deleted_frontier ||
        local_stats_.inserted_frontier != triad_stats_.inserted_frontier ||
        local_stats_.affected != triad_stats_.affected ||
        local_stats_.batch.inserted_ids != triad_stats_.batch.inserted_ids)
      throw std::runtime_error(at + "the stats differ from triad_count's");
  }

  [[nodiscard]] std::vector<std::uint64_t> counts() const override {
    return to_vector(local_->counts());
  }

  [[nodiscard]] std::vector<std::uint64_t> recount(const resources& res,
                                                   const hgraph& h) const override {
    return to_vector(dx::count_local_patterns<op_t>::compute(res, h).counts());
  }

  [[nodiscard]] std::uint64_t graph_version() const override {
    return local_->graph_version();
  }

 private:
  [[nodiscard]] std::string name() const {
    return std::string(tc::to_string(kind_));
  }

  tc::pattern kind_;
  std::optional<tc::result> triad_;
  std::optional<dx::local_patterns_result<op_t>> local_;
  tc::stats triad_stats_;
  dx::local_patterns_stats local_stats_;
};

std::unique_ptr<slot> make_slot(testing::triad_pattern p) {
  switch (p) {
    case testing::triad_pattern::h_motif:
      return std::make_unique<op_slot<triad::motif_op>>(tc::pattern::h_motif);
    case testing::triad_pattern::stathyper_type1:
      return std::make_unique<op_slot<triad::choose3_op>>(tc::pattern::stathyper_type1);
    case testing::triad_pattern::stathyper_type2:
      return std::make_unique<op_slot<triad::type2_op>>(tc::pattern::stathyper_type2);
    case testing::triad_pattern::stathyper_type3:
      return std::make_unique<op_slot<triad::type3_op>>(tc::pattern::stathyper_type3);
  }
  throw std::invalid_argument("unknown pattern");
}

/// The counter: one slot per requested pattern, one commit per batch for all of them.
class local_patterns_counter final : public counter {
 public:
  explicit local_patterns_counter(const options& opt) : opt_(opt) {}

  void build(const resources& res, hgraph& h) override {
    slots_.clear();
    for (const auto p : opt_.patterns) {
      slots_.push_back(make_slot(p));
      slots_.back()->compute(res, h);
    }
  }

  hyper_apply_summary apply(const resources& res, hgraph& h, const batch_view& b) override {
    std::vector<std::unique_ptr<participant>> owned;
    for (const auto& s : slots_) s->participants(owned);
    std::vector<participant*> raw;
    for (const auto& p : owned) raw.push_back(p.get());
    const hyper_apply_summary summary =
        detail::participant_of<hgraph>::run(res, h, b, raw.data(), raw.size(), "update.commit");
    for (const auto& s : slots_) s->check("after a batch");
    return summary;
  }

  pattern_counts counts(const resources& res, const hgraph& h) override {
    pattern_counts out;
    for (std::size_t i = 0; i < slots_.size(); ++i) {
      const std::vector<std::uint64_t> c = slots_[i]->counts();
      if (opt_.recount && h.version() != slots_[i]->graph_version())
        throw std::runtime_error("count_local_patterns: the result is stale");
      if (opt_.recount && evaluations > 0 && slots_[i]->recount(res, h) != c)
        throw std::runtime_error(std::string("count_local_patterns: baseline + deltas != recount "
                                             "(pattern ") +
                                 testing::triad_pattern_name(opt_.patterns[i]) + ")");
      std::vector<std::int64_t> v(c.size());
      for (std::size_t k = 0; k < c.size(); ++k) v[k] = static_cast<std::int64_t>(c[k]);
      out[testing::triad_pattern_name(opt_.patterns[i])] = std::move(v);
    }
    ++evaluations;
    return out;
  }

 private:
  const options& opt_;
  std::vector<std::unique_ptr<slot>> slots_;
};

}  // namespace

std::unique_ptr<counter> make_local_patterns_counter(const options& opt) {
  return std::make_unique<local_patterns_counter>(opt);
}

}  // namespace dyng::test::triad_replay
