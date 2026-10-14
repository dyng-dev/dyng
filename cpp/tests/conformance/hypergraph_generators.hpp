// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file hypergraph_generators.hpp
 * @brief The hypergraph axis of the conformance kit (ADR 0043): a host model of a hypergraph, the
 *        hyperedge batch mixes generated from it, and container_traits<hypergraph<V,W>>
 *        (containers.hpp has the vocabulary).
 *
 * The model keeps the live hyperedges by id (sorted, distinct members and a weight) and the id
 * bound. A generated batch names live ids for its deletions (and, in the mixes that test the
 * skip rules, dead, out-of-range and repeated ids) and member lists for its insertions. Ids of
 * inserted hyperedges are decided by the container (best-fit reuse; never predicted), so
 * random_batch() applies the deletions to the model and observe() records the inserted ids from
 * the commit's hyper_apply_summary. A batch is undone (C5) by the inverse built from that summary
 * and the deleted member sets: delete the inserted ids, insert the deleted hyperedges again.
 *
 * The mixes (M8b criterion 11): insertions only, deletions only, mixed, delete-then-reuse within
 * one batch (the freed slots fit the new rows), duplicate members (and hyperedges equal to live
 * ones), dead-id deletions (dead, out-of-range, negative and repeated ids), the empty batch,
 * delete-all, re-insert (delete hyperedges and insert their member sets again), a batch larger
 * than the hypergraph, ids at the index limits (members 0 and n - 1, the largest live id,
 * INT64_MIN / INT64_MAX deletions) and skewed cardinality (a hyperedge of min(n, 1,100) members,
 * >= 1,024 on the medium shapes). The id_reuse orders and the capacity presets are the property
 * presets: {erase_first, insert_first} x {escher_gpu(), mosp_escher(), the default}.
 */
#pragma once

#include "conformance/containers.hpp"
#include "conformance/test_traits.hpp"

#include <dyng/core/array_view.hpp>
#include <dyng/core/registry.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/core/types.hpp>
#include <dyng/hypergraph/hyper_apply_summary.hpp>
#include <dyng/hypergraph/hyperedge_batch.hpp>
#include <dyng/hypergraph/hyperedge_list.hpp>
#include <dyng/hypergraph/hypergraph.hpp>
#include <dyng/hypergraph/hypergraph_properties.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <map>
#include <random>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng::conformance {

/// The hyperedge batch mixes of the hypergraph axis (see the file comment).
enum class hyper_mix : std::uint8_t {
  insert_only,  ///< insertions only
  delete_only,  ///< deletions of live ids only
  mixed,        ///< half deletions, half insertions
  reuse,        ///< deletions of large hyperedges and insertions that fit their freed slots
  duplicates,   ///< insertions with repeated members, and copies of live hyperedges
  dead_ids,     ///< deletions of dead, out-of-range, negative and repeated ids (and live ones)
  empty,        ///< no operation
  delete_all,   ///< every live hyperedge deleted
  reinsert,     ///< hyperedges deleted and their member sets inserted again
  larger,       ///< more insertions than the hypergraph has hyperedges
  limits,       ///< ids at the index limits
  skewed,       ///< a hyperedge of min(n, 1100) members (>= 1,024 on the medium shapes)
};

/// The mix's name (for traces).
inline std::string_view to_string(hyper_mix mix) noexcept {
  switch (mix) {
    case hyper_mix::insert_only:
      return "insert_only";
    case hyper_mix::delete_only:
      return "delete_only";
    case hyper_mix::mixed:
      return "mixed";
    case hyper_mix::reuse:
      return "reuse";
    case hyper_mix::duplicates:
      return "duplicates";
    case hyper_mix::dead_ids:
      return "dead_ids";
    case hyper_mix::empty:
      return "empty";
    case hyper_mix::delete_all:
      return "delete_all";
    case hyper_mix::reinsert:
      return "reinsert";
    case hyper_mix::larger:
      return "larger";
    case hyper_mix::limits:
      return "limits";
    case hyper_mix::skewed:
      return "skewed";
  }
  return "?";
}

/// Every hyperedge mix, in the order C2 runs them.
inline std::vector<hyper_mix> all_hyper_mixes() {
  return {hyper_mix::insert_only, hyper_mix::delete_only, hyper_mix::mixed,  hyper_mix::reuse,
          hyper_mix::duplicates,  hyper_mix::dead_ids,    hyper_mix::empty,  hyper_mix::delete_all,
          hyper_mix::reinsert,    hyper_mix::larger,      hyper_mix::limits, hyper_mix::skewed};
}

/// The shape of a generated hypergraph.
struct hypergraph_shape {
  std::int64_t vertices = 0;    ///< n
  std::int64_t hyperedges = 0;  ///< live hyperedges
  int max_cardinality = 5;      ///< members are drawn per hyperedge from [1, max_cardinality]
  int max_weight = 9;           ///< weights are drawn from [1, max_weight]
};

/**
 * @brief A host model of a hypergraph: the live hyperedges by id and the id bound.
 * @tparam hypergraph_t The hypergraph type.
 */
template <typename hypergraph_t>
struct hypergraph_model {
  using vertex_type = typename hypergraph_t::vertex_type;  ///< vertex ids
  using weight_type = typename hypergraph_t::weight_type;  ///< weights (or unweighted)
  /// Whether the hypergraph has weights.
  static constexpr bool weighted = !is_unweighted_v<weight_type>;

  /// One hyperedge.
  struct row {
    std::vector<vertex_type> members;  ///< sorted, distinct
    std::int32_t weight = 1;           ///< in [1, max_weight] (ignored if unweighted)

    /// Equal members and weight.
    bool operator==(const row& o) const {
      return members == o.members && weight == o.weight;
    }
  };

  vertex_type num_vertices = 0;      ///< n
  std::map<std::int64_t, row> live;  ///< the live hyperedges by id
  std::int64_t id_bound = 0;         ///< hyperedge_id_bound()
  int max_cardinality = 5;           ///< new hyperedges have [1, max_cardinality] members
  int max_weight = 9;                ///< new weights are drawn from [1, max_weight]
  /// Insertions generated but not yet observed (observe() records their ids).
  std::vector<row> pending;

  /// The hypergraph of the model: the live hyperedges in id order, then the pending ones. Its ids
  /// are the model's when the live ids are 0..k-1 and nothing is pending (a fresh model).
  [[nodiscard]] hypergraph_t build(const resources& res, const hypergraph_properties& props) const {
    hyperedge_list<vertex_type, weight_type> list;
    list.num_vertices = num_vertices;
    const auto add = [&](const row& r) {
      if constexpr (weighted) {
        list.add_hyperedge(host_view(r.members), static_cast<weight_type>(r.weight));
      } else {
        list.add_hyperedge(host_view(r.members));
      }
    };
    for (const auto& [id, r] : live) {
      (void)id;
      add(r);
    }
    for (const row& r : pending) {
      add(r);
    }
    return hypergraph_t::from_hyperedges(res, list.view(), props);
  }
};

/// A random member set of `cardinality` distinct vertices of [0, n), sorted.
template <typename vertex_t>
std::vector<vertex_t> random_members(std::int64_t n, std::int64_t cardinality,
                                     std::mt19937_64& rng) {
  cardinality = std::max<std::int64_t>(1, std::min(cardinality, n));
  std::vector<vertex_t> out;
  if (cardinality * 2 > n) {
    std::vector<vertex_t> all(static_cast<std::size_t>(n));
    for (std::int64_t v = 0; v < n; ++v)
      all[static_cast<std::size_t>(v)] = static_cast<vertex_t>(v);
    std::shuffle(all.begin(), all.end(), rng);
    out.assign(all.begin(), all.begin() + cardinality);
  } else {
    std::uniform_int_distribution<std::int64_t> vertex(0, n - 1);
    while (static_cast<std::int64_t>(out.size()) < cardinality) {
      const auto v = static_cast<vertex_t>(vertex(rng));
      if (std::find(out.begin(), out.end(), v) == out.end()) out.push_back(v);
    }
  }
  std::sort(out.begin(), out.end());
  return out;
}

/// A random model of `shape`: hyperedges 0..k-1 with [1, max_cardinality] distinct members.
template <typename hypergraph_t>
hypergraph_model<hypergraph_t> random_hypergraph_model(const hypergraph_shape& shape,
                                                       std::mt19937_64& rng) {
  using vertex_type = typename hypergraph_t::vertex_type;
  hypergraph_model<hypergraph_t> m;
  m.num_vertices = static_cast<vertex_type>(shape.vertices);
  m.max_cardinality = shape.max_cardinality;
  m.max_weight = shape.max_weight;
  if (shape.vertices <= 0) return m;
  std::uniform_int_distribution<int> cardinality(1, std::max(1, shape.max_cardinality));
  std::uniform_int_distribution<int> weight(1, shape.max_weight);
  for (std::int64_t id = 0; id < shape.hyperedges; ++id) {
    typename hypergraph_model<hypergraph_t>::row r;
    r.members = random_members<vertex_type>(shape.vertices, cardinality(rng), rng);
    r.weight = weight(rng);
    m.live.emplace(id, std::move(r));
  }
  m.id_bound = shape.hyperedges;
  return m;
}

/**
 * @brief One generated hyperedge batch: the batch, what it asks for, and what the commit must
 *        report about it.
 * @tparam hypergraph_t The hypergraph type.
 */
template <typename hypergraph_t>
struct generated_hyper_batch {
  using vertex_type = typename hypergraph_t::vertex_type;    ///< vertex ids
  using weight_type = typename hypergraph_t::weight_type;    ///< weights
  using row = typename hypergraph_model<hypergraph_t>::row;  ///< a hyperedge
  hyperedge_batch<vertex_type, weight_type> batch;           ///< the batch
  std::vector<row> inserted;          ///< the inserted hyperedges, normalized, batch order
  std::vector<std::int64_t> deleted;  ///< the live ids deleted (distinct, batch order)
  std::vector<row> deleted_rows;      ///< their hyperedges
  std::int64_t expected_ignored = 0;  ///< deletions the commit must skip
  std::int64_t expected_dropped = 0;  ///< repeated members the commit must drop
  bool invertible = true;             ///< C5 may undo it (always, from the summary)
};

namespace hyper_generators_detail {

/// Add the insertion of `members` (possibly with repeats) as `r` (normalized) to the batch.
template <typename hypergraph_t>
void insert(generated_hyper_batch<hypergraph_t>& out,
            const std::vector<typename hypergraph_t::vertex_type>& members, std::int32_t weight) {
  using vertex_type = typename hypergraph_t::vertex_type;
  using weight_type = typename hypergraph_t::weight_type;
  if constexpr (!is_unweighted_v<weight_type>) {
    out.batch.insert_hyperedge(host_view(members), static_cast<weight_type>(weight));
  } else {
    out.batch.insert_hyperedge(host_view(members));
  }
  typename generated_hyper_batch<hypergraph_t>::row r;
  r.members = members;
  std::sort(r.members.begin(), r.members.end());
  const auto last = std::unique(r.members.begin(), r.members.end());
  out.expected_dropped += static_cast<std::int64_t>(r.members.end() - last);
  r.members.erase(last, r.members.end());
  r.weight = weight;
  out.inserted.push_back(std::move(r));
  (void)sizeof(vertex_type);
}

/// Delete the live id `id` (in the model and the batch).
template <typename hypergraph_t>
void erase(hypergraph_model<hypergraph_t>& m, generated_hyper_batch<hypergraph_t>& out,
           std::int64_t id) {
  out.batch.delete_hyperedge(id);
  const auto it = m.live.find(id);
  out.deleted.push_back(id);
  out.deleted_rows.push_back(it->second);
  m.live.erase(it);
}

}  // namespace hyper_generators_detail

/**
 * @brief A batch of `mix` from the model; its deletions are applied to the model, its insertions
 *        wait for observe().
 * @tparam hypergraph_t The hypergraph type.
 * @param[in,out] m   The model (nothing may be pending).
 * @param[in]     mix The mix.
 * @param[in,out] rng The generator.
 * @return The batch and what the commit must report.
 */
template <typename hypergraph_t>
generated_hyper_batch<hypergraph_t> random_hyper_batch(hypergraph_model<hypergraph_t>& m,
                                                       hyper_mix mix, std::mt19937_64& rng) {
  using vertex_type = typename hypergraph_t::vertex_type;
  namespace d = hyper_generators_detail;
  generated_hyper_batch<hypergraph_t> out;
  if (!m.pending.empty()) {
    ADD_FAILURE() << "the kit generated a batch before observing the previous one's ids";
    m.pending.clear();
  }
  const std::int64_t n = m.num_vertices;
  const auto live_count = static_cast<std::int64_t>(m.live.size());
  const std::int64_t k = std::max<std::int64_t>(1, live_count / 10);
  std::uniform_int_distribution<int> cardinality(1, std::max(1, m.max_cardinality));
  std::uniform_int_distribution<int> weight(1, m.max_weight);
  const auto fresh = [&](std::int64_t card) { return random_members<vertex_type>(n, card, rng); };
  // The live ids in a random order.
  std::vector<std::int64_t> ids;
  ids.reserve(m.live.size());
  for (const auto& entry : m.live) ids.push_back(entry.first);
  std::shuffle(ids.begin(), ids.end(), rng);
  const auto take = [&](std::int64_t count) {
    count = std::min<std::int64_t>(count, static_cast<std::int64_t>(ids.size()));
    std::vector<std::int64_t> chosen(ids.begin(), ids.begin() + count);
    ids.erase(ids.begin(), ids.begin() + count);
    return chosen;
  };
  if (n <= 0) {
    return out;  // no vertex: only the empty batch
  }
  switch (mix) {
    case hyper_mix::insert_only:
      for (std::int64_t i = 0; i < k; ++i) d::insert(out, fresh(cardinality(rng)), weight(rng));
      break;
    case hyper_mix::delete_only:
      for (const std::int64_t id : take(k)) d::erase(m, out, id);
      break;
    case hyper_mix::mixed:
      for (const std::int64_t id : take((k + 1) / 2)) d::erase(m, out, id);
      for (std::int64_t i = 0; i < (k + 1) / 2; ++i) {
        d::insert(out, fresh(cardinality(rng)), weight(rng));
      }
      break;
    case hyper_mix::reuse: {
      // The largest hyperedges go; the new ones are no larger than the smallest of them, so under
      // erase_first they fit the freed slots (best fit) and take their ids.
      std::sort(ids.begin(), ids.end(), [&](std::int64_t a, std::int64_t b) {
        const auto sa = m.live.at(a).members.size();
        const auto sb = m.live.at(b).members.size();
        return sa != sb ? sa > sb : a < b;
      });
      const std::vector<std::int64_t> gone = take(k);
      std::int64_t smallest = m.max_cardinality;
      for (const std::int64_t id : gone) {
        smallest = std::min<std::int64_t>(smallest,
                                          static_cast<std::int64_t>(m.live.at(id).members.size()));
        d::erase(m, out, id);
      }
      std::uniform_int_distribution<std::int64_t> small(1, std::max<std::int64_t>(1, smallest));
      for (std::size_t i = 0; i < gone.size(); ++i) d::insert(out, fresh(small(rng)), weight(rng));
      break;
    }
    case hyper_mix::duplicates: {
      for (std::int64_t i = 0; i < k; ++i) {
        std::vector<vertex_type> members = fresh(cardinality(rng));
        const std::vector<vertex_type> copy = members;
        members.insert(members.end(), copy.begin(), copy.end());  // every member twice
        std::shuffle(members.begin(), members.end(), rng);
        d::insert(out, members, weight(rng));
      }
      if (!m.live.empty()) {  // a hyperedge equal to a live one (parallel hyperedges)
        d::insert(out, m.live.begin()->second.members, weight(rng));
      }
      break;
    }
    case hyper_mix::dead_ids: {
      const std::vector<std::int64_t> gone = take(std::min<std::int64_t>(k, 2));
      for (const std::int64_t id : gone) d::erase(m, out, id);
      // The same live ids again (repeated deletions), a dead id below the bound, the bound,
      // a negative id.
      for (const std::int64_t id : gone) {
        out.batch.delete_hyperedge(id);
        ++out.expected_ignored;
      }
      for (std::int64_t id = 0; id < m.id_bound; ++id) {
        if (m.live.count(id) == 0 && std::find(gone.begin(), gone.end(), id) == gone.end()) {
          out.batch.delete_hyperedge(id);
          ++out.expected_ignored;
          break;
        }
      }
      out.batch.delete_hyperedge(m.id_bound);
      out.batch.delete_hyperedge(-1);
      out.expected_ignored += 2;
      d::insert(out, fresh(cardinality(rng)), weight(rng));
      break;
    }
    case hyper_mix::empty:
      break;
    case hyper_mix::delete_all:
      for (const std::int64_t id : take(live_count)) d::erase(m, out, id);
      break;
    case hyper_mix::reinsert:
      for (const std::int64_t id : take(k)) {
        d::erase(m, out, id);
        const auto& r = out.deleted_rows.back();
        d::insert(out, r.members, r.weight);
      }
      break;
    case hyper_mix::larger: {
      for (const std::int64_t id : take(live_count / 2)) d::erase(m, out, id);
      const std::int64_t count = 2 * live_count + 3;
      for (std::int64_t i = 0; i < count; ++i) {
        d::insert(out, fresh(cardinality(rng)), weight(rng));
      }
      break;
    }
    case hyper_mix::limits: {
      // The largest live id goes; members 0 and n - 1; deletions at the int64 limits.
      if (!m.live.empty()) d::erase(m, out, m.live.rbegin()->first);
      out.batch.delete_hyperedge(std::numeric_limits<std::int64_t>::min());
      out.batch.delete_hyperedge(std::numeric_limits<std::int64_t>::max());
      out.batch.delete_hyperedge(m.id_bound);
      out.expected_ignored += 3;
      d::insert(out, std::vector<vertex_type>{0, static_cast<vertex_type>(n - 1)}, weight(rng));
      d::insert(out, std::vector<vertex_type>{static_cast<vertex_type>(n - 1)}, 1);
      d::insert(out, std::vector<vertex_type>{0}, m.max_weight);
      break;
    }
    case hyper_mix::skewed: {
      for (const std::int64_t id : take(std::min<std::int64_t>(k, 2))) d::erase(m, out, id);
      d::insert(out, fresh(std::min<std::int64_t>(n, 1100)), weight(rng));
      d::insert(out, fresh(1), weight(rng));
      break;
    }
  }
  m.pending = out.inserted;
  return out;
}

/// Record the ids the commit gave the insertions of `gen` (and the new id bound).
template <typename hypergraph_t>
void observe_hyper_batch(hypergraph_model<hypergraph_t>& m,
                         const generated_hyper_batch<hypergraph_t>& gen,
                         const hyper_apply_summary& summary) {
  if (summary.inserted_ids.size() != gen.inserted.size()) {
    ADD_FAILURE() << "the commit reports " << summary.inserted_ids.size() << " ids for "
                  << gen.inserted.size() << " inserted hyperedges";
    m.pending.clear();
    return;
  }
  for (std::size_t i = 0; i < gen.inserted.size(); ++i) {
    const std::int64_t id = summary.inserted_ids[i];
    if (!m.live.emplace(id, gen.inserted[i]).second) {
      ADD_FAILURE() << "the commit gave the live id " << id << " to inserted hyperedge " << i;
    }
  }
  m.id_bound = summary.id_bound_after;
  m.pending.clear();
}

/// The hypergraph's container axis (containers.hpp).
template <typename vertex_t, typename weight_t>
struct container_traits<hypergraph<vertex_t, weight_t>> {
  using container_type = hypergraph<vertex_t, weight_t>;              ///< the container
  using properties_type = hypergraph_properties;                      ///< its properties
  using shape_type = hypergraph_shape;                                ///< test_traits::shape()
  using model_type = hypergraph_model<container_type>;                ///< the host model
  using batch_type = hyperedge_batch<vertex_t, weight_t>;             ///< the owning batch
  using batch_view_type = hyperedge_batch_view<vertex_t, weight_t>;   ///< update()'s batch
  using summary_type = hyper_apply_summary;                           ///< stats::batch
  using generated_type = generated_hyper_batch<container_type>;       ///< a generated batch
  using mix_type = hyper_mix;                                         ///< the batch mixes
  static constexpr container_kind kind = container_kind::hypergraph;  ///< the manifest's

  /// "hv32_unweighted" / "hv32_w32".
  static std::string case_name() {
    return "hv" + std::to_string(8 * sizeof(vertex_t)) +
           (is_unweighted_v<weight_t> ? std::string("_unweighted")
                                      : "_w" + std::to_string(8 * sizeof(weight_t)));
  }

  /// The presets: {erase_first, insert_first} x {escher_gpu(), mosp_escher(), the default}, each
  /// with require(), and the traits' extras.
  template <typename traits_t>
  static std::vector<kit_preset<hypergraph_properties>> presets() {
    std::vector<kit_preset<hypergraph_properties>> out;
    for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
      for (int c = 0; c < 3; ++c) {
        hypergraph_properties p;
        p.reuse = reuse;
        p.capacity = c == 0   ? capacity_policy::escher_gpu()
                     : c == 1 ? capacity_policy::mosp_escher()
                              : capacity_policy{};
        traits_t::require(p);
        out.push_back({std::string(to_string(reuse)) + ", " +
                           (c == 0   ? "escher_gpu()"
                            : c == 1 ? "mosp_escher()"
                                     : "default"),
                       p});
      }
    }
    if constexpr (kit_detail::has_extra_properties<traits_t>::value) {
      int i = 0;
      for (hypergraph_properties p : traits_t::extra_properties()) {
        traits_t::require(p);
        out.push_back({"extra properties " + std::to_string(i++), p});
      }
    }
    return out;
  }

  /// The mixes of each check.
  static std::vector<hyper_mix> mixes(kit_check check) {
    switch (check) {
      case kit_check::c2:
      case kit_check::c4:
      case kit_check::c9:
        return all_hyper_mixes();
      case kit_check::c3:
        return {hyper_mix::mixed,  hyper_mix::reuse,  hyper_mix::duplicates, hyper_mix::dead_ids,
                hyper_mix::larger, hyper_mix::limits, hyper_mix::delete_all, hyper_mix::skewed};
      case kit_check::c5:
        return {hyper_mix::insert_only, hyper_mix::delete_only, hyper_mix::mixed,
                hyper_mix::reuse,       hyper_mix::duplicates,  hyper_mix::delete_all,
                hyper_mix::reinsert,    hyper_mix::larger};
      case kit_check::c6:
        return {hyper_mix::mixed, hyper_mix::reuse, hyper_mix::larger};
      case kit_check::c8:
        return {hyper_mix::mixed, hyper_mix::reuse};
    }
    return {};
  }

  /// Every mix applies to every preset.
  static bool mix_applies(hyper_mix /*mix*/, const hypergraph_properties& /*props*/) {
    return true;
  }

  /// A random model of a size class.
  template <typename traits_t>
  static model_type random_model(size_class size, std::mt19937_64& rng) {
    return random_hypergraph_model<container_type>(traits_t::shape(size), rng);
  }

  /// The hypergraph of the model.
  static container_type build(const model_type& m, const resources& res,
                              const hypergraph_properties& props) {
    return m.build(res, props);
  }

  /// A batch of `mix` from the model (its deletions applied to the model).
  static generated_type random_batch(model_type& m, hyper_mix mix, std::mt19937_64& rng,
                                     const hypergraph_properties& /*props*/) {
    return random_hyper_batch(m, mix, rng);
  }

  /// @copydoc random_batch
  static generated_type random_batch(model_type& m, hyper_mix mix, std::mt19937_64& rng) {
    return random_hyper_batch(m, mix, rng);
  }

  /// Record the inserted ids of the commit in the model.
  static void observe(model_type& m, const generated_type& gen, const summary_type& summary) {
    observe_hyper_batch(m, gen, summary);
  }

  /// The batch of a generated batch.
  static const batch_type& batch_of(const generated_type& gen) {
    return gen.batch;
  }

  /// Every hyperedge batch can be undone (from its summary and the deleted member sets).
  static bool invertible(const generated_type& gen) {
    return gen.invertible;
  }

  /// The inverse of an observed batch: delete the inserted ids, insert the deleted hyperedges
  /// again (with their weights); applied to the model like random_batch().
  static generated_type inverse(model_type& m, const generated_type& gen,
                                const summary_type& summary) {
    namespace d = hyper_generators_detail;
    generated_type out;
    for (const std::int64_t id : summary.inserted_ids) {
      if (m.live.count(id) != 0) d::erase(m, out, id);
    }
    for (const auto& r : gen.deleted_rows) d::insert(out, r.members, r.weight);
    m.pending = out.inserted;
    return out;
  }

  /// The deterministic counters of the batch summary, the inserted ids included.
  static std::vector<std::int64_t> summary_counters(const summary_type& b) {
    std::vector<std::int64_t> out = {
        b.inserted_hyperedges, b.deleted_hyperedges,   b.inserted_incidences,
        b.deleted_incidences,  b.skipped_ops,          b.ignored_deletions,
        b.dropped_members,     b.num_hyperedges_after, b.id_bound_after};
    out.insert(out.end(), b.inserted_ids.begin(), b.inserted_ids.end());
    return out;
  }

  /// What an empty batch must leave at zero.
  static std::int64_t changed(const summary_type& b) {
    return b.inserted_hyperedges + b.deleted_hyperedges + b.inserted_incidences +
           b.deleted_incidences + static_cast<std::int64_t>(b.inserted_ids.size());
  }

  /// The hypergraph has the model's vertices, live hyperedges and id bound.
  static ::testing::AssertionResult matches(const container_type& h, const model_type& m) {
    if (static_cast<std::int64_t>(h.num_vertices()) != static_cast<std::int64_t>(m.num_vertices)) {
      return ::testing::AssertionFailure() << "the hypergraph has " << h.num_vertices()
                                           << " vertices, the model " << m.num_vertices;
    }
    if (h.num_hyperedges() != static_cast<std::int64_t>(m.live.size())) {
      return ::testing::AssertionFailure() << "the hypergraph has " << h.num_hyperedges()
                                           << " hyperedges, the model " << m.live.size();
    }
    if (h.hyperedge_id_bound() != m.id_bound) {
      return ::testing::AssertionFailure()
             << "the hypergraph's id bound is " << h.hyperedge_id_bound() << ", the model's "
             << m.id_bound;
    }
    return ::testing::AssertionSuccess();
  }

  /// C10: both orders of a batch, each under the capacity preset of its original, with both
  /// algorithms' requirements.
  template <typename traits_t, typename partner_t>
  static std::vector<kit_preset<hypergraph_properties>> composition_presets() {
    std::vector<kit_preset<hypergraph_properties>> out;
    for (const id_reuse reuse : {id_reuse::erase_first, id_reuse::insert_first}) {
      hypergraph_properties p;
      p.reuse = reuse;
      p.capacity = reuse == id_reuse::erase_first ? capacity_policy::escher_gpu()
                                                  : capacity_policy::mosp_escher();
      traits_t::require(p);
      partner_t::require(p);
      out.push_back({std::string(to_string(reuse)), p});
    }
    return out;
  }

  /// C10: the model (the smaller of the two small shapes).
  template <typename traits_t, typename partner_t>
  static model_type composition_model(const hypergraph_properties& /*props*/,
                                      std::mt19937_64& rng) {
    hypergraph_shape shape = traits_t::shape(size_class::small);
    const hypergraph_shape theirs = partner_t::shape(size_class::small);
    shape.vertices = std::min(shape.vertices, theirs.vertices);
    shape.hyperedges = std::min(shape.hyperedges, theirs.hyperedges);
    return random_hypergraph_model<container_type>(shape, rng);
  }

  /// C10's mix of each of its three batches.
  static hyper_mix composition_mix(int step) {
    return step == 2 ? hyper_mix::larger : step == 1 ? hyper_mix::reuse : hyper_mix::mixed;
  }

  /// C11's batch (applied by a separate h.apply()).
  static hyper_mix stale_mix() {
    return hyper_mix::mixed;
  }

  /// C12's mix of each of its four batches.
  static hyper_mix stream_mix(int step) {
    return step == 3 ? hyper_mix::larger : step == 2 ? hyper_mix::delete_all : hyper_mix::reuse;
  }
};

}  // namespace dyng::conformance
