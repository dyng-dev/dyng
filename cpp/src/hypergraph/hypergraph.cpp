// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:structure/dynamic_hypergraph.cu (update: the erase_first order)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/DynamicHypergraph.cpp (beginBatch,
// finishBatch: the insert_first order)
// Derived from MOSP_ESCHER@4b86159:hypergraph/src/HostHypergraph.cpp (applyBatch: the skip rules)
/**
 * @file hypergraph.cpp
 * @brief hypergraph<V,W> member functions, hypergraph_access, and their explicit instantiations
 *        (ADR 0041).
 */
#include "core/budget_counters.hpp"
#include "core/resources_access.hpp"
#include "core/staging.hpp"
#include "hypergraph/capacity.hpp"
#include "hypergraph/escher_store.hpp"
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/incidence_model.hpp"
#include "hypergraph/instantiate.hpp"
#include "util/allocation.hpp"

#include <dyng/config.hpp>
#include <dyng/core/copy.hpp>
#include <dyng/core/error.hpp>
#include <dyng/core/profiler.hpp>
#include <dyng/hypergraph/hypergraph.hpp>

#include <algorithm>
#include <climits>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace dyng {

namespace {

/// The largest weight (PLAN 4.4.2: integer weights in [1, 2^31 - 1]).
template <typename weight_t>
void check_weight(weight_t w, std::int64_t index, const char* what) {
  if constexpr (!is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(static_cast<std::int64_t>(w) >= 1 && static_cast<std::int64_t>(w) <= INT_MAX, what,
                 ": the weight of hyperedge ", index, " is ", static_cast<std::int64_t>(w),
                 "; weights must lie in [1, 2^31 - 1]");
  } else {
    (void)w;
    (void)index;
    (void)what;
  }
}

/// The rows of a CSR of member lists, checked and normalized: offsets from 0, non-decreasing,
/// ending at the member count; every row non-empty; members in [0, n); each row sorted and its
/// repeated members dropped (counted in `dropped`).
template <typename vertex_t>
void normalize_rows(array_view<const std::int64_t> offsets, array_view<const vertex_t> members,
                    vertex_t num_vertices, const char* what, detail::hyper::member_rows& out,
                    std::int64_t& dropped, bool batch_payload = false) {
  (void)batch_payload;  // only a mutation reads it
  out.clear();
  dropped = 0;
  if (offsets.empty()) {
    DYNG_EXPECTS(members.empty(), what, ": ", members.size(),
                 " members are given without offsets (offsets need one entry more than the "
                 "hyperedges)");
    return;
  }
  const auto num_rows = static_cast<std::int64_t>(offsets.size()) - 1;
  DYNG_EXPECTS(offsets[0] == 0, what, ": the first offset is ", offsets[0], ", not 0");
  DYNG_EXPECTS(offsets[offsets.size() - 1] == static_cast<std::int64_t>(members.size()), what,
               ": the last offset is ", offsets[offsets.size() - 1], ", not the member count ",
               members.size());
  if (members.size() > static_cast<std::size_t>(INT_MAX / 3)) {
    throw capacity_error(detail::concat_message(
        "dyng: ", what, ": ", members.size(),
        " members exceed the int32 payload of the hypergraph store (at most ", INT_MAX / 3, ")"));
  }
  out.offsets.reserve(static_cast<std::size_t>(num_rows) + 1);
  out.members.reserve(members.size());
  std::vector<std::int32_t> row;
  for (std::int64_t i = 0; i < num_rows; ++i) {
    const std::int64_t begin = offsets[static_cast<std::size_t>(i)];
    const std::int64_t end = offsets[static_cast<std::size_t>(i) + 1];
    DYNG_EXPECTS(begin <= end, what, ": the offsets decrease at hyperedge ", i);
    DYNG_EXPECTS(begin < end, what, ": hyperedge ", i, " is empty (a hyperedge needs a member)");
    row.clear();
    for (std::int64_t k = begin; k < end; ++k) {
      const vertex_t v = members[static_cast<std::size_t>(k)];
      DYNG_EXPECTS(v >= 0 && v < num_vertices, what, ": hyperedge ", i, " has the vertex ",
                   static_cast<std::int64_t>(v), ", outside [0, ",
                   static_cast<std::int64_t>(num_vertices), ")");
      row.push_back(static_cast<std::int32_t>(v));
    }
    std::sort(row.begin(), row.end());
#if defined(DYNG_MUTATION_HYPERGRAPH_H2V_NOT_DEDUPLICATED)
    // MUTATION (PLAN 8.4, MOSP_ESCHER #6; M8b criterion 20): the h2v insert payload of a batch is
    // not deduplicated (repeated members stored).
    const auto last = batch_payload ? row.end() : std::unique(row.begin(), row.end());
#else
    const auto last = std::unique(row.begin(), row.end());
#endif
    dropped += static_cast<std::int64_t>(row.end() - last);
    out.members.insert(out.members.end(), row.begin(), last);
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
  }
}

/// The weights of `num` hyperedges, checked (empty for unweighted).
template <typename weight_t>
std::vector<weight_t> checked_weights(array_view<const weight_t> weights, std::int64_t num,
                                      const char* what) {
  std::vector<weight_t> out;
  if constexpr (is_unweighted_v<weight_t>) {
    DYNG_EXPECTS(weights.empty(), what, ": an unweighted hypergraph takes no weights, got ",
                 weights.size());
  } else {
    DYNG_EXPECTS(static_cast<std::int64_t>(weights.size()) == num, what, ": ", weights.size(),
                 " weights for ", num, " hyperedges (one weight per hyperedge)");
    out.assign(weights.begin(), weights.end());
    for (std::int64_t i = 0; i < num; ++i) check_weight(out[static_cast<std::size_t>(i)], i, what);
  }
  return out;
}

/// Record the placement of a new hypergraph (PLAN 4.6 rule 5).
template <typename impl_t>
void set_home(impl_t& impl, const resources& res) {
  impl.home = res.get_backend();
  impl.home_device = res.get_backend() == backend::cuda ? res.device() : -1;
}

/// A store for the backend of `res` from a snapshot of any store.
std::unique_ptr<detail::hyper::incidence_store> store_from_snapshot(
    const resources& res, const detail::hyper::store_snapshot& snapshot,
    const capacity_policy& capacity) {
  if (res.get_backend() == backend::cuda) {
    return detail::hyper::make_escher_store(res, snapshot, capacity);
  }
  return std::make_unique<detail::hyper::incidence_model>(res, snapshot, capacity);
}

}  // namespace

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::hypergraph(std::unique_ptr<impl_type> impl) noexcept
    : impl_(std::move(impl)) {}

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::hypergraph(hypergraph&& other) noexcept = default;

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>& hypergraph<vertex_t, weight_t>::operator=(
    hypergraph&& other) noexcept = default;

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t>::~hypergraph() = default;

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::impl_type& hypergraph<vertex_t, weight_t>::impl() {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from hypergraph");
  return *impl_;
}

template <typename vertex_t, typename weight_t>
const typename hypergraph<vertex_t, weight_t>::impl_type& hypergraph<vertex_t, weight_t>::impl()
    const {
  DYNG_EXPECTS(impl_ != nullptr, "use of a moved-from hypergraph");
  return *impl_;
}

namespace {

/// The state of a hypergraph for a call that reads it: not moved from, not poisoned.
template <typename impl_t>
const impl_t& readable(const impl_t& impl, const char* what) {
  DYNG_EXPECTS(!impl.poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  return impl;
}

}  // namespace

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::from_hyperedges(
    const resources& res, hyperedge_list_view<vertex_t, weight_t> hyperedges,
    const hypergraph_properties& props) try {
  static_assert(std::is_same_v<vertex_t, std::int32_t>,
                "hypergraph supports vertex_t = std::int32_t in this release");
  detail::hyper::validate_properties(props);
  DYNG_EXPECTS(hyperedges.num_vertices >= 0, "hypergraph::from_hyperedges: num_vertices ",
               static_cast<std::int64_t>(hyperedges.num_vertices), " is negative");
  auto impl = std::make_unique<impl_type>();
  impl->props = props;
  set_home(*impl, res);
  detail::hyper::member_rows rows;
  {
    // The host side of the construction (ESCHER-GPU's init.v2h_build / flatten region).
    const scoped_stage stage(res, "hypergraph.build");
    const detail::host_input<std::int64_t> offsets(
        res, hyperedges.offsets, "hypergraph::from_hyperedges: hyperedge_list_view::offsets");
    const detail::host_input<vertex_t> members(
        res, hyperedges.members, "hypergraph::from_hyperedges: hyperedge_list_view::members");
    const detail::host_input<weight_t> weights(
        res, hyperedges.weights, "hypergraph::from_hyperedges: hyperedge_list_view::weights");
    std::int64_t dropped = 0;
    normalize_rows(offsets.view(), members.view(), hyperedges.num_vertices,
                   "hypergraph::from_hyperedges", rows, dropped);
    impl->weights = checked_weights(weights.view(), rows.size(), "hypergraph::from_hyperedges");
    if (res.get_backend() != backend::cuda) {
      impl->store = std::make_unique<detail::hyper::incidence_model>(
          res, static_cast<std::int32_t>(hyperedges.num_vertices), rows, props.capacity,
          props.store_v2h);
    }
  }
  if (res.get_backend() == backend::cuda) {
    const detail::container_scope container;  // the container's own work (budgets, I9)
    impl->store =
        detail::hyper::make_escher_store(res, static_cast<std::int32_t>(hyperedges.num_vertices),
                                         rows, props.capacity, props.store_v2h);
  }
  return hypergraph(std::move(impl));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::from_hyperedges (", hyperedges.num_hyperedges(),
                                  " hyperedges, ", hyperedges.members.size(), " members)")

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::clone(const resources& res) const
    try {
  const auto& self = readable(impl(), "hypergraph::clone");
  auto copy = std::make_unique<impl_type>();
  copy->props = self.props;
  copy->version = self.version;
  copy->state_id = self.state_id;
  copy->weights = self.weights;
  set_home(*copy, res);
  const bool from_device = self.home == backend::cuda;
  const bool to_device = res.get_backend() == backend::cuda;
  if (from_device == to_device) {
    copy->store = self.store->clone(res);
    copy->store->set_host_threads(detail::resources_access::host_threads(res));
  } else {
    copy->store = store_from_snapshot(res, self.store->snapshot(), self.props.capacity);
  }
  return hypergraph(std::move(copy));
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::clone (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
hypergraph<vertex_t, weight_t> hypergraph<vertex_t, weight_t>::to_backend(
    const resources& res) const {
  return clone(res);
}

template <typename vertex_t, typename weight_t>
vertex_t hypergraph<vertex_t, weight_t>::num_vertices() const noexcept {
  return impl_ && impl_->store ? static_cast<vertex_t>(impl_->store->num_vertices()) : vertex_t{0};
}

template <typename vertex_t, typename weight_t>
std::int64_t hypergraph<vertex_t, weight_t>::num_hyperedges() const noexcept {
  return impl_ && impl_->store ? impl_->store->num_live() : 0;
}

template <typename vertex_t, typename weight_t>
std::int64_t hypergraph<vertex_t, weight_t>::hyperedge_id_bound() const noexcept {
  return impl_ && impl_->store ? impl_->store->id_bound() : 0;
}

template <typename vertex_t, typename weight_t>
const hypergraph_properties& hypergraph<vertex_t, weight_t>::properties() const noexcept {
  static const hypergraph_properties moved_from{};
  return impl_ ? impl_->props : moved_from;
}

template <typename vertex_t, typename weight_t>
memory_space hypergraph<vertex_t, weight_t>::space() const noexcept {
  return impl_ && impl_->home == backend::cuda ? memory_space::device : memory_space::host;
}

template <typename vertex_t, typename weight_t>
std::uint64_t hypergraph<vertex_t, weight_t>::version() const noexcept {
  return impl_ ? impl_->version : 0;
}

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::h2v_view_type hypergraph<vertex_t, weight_t>::h2v() const
    try {
  const auto& self = readable(impl(), "hypergraph::h2v");
  h2v_view_type out;
  array_view<const std::int32_t> cols;
  self.store->h2v_view(out.row_ptr, cols);
  out.col_ind = array_view<const vertex_t>(cols.data(), cols.size(), cols.space(), cols.device());
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::h2v (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::v2h_view_type hypergraph<vertex_t, weight_t>::v2h() const
    try {
  const auto& self = readable(impl(), "hypergraph::v2h");
  DYNG_EXPECTS(self.props.store_v2h,
               "hypergraph::v2h: the hypergraph was built with store_v2h = false");
  v2h_view_type out;
  self.store->v2h_view(out.row_ptr, out.col_ind);
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::v2h (", static_cast<std::int64_t>(num_vertices()),
                                  " vertices)")

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::line_graph_type hypergraph<vertex_t, weight_t>::h2h()
    const {
  const auto& self = readable(impl(), "hypergraph::h2h");
  DYNG_EXPECTS(self.props.line_graph != line_graph_policy::none,
               "hypergraph::h2h: the hypergraph was built with line_graph_policy::none");
  throw not_supported_error(
      "dyng: hypergraph::h2h: the line graph (line_graph_policy::automatic) is not available in "
      "this build yet");
}

template <typename vertex_t, typename weight_t>
array_view<const weight_t> hypergraph<vertex_t, weight_t>::weights() const {
  return host_view(impl().weights);
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hypergraph<vertex_t, weight_t>::apply(const resources& res,
                                                          const batch_view_type& batch) {
  return detail::hypergraph_access::apply(res, *this, batch, false);
}

template <typename vertex_t, typename weight_t>
typename hypergraph<vertex_t, weight_t>::list_type
hypergraph<vertex_t, weight_t>::to_hyperedge_list(const resources& res) const try {
  detail::hypergraph_access::expect_placement(res, *this, "hypergraph::to_hyperedge_list");
  const auto& self = readable(impl(), "hypergraph::to_hyperedge_list");
  array_view<const std::int64_t> row_ptr;
  array_view<const std::int32_t> col_ind;
  self.store->h2v_view(row_ptr, col_ind);
  const std::vector<std::int64_t> ptr = to_vector(res, row_ptr);
  const std::vector<std::int32_t> cols = to_vector(res, col_ind);
  list_type out;
  out.num_vertices = num_vertices();
  out.offsets.assign(1, 0);
  for (std::size_t id = 0; id + 1 < ptr.size(); ++id) {
    if (ptr[id + 1] == ptr[id]) continue;  // a dead id
    out.members.insert(out.members.end(), cols.begin() + ptr[id], cols.begin() + ptr[id + 1]);
    out.offsets.push_back(static_cast<std::int64_t>(out.members.size()));
    if constexpr (!is_unweighted_v<weight_t>) out.weights.push_back(self.weights[id]);
  }
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::to_hyperedge_list (", num_hyperedges(),
                                  " hyperedges)")

template <typename vertex_t, typename weight_t>
storage_stats hypergraph<vertex_t, weight_t>::store_stats() const noexcept {
  if (!impl_ || !impl_->store) return {};
  try {
    return impl_->store->stats();
  } catch (...) {
    return {};
  }
}

template <typename vertex_t, typename weight_t>
void hypergraph<vertex_t, weight_t>::check_integrity(const resources& res) const try {
  detail::hypergraph_access::expect_placement(res, *this, "hypergraph::check_integrity");
  const auto& self = readable(impl(), "hypergraph::check_integrity");
  self.store->check_integrity();
  if constexpr (!is_unweighted_v<weight_t>) {
    const std::int64_t bound = self.store->id_bound();
    if (static_cast<std::int64_t>(self.weights.size()) != bound) {
      DYNG_FAIL("hypergraph integrity: ", self.weights.size(), " weights for an id bound of ",
                bound);
    }
    for (std::int64_t id = 0; id < bound; ++id) {
      const auto w = static_cast<std::int64_t>(self.weights[static_cast<std::size_t>(id)]);
      const bool alive = self.store->is_live(id);
      if (alive ? w < 1 : w != 0) {
        DYNG_FAIL("hypergraph integrity: id ", id, (alive ? " is live" : " is dead"),
                  " with the weight ", w);
      }
    }
  }
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::check_integrity")

namespace detail {

#if !DYNG_HAS_CUDA
namespace hyper {
std::unique_ptr<incidence_store> make_escher_store(const resources& /*res*/,
                                                   std::int32_t /*num_vertices*/,
                                                   const member_rows& /*rows*/,
                                                   const capacity_policy& /*capacity*/,
                                                   bool /*has_v2h*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
std::unique_ptr<incidence_store> make_escher_store(const resources& /*res*/,
                                                   const store_snapshot& /*snapshot*/,
                                                   const capacity_policy& /*capacity*/) {
  throw not_supported_error(
      "dyng: the cuda backend is not built (configure with "
      "DYNG_ENABLE_CUDA=ON)");
}
}  // namespace hyper
#endif

template <typename vertex_t, typename weight_t>
void hypergraph_access::expect_placement(const resources& res,
                                         const hypergraph<vertex_t, weight_t>& h,
                                         const char* what) {
  const auto& state = h.impl();
  const backend b = res.get_backend();
  if (b == backend::cuda) {
    DYNG_EXPECTS(state.home == backend::cuda && state.home_device == res.device(), what,
                 ": the hypergraph was built for ", to_string(state.home),
                 state.home == backend::cuda ? " on CUDA device " : "",
                 state.home == backend::cuda ? std::to_string(state.home_device) : std::string(),
                 " but the resources are cuda on device ", res.device(),
                 "; build the hypergraph with these resources or move it with h.to_backend(res)");
  } else {
    DYNG_EXPECTS(state.home != backend::cuda, what, ": the hypergraph is resident on CUDA device ",
                 state.home_device, " but the resources are ", to_string(b),
                 "; move it with h.to_backend(res)");
  }
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hypergraph_access::apply(const resources& res,
                                             hypergraph<vertex_t, weight_t>& h,
                                             const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                             bool record_incidence_delta) try {
  auto& state = h.impl();
  expect_placement(res, h, "hypergraph::apply");
  DYNG_EXPECTS(!state.poisoned,
               "hypergraph::apply: the hypergraph was poisoned by a failed apply(); rebuild it "
               "with from_hyperedges()");
  const scoped_stage stage(res, "hypergraph.apply");
  // ---- Validation: everything that can fail before the store changes (strong guarantee).
  if (!batch.delete_incidence_hyperedge.empty() || !batch.delete_incidence_vertex.empty() ||
      !batch.insert_incidence_hyperedge.empty() || !batch.insert_incidence_vertex.empty()) {
    throw not_supported_error(
        "dyng: hypergraph::apply: incidence edits (delete_incidence / insert_incidence) are "
        "planned for 0.4; delete the hyperedge and insert it with its new members instead");
  }
  const host_input<std::int64_t> deletions(res, batch.delete_hyperedges,
                                           "hypergraph::apply: delete_hyperedges");
  const host_input<std::int64_t> offsets(res, batch.insert_offsets,
                                         "hypergraph::apply: insert_offsets");
  const host_input<vertex_t> members(res, batch.insert_members,
                                     "hypergraph::apply: insert_members");
  const host_input<weight_t> weights(res, batch.insert_weights,
                                     "hypergraph::apply: insert_weights");
  hyper::incidence_store& store = *state.store;
  hyper::member_rows rows;
  std::int64_t dropped = 0;
  normalize_rows(offsets.view(), members.view(), static_cast<vertex_t>(store.num_vertices()),
                 "hypergraph::apply", rows, dropped, /*batch_payload=*/true);
  const std::vector<weight_t> inserted_weights =
      checked_weights(weights.view(), rows.size(), "hypergraph::apply");
  // Deletions are checked against the state before the batch (MOSP_ESCHER's applyBatch, phase
  // 1): a negative, out-of-range, dead or repeated id is skipped and counted.
  std::vector<std::int64_t> live_deletions;
  std::int64_t ignored = 0;
  {
    std::vector<std::int64_t> sorted;
    for (const std::int64_t id : deletions.view()) {
      if (id < 0 || id >= store.id_bound() || !store.is_live(id)) {
        ++ignored;
        continue;
      }
      live_deletions.push_back(id);
    }
    // Repeated ids: the first deletion counts, the others are skipped.
    sorted = live_deletions;
    std::sort(sorted.begin(), sorted.end());
    if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
      std::vector<std::int64_t> unique_order;
      std::vector<std::int64_t> seen;
      unique_order.reserve(live_deletions.size());
      for (const std::int64_t id : live_deletions) {
        const auto it = std::lower_bound(seen.begin(), seen.end(), id);
        if (it != seen.end() && *it == id) {
          ++ignored;
          continue;
        }
        seen.insert(it, id);
        unique_order.push_back(id);
      }
      live_deletions = std::move(unique_order);
    }
  }
  // ---- The store phase (basic guarantee: a failure poisons the hypergraph).
  hyper_applied applied;
  std::vector<weight_t> next_weights;
  try {
    const container_scope container;  // the container's own work (budgets, I9)
    store.set_host_threads(resources_access::host_threads(res));
    hyper::grouped_incidences groups;
    std::vector<std::int64_t> ids;
    // The members of the deleted ids, read from the store (the callers do not supply them).
    store.read_rows(live_deletions, applied.deleted_members);
    if (state.props.reuse == id_reuse::erase_first) {
      // ESCHER-GPU's DynamicHypergraph::update: H2V erase, V2H unfill, H2V insert, V2H fill with
      // the keys the insert assigned.
      {
        const scoped_stage delete_stage(res, "hypergraph.apply.delete");
        store.erase(live_deletions);
        hyper::group_by_vertex(applied.deleted_members, live_deletions, groups);
        store.v2h_remove(groups);
      }
      {
        const scoped_stage insert_stage(res, "hypergraph.apply.insert");
        store.insert(rows, ids);
        hyper::group_by_vertex(rows, ids, groups);
        store.v2h_append(groups);
      }
    } else {
      // MOSP_ESCHER's beginBatch / finishBatch: the H2V insert decides the ids first; then H2V
      // erase, V2H unfill, V2H fill.
      {
        const scoped_stage insert_stage(res, "hypergraph.apply.insert");
        store.insert(rows, ids);
      }
      {
        const scoped_stage delete_stage(res, "hypergraph.apply.delete");
        store.erase(live_deletions);
        hyper::group_by_vertex(applied.deleted_members, live_deletions, groups);
        store.v2h_remove(groups);
      }
      {
        const scoped_stage insert_stage(res, "hypergraph.apply.insert");
        hyper::group_by_vertex(rows, ids, groups);
        store.v2h_append(groups);
      }
    }
    // The applied record.
    applied.inserted_ids = ids;
    if (state.home == backend::cuda && !ids.empty()) {
      applied.inserted_ids_device = buffer<std::int64_t>(res, ids.size());
      copy<std::int64_t>(res, host_view(applied.inserted_ids), applied.inserted_ids_device.view());
    }
    applied.deleted_ids = live_deletions;
    applied.touched_hyperedges = live_deletions;
    applied.touched_hyperedges.insert(applied.touched_hyperedges.end(), ids.begin(), ids.end());
    std::sort(applied.touched_hyperedges.begin(), applied.touched_hyperedges.end());
    applied.touched_hyperedges.erase(
        std::unique(applied.touched_hyperedges.begin(), applied.touched_hyperedges.end()),
        applied.touched_hyperedges.end());
    applied.touched_vertices = applied.deleted_members.members;
    applied.touched_vertices.insert(applied.touched_vertices.end(), rows.members.begin(),
                                    rows.members.end());
    std::sort(applied.touched_vertices.begin(), applied.touched_vertices.end());
    applied.touched_vertices.erase(
        std::unique(applied.touched_vertices.begin(), applied.touched_vertices.end()),
        applied.touched_vertices.end());
    applied.ignored_deletions = ignored;
    applied.dropped_members = dropped;
    if (record_incidence_delta) {
      applied.has_incidence_delta = true;
      auto& delta = applied.incidence_delta;
      const auto add = [&](const hyper::member_rows& r, const std::vector<std::int64_t>& rid,
                           std::int8_t sign) {
        for (std::int64_t i = 0; i < r.size(); ++i) {
          for (std::int64_t k = r.offsets[static_cast<std::size_t>(i)];
               k < r.offsets[static_cast<std::size_t>(i) + 1]; ++k) {
            delta.vertices.push_back(r.members[static_cast<std::size_t>(k)]);
            delta.ids.push_back(rid[static_cast<std::size_t>(i)]);
            delta.signs.push_back(sign);
          }
        }
      };
      add(applied.deleted_members, live_deletions, -1);
      add(rows, ids, +1);
    }
    if constexpr (!is_unweighted_v<weight_t>) {
      next_weights = state.weights;
      for (const std::int64_t id : live_deletions) next_weights[static_cast<std::size_t>(id)] = 0;
      next_weights.resize(static_cast<std::size_t>(store.id_bound()), weight_t{0});
      for (std::size_t i = 0; i < ids.size(); ++i) {
        next_weights[static_cast<std::size_t>(ids[i])] = inserted_weights[i];
      }
    }
    // One read of the device error words, then the row views of the new state.
    store.finish_apply();
  } catch (...) {
    state.poisoned = true;
    throw;
  }
  // ---- The commit (no-throw).
  applied.version_before = state.version;
  applied.state_before = state.state_id;
  ++state.version;
  state.state_id = next_graph_state_id();
  applied.version_after = state.version;
  applied.state_after = state.state_id;
  if constexpr (!is_unweighted_v<weight_t>) state.weights.swap(next_weights);
  hyper_apply_summary summary;
  summary.inserted_hyperedges = static_cast<std::int64_t>(applied.inserted_ids.size());
  summary.deleted_hyperedges = static_cast<std::int64_t>(live_deletions.size());
  summary.ignored_deletions = ignored;
  summary.dropped_members = dropped;
  summary.skipped_ops = ignored + dropped;
  summary.num_hyperedges_after = store.num_live();
  summary.id_bound_after = store.id_bound();
  summary.inserted_ids = applied.inserted_ids;
  state.applied = std::move(applied);
  return summary;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::apply (", h.hyperedge_id_bound(), " ids; batch of ",
                                  batch.num_insertions(), " insertions, ", batch.num_deletions(),
                                  " deletions)")

}  // namespace detail

#define DYNG_INSTANTIATE_HYPERGRAPH(V, W)                                            \
  template class hypergraph<V, W>;                                                   \
  template hyper_apply_summary detail::hypergraph_access::apply<V, W>(               \
      const resources&, hypergraph<V, W>&, const hyperedge_batch_view<V, W>&, bool); \
  template void detail::hypergraph_access::expect_placement<V, W>(                   \
      const resources&, const hypergraph<V, W>&, const char*);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPERGRAPH)
#undef DYNG_INSTANTIATE_HYPERGRAPH

}  // namespace dyng
