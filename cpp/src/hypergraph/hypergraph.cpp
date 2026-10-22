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
#include "hypergraph/transaction.hpp"
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
  // A batch's rows are its insertions: "inserted hyperedge i of the batch", not the id i (R030
  // review), and a member outside the vertex set names where that set is fixed.
  const char* row_name = batch_payload ? "inserted hyperedge " : "hyperedge ";
  const char* of_batch = batch_payload ? " of the batch" : "";
  const char* fixed = batch_payload
                          ? " (the vertex count is fixed when the hypergraph is built: its "
                            "num_vertices)"
                          : "";
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
    DYNG_EXPECTS(begin <= end, what, ": the offsets decrease at ", row_name, i, of_batch);
    DYNG_EXPECTS(begin < end, what, ": ", row_name, i, of_batch,
                 " is empty (a hyperedge needs a member)");
    row.clear();
    for (std::int64_t k = begin; k < end; ++k) {
      const vertex_t v = members[static_cast<std::size_t>(k)];
      DYNG_EXPECTS(v >= 0 && v < num_vertices, what, ": ", row_name, i, of_batch,
                   " has the vertex ", static_cast<std::int64_t>(v), ", outside [0, ",
                   static_cast<std::int64_t>(num_vertices), ")", fixed);
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
  const bool line_graph = props.line_graph == line_graph_policy::sorted_merge;
  if (res.get_backend() == backend::cuda) {
    const detail::container_scope container;  // the container's own work (budgets, I9)
    impl->store =
        detail::hyper::make_escher_store(res, static_cast<std::int32_t>(hyperedges.num_vertices),
                                         rows, props.capacity, props.store_v2h, line_graph);
  } else if (line_graph) {
    // The line graph of sorted_merge from the start (ESCHER-GPU's init.h2h_build).
    const scoped_stage stage(res, "hypergraph.build");
    impl->store->enable_line_graph();
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
typename hypergraph<vertex_t, weight_t>::line_graph_type hypergraph<vertex_t, weight_t>::h2h() const
    try {
  const auto& self = readable(impl(), "hypergraph::h2h");
  detail::hypergraph_access::require_line_graph(*this, "hypergraph::h2h");
  line_graph_type out;
  self.store->h2h_view(out.row_ptr, out.col_ind, out.out);
  if constexpr (!is_unweighted_v<weight_t>) out.weights = host_view(self.weights);
  out.version = self.version;
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("hypergraph::h2h (", hyperedge_id_bound(), " ids)")

template <typename vertex_t, typename weight_t>
array_view<const weight_t> hypergraph<vertex_t, weight_t>::weights() const {
  return host_view(readable(impl(), "hypergraph::weights").weights);
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
                                                   bool /*has_v2h*/, bool /*line_graph*/) {
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
void hypergraph_access::require_line_graph(const hypergraph<vertex_t, weight_t>& h,
                                           const char* what) {
  const auto& state = h.impl();
  DYNG_EXPECTS(!state.poisoned, what,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  DYNG_EXPECTS(state.props.line_graph != line_graph_policy::none, what,
               ": the hypergraph was built with line_graph_policy::none (no line graph); build it "
               "with line_graph_policy::automatic or sorted_merge");
  DYNG_EXPECTS(state.props.store_v2h, what,
               ": the line graph needs v2h; build the hypergraph with store_v2h = true");
  // line_graph_policy::automatic builds it here, once, and maintains it from then on.
  state.store->enable_line_graph();
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hypergraph_access::apply(const resources& res,
                                             hypergraph<vertex_t, weight_t>& h,
                                             const hyperedge_batch_view<vertex_t, weight_t>& batch,
                                             bool record_incidence_delta) {
  // A transaction without participants (ADR 0043): begin (Step 0), then the commit.
  const scoped_stage stage(res, "hypergraph.apply");
  hyper_transaction<vertex_t, weight_t> txn(res, h, batch, "hypergraph::apply");
  return txn.commit(record_incidence_delta);
}

namespace {

/// begin's checks before the batch is staged: placement, a live hypergraph, no incidence edits.
template <typename vertex_t, typename weight_t>
const char* begin_checks(const resources& res, hypergraph<vertex_t, weight_t>& h,
                         const hyperedge_batch_view<vertex_t, weight_t>& batch,
                         const char* function) {
  hypergraph_access::expect_placement(res, h, function);
  DYNG_EXPECTS(!hypergraph_access::impl(h).poisoned, function,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  if (!batch.delete_incidence_hyperedge.empty() || !batch.delete_incidence_vertex.empty() ||
      !batch.insert_incidence_hyperedge.empty() || !batch.insert_incidence_vertex.empty()) {
    throw not_supported_error(concat_message(
        "dyng: ", function,
        ": incidence edits (delete_incidence / insert_incidence) are planned for 0.4; delete the "
        "hyperedge and insert it with its new members instead"));
  }
  return function;
}

}  // namespace

template <typename vertex_t, typename weight_t>
hyper_transaction<vertex_t, weight_t>::hyper_transaction(const resources& res, container_type& h,
                                                         const batch_view_type& batch,
                                                         const char* function) try
    : res_(res),
      h_(h),
      function_(begin_checks(res, h, batch, function)),
      deletions_(res, batch.delete_hyperedges, concat_message(function, ": delete_hyperedges")),
      offsets_(res, batch.insert_offsets, concat_message(function, ": insert_offsets")),
      members_(res, batch.insert_members, concat_message(function, ": insert_members")),
      weights_(res, batch.insert_weights, concat_message(function, ": insert_weights")) {
  // ---- Step 0, once: everything that can fail before the store changes (strong guarantee).
  host_.delete_hyperedges = deletions_.view();
  host_.insert_offsets = offsets_.view();
  host_.insert_members = members_.view();
  host_.insert_weights = weights_.view();
  const hyper::incidence_store& store = *hypergraph_access::impl(h).store;
  staged_.version = h.version();
  staged_.id_bound = store.id_bound();
  staged_.requested_insertions = static_cast<std::int64_t>(host_.num_insertions());
  staged_.requested_deletions = static_cast<std::int64_t>(host_.num_deletions());
  normalize_rows(offsets_.view(), members_.view(), static_cast<vertex_t>(store.num_vertices()),
                 function, staged_.rows, staged_.dropped_members, /*batch_payload=*/true);
  staged_.weights = checked_weights(weights_.view(), staged_.rows.size(), function);
  // Deletions are checked against G_t (MOSP_ESCHER's applyBatch, phase 1): a negative,
  // out-of-range, dead or repeated id is skipped and counted.
  std::vector<std::int64_t>& live = staged_.deletions;
  std::int64_t ignored = 0;
  for (const std::int64_t id : deletions_.view()) {
    if (id < 0 || id >= store.id_bound() || !store.is_live(id)) {
      ++ignored;
      continue;
    }
    live.push_back(id);
  }
  // Repeated ids: the first deletion counts, the others are skipped. Only a batch with a repeated
  // id pays for more than the sort: its (id, position) pairs are sorted, and the first position
  // of every id is kept (O(k log k) in the batch, independent of the id bound).
  std::vector<std::int64_t> sorted = live;
  std::sort(sorted.begin(), sorted.end());
  if (std::adjacent_find(sorted.begin(), sorted.end()) != sorted.end()) {
    std::vector<std::pair<std::int64_t, std::size_t>> at(live.size());
    for (std::size_t i = 0; i < live.size(); ++i) at[i] = {live[i], i};
    std::sort(at.begin(), at.end());
    std::vector<std::uint8_t> keep(live.size(), 0);
    for (std::size_t i = 0; i < at.size(); ++i) {
      if (i == 0 || at[i].first != at[i - 1].first) {
        keep[at[i].second] = 1;
      } else {
        ++ignored;
      }
    }
    std::vector<std::int64_t> unique_order;
    unique_order.reserve(live.size());
    for (std::size_t i = 0; i < live.size(); ++i)
      if (keep[i] != 0) unique_order.push_back(live[i]);
    live = std::move(unique_order);
  }
  staged_.ignored_deletions = ignored;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE(function, " (", h.hyperedge_id_bound(), " ids; batch of ",
                                  batch.num_insertions(), " insertions, ", batch.num_deletions(),
                                  " deletions)")

template <typename vertex_t, typename weight_t>
hyper_transaction<vertex_t, weight_t>::~hyper_transaction() = default;

template <typename vertex_t, typename weight_t>
framework::old_view<hypergraph<vertex_t, weight_t>> hyper_transaction<vertex_t, weight_t>::before()
    const {
  if (committed_) {
    DYNG_FAIL(function_, ": the transaction's G_t was asked for after its commit (invariant I1)");
  }
  return framework::old_view<container_type>(h_);
}

template <typename vertex_t, typename weight_t>
framework::new_view<hypergraph<vertex_t, weight_t>> hyper_transaction<vertex_t, weight_t>::after()
    const {
  if (!committed_) {
    DYNG_FAIL(function_, ": the transaction's G_{t+1} was asked for before its commit");
  }
  return framework::new_view<container_type>(h_);
}

template <typename vertex_t, typename weight_t>
const hyper_applied& hyper_transaction<vertex_t, weight_t>::applied() const {
  if (!committed_) {
    DYNG_FAIL(function_, ": the transaction's applied record was asked for before its commit");
  }
  return hypergraph_access::applied(h_);
}

template <typename vertex_t, typename weight_t>
hyper_apply_summary hyper_transaction<vertex_t, weight_t>::commit(bool record_incidence_delta) try {
  if (committed_) {
    DYNG_FAIL(function_, ": a transaction commits once");
  }
  auto& state = hypergraph_access::state(h_);
  DYNG_EXPECTS(!state.poisoned, function_,
               ": the hypergraph was poisoned by a failed apply(); rebuild it with "
               "from_hyperedges()");
  DYNG_EXPECTS(state.version == staged_.version, function_, ": the hypergraph changed (version ",
               staged_.version, " -> ", state.version,
               ") between the transaction's begin and its commit");
  const resources& res = res_;
  const hyper::member_rows& rows = staged_.rows;
  const std::vector<std::int64_t>& live_deletions = staged_.deletions;
  const std::int64_t ignored = staged_.ignored_deletions;
  const std::int64_t dropped = staged_.dropped_members;
  hyper::incidence_store& store = *state.store;
  // ---- The store phase (basic guarantee: a failure poisons the hypergraph).
  hyper_applied applied;
  hyper_apply_summary summary;
  std::vector<weight_t> next_weights;
  try {
    const container_scope container;  // the container's own work (budgets, I9)
    store.set_host_threads(resources_access::host_threads(res));
    // The store works on the stream it was built with: its work is ordered after the work queued
    // so far on the stream of `res` (a reader of G_t's views there, count(-) of dyng::update),
    // and the work later queued on that stream after the store's (ADR 0041 section 4).
    store.wait_for(res);
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
    if (store.line_graph_enabled()) {
      // ESCHER-GPU's maintain.h2h_merge: the sorted delta merge of the line graph.
      const scoped_stage line_graph_stage(res, "hypergraph.apply.line_graph");
      store.update_line_graph(live_deletions, ids, rows);
    }
    // The applied record.
    applied.inserted_ids = ids;
    if (state.home == backend::cuda && !ids.empty()) {
      applied.inserted_ids_device = buffer<std::int64_t>(res, ids.size());
      copy<std::int64_t>(res, host_view(applied.inserted_ids), applied.inserted_ids_device.view());
    }
    applied.deleted_ids = live_deletions;
    // The touched hyperedges and vertices, sorted and unique, from the batch alone (reusable
    // marks: no O(|E| + |V|) pass per batch; sorting all incidences took about 16 ms of a 50K
    // batch on synthetic 1M, parity/results/M8b.md section 2.4, and a scan of every id and vertex
    // about 9 ms of every coauth batch, the M8b review).
    {
      std::int64_t id_end = store.id_bound();
      for (const std::int64_t id : live_deletions) id_end = std::max(id_end, id + 1);
      state.touched_ids.begin(static_cast<std::size_t>(id_end));
      for (const std::int64_t id : live_deletions) state.touched_ids.mark(id);
      for (const std::int64_t id : ids) state.touched_ids.mark(id);
      state.touched_ids.take_sorted(applied.touched_hyperedges);
      state.touched_vertices.begin(static_cast<std::size_t>(store.num_vertices()));
      for (const std::int32_t v : applied.deleted_members.members) state.touched_vertices.mark(v);
      for (const std::int32_t v : rows.members) state.touched_vertices.mark(v);
      state.touched_vertices.take_sorted(applied.touched_vertices);
    }
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
        next_weights[static_cast<std::size_t>(ids[i])] = staged_.weights[i];
      }
    }
    // One read of the device error words, then the row views of the new state.
    store.finish_apply();
    store.signal_to(res);
    // The summary, built here: everything that allocates comes before the commit point.
    summary.inserted_hyperedges = static_cast<std::int64_t>(applied.inserted_ids.size());
    summary.deleted_hyperedges = static_cast<std::int64_t>(live_deletions.size());
    summary.ignored_deletions = ignored;
    summary.dropped_members = dropped;
    summary.skipped_ops = ignored + dropped;
    summary.num_hyperedges_after = store.num_live();
    summary.id_bound_after = store.id_bound();
    summary.inserted_ids = applied.inserted_ids;
  } catch (...) {
    state.poisoned = true;
    throw;
  }
  // ---- The commit (no-throw: integer assignments, a swap and moves only).
  applied.version_before = state.version;
  applied.state_before = state.state_id;
  ++state.version;
  state.state_id = next_graph_state_id();
  applied.version_after = state.version;
  applied.state_after = state.state_id;
  if constexpr (!is_unweighted_v<weight_t>) state.weights.swap(next_weights);
  static_assert(std::is_nothrow_move_assignable_v<hyper_applied>,
                "the commit moves the applied record after the commit point");
  state.applied = std::move(applied);
  committed_ = true;
  return summary;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE(function_, " (", h_.hyperedge_id_bound(), " ids; batch of ",
                                  staged_.requested_insertions, " insertions, ",
                                  staged_.requested_deletions, " deletions)")

}  // namespace detail

#define DYNG_INSTANTIATE_HYPERGRAPH(V, W)                                                    \
  template class hypergraph<V, W>;                                                           \
  template class detail::hyper_transaction<V, W>;                                            \
  template hyper_apply_summary detail::hypergraph_access::apply<V, W>(                       \
      const resources&, hypergraph<V, W>&, const hyperedge_batch_view<V, W>&, bool);         \
  template void detail::hypergraph_access::expect_placement<V, W>(                           \
      const resources&, const hypergraph<V, W>&, const char*);                               \
  template void detail::hypergraph_access::require_line_graph<V, W>(const hypergraph<V, W>&, \
                                                                    const char*);
DYNG_FOR_EACH_HYPERGRAPH_TYPE(DYNG_INSTANTIATE_HYPERGRAPH)
#undef DYNG_INSTANTIATE_HYPERGRAPH

}  // namespace dyng
