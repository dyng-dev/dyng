// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
/**
 * @file m9_triad_count_fitness.cpp
 * @brief Compile-only fitness skeleton of the frozen hypergraph API for M9, retargeted at the real
 *        triad_count API (M9a step 3; the hypergraph API review, ADR 0047; ADR 0052).
 *
 * Nothing here runs; the target dyng_api_fitness only compiles it (in every build), so a change
 * that removes what M9 needs fails the build:
 *  1. triad_count::compute() of every pattern on every backend, the result's accessors and the
 *     options (kind fixed, cuda_engine a tunable);
 *  2. triad_count::update() and dyng::update(res, h, batch, r1, r2) (update_traits /
 *     participant_of<hypergraph>, detail::make_triad_count_participant): one commit for several
 *     results, the stats (the frontier sizes, triads removed and added, the per-bin delta);
 *  3. replay by key (a .dgb file's expected ids against inserted_ids) and by content (its deleted
 *     member sets resolved against h2v(): the public view is the replay's, never the counting
 *     path's, which reads the internal row views of detail::hypergraph_access::views()).
 */
#include "hypergraph/hypergraph_impl.hpp"
#include "hypergraph/row_views.hpp"

#include <dyng/core/copy.hpp>
#include <dyng/dyng.hpp>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

namespace dyng_fitness_m9 {

using hgraph = dyng::hypergraph<std::int32_t, dyng::unweighted>;
using batch_view = dyng::hyperedge_batch_view<std::int32_t, dyng::unweighted>;
namespace tc = dyng::triad_count;

static_assert(dyng::detail::triad_count_supported_v<std::int32_t, dyng::unweighted>);
static_assert(dyng::detail::triad_count_supported_v<std::int32_t, std::int32_t>);
static_assert(!dyng::detail::triad_count_supported_v<std::int64_t, dyng::unweighted>);
static_assert(std::is_same_v<dyng::detail::update_traits<tc::result>::stats_type, tc::stats>);

/// What the kernels of count(-) / count(+) read (internal-stable): the rows of ESCHER-GPU's layout
/// (keys = id + 1) and, on CUDA, the applied record's device copy of the ASSIGNED inserted ids.
struct engine_inputs {
  dyng::detail::hyper::row_views rows;  ///< h2v (keys 1..bound), h2h (keys 1..max bound, out)
  const std::int64_t* inserted_ids = nullptr;
  std::size_t count = 0;
};

[[maybe_unused]] engine_inputs inputs_of(const hgraph& h) {
  engine_inputs in;
  in.rows = dyng::detail::hypergraph_access::views(h);
  const dyng::detail::hyper_applied& applied = dyng::detail::hypergraph_access::applied(h);
  in.inserted_ids = applied.inserted_ids_device.data();
  in.count = applied.inserted_ids.size();
  return in;
}

/// One update on every backend; replay of a .dgb file by key and by content.
[[maybe_unused]] void fitness(const std::string& hg_file, const std::string& dgb_file) {
  for (const dyng::resources& res :
       {dyng::resources::sequential(), dyng::resources::openmp(8), dyng::resources::cuda(0)}) {
    const auto list = dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>(
        hg_file, dyng::io::hypergraph_read_options::escher_gpu());
    dyng::hypergraph_properties props;
    props.line_graph = dyng::line_graph_policy::sorted_merge;
    props.capacity = dyng::capacity_policy::escher_gpu();
    hgraph by_key = hgraph::from_hyperedges(res, list.view(), props);
    hgraph by_content = by_key.clone(res);
    tc::options motif;
    motif.cuda_engine = dyng::engine::fused;
    tc::result r = tc::compute(res, by_key, motif);
    tc::options type1;
    type1.kind = tc::pattern::stathyper_type1;
    tc::result r1 = tc::compute(res, by_key, type1);
    tc::result r2 = tc::compute(res, by_content, motif);
    if (r.num_bins() != tc::max_bins || r1.num_bins() != 1 || r.kind() != tc::pattern::h_motif)
      throw dyng::internal_error("bins");
    const dyng::array_view<const std::uint64_t> counts = r.counts();
    (void)counts.size();
    (void)r.count(0);
    (void)r.total();
    (void)tc::to_string(r1.kind());
    tc::result copy = r.clone(dyng::resources::sequential());
    copy.set_options(motif);
    dyng::io::batch_reader in(res, dgb_file);
    while (in.next()) {
      const batch_view b = in.hyperedge_batch<std::int32_t, dyng::unweighted>();
      const auto notes = in.annotations<std::int32_t>();
      // by key: the generating run's ids; two results, one commit
      const auto [st, st1] = dyng::update(res, by_key, b, r, r1);
      const std::vector<std::int64_t> want = dyng::to_vector(res, notes.expected_ids);
      if (st.batch.inserted_ids != want || st1.inserted_frontier != st.inserted_frontier)
        throw dyng::internal_error("ids differ");
      std::int64_t net = 0;
      for (const std::int64_t d : st.delta) net += d;
      if (static_cast<std::uint64_t>(net) != st.triads_added - st.triads_removed)
        throw dyng::internal_error("delta");
      (void)st.deleted_frontier;
      // by content: each deleted member set resolved to a live id of the same set
      const auto h2v = by_content.h2v();
      const std::vector<std::int64_t> ptr = dyng::to_vector(res, h2v.row_ptr);
      const std::vector<std::int32_t> col = dyng::to_vector(res, h2v.col_ind);
      std::multimap<std::vector<std::int32_t>, std::int64_t> ids;
      for (std::size_t i = 0; i + 1 < ptr.size(); ++i)
        if (ptr[i + 1] > ptr[i])
          ids.emplace(std::vector<std::int32_t>(col.begin() + ptr[i], col.begin() + ptr[i + 1]),
                      static_cast<std::int64_t>(i));
      std::vector<std::int64_t> deletions;
      const std::vector<std::int64_t> doff = dyng::to_vector(res, notes.deleted_offsets);
      const std::vector<std::int32_t> dmem = dyng::to_vector(res, notes.deleted_members);
      for (std::size_t i = 0; i + 1 < doff.size(); ++i) {
        const auto it =
            ids.find(std::vector<std::int32_t>(dmem.begin() + doff[i], dmem.begin() + doff[i + 1]));
        if (it != ids.end()) {
          deletions.push_back(it->second);
          ids.erase(it);
        }
      }
      batch_view c = b;
      c.delete_hyperedges = dyng::host_view(deletions);
      (void)tc::update(res, by_content, c, r2);
    }
    if (r.counts().size() != r2.counts().size()) throw dyng::internal_error("modes differ");
  }
}

}  // namespace dyng_fitness_m9
