// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/src/hsospDevice.cu
/**
 * @file engine.cu
 * @brief hyper_sssp's CUDA engine (engine::fused): the device state's lifecycle and MOSP_ESCHER's
 *        host loops, the shared propagation of the recompute (propagate), hsospRecompute and
 *        hsospUpdate with their budget and fallback.
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hyper_sssp.toml
 * (M11a, ADR 0091).
 */

#include "algorithms/hyper_sssp/kernels.cuh"
#include "algorithms/hyper_sssp/state.cuh"
#include "core/resources_access.hpp"
#include "util/cuda_check.hpp"

#include <dyng/core/error.hpp>

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

namespace dyng::detail::hsosp {

// ---------------------------------------------------------------------------
// device_state
// ---------------------------------------------------------------------------

void device_state::free() {
  *this = device_state();
}

void device_state::allocate(const resources& res, int n) {
  free();
  stream = static_cast<cudaStream_t>(res.stream().get());
  max_nodes = n;
  const auto un = static_cast<std::size_t>(n);
  b_dist = buffer<long long>(res, un);
  b_parent = buffer<int>(res, un);
  b_is_affected = buffer<int>(res, un);
  b_is_candidate = buffer<int>(res, un);
  b_cand_list = buffer<int>(res, un);
  b_aff_list = buffer<int>(res, un);
  b_counters = buffer<int>(res, 2);
  b_packed = buffer<unsigned long long>(res, un);
  b_inv_a = buffer<int>(res, un);
  b_inv_b = buffer<int>(res, un);
  b_jump_a = buffer<int>(res, un);
  b_jump_b = buffer<int>(res, un);
  b_front = buffer<int>(res, un);
  b_next = buffer<int>(res, un);
  b_stamp = buffer<int>(res, un);
  b_upd_counters = buffer<unsigned long long>(res, upd_counters);
  b_host_upd_counters = buffer<unsigned long long>(
      upd_counters, res.stream(), resources_access::staging_memory(res), res.device());
  d_dist = b_dist.data();
  d_parent = b_parent.data();
  d_is_affected = b_is_affected.data();
  d_is_candidate = b_is_candidate.data();
  d_cand_list = b_cand_list.data();
  d_aff_list = b_aff_list.data();
  d_counters = b_counters.data();
  d_packed = b_packed.data();
  d_inv_a = b_inv_a.data();
  d_inv_b = b_inv_b.data();
  d_jump_a = b_jump_a.data();
  d_jump_b = b_jump_b.data();
  d_front = b_front.data();
  d_next = b_next.data();
  d_stamp = b_stamp.data();
  d_upd_counters = b_upd_counters.data();
  h_upd_counters = b_host_upd_counters.data();
  DYNG_CUDA_TRY(cudaMemsetAsync(d_stamp, 0, sizeof(int) * n, stream));
  epoch = 0;

  const int block = 256;
  fill_ll_kernel<<<grid_for(n, block), block, 0, stream>>>(d_dist, n, inf_value);
  fill_int_kernel<<<grid_for(n, block), block, 0, stream>>>(d_parent, n, -1);
  DYNG_CUDA_TRY(cudaMemsetAsync(d_is_affected, 0, sizeof(int) * n, stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(d_is_candidate, 0, sizeof(int) * n, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
}

void device_state::allocate_holds(const resources& res) {
  b_holds = buffer<unsigned char>(res, static_cast<std::size_t>(max_nodes));
  d_holds = b_holds.data();
  DYNG_CUDA_TRY(cudaMemsetAsync(d_holds, 0, static_cast<std::size_t>(max_nodes), stream));
}

long long device_state::device_bytes() const {
  // dynG (transplant hunk): + the public mode's source flags (none in the virtual mode, whose bytes
  // are MOSP_ESCHER's).
  return static_cast<long long>(max_nodes) *
             (sizeof(long long) + 5 * sizeof(int) + sizeof(unsigned long long) + 8 * sizeof(int)) +
         2 * sizeof(int) + upd_counters * sizeof(unsigned long long) +
         (d_holds != nullptr ? static_cast<long long>(max_nodes) : 0);
}

void device_state::download_distances(std::vector<long long>& dist, int n) const {
  dist.resize(n);
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(dist.data(), d_dist, sizeof(long long) * n, cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
}

void device_state::download_parents(std::vector<int>& parent, int n) const {
  parent.resize(n);
  DYNG_CUDA_TRY(
      cudaMemcpyAsync(parent.data(), d_parent, sizeof(int) * n, cudaMemcpyDeviceToHost, stream));
  DYNG_CUDA_TRY(cudaStreamSynchronize(stream));
}

void device_state::download_parent_ids(std::vector<int>& parent_ids, int n) const {
  download_parents(parent_ids, n);
  for (int& p : parent_ids)
    if (p >= 0) p += 1;
}

// ---------------------------------------------------------------------------
// Propagation loop
// ---------------------------------------------------------------------------

namespace {

/**
 * Core loop shared by update and recompute.
 *
 * @param start_with_candidates  true: initial_list holds candidate nodes to
 *        evaluate first (dynamic update seeds). false: initial_list holds
 *        affected nodes whose neighbors are collected first (recompute
 *        seeded at the source).
 * @return iterations used, or -1 if the cap was hit before convergence.
 */
int propagate(const device_graph& dev, device_state& st, const std::vector<int>& initial_list,
              bool start_with_candidates, int source0, const engine_config& cfg,
              int* max_frontier_out) {
  const int block = cfg.block_size;
  int* d_aff_count = st.d_counters;
  int* d_cand_count = st.d_counters + 1;

  int list_count = static_cast<int>(initial_list.size());
  if (list_count == 0) return 0;

  int* d_initial = start_with_candidates ? st.d_cand_list : st.d_aff_list;
  DYNG_CUDA_TRY(cudaMemcpyAsync(d_initial, initial_list.data(), sizeof(int) * list_count,
                                cudaMemcpyHostToDevice, st.stream));

  int iterations = 0;
  int max_frontier = list_count;
  int cand_count = 0;

  if (!start_with_candidates) {
    // Prime: collect candidates from the initial affected list.
    DYNG_CUDA_TRY(cudaMemsetAsync(d_cand_count, 0, sizeof(int), st.stream));
    collect_candidates_warp_kernel<<<grid_for(32LL * list_count, block), block, 0, st.stream>>>(
        st.d_aff_list, list_count, dev.d_row_start, dev.d_deg, dev.d_col_ind, st.d_is_candidate,
        st.d_cand_list, d_cand_count, st.d_is_affected, source0);
    DYNG_CUDA_TRY(cudaGetLastError());
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(&cand_count, d_cand_count, sizeof(int), cudaMemcpyDeviceToHost, st.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
  } else {
    cand_count = list_count;
  }

  while (cand_count > 0) {
    if (iterations >= cfg.max_iterations) return -1;
    ++iterations;
    max_frontier = std::max(max_frontier, cand_count);

    DYNG_CUDA_TRY(cudaMemsetAsync(d_aff_count, 0, sizeof(int), st.stream));
    pull_distances_warp_kernel<<<grid_for(32LL * cand_count, block), block, 0, st.stream>>>(
        st.d_cand_list, cand_count, dev.d_row_start, dev.d_deg, dev.d_col_ind, dev.d_node_w,
        st.d_dist, st.d_parent, st.d_is_candidate, st.d_is_affected, st.d_aff_list, d_aff_count,
        source0, dev.zero_a, dev.zero_b, dev.d_holds);
    DYNG_CUDA_TRY(cudaGetLastError());

    int aff_count = 0;
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(&aff_count, d_aff_count, sizeof(int), cudaMemcpyDeviceToHost, st.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
    if (aff_count == 0) {
      cand_count = 0;
      break;
    }

    DYNG_CUDA_TRY(cudaMemsetAsync(d_cand_count, 0, sizeof(int), st.stream));
    collect_candidates_warp_kernel<<<grid_for(32LL * aff_count, block), block, 0, st.stream>>>(
        st.d_aff_list, aff_count, dev.d_row_start, dev.d_deg, dev.d_col_ind, st.d_is_candidate,
        st.d_cand_list, d_cand_count, st.d_is_affected, source0);
    DYNG_CUDA_TRY(cudaGetLastError());
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(&cand_count, d_cand_count, sizeof(int), cudaMemcpyDeviceToHost, st.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
  }

  if (max_frontier_out) *max_frontier_out = max_frontier;
  return iterations;
}

/** The recompute kernels run one warp per node and reduce with full-warp
 *  shuffles, so a block must hold whole warps. */
void check_block_size(const engine_config& cfg, const char* where) {
  if (cfg.block_size < 32 || cfg.block_size > 1024 || cfg.block_size % 32) {
    throw invalid_argument_error(std::string("dyng: hyper_sssp: ") + where +
                                 ": block_size must be a multiple of 32 in [32, 1024], got " +
                                 std::to_string(cfg.block_size));
  }
}

void reset_for_recompute(const device_graph& dev, device_state& st, int source0) {
  const int n = dev.num_nodes;
  const int block = 256;
  fill_ll_kernel<<<grid_for(n, block), block, 0, st.stream>>>(st.d_dist, n, inf_value);
  fill_int_kernel<<<grid_for(n, block), block, 0, st.stream>>>(st.d_parent, n, -1);
  DYNG_CUDA_TRY(cudaMemsetAsync(st.d_is_affected, 0, sizeof(int) * n, st.stream));
  DYNG_CUDA_TRY(cudaMemsetAsync(st.d_is_candidate, 0, sizeof(int) * n, st.stream));
  if (source0 >= 0) {
    const long long zero = 0;
    DYNG_CUDA_TRY(cudaMemcpyAsync(st.d_dist + source0, &zero, sizeof(long long),
                                  cudaMemcpyHostToDevice, st.stream));
  } else if (dev.base != nullptr && !dev.base->empty()) {
    // dynG (transplant hunk), the public mode: the hyperedges holding the source at their own
    // weight (the source row's role; propagate collects their neighbours first).
    const int nb = static_cast<int>(dev.base->size());
    DYNG_CUDA_TRY(cudaMemcpyAsync(st.d_aff_list, dev.base->data(), sizeof(int) * nb,
                                  cudaMemcpyHostToDevice, st.stream));
    seed_base_kernel<<<grid_for(nb, block), block, 0, st.stream>>>(st.d_aff_list, nb, dev.d_node_w,
                                                                   st.d_dist);
  }
  DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
}

}  // namespace

engine_stats recompute(const device_graph& dev, device_state& st, int source_id,
                       const engine_config& cfg) {
  check_block_size(cfg, "recompute");
  engine_stats stats;
  const int source0 = source_id - 1;
  reset_for_recompute(dev, st, source0);

  engine_config rcfg = cfg;
  // Recompute-from-blank converges in <= eccentricity(source) rounds; the
  // stale-loop hazard of the dynamic path cannot occur. Cap generously.
  rcfg.max_iterations = std::max<std::int64_t>(cfg.max_iterations, dev.num_nodes + 1);

  // dynG (transplant hunk): the public mode starts from the hyperedges holding the source.
  std::vector<int> initial = {source0};
  if (source0 < 0) initial = dev.base != nullptr ? *dev.base : std::vector<int>{};
  int it = propagate(dev, st, initial, /*start_with_candidates=*/false, source0, rcfg,
                     &stats.max_frontier);
  stats.iterations = (it < 0) ? static_cast<int>(rcfg.max_iterations) : it;
  return stats;
}

engine_stats update(const device_graph& dev, device_state& st, int source_id,
                    const engine_config& cfg) {
  // Checked up front: the fallback recompute would otherwise throw
  // after the update had already modified the state.
  check_block_size(cfg, "update");
  engine_stats stats;
  const int n = dev.num_nodes;
  const int source0 = source_id - 1;
  const int block = 256;
  const device_delta& dd = dev.delta;
  const int2* del_pairs = dd.d_pairs;
  const int2* ins_pairs = dd.d_pairs + dd.num_del;
  const int* new_ids = dd.d_ids;
  u64* cnt = st.d_upd_counters;
  u64* hcnt = st.h_upd_counters;
  auto read_counters = [&]() {
    DYNG_CUDA_TRY(
        cudaMemcpyAsync(hcnt, cnt, sizeof(u64) * upd_counters, cudaMemcpyDeviceToHost, st.stream));
    DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
  };

  // dynG (transplant hunk): the dead nodes' degrees are zeroed by the slack_csr store's CSR apply
  // (ADR 0089), so the engine writes no container state.

  // 1. Pack the pre-batch state; mark the roots.
  DYNG_CUDA_TRY(cudaMemsetAsync(cnt, 0, sizeof(u64) * upd_counters, st.stream));
  pack_state_kernel<<<grid_for(n, block), block, 0, st.stream>>>(st.d_dist, st.d_parent,
                                                                 st.d_packed, n, cnt);
  DYNG_CUDA_TRY(cudaMemsetAsync(st.d_inv_a, 0, sizeof(int) * n, st.stream));
  if (dd.num_del > 0)
    mark_deleted_tree_edges_kernel<<<grid_for(dd.num_del, block), block, 0, st.stream>>>(
        del_pairs, dd.num_del, st.d_packed, st.d_inv_a);
  if (dd.num_new + dd.num_dead > 0)
    mark_ids_kernel<<<grid_for(dd.num_new + dd.num_dead, block), block, 0, st.stream>>>(
        new_ids, dd.num_new + dd.num_dead, st.d_inv_a);
  // dynG (transplant hunk), the public mode: a hyperedge that lost the source is a root if it was
  // reached from the source directly (uploaded into the unused candidate list).
  if (dev.lost != nullptr && !dev.lost->empty()) {
    const int nl = static_cast<int>(dev.lost->size());
    DYNG_CUDA_TRY(cudaMemcpyAsync(st.d_cand_list, dev.lost->data(), sizeof(int) * nl,
                                  cudaMemcpyHostToDevice, st.stream));
    mark_lost_source_kernel<<<grid_for(nl, block), block, 0, st.stream>>>(st.d_cand_list, nl,
                                                                          st.d_packed, st.d_inv_a);
  }
  DYNG_CUDA_TRY(cudaGetLastError());

  // 2. Invalidate the subtrees of the roots by pointer jumping. A parent
  //    cycle cannot occur with positive weights; the round cap only
  //    guards against a corrupt tree (then: fall back).
  init_jump_kernel<<<grid_for(n, block), block, 0, st.stream>>>(st.d_packed, st.d_jump_a, n);
  int* inv_in = st.d_inv_a;
  int* inv_out = st.d_inv_b;
  int* jump_in = st.d_jump_a;
  int* jump_out = st.d_jump_b;
  bool tree_ok = false;
  for (int round = 0; round < 64; ++round) {
    DYNG_CUDA_TRY(cudaMemsetAsync(cnt + cnt_active, 0, sizeof(u64), st.stream));
    jump_kernel<<<grid_for(n, block), block, 0, st.stream>>>(inv_in, inv_out, jump_in, jump_out, n,
                                                             cnt);
    DYNG_CUDA_TRY(cudaGetLastError());
    std::swap(inv_in, inv_out);
    std::swap(jump_in, jump_out);
    ++stats.jump_rounds;
    read_counters();
    if (hcnt[cnt_active] == 0) {
      tree_ok = true;
      break;
    }
  }
  const double work_limit = cfg.work_budget * static_cast<double>(std::max(1LL, dev.num_entries));
  bool budget_exceeded = !tree_ok;
  if (!tree_ok) stats.reason = hyper_sssp::fallback_reason::invalidation_depth;

  if (!budget_exceeded) {
    invalidate_kernel<<<grid_for(n, block), block, 0, st.stream>>>(inv_in, st.d_packed, st.d_front,
                                                                   cnt, n, source0);
    read_counters();
    const int n_inv = static_cast<int>(hcnt[cnt_list]);
    stats.seed_count = n_inv;

    // 3. Pull for the invalidated nodes, relax the inserted pairs; the
    //    improved nodes form the first frontier (epoch stamps).
    const int ep0 = ++st.epoch;
    DYNG_CUDA_TRY(cudaMemsetAsync(cnt + cnt_next, 0, sizeof(u64), st.stream));
    if (n_inv > 0)
      pull_invalidated_kernel<<<grid_for(32LL * n_inv, block), block, 0, st.stream>>>(
          st.d_front, n_inv, dev.d_row_start, dev.d_deg, dev.d_col_ind, dev.d_node_w, st.d_packed,
          st.d_stamp, ep0, st.d_next, cnt, dev.zero_a, dev.zero_b, dev.d_holds);
    if (dd.num_ins > 0)
      relax_inserted_kernel<<<grid_for(dd.num_ins, block), block, 0, st.stream>>>(
          ins_pairs, dd.num_ins, dev.d_node_w, st.d_packed, st.d_stamp, ep0, st.d_next, cnt,
          source0, dev.zero_a, dev.zero_b);
    // dynG (transplant hunk), the public mode: the hyperedges that gained the source are offered
    // their own weight (uploaded into the unused candidate list).
    if (dev.gained != nullptr && !dev.gained->empty()) {
      const int ng = static_cast<int>(dev.gained->size());
      DYNG_CUDA_TRY(cudaMemcpyAsync(st.d_cand_list, dev.gained->data(), sizeof(int) * ng,
                                    cudaMemcpyHostToDevice, st.stream));
      relax_base_kernel<<<grid_for(ng, block), block, 0, st.stream>>>(
          st.d_cand_list, ng, dev.d_node_w, st.d_packed, st.d_stamp, ep0, st.d_next, cnt);
    }
    DYNG_CUDA_TRY(cudaGetLastError());
    read_counters();

    // 4. Push until no distance decreases (or the budget is spent).
    int frontier = static_cast<int>(hcnt[cnt_next]);
    stats.max_frontier = frontier;
    while (frontier > 0) {
      if (stats.iterations >= cfg.max_iterations ||
          static_cast<double>(hcnt[cnt_work]) > work_limit || hcnt[cnt_overflow] != 0) {
        budget_exceeded = true;
        // dynG (transplant hunk): the reason, in MOSP_ESCHER's order of the test (HD12).
        stats.reason = stats.iterations >= cfg.max_iterations
                           ? hyper_sssp::fallback_reason::iterations
                       : static_cast<double>(hcnt[cnt_work]) > work_limit
                           ? hyper_sssp::fallback_reason::work_budget
                           : hyper_sssp::fallback_reason::distance_width;
        break;
      }
      ++stats.iterations;
      std::swap(st.d_front, st.d_next);
      const int ep = ++st.epoch;
      DYNG_CUDA_TRY(cudaMemsetAsync(cnt + cnt_next, 0, sizeof(u64), st.stream));
      push_kernel<<<grid_for(32LL * frontier, block), block, 0, st.stream>>>(
          st.d_front, frontier, dev.d_row_start, dev.d_deg, dev.d_col_ind, dev.d_node_w,
          st.d_packed, st.d_stamp, ep, st.d_next, cnt, source0, dev.zero_a, dev.zero_b);
      DYNG_CUDA_TRY(cudaGetLastError());
      read_counters();
      frontier = static_cast<int>(hcnt[cnt_next]);
      stats.max_frontier = std::max(stats.max_frontier, frontier);
    }
    stats.work = static_cast<long long>(hcnt[cnt_work]);
    if (hcnt[cnt_overflow] != 0) {
      if (!budget_exceeded) stats.reason = hyper_sssp::fallback_reason::distance_width;
      budget_exceeded = true;
    }
  }

  // dynG (transplant hunk): budget_action::error for a spent work or iteration budget (HD8).
  if (budget_exceeded && cfg.error_on_budget &&
      (stats.reason == hyper_sssp::fallback_reason::work_budget ||
       stats.reason == hyper_sssp::fallback_reason::iterations)) {
    throw convergence_error(
        std::string("dyng: hyper_sssp::update: the update exceeded its ") +
        (stats.reason == hyper_sssp::fallback_reason::iterations
             ? "iteration budget (options::max_iterations push rounds)"
             : "work budget (options::work_budget x the live entries)") +
        " under budget_action::error; the hypergraph holds the batch and the result is "
        "poisoned: recompute it");
  }

  if (budget_exceeded) {
    // Exact fallback: recompute from scratch (64-bit distances).
    stats.fallback_recompute = true;
    engine_stats rs = recompute(dev, st, source_id, cfg);
    stats.fallback_iterations = rs.iterations;
    stats.max_frontier = std::max(stats.max_frontier, rs.max_frontier);
  } else {
    unpack_state_kernel<<<grid_for(n, block), block, 0, st.stream>>>(st.d_packed, st.d_dist,
                                                                     st.d_parent, n);
    DYNG_CUDA_TRY(cudaGetLastError());
    DYNG_CUDA_TRY(cudaStreamSynchronize(st.stream));
  }
  return stats;
}

}  // namespace dyng::detail::hsosp
