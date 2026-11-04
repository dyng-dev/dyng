// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from MOSP_ESCHER@4b86159:hsosp/include/hsosp.cuh
/**
 * @file state.cuh
 * @brief The device state of a CUDA hyper_sssp result (MOSP_ESCHER's HsospState: the distances and
 *        parents, the packed words, the invalidation and frontier arrays, the counters), the
 *        update's configuration and counters (UpdateConfig, UpdateStats), and the device graph as
 *        the engine reads it.
 *
 * Ported from MOSP_ESCHER@4b86159 with the renames of parity/escher_port/rename_map_hyper_sssp.toml
 * (M11a, ADR 0091).
 */
#pragma once
#include <dyng/core/buffer.hpp>
#include <dyng/core/resources.hpp>
#include <dyng/hyper_sssp.hpp>

#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace dyng::detail::hsosp {

/** Persistent SOSP state (distances over hyperedge nodes). */
struct device_state {
  int max_nodes = 0;
  long long* d_dist = nullptr;
  int* d_parent = nullptr;
  int* d_is_affected = nullptr;
  int* d_is_candidate = nullptr;
  int* d_cand_list = nullptr;
  int* d_aff_list = nullptr;
  int* d_counters = nullptr;  // [0] = affected, [1] = candidates

  // Incremental update (update): packed (dist << 32 | parent) words,
  // invalidation marks and pointer-jumping ancestors (double buffered),
  // frontier lists with epoch stamps, device counters and their pinned
  // host mirror.
  unsigned long long* d_packed = nullptr;
  int* d_inv_a = nullptr;
  int* d_inv_b = nullptr;
  int* d_jump_a = nullptr;
  int* d_jump_b = nullptr;
  int* d_front = nullptr;
  int* d_next = nullptr;
  int* d_stamp = nullptr;
  unsigned long long* d_upd_counters = nullptr;
  unsigned long long* h_upd_counters = nullptr;
  int epoch = 0;
  /// dynG: the stream of every operation (the resources' of the call; set by its entry point).
  cudaStream_t stream = nullptr;
  // dynG (transplant hunk): the public mode's source flags (one byte per node: the hyperedge holds
  // the source vertex), allocated by allocate_holds() for a result of the public mode only.
  unsigned char* d_holds = nullptr;
  // dynG: the arrays above are owned by buffers of the result's resources (their memory resource
  // and stream; the counters' host mirror from the pinned staging resource); the raw pointers
  // alias them (MOSP_ESCHER swaps d_front and d_next).
  buffer<long long> b_dist;
  buffer<int> b_parent, b_is_affected, b_is_candidate, b_cand_list, b_aff_list, b_counters;
  buffer<unsigned long long> b_packed;
  buffer<int> b_inv_a, b_inv_b, b_jump_a, b_jump_b, b_front, b_next, b_stamp;
  buffer<unsigned long long> b_upd_counters, b_host_upd_counters;
  buffer<unsigned char> b_holds;

  void allocate(const resources& res, int max_nodes);
  /// dynG (transplant hunk): the public mode's source flags, all 0.
  void allocate_holds(const resources& res);
  long long device_bytes() const;
  void free();

  void download_distances(std::vector<long long>& dist, int n) const;
  /** Parents as 0-based node indices (the device convention), -1 for
     *  none. */
  void download_parents(std::vector<int>& parent, int n) const;
  /** Parents as 1-based hyperedge ids (the host convention, e.g. for
     *  generateBatch), -1 for none. */
  void download_parent_ids(std::vector<int>& parent_ids, int n) const;
};

struct engine_config {
  /// Update budget: push iterations before falling back to a recompute (dynG: 64-bit, HD8).
  std::int64_t max_iterations = 4096;
  /// Update budget: edge relaxations (pull + push) before falling back,
  /// as a multiple of the live adjacency entries.
  double work_budget = 1.0;
  /// Threads per block of the recompute kernels (recompute and the
  /// update's fallback), which run one warp per node: a multiple of 32
  /// in [32, 1024], else both entry points throw std::invalid_argument.
  /// The update kernels always use 256.
  int block_size = 256;
  /// dynG (transplant hunk): budget_action::error (HD8): a spent work or iteration budget throws
  /// convergence_error instead of the fallback (the width and the jump cap always recompute).
  bool error_on_budget = false;
  /// dynG: the profiler stages of the update's parts (hyper_sssp.identify_affected, .seed, .loop,
  /// .recompute, .finalize; parity/timed_regions/hyper_sssp.toml) on these resources; none when
  /// null.
  const resources* res = nullptr;
  const std::string* stage_identify = nullptr;
  const std::string* stage_seed = nullptr;
  const std::string* stage_loop = nullptr;
  const std::string* stage_recompute = nullptr;
  const std::string* stage_finalize = nullptr;
};

struct engine_stats {
  int iterations = 0;  ///< update: push iterations; recompute: rounds
  bool fallback_recompute = false;
  int fallback_iterations = 0;  ///< rounds of the fallback recompute
  int max_frontier = 0;
  long long seed_count = 0;  ///< update: invalidated nodes
  int jump_rounds = 0;       ///< pointer-jumping rounds
  long long work = 0;        ///< edge relaxations of the update
  /// dynG (transplant hunk): why the update fell back (fallback_reason, HD12).
  hyper_sssp::fallback_reason reason = hyper_sssp::fallback_reason::none;
};

// dynG (transplant hunk): MOSP_ESCHER's DeviceDelta and DeviceH2H as the engine reads them: the
// slack_csr store's last delta and device line graph (slack_access::last_delta() and rows(),
// M11a step 3; the store owns them), plus the endpoints of hyper_sssp's two modes (HD2).

/** The last batch delta in device memory (0-based node indices): deleted
 *  pairs first, then inserted pairs; ids = new (inserted or recreated)
 *  nodes, then dead ones. */
struct device_delta {
  const int2* d_pairs = nullptr;
  const int* d_ids = nullptr;
  int num_del = 0;
  int num_ins = 0;
  int num_new = 0;
  int num_dead = 0;
};

/** The device line graph: slack rows (row_start, deg) over one symmetric
 *  adjacency (col_ind) and the per-node weights. */
struct device_graph {
  int num_nodes = 0;
  long long num_entries = 0;  ///< the live entries (each pair in both rows)
  const long long* d_row_start = nullptr;
  const int* d_deg = nullptr;
  const int* d_col_ind = nullptr;
  const long long* d_node_w = nullptr;
  device_delta delta;
  // The virtual-endpoint mode: the ids whose weight reads as 0 (-1: none).
  int zero_a = -1;
  int zero_b = -1;
  // The public mode: the source flags, the live hyperedges of the source (the recompute's base),
  // the live ids that lost the source (a root if their parent is -1) and those that gained it.
  const unsigned char* d_holds = nullptr;
  const std::vector<int>* base = nullptr;
  const std::vector<int>* lost = nullptr;
  const std::vector<int>* gained = nullptr;
  /// dynG: the host scratch of the recompute's first list (the pooled workspace's: a steady-state
  /// update allocates nothing; null: a list of its own).
  std::vector<int>* initial = nullptr;
};

/**
 * Exact dynamic SOSP update for the delta last applied by
 * applyBatch (dev.delta):
 *  1. roots: the endpoint b of every deleted pair (a, b) whose tree parent
 *     was a, and every new, recreated or dead node;
 *  2. every descendant of a root in the pre-batch shortest-path tree is
 *     invalidated (pointer jumping over the parent array): its distance
 *     becomes INF, so every finite distance left is realised by a path of
 *     the new graph (no stale value can count to infinity);
 *  3. every invalidated node pulls the best (distance, id) over its
 *     neighbours, and both directions of every inserted pair are relaxed;
 *  4. improved nodes push to their neighbours with a packed 64-bit
 *     atomicMin on (distance << 32 | parent) until no distance decreases.
 * Distances only decrease after step 2, so the loop needs no cap; ties go
 * to the lowest parent id, so a canonical input tree gives the canonical
 * tree of the new graph. If the update exceeds its budget (cfg) or a
 * distance does not fit 32 bits, it falls back to recompute; the
 * fallback is reported in the stats and included in the call's time.
 */
engine_stats update(const device_graph& dev, device_state& st, int source_id,
                    const engine_config& cfg);

/** Static baseline: recompute from blank (GPU Bellman-Ford with frontier
 *  dedup, seeded at the source), on the current device graph. Parent ties
 *  go to the lowest node id (canonical tree). */
engine_stats recompute(const device_graph& dev, device_state& st, int source_id,
                       const engine_config& cfg);

}  // namespace dyng::detail::hsosp
