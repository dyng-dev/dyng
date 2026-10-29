// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from LabelPropagation-CUDA@a276a3a:src/generateSparseGraph.cu (EdgeSet, edgeListToCSR, generateSparseGraph)
// Derived from LabelPropagation-CUDA@a276a3a:src/generateBatches.cu (generateBatches)
// Derived from LabelPropagation-CUDA@a276a3a:src/dynLP.cu (computeAvgEdgeWeight)
/**
 * @file legacy_dynlp.cpp
 * @brief generators::legacy::dynlp_vertex_arrival_stream(): LabelPropagation-CUDA@a276a3a's DynLP
 *        generator, byte for byte (ADR 0067).
 *
 * The random streams are consumed exactly as the original consumes them: std::mt19937(seed) for
 * the graph (two shuffles, the spanning tree's weights, then the random edges), and
 * std::mt19937(seed + 12345) for the batches (the class shuffles, the remaining vertices, one
 * shuffle per batch for its ground truth, one per batch for its deletions). libstdc++ 12's
 * std::shuffle, std::uniform_int_distribution<int> and std::uniform_real_distribution<float> are
 * reproduced by util/rng.hpp (legacy_shuffle32, legacy_uniform_int, legacy_uniform_real_float), so
 * the stream does not depend on the standard library. The float products of the original
 * (int(size * fraction), the weights' (b - a) * u + a) are evaluated as separately rounded float
 * operations: this file is compiled with -ffp-contract=off (cpp/CMakeLists.txt), as the
 * original's x86-64 build had no fused multiply-add to contract them into.
 *
 * Parallel work, where the original runs it in parallel, does not change the output: the CSR is
 * filled with atomic offsets and each row is then sorted by column (columns within a row are
 * distinct, so the sorted row is unique), and the batch edge lists are counted and filled per
 * chunk of rows at prefix offsets, which keeps the sequential order. No region allocates.
 */
#include "core/resources_access.hpp"
#include "generators/dynlp_stream.hpp"
#include "util/allocation.hpp"
#include "util/rng.hpp"

#include <dyng/config.hpp>
#include <dyng/core/error.hpp>
#include <dyng/generators/vertex_arrival.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <numeric>
#include <random>
#include <utility>
#include <vector>

namespace dyng::detail {

namespace {

/// The original's open-addressing set of undirected edges {a, b} (a < b).
class edge_set {
 public:
  explicit edge_set(std::int64_t expected_edges) {
    std::size_t capacity = 16;
    while (capacity < static_cast<std::size_t>(2 * expected_edges + 16)) {
      capacity <<= 1U;
    }
    slots_.assign(capacity, empty);
    mask_ = capacity - 1;
  }

  /// Inserts {a, b}; false if it was present.
  bool insert(std::int32_t a, std::int32_t b) {
    const std::uint64_t key = (static_cast<std::uint64_t>(static_cast<std::uint32_t>(a)) << 32U) |
                              static_cast<std::uint32_t>(b);
    std::size_t i = static_cast<std::size_t>(hash(key)) & mask_;
    while (true) {
      if (slots_[i] == empty) {
        slots_[i] = key;
        return true;
      }
      if (slots_[i] == key) {
        return false;
      }
      i = (i + 1) & mask_;
    }
  }

 private:
  static constexpr std::uint64_t empty = ~0ULL;
  static std::uint64_t hash(std::uint64_t x) {
    x ^= x >> 33U;
    x *= 0xff51afd7ed558ccdULL;
    x ^= x >> 33U;
    x *= 0xc4ceb9fe1a85ec53ULL;
    x ^= x >> 33U;
    return x;
  }
  std::vector<std::uint64_t> slots_;
  std::size_t mask_ = 0;
};

void check_options(const generators::vertex_arrival_options& opt) {
  DYNG_EXPECTS(
      opt.total_vertices > 1 && opt.total_vertices <= std::numeric_limits<std::int32_t>::max(),
      "dynlp_vertex_arrival_stream: total_vertices must be in [2, 2^31 - 1], got ",
      opt.total_vertices);
  DYNG_EXPECTS(opt.average_degree > 0,
               "dynlp_vertex_arrival_stream: average_degree must be positive, got ",
               opt.average_degree);
  DYNG_EXPECTS(opt.num_batches > 0,
               "dynlp_vertex_arrival_stream: num_batches must be positive, got ", opt.num_batches);
  DYNG_EXPECTS(opt.ground_truth_fraction >= 0.0F && opt.ground_truth_fraction <= 1.0F,
               "dynlp_vertex_arrival_stream: ground_truth_fraction must be in [0, 1], got ",
               opt.ground_truth_fraction);
  DYNG_EXPECTS(opt.deleted_fraction >= 0.0F && opt.deleted_fraction <= 1.0F,
               "dynlp_vertex_arrival_stream: deleted_fraction must be in [0, 1], got ",
               opt.deleted_fraction);
}

/// The original's int(count * fraction): the count converted to float, one float product.
std::int64_t scaled_count(std::size_t count, float fraction) {
  const float product = static_cast<float>(count) * fraction;
  return static_cast<std::int64_t>(product);
}

/// generateSparseGraph: the true labels, the edge list (with weights) and the CSR.
void generate_graph(const generators::vertex_arrival_options& opt, int threads,
                    dynlp_legacy_output& out) {
  const auto n = static_cast<std::int32_t>(opt.total_vertices);
  std::int64_t target = static_cast<std::int64_t>(n) * opt.average_degree / 2;
  if (target < static_cast<std::int64_t>(n) - 1) {
    target = n - 1;  // at least the spanning tree
  }
  DYNG_EXPECTS(2 * target <= std::int64_t{2147483647}, "dynlp_vertex_arrival_stream: ", 2 * target,
               " directed edges exceed the original's 32-bit CSR offsets (at most 2147483647)");

  std::mt19937 rng(opt.seed);

  // Step 1: balanced classes, shuffled.
  std::vector<float>& labels = out.true_labels;
  labels.resize(static_cast<std::size_t>(n));
  for (std::int32_t i = 0; i < n; ++i) {
    labels[static_cast<std::size_t>(i)] = i < n / 2 ? 0.0F : 1.0F;
  }
  legacy_shuffle32(labels, rng);

  // Step 2: the random spanning tree through a shuffled permutation.
  std::vector<std::int32_t> perm(static_cast<std::size_t>(n));
  std::iota(perm.begin(), perm.end(), 0);
  legacy_shuffle32(perm, rng);

  edge_set seen(target);
  std::vector<std::pair<std::int32_t, std::int32_t>> edges;
  std::vector<float> weights;
  edges.reserve(static_cast<std::size_t>(target));
  weights.reserve(static_cast<std::size_t>(target));
  const auto weight_of = [&](std::int32_t u, std::int32_t v) {
    return labels[static_cast<std::size_t>(u)] == labels[static_cast<std::size_t>(v)]
               ? legacy_uniform_real_float(rng, 0.5F, 1.0F)
               : legacy_uniform_real_float(rng, 0.01F, 0.5F);
  };
  for (std::int32_t i = 0; i + 1 < n; ++i) {
    const std::int32_t u =
        std::min(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(i) + 1]);
    const std::int32_t v =
        std::max(perm[static_cast<std::size_t>(i)], perm[static_cast<std::size_t>(i) + 1]);
    if (u == v) {
      continue;
    }
    seen.insert(u, v);
    const float w = weight_of(u, v);
    edges.emplace_back(u, v);
    weights.push_back(w);
  }

  // Step 3: random edges until the target, or 10 x target consecutive failed draws.
  std::int64_t attempts = 0;
  const std::int64_t max_attempts = target * 10;
  while (static_cast<std::int64_t>(edges.size()) < target && attempts < max_attempts) {
    const auto u = static_cast<std::int32_t>(legacy_uniform_int(rng, 0, n - 1));
    const auto v = static_cast<std::int32_t>(legacy_uniform_int(rng, 0, n - 1));
    if (u == v) {
      ++attempts;
      continue;
    }
    const std::int32_t a = std::min(u, v);
    const std::int32_t b = std::max(u, v);
    if (!seen.insert(a, b)) {
      ++attempts;
      continue;
    }
    const float w = weight_of(a, b);
    edges.emplace_back(a, b);
    weights.push_back(w);
    attempts = 0;
  }
  out.undirected_edges = static_cast<std::int64_t>(edges.size());

  // Step 4: the undirected CSR, rows sorted by column (edgeListToCSR). As in the original, the
  // degrees are counted and the rows filled in parallel with atomic offsets: the fill order
  // within a row then varies, but the rows are sorted below and their columns are distinct, so
  // the result does not.
  out.num_nodes = n;
  std::vector<std::int32_t>& row_ptr = out.row_ptr;
  row_ptr.assign(static_cast<std::size_t>(n) + 1, 0);
  const auto num_edges = static_cast<std::int64_t>(edges.size());
  const std::pair<std::int32_t, std::int32_t>* const edge = edges.data();
  std::int32_t* const degree = row_ptr.data() + 1;
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
#endif
  for (std::int64_t e = 0; e < num_edges; ++e) {
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp atomic
#endif
    ++degree[edge[e].first];
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp atomic
#endif
    ++degree[edge[e].second];
  }
  for (std::size_t i = 0; i < static_cast<std::size_t>(n); ++i) {
    row_ptr[i + 1] += row_ptr[i];
  }
  const auto entries = static_cast<std::size_t>(row_ptr[static_cast<std::size_t>(n)]);
  out.col_ind.resize(entries);
  out.values.resize(entries);
  {
    std::vector<std::int32_t> offset(row_ptr.begin(), row_ptr.end() - 1);
    std::int32_t* const next = offset.data();
    std::int32_t* const cols = out.col_ind.data();
    float* const vals = out.values.data();
    const float* const w = weights.data();
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(static) num_threads(threads) if (threads > 1)
#endif
    for (std::int64_t e = 0; e < num_edges; ++e) {
      const std::int32_t u = edge[e].first;
      const std::int32_t v = edge[e].second;
      std::int32_t pu = 0;
      std::int32_t pv = 0;
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp atomic capture
#endif
      pu = next[u]++;
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp atomic capture
#endif
      pv = next[v]++;
      cols[pu] = v;
      vals[pu] = w[e];
      cols[pv] = u;
      vals[pv] = w[e];
    }
  }
  edges.clear();
  edges.shrink_to_fit();
  weights.clear();
  weights.shrink_to_fit();
  std::int32_t* const col = out.col_ind.data();
  float* const val = out.values.data();
  const std::int32_t* const rp = row_ptr.data();
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 4096) num_threads(threads) if (threads > 1)
#else
  (void)threads;
#endif
  for (std::int32_t u = 0; u < n; ++u) {
    // The original's insertion sort (columns are distinct: the sorted row is unique).
    const std::int32_t start = rp[u];
    const std::int32_t end = rp[u + 1];
    for (std::int32_t i = start + 1; i < end; ++i) {
      const std::int32_t key_col = col[i];
      const float key_val = val[i];
      std::int32_t j = i - 1;
      while (j >= start && col[j] > key_col) {
        col[j + 1] = col[j];
        val[j + 1] = val[j];
        --j;
      }
      col[j + 1] = key_col;
      val[j + 1] = key_val;
    }
  }
}

/// generateBatches: phases A (vertices), B (edges) and C (deletions).
void generate_batches(const generators::vertex_arrival_options& opt, int threads,
                      dynlp_legacy_output& out) {
  const std::int32_t n = out.num_nodes;
  const auto un = static_cast<std::size_t>(n);
  const std::int32_t num_batches = opt.num_batches;
  const std::vector<float>& labels = out.true_labels;
  std::mt19937 rng(static_cast<std::uint32_t>(opt.seed + 12345U));

  // Phase A: the initial ground truth (b0), then the remaining vertices.
  std::vector<std::int32_t> class0;
  std::vector<std::int32_t> class1;
  for (std::int32_t i = 0; i < n; ++i) {
    (labels[static_cast<std::size_t>(i)] < 0.5F ? class0 : class1).push_back(i);
  }
  legacy_shuffle32(class0, rng);
  legacy_shuffle32(class1, rng);
  const std::int64_t num_initial_gt =
      std::max<std::int64_t>(2, scaled_count(un, opt.ground_truth_fraction));
  const std::int64_t per_class = num_initial_gt / 2;

  std::vector<dynlp_legacy_batch>& batches = out.batches;
  batches.assign(static_cast<std::size_t>(num_batches), dynlp_legacy_batch{});
  std::vector<char> is_initial_gt(un, 0);
  const auto take_initial = [&](const std::vector<std::int32_t>& cls, float label) {
    for (std::int64_t i = 0; i < per_class && i < static_cast<std::int64_t>(cls.size()); ++i) {
      const std::int32_t v = cls[static_cast<std::size_t>(i)];
      batches[0].inserted.push_back(v);
      batches[0].gt_ids.push_back(v);
      batches[0].gt_labels.push_back(label);
      is_initial_gt[static_cast<std::size_t>(v)] = 1;
    }
  };
  take_initial(class0, 0.0F);
  take_initial(class1, 1.0F);

  std::vector<std::int32_t> remaining;
  remaining.reserve(un);
  for (std::int32_t i = 0; i < n; ++i) {
    if (is_initial_gt[static_cast<std::size_t>(i)] == 0) {
      remaining.push_back(i);
    }
  }
  legacy_shuffle32(remaining, rng);
  const auto num_remaining = static_cast<std::int64_t>(remaining.size());
  const std::int64_t spread = std::max<std::int64_t>(1, num_batches - 1);
  const std::int64_t per_batch = num_remaining / spread;
  const std::int64_t leftover = num_remaining - per_batch * spread;
  std::int64_t idx = 0;
  std::vector<std::int32_t> shuffled;
  for (std::int32_t b = 1; b < num_batches; ++b) {
    dynlp_legacy_batch& batch = batches[static_cast<std::size_t>(b)];
    const std::int64_t count = per_batch + (b <= leftover ? 1 : 0);
    for (std::int64_t j = 0; j < count && idx < num_remaining; ++j) {
      batch.inserted.push_back(remaining[static_cast<std::size_t>(idx++)]);
    }
    const std::int64_t num_gt =
        std::max<std::int64_t>(1, scaled_count(batch.inserted.size(), opt.ground_truth_fraction));
    shuffled = batch.inserted;
    legacy_shuffle32(shuffled, rng);
    for (std::int64_t j = 0; j < num_gt && j < static_cast<std::int64_t>(shuffled.size()); ++j) {
      const std::int32_t v = shuffled[static_cast<std::size_t>(j)];
      batch.gt_ids.push_back(v);
      batch.gt_labels.push_back(labels[static_cast<std::size_t>(v)]);
    }
  }

  // Phase B: every edge (u, v), u < v, goes to the batch of its later endpoint, in (u, row)
  // order: counted per chunk of rows, then written at the chunk's prefix offsets.
  std::vector<std::int32_t> vertex_batch(un, -1);
  for (std::int32_t b = 0; b < num_batches; ++b) {
    for (const std::int32_t v : batches[static_cast<std::size_t>(b)].inserted) {
      vertex_batch[static_cast<std::size_t>(v)] = b;
    }
  }
  const std::int32_t* const rp = out.row_ptr.data();
  const std::int32_t* const col = out.col_ind.data();
  const float* const val = out.values.data();
  const std::int32_t* const vb = vertex_batch.data();
  const auto edge_batch = [&](std::int32_t u, std::int32_t e) {
    const std::int32_t v = col[e];
    if (v < 0 || vb[v] < 0 || u >= v) {
      return -1;
    }
    return std::max(vb[u], vb[v]);
  };
  constexpr std::int32_t num_chunks = 1024;
  const std::int32_t chunk_size =
      static_cast<std::int32_t>((static_cast<std::int64_t>(n) + num_chunks - 1) / num_chunks);
  const auto nb = static_cast<std::size_t>(num_batches);
  std::vector<std::int64_t> chunk_count(static_cast<std::size_t>(num_chunks) * nb, 0);
  std::int64_t* const counts = chunk_count.data();
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1)
#else
  (void)threads;
#endif
  for (std::int32_t c = 0; c < num_chunks; ++c) {
    std::int64_t* const count = counts + static_cast<std::size_t>(c) * nb;
    const std::int64_t u_end =
        std::min<std::int64_t>(n, static_cast<std::int64_t>(c + 1) * chunk_size);
    for (std::int64_t u = static_cast<std::int64_t>(c) * chunk_size; u < u_end; ++u) {
      if (vb[u] < 0) {
        continue;
      }
      for (std::int32_t e = rp[u]; e < rp[u + 1]; ++e) {
        const std::int32_t b = edge_batch(static_cast<std::int32_t>(u), e);
        if (b >= 0) {
          ++count[b];
        }
      }
    }
  }
  std::vector<std::int64_t> chunk_offset(chunk_count.size(), 0);
  for (std::size_t b = 0; b < nb; ++b) {
    std::int64_t total = 0;
    for (std::size_t c = 0; c < static_cast<std::size_t>(num_chunks); ++c) {
      chunk_offset[c * nb + b] = total;
      total += chunk_count[c * nb + b];
    }
    batches[b].edge_src.resize(static_cast<std::size_t>(total));
    batches[b].edge_dst.resize(static_cast<std::size_t>(total));
    batches[b].edge_weight.resize(static_cast<std::size_t>(total));
  }
  std::vector<std::int32_t*> src(nb);
  std::vector<std::int32_t*> dst(nb);
  std::vector<float*> wts(nb);
  for (std::size_t b = 0; b < nb; ++b) {
    src[b] = batches[b].edge_src.data();
    dst[b] = batches[b].edge_dst.data();
    wts[b] = batches[b].edge_weight.data();
  }
  std::int64_t* const offsets = chunk_offset.data();
#if DYNG_HAS_OPENMP && defined(_OPENMP)
#pragma omp parallel for schedule(dynamic, 1) num_threads(threads) if (threads > 1)
#endif
  for (std::int32_t c = 0; c < num_chunks; ++c) {
    std::int64_t* const offset = offsets + static_cast<std::size_t>(c) * nb;
    const std::int64_t u_end =
        std::min<std::int64_t>(n, static_cast<std::int64_t>(c + 1) * chunk_size);
    for (std::int64_t u = static_cast<std::int64_t>(c) * chunk_size; u < u_end; ++u) {
      if (vb[u] < 0) {
        continue;
      }
      for (std::int32_t e = rp[u]; e < rp[u + 1]; ++e) {
        const std::int32_t b = edge_batch(static_cast<std::int32_t>(u), e);
        if (b < 0) {
          continue;
        }
        const std::int64_t pos = offset[b]++;
        src[static_cast<std::size_t>(b)][pos] = static_cast<std::int32_t>(u);
        dst[static_cast<std::size_t>(b)][pos] = col[e];
        wts[static_cast<std::size_t>(b)][pos] = val[e];
      }
    }
  }

  // Phase C: deletions among the active unlabelled vertices (a sorted vector: the iteration order
  // of the original's std::set, which the shuffle depends on).
  std::vector<char> is_gt(is_initial_gt);
  std::vector<std::int32_t> active;
  std::vector<std::int32_t> added;
  std::vector<std::int32_t> merged;
  std::vector<char> removed(un, 0);
  std::vector<std::int32_t> candidates;
  for (std::int32_t b = 1; b < num_batches; ++b) {
    dynlp_legacy_batch& batch = batches[static_cast<std::size_t>(b)];
    for (const std::int32_t v : batch.gt_ids) {
      is_gt[static_cast<std::size_t>(v)] = 1;
    }
    added.clear();
    for (const std::int32_t v : batch.inserted) {
      if (is_gt[static_cast<std::size_t>(v)] == 0) {
        added.push_back(v);
      }
    }
    std::sort(added.begin(), added.end());
    merged.resize(active.size() + added.size());
    std::merge(active.begin(), active.end(), added.begin(), added.end(), merged.begin());
    active.swap(merged);

    std::int64_t num_delete = scaled_count(batch.inserted.size(), opt.deleted_fraction);
    num_delete = std::min<std::int64_t>(num_delete, static_cast<std::int64_t>(active.size()));
    if (num_delete > 0) {
      candidates = active;
      legacy_shuffle32(candidates, rng);
      for (std::int64_t j = 0; j < num_delete; ++j) {
        const std::int32_t v = candidates[static_cast<std::size_t>(j)];
        batch.deleted.push_back(v);
        removed[static_cast<std::size_t>(v)] = 1;
      }
      active.erase(
          std::remove_if(active.begin(), active.end(),
                         [&](std::int32_t v) { return removed[static_cast<std::size_t>(v)] != 0; }),
          active.end());
      for (const std::int32_t v : batch.deleted) {
        removed[static_cast<std::size_t>(v)] = 0;
      }
    }
  }
}

/// computeAvgEdgeWeight: the mean over the directed entries, summed in double in row order.
float average_edge_weight(const dynlp_legacy_output& out) {
  if (out.col_ind.empty()) {
    return 0.5F;
  }
  double sum = 0.0;
  std::int64_t count = 0;
  for (std::size_t i = 0; i < out.col_ind.size(); ++i) {
    if (out.col_ind[i] >= 0) {
      sum += out.values[i];
      ++count;
    }
  }
  return count > 0 ? static_cast<float>(sum / static_cast<double>(count)) : 0.5F;
}

}  // namespace

dynlp_legacy_output dynlp_legacy_generate(const generators::vertex_arrival_options& opt,
                                          int threads) try {
  check_options(opt);
  dynlp_legacy_output out;
  const int t = std::max(threads, 1);
  generate_graph(opt, t, out);
  generate_batches(opt, t, out);
  out.tau = average_edge_weight(out);
  return out;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::dynlp_vertex_arrival_stream")

bool dynlp_draws_match_standard_library() try {
#if defined(__GLIBCXX__)
  bool same = true;
  for (const std::uint32_t seed : {1U, 42U, 12387U, 4294967295U}) {
    std::mt19937 ours(seed);
    std::mt19937 theirs(seed);
    for (const std::size_t count :
         {std::size_t{0}, std::size_t{1}, std::size_t{2}, std::size_t{7}, std::size_t{5000},
          std::size_t{65535}, std::size_t{65536}, std::size_t{70001}}) {
      std::vector<std::int32_t> x(count);
      std::iota(x.begin(), x.end(), 0);
      std::vector<std::int32_t> y = x;
      legacy_shuffle32(x, ours);
      std::shuffle(y.begin(), y.end(), theirs);
      same = same && x == y && ours() == theirs();
    }
    std::uniform_real_distribution<float> a(0.5F, 1.0F);
    std::uniform_real_distribution<float> b(0.01F, 0.5F);
    for (int i = 0; i < 4096; ++i) {
      same = same && legacy_uniform_real_float(ours, 0.5F, 1.0F) == a(theirs);
      same = same && legacy_uniform_real_float(ours, 0.01F, 0.5F) == b(theirs);
    }
  }
  return same;
#else
  return false;
#endif
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::standard_library_matches_legacy_streams")

}  // namespace dyng::detail

namespace dyng::generators::legacy {

template <typename vertex_t, typename edge_t>
vertex_arrival_stream<vertex_t, edge_t> dynlp_vertex_arrival_stream(
    const resources& res, const vertex_arrival_options& opt) try {
  static_assert(sizeof(vertex_t) == 4, "dynlp_vertex_arrival_stream: 32-bit vertex ids");
  detail::dynlp_legacy_output raw =
      detail::dynlp_legacy_generate(opt, detail::resources_access::host_threads(res));
  vertex_arrival_stream<vertex_t, edge_t> s;
  const auto n = static_cast<std::size_t>(raw.num_nodes);
  s.graph.num_weights = 1;
  s.graph.row_ptr.assign(raw.row_ptr.begin(), raw.row_ptr.end());
  s.graph.col_ind = std::move(raw.col_ind);
  s.graph.weights = std::move(raw.values);
  s.true_labels.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    s.true_labels[i] = raw.true_labels[i] >= 0.5F ? 1 : 0;
  }
  s.row_capacity.resize(n);
  for (std::size_t i = 0; i < n; ++i) {
    s.row_capacity[i] = static_cast<edge_t>(raw.row_ptr[i + 1] - raw.row_ptr[i]);
  }
  s.similarity_threshold = raw.tau;
  std::vector<std::int64_t> position(n, -1);
  s.batches.resize(raw.batches.size());
  for (std::size_t b = 0; b < raw.batches.size(); ++b) {
    detail::dynlp_legacy_batch& in = raw.batches[b];
    vertex_arrival_batch<vertex_t>& batch = s.batches[b];
    batch.insert_vertex_labels.assign(in.inserted.size(), -1);
    for (std::size_t i = 0; i < in.inserted.size(); ++i) {
      position[static_cast<std::size_t>(in.inserted[i])] = static_cast<std::int64_t>(i);
    }
    for (std::size_t i = 0; i < in.gt_ids.size(); ++i) {
      const std::int64_t p = position[static_cast<std::size_t>(in.gt_ids[i])];
      if (p < 0) {
        DYNG_FAIL("dynlp_vertex_arrival_stream: a ground-truth vertex that does not arrive");
      }
      // setGroundTruthKernel: class = label < 0.5f ? 0 : 1.
      batch.insert_vertex_labels[static_cast<std::size_t>(p)] = in.gt_labels[i] < 0.5F ? 0 : 1;
    }
    for (const std::int32_t v : in.inserted) {
      position[static_cast<std::size_t>(v)] = -1;
    }
    batch.insert_vertices.assign(in.inserted.begin(), in.inserted.end());
    batch.edge_sources.assign(in.edge_src.begin(), in.edge_src.end());
    batch.edge_destinations.assign(in.edge_dst.begin(), in.edge_dst.end());
    batch.edge_weights = std::move(in.edge_weight);
    batch.delete_vertices.assign(in.deleted.begin(), in.deleted.end());
    in = detail::dynlp_legacy_batch{};
    s.max_batch_vertices = std::max(s.max_batch_vertices, batch.insert_vertices.size());
    s.max_batch_edges = std::max(s.max_batch_edges, batch.edge_sources.size());
  }
  return s;
}
DYNG_TRANSLATE_ALLOCATION_FAILURE("generators::legacy::dynlp_vertex_arrival_stream")

template vertex_arrival_stream<std::int32_t, std::int32_t>
dynlp_vertex_arrival_stream<std::int32_t, std::int32_t>(const resources&,
                                                        const vertex_arrival_options&);
template vertex_arrival_stream<std::int32_t, std::int64_t>
dynlp_vertex_arrival_stream<std::int32_t, std::int64_t>(const resources&,
                                                        const vertex_arrival_options&);

}  // namespace dyng::generators::legacy
