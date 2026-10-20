// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:utils/reference.cpp (referenceLineGraph, forEachTriangle,
// referenceMotifCounts, referenceType1/2/3)
/**
 * @file triad_reference.cpp
 * @brief dyng::testing::triad_reference (ESCHER-GPU's CPU reference algorithm; exact counts, a
 *        count past 2^64 - 1 refused where the original wraps or overflows). Standard
 *        library (and OpenMP) only: it shares no code with triad_count or the hypergraph container
 *        (triad_reference_isolation). The h-motif classes are derived from the six permutations
 *        of the Venn regions and the documented representatives, not copied from the original's
 *        table (a test compares the two).
 */
#include <dyng/testing/triad_reference.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace dyng::testing {

namespace {

using row_t = std::vector<std::int32_t>;
using rows_t = std::vector<row_t>;

// The canonical Venn id of classes 1-30 in ESCHER's numbering (the table of the header).
constexpr std::array<std::uint8_t, triad_num_classes> listed = {
    64, 65, 67, 71, 72, 73, 76, 75, 77, 79, 88, 90, 89, 91, 93, 95, 120, 121, 123, 127,  // ABC
    24, 26, 25, 27, 29, 31,                                                              // open
    56, 57, 59, 63};  // AB, BC and CA without ABC

constexpr std::array<std::uint8_t, triad_num_classes> make_representatives() {
  std::array<std::uint8_t, triad_num_classes> r = listed;
#if defined(DYNG_MUTATION_TRIAD_REFERENCE_CLASS_MAP)
  // The planted mutation (parity/mutate.py, triad_reference_class_map): classes 2 and 3 swapped.
  const std::uint8_t two = r[1];
  r[1] = r[2];
  r[2] = two;
#endif
  return r;
}

constexpr std::array<std::uint8_t, triad_num_classes> representatives = make_representatives();

void check_venn_id(int venn_id) {
  if (venn_id < 0 || venn_id >= triad_num_venn_ids)
    throw std::invalid_argument("triad_reference: Venn id " + std::to_string(venn_id) +
                                " is outside [0, 128)");
}

// Whether a pair of hyperedges intersects: its pairwise-only region or the triple region.
int intersecting_pairs(int venn_id) {
  const bool triple = (venn_id >> 6) & 1;
  int n = 0;
  for (int bit = 3; bit <= 5; ++bit) n += (((venn_id >> bit) & 1) != 0) || triple;
  return n;
}

int threads_of(const triad_reference_options& options) {
  if (options.threads < 0)
    throw std::invalid_argument("triad_reference: threads must be >= 0, got " +
                                std::to_string(options.threads));
#ifdef _OPENMP
  return options.threads > 0 ? options.threads : omp_get_max_threads();
#else
  return 1;
#endif
}

// Exact arithmetic: the products and the sums in unsigned 128-bit integers (a row has fewer than
// 2^31 members, so every product is below 2^124), a count past 2^64 - 1 refused instead of
// wrapped. The oracle checks the engines at the 64-bit limit too (M9a review), so it shares none
// of their overflow tests. (__extension__: the GCC and Clang type is outside ISO C++.)
__extension__ typedef unsigned __int128 wide_t;

constexpr wide_t count_limit = static_cast<wide_t>(UINT64_MAX);

// A sum of counts: exact while it stays at or below 2^64 - 1, then marked over (and not grown).
struct exact_sum {
  wide_t value = 0;
  bool over = false;

  void add(wide_t v) {
    if (over) return;
    value += v;  // value <= 2^64 - 1 and v < 2^124: no wrap
    if (value > count_limit) over = true;
  }
  void add(const exact_sum& other) {
    if (other.over) {
      over = true;
    } else {
      add(other.value);
    }
  }
};

// The 64-bit value of a sum, or std::overflow_error naming the pattern.
std::uint64_t narrow(const exact_sum& s, const char* pattern) {
  if (s.over)
    throw std::overflow_error(std::string("triad_reference: the ") + pattern +
                              " count exceeds 2^64 - 1");
  return static_cast<std::uint64_t>(s.value);
}

// The rows by id, each sorted and without repeats, after checking the layout.
rows_t read_rows(const std::vector<std::int64_t>& offsets, const std::vector<std::int32_t>& members,
                 std::int32_t num_vertices) {
  if (num_vertices < 0) throw std::invalid_argument("triad_reference: negative vertex count");
  if (offsets.empty() || offsets.front() != 0 ||
      offsets.back() != static_cast<std::int64_t>(members.size()))
    throw std::invalid_argument(
        "triad_reference: the offsets must start at 0 and end at the member count");
  const std::size_t m = offsets.size() - 1;
  if (m > static_cast<std::size_t>(INT32_MAX))
    throw std::invalid_argument("triad_reference: more than 2^31 - 1 ids");
  rows_t rows(m);
  for (std::size_t id = 0; id < m; ++id) {
    if (offsets[id + 1] < offsets[id])
      throw std::invalid_argument("triad_reference: the offsets decrease at id " +
                                  std::to_string(id));
    row_t& r = rows[id];
    r.assign(members.begin() + offsets[id], members.begin() + offsets[id + 1]);
    for (const std::int32_t v : r) {
      if (v < 0 || v >= num_vertices)
        throw std::invalid_argument("triad_reference: member " + std::to_string(v) + " of id " +
                                    std::to_string(id) + " is outside [0, " +
                                    std::to_string(num_vertices) + ")");
    }
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
  }
  return rows;
}

std::int64_t intersect2(const row_t& a, const row_t& b) {
  std::int64_t c = 0;
  std::size_t i = 0, j = 0;
  while (i < a.size() && j < b.size()) {
    if (a[i] == b[j]) {
      ++c;
      ++i;
      ++j;
    } else if (a[i] < b[j]) {
      ++i;
    } else {
      ++j;
    }
  }
  return c;
}

std::int64_t intersect3(const row_t& a, const row_t& b, const row_t& c) {
  std::int64_t n = 0;
  std::size_t i = 0, j = 0, k = 0;
  while (i < a.size() && j < b.size() && k < c.size()) {
    const std::int32_t top = std::max(a[i], std::max(b[j], c[k]));
    if (a[i] == top && b[j] == top && c[k] == top) {
      ++n;
      ++i;
      ++j;
      ++k;
      continue;
    }
    if (a[i] < top) ++i;
    if (j < b.size() && b[j] < top) ++j;
    if (k < c.size() && c[k] < top) ++k;
  }
  return n;
}

// C(n, 3), exact (n < 2^31: n^3 < 2^93).
wide_t choose3(std::size_t n) {
  const auto w = static_cast<wide_t>(n);
  return n >= 3 ? w * (w - 1) * (w - 2) / 6 : 0;
}

// The line graph (referenceLineGraph): row i = the sorted ids of the other live hyperedges that
// share a vertex with i, gathered from the hyperedges of every member.
rows_t line_graph(const rows_t& rows, int threads) {
  // Vertex -> hyperedges, from the incidences sorted by (vertex, id): no array indexed by vertex.
  std::vector<std::pair<std::int32_t, std::int32_t>> inc;
  std::size_t pins = 0;
  for (const row_t& r : rows) pins += r.size();
  inc.reserve(pins);
  for (std::size_t id = 0; id < rows.size(); ++id)
    for (const std::int32_t v : rows[id]) inc.emplace_back(v, static_cast<std::int32_t>(id));
  std::sort(inc.begin(), inc.end());
  std::vector<std::int32_t> ids(inc.size());
  for (std::size_t k = 0; k < inc.size(); ++k) ids[k] = inc[k].second;
  // For every incidence of a row, the range of its vertex's hyperedges in `ids`.
  std::vector<std::size_t> group_end(inc.size());
  std::vector<std::size_t> group_begin(inc.size());
  for (std::size_t k = 0; k < inc.size();) {
    std::size_t e = k;
    while (e < inc.size() && inc[e].first == inc[k].first) ++e;
    for (std::size_t q = k; q < e; ++q) {
      group_begin[q] = k;
      group_end[q] = e;
    }
    k = e;
  }
  // The position of incidence (v, id) in `inc`, per row in member order.
  std::vector<std::size_t> row_start(rows.size() + 1, 0);
  for (std::size_t id = 0; id < rows.size(); ++id)
    row_start[id + 1] = row_start[id] + rows[id].size();
  std::vector<std::size_t> where(inc.size());
  {
    std::vector<std::size_t> next(row_start.begin(), row_start.end() - 1);
    // inc is sorted by (vertex, id) and every row is sorted, so the incidences of a row appear
    // in member order when scanned by vertex.
    for (std::size_t k = 0; k < inc.size(); ++k)
      where[next[static_cast<std::size_t>(inc[k].second)]++] = k;
  }
  rows_t adj(rows.size());
  const auto n = static_cast<std::int64_t>(rows.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(dynamic, 256) num_threads(threads) if (threads > 1)
#endif
  for (std::int64_t e = 0; e < n; ++e) {
    const auto x = static_cast<std::size_t>(e);
    row_t row;
    for (std::size_t q = row_start[x]; q < row_start[x + 1]; ++q) {
      const std::size_t k = where[q];
      row.insert(row.end(), ids.begin() + static_cast<std::ptrdiff_t>(group_begin[k]),
                 ids.begin() + static_cast<std::ptrdiff_t>(group_end[k]));
    }
    std::sort(row.begin(), row.end());
    row.erase(std::unique(row.begin(), row.end()), row.end());
    row.erase(std::remove(row.begin(), row.end(), static_cast<std::int32_t>(e)), row.end());
    adj[x] = std::move(row);
  }
  (void)threads;
  return adj;
}

// The per-thread accumulator of the triangle patterns.
struct triangle_acc {
  std::array<std::uint64_t, triad_num_classes> bins{};  // unit counts (cannot reach 2^64)
  exact_sum type3;
  bool invalid = false;
};

// Calls f(acc, i, j, k) for every triangle i < j < k of `adj` (forEachTriangle), each thread with
// its own accumulator; the accumulators are returned.
template <typename visit_t>
std::vector<triangle_acc> for_each_triangle(const rows_t& adj, int threads, visit_t f) {
  std::vector<triangle_acc> parts;
  const auto n = static_cast<std::int64_t>(adj.size());
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (threads > 1)
#endif
  {
    triangle_acc local;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
    for (std::int64_t x = 0; x < n; ++x) {
      const auto i = static_cast<std::int32_t>(x);
      const row_t& ni = adj[static_cast<std::size_t>(x)];
      for (const std::int32_t j : ni) {
        if (j <= i) continue;
        const row_t& nj = adj[static_cast<std::size_t>(j)];
        std::size_t p = 0, q = 0;
        while (p < ni.size() && q < nj.size()) {
          if (ni[p] == nj[q]) {
            if (ni[p] > j) f(local, i, j, ni[p]);
            ++p;
            ++q;
          } else if (ni[p] < nj[q]) {
            ++p;
          } else {
            ++q;
          }
        }
      }
    }
#ifdef _OPENMP
#pragma omp critical(dyng_triad_reference)
#endif
    parts.push_back(local);
  }
  (void)threads;
  return parts;
}

// The Venn id of three sorted rows.
int venn_of(const row_t& a, const row_t& b, const row_t& c) {
  const std::int64_t ab = intersect2(a, b), bc = intersect2(b, c), ca = intersect2(c, a);
  const std::int64_t g = intersect3(a, b, c);
  const std::int64_t only_a = static_cast<std::int64_t>(a.size()) - (ab + ca) + g;
  const std::int64_t only_b = static_cast<std::int64_t>(b.size()) - (bc + ab) + g;
  const std::int64_t only_c = static_cast<std::int64_t>(c.size()) - (ca + bc) + g;
  return static_cast<int>(only_a > 0) | (static_cast<int>(only_b > 0) << 1) |
         (static_cast<int>(only_c > 0) << 2) | (static_cast<int>(ab - g > 0) << 3) |
         (static_cast<int>(bc - g > 0) << 4) | (static_cast<int>(ca - g > 0) << 5) |
         (static_cast<int>(g > 0) << 6);
}

bool wants(const std::vector<triad_pattern>& patterns, triad_pattern p) {
  return std::find(patterns.begin(), patterns.end(), p) != patterns.end();
}

triad_reference_counts count_rows(const rows_t& rows, const std::vector<triad_pattern>& patterns,
                                  int threads) {
  triad_reference_counts out;
  if (wants(patterns, triad_pattern::stathyper_type1)) {
    exact_sum type1;
    for (const row_t& r : rows) type1.add(choose3(r.size()));
    out.type1 = narrow(type1, "stathyper_type1");
  }
  const bool motif = wants(patterns, triad_pattern::h_motif);
  const bool type2 = wants(patterns, triad_pattern::stathyper_type2);
  const bool type3 = wants(patterns, triad_pattern::stathyper_type3);
  if (!motif && !type2 && !type3) return out;
  const rows_t adj = line_graph(rows, threads);
  if (type2) {
    exact_sum total;
    const auto n = static_cast<std::int64_t>(adj.size());
#ifdef _OPENMP
#pragma omp parallel num_threads(threads) if (threads > 1)
#endif
    {
      exact_sum local;
#ifdef _OPENMP
#pragma omp for schedule(dynamic, 64)
#endif
      for (std::int64_t x = 0; x < n; ++x) {
        const auto i = static_cast<std::int32_t>(x);
        const row_t& e = rows[static_cast<std::size_t>(x)];
        for (const std::int32_t j : adj[static_cast<std::size_t>(x)]) {
          if (j <= i) continue;
          const row_t& f = rows[static_cast<std::size_t>(j)];
          const auto a = static_cast<wide_t>(intersect2(e, f));
          const wide_t b = static_cast<wide_t>(e.size()) - a;
          const wide_t c = static_cast<wide_t>(f.size()) - a;
          if (a >= 2 && b > 0 && c > 0) local.add(a * (a - 1) / 2 * b * c);  // < 2^123
        }
      }
#ifdef _OPENMP
#pragma omp critical(dyng_triad_reference_type2)
#endif
      total.add(local);
    }
    out.type2 = narrow(total, "stathyper_type2");
  }
  if (motif || type3) {
    const std::array<std::int8_t, triad_num_venn_ids> map = triad_motif_class_map();
    const auto parts = for_each_triangle(
        adj, threads, [&](triangle_acc& acc, std::int32_t i, std::int32_t j, std::int32_t k) {
          const row_t& a = rows[static_cast<std::size_t>(i)];
          const row_t& b = rows[static_cast<std::size_t>(j)];
          const row_t& c = rows[static_cast<std::size_t>(k)];
          if (motif) {
            const int cls = map[static_cast<std::size_t>(venn_of(a, b, c))];
            if (cls <= 0) {
              acc.invalid = true;
            } else {
              ++acc.bins[static_cast<std::size_t>(cls - 1)];
            }
          }
          if (type3) {
            const std::int64_t g = intersect3(a, b, c);
            const std::int64_t x = intersect2(a, b) - g;
            const std::int64_t y = intersect2(b, c) - g;
            const std::int64_t z = intersect2(a, c) - g;
            if (x > 0 && y > 0 && z > 0)
              acc.type3.add(static_cast<wide_t>(x) * static_cast<wide_t>(y) *
                            static_cast<wide_t>(z));  // < 2^93
          }
        });
    std::array<exact_sum, triad_num_classes> bins{};
    exact_sum type3_total;
    for (const triangle_acc& p : parts) {
      if (p.invalid)
        throw std::logic_error(
            "triad_reference: a triangle of the line graph has no h-motif class");
      for (int t = 0; t < triad_num_classes; ++t)
        bins[static_cast<std::size_t>(t)].add(p.bins[static_cast<std::size_t>(t)]);
      type3_total.add(p.type3);
    }
    if (motif) {
      for (int t = 0; t < triad_num_classes; ++t)
        out.h_motif[static_cast<std::size_t>(t)] =
            narrow(bins[static_cast<std::size_t>(t)], "h_motif");
    }
    if (type3) out.type3 = narrow(type3_total, "stathyper_type3");
  }
  return out;
}

// counts(G) - counts(G \ ids).
std::vector<std::uint64_t> touching(const std::vector<std::int64_t>& offsets,
                                    const std::vector<std::int32_t>& members,
                                    std::int32_t num_vertices, const std::vector<std::int64_t>& ids,
                                    triad_pattern pattern, const triad_reference_options& options) {
  const int threads = threads_of(options);
  rows_t rows = read_rows(offsets, members, num_vertices);
  const std::vector<std::uint64_t> with = count_rows(rows, {pattern}, threads).of(pattern);
  for (const std::int64_t id : ids) {
    if (id >= 0 && id < static_cast<std::int64_t>(rows.size()))
      rows[static_cast<std::size_t>(id)].clear();
  }
  const std::vector<std::uint64_t> without = count_rows(rows, {pattern}, threads).of(pattern);
  std::vector<std::uint64_t> out(with.size());
  for (std::size_t t = 0; t < with.size(); ++t) out[t] = with[t] - without[t];
  return out;
}

}  // namespace

const char* triad_pattern_name(triad_pattern pattern) noexcept {
  switch (pattern) {
    case triad_pattern::h_motif:
      return "h_motif";
    case triad_pattern::stathyper_type1:
      return "stathyper_type1";
    case triad_pattern::stathyper_type2:
      return "stathyper_type2";
    case triad_pattern::stathyper_type3:
      return "stathyper_type3";
  }
  return "unknown";
}

int triad_num_bins(triad_pattern pattern) noexcept {
  return pattern == triad_pattern::h_motif ? triad_num_classes : 1;
}

const std::array<std::uint8_t, triad_num_classes>& triad_class_representatives() noexcept {
  return representatives;
}

int triad_venn_permuted(int venn_id, const std::array<int, 3>& image) {
  check_venn_id(venn_id);
  std::array<bool, 3> seen{};
  for (const int p : image) {
    if (p < 0 || p > 2 || seen[static_cast<std::size_t>(p)])
      throw std::invalid_argument("triad_venn_permuted: the image is not a permutation of 0, 1, 2");
    seen[static_cast<std::size_t>(p)] = true;
  }
  // The pairwise-only region of hyperedges x < y: bit 3 for (A, B), 4 for (B, C), 5 for (C, A).
  const auto pair_bit = [](int x, int y) {
    const int lo = std::min(x, y), hi = std::max(x, y);
    return lo == 0 ? (hi == 1 ? 3 : 5) : 4;
  };
  int out = venn_id & (1 << 6);
  for (int x = 0; x < 3; ++x) {
    if ((venn_id >> x) & 1) out |= 1 << image[static_cast<std::size_t>(x)];
  }
  for (const auto& [x, y] : {std::pair<int, int>{0, 1}, {1, 2}, {0, 2}}) {
    if ((venn_id >> pair_bit(x, y)) & 1)
      out |= 1 << pair_bit(image[static_cast<std::size_t>(x)], image[static_cast<std::size_t>(y)]);
  }
  return out;
}

int triad_venn_canonical(int venn_id) {
  check_venn_id(venn_id);
  std::array<int, 3> image = {0, 1, 2};
  int best = venn_id;
  do {
    best = std::min(best, triad_venn_permuted(venn_id, image));
  } while (std::next_permutation(image.begin(), image.end()));
  return best;
}

int triad_motif_class(int venn_id) {
  check_venn_id(venn_id);
  if (intersecting_pairs(venn_id) < 2) return 0;
  const int canonical = triad_venn_canonical(venn_id);
  for (std::size_t k = 0; k < representatives.size(); ++k) {
    if (representatives[k] == canonical) return static_cast<int>(k) + 1;
  }
  throw std::logic_error("triad_motif_class: Venn id " + std::to_string(venn_id) + " (canonical " +
                         std::to_string(canonical) + ") is in no listed class");
}

std::array<std::int8_t, triad_num_venn_ids> triad_motif_class_map() {
  std::array<std::int8_t, triad_num_venn_ids> map{};
  for (int id = 0; id < triad_num_venn_ids; ++id)
    map[static_cast<std::size_t>(id)] = static_cast<std::int8_t>(triad_motif_class(id));
  return map;
}

int triad_venn_id(const std::vector<std::int32_t>& a, const std::vector<std::int32_t>& b,
                  const std::vector<std::int32_t>& c) {
  const auto sorted = [](row_t r) {
    std::sort(r.begin(), r.end());
    r.erase(std::unique(r.begin(), r.end()), r.end());
    return r;
  };
  return venn_of(sorted(a), sorted(b), sorted(c));
}

std::vector<std::uint64_t> triad_reference_counts::of(triad_pattern pattern) const {
  switch (pattern) {
    case triad_pattern::h_motif:
      return {h_motif.begin(), h_motif.end()};
    case triad_pattern::stathyper_type1:
      return {type1};
    case triad_pattern::stathyper_type2:
      return {type2};
    case triad_pattern::stathyper_type3:
      return {type3};
  }
  throw std::invalid_argument("triad_reference_counts::of: unknown pattern");
}

triad_reference_counts triad_reference_all(const std::vector<std::int64_t>& offsets,
                                           const std::vector<std::int32_t>& members,
                                           std::int32_t num_vertices,
                                           const std::vector<triad_pattern>& patterns,
                                           const triad_reference_options& options) {
  const int threads = threads_of(options);
  return count_rows(read_rows(offsets, members, num_vertices), patterns, threads);
}

std::vector<std::uint64_t> triad_reference(const std::vector<std::int64_t>& offsets,
                                           const std::vector<std::int32_t>& members,
                                           std::int32_t num_vertices, triad_pattern pattern,
                                           const triad_reference_options& options) {
  return triad_reference_all(offsets, members, num_vertices, {pattern}, options).of(pattern);
}

std::vector<std::uint64_t> triad_reference_removed(const std::vector<std::int64_t>& offsets,
                                                   const std::vector<std::int32_t>& members,
                                                   std::int32_t num_vertices,
                                                   const std::vector<std::int64_t>& ids,
                                                   triad_pattern pattern,
                                                   const triad_reference_options& options) {
  return touching(offsets, members, num_vertices, ids, pattern, options);
}

std::vector<std::uint64_t> triad_reference_added(const std::vector<std::int64_t>& offsets,
                                                 const std::vector<std::int32_t>& members,
                                                 std::int32_t num_vertices,
                                                 const std::vector<std::int64_t>& ids,
                                                 triad_pattern pattern,
                                                 const triad_reference_options& options) {
  return touching(offsets, members, num_vertices, ids, pattern, options);
}

}  // namespace dyng::testing
