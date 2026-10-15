// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from ESCHER-GPU@abcf9b2:kernel/wedge.cuh (intersectSize, venn3, intersects3)
/**
 * @file sets.hpp
 * @brief The rows the counting operations read and the set primitives on them (ESCHER-GPU's
 *        intersectSize, venn3 and intersects3), on the host and the device.
 *
 * The Op form of the engine (c6 of the port, ADR 0053): an operation sees 0-based hyperedge ids
 * and a rows handle (hyperedge_rows) that gives the members of an id as a row_ref; the store's
 * encoding (the row of id i at key i + 1, its members + 1, sorted) stays inside the handle and the
 * set primitives, which only compare members. Host-compilable: the host engine of M9a step 4 reads
 * the same rows with the same operations.
 */
#pragma once

#include <climits>

#if defined(__CUDACC__)
/// A function of the engine inlined on the host and the device.
#define DYNG_TRIAD_INLINE __host__ __device__ __forceinline__
/// A member of a counting operation (host and device).
#define DYNG_TRIAD_HD __host__ __device__
#else
/// A function of the engine inlined on the host and the device.
#define DYNG_TRIAD_INLINE inline
/// A member of a counting operation (host and device).
#define DYNG_TRIAD_HD
#endif

namespace dyng::detail::triad {

/// The members of one hyperedge: `size` sorted values from `data` (the store's encoding).
struct row_ref {
  const int* data;
  int size;
};

/// The rows handle of the operations: the members of a hyperedge by its 0-based id (the h2v row
/// view: the row of id i is at key i + 1).
struct hyperedge_rows {
  const int* flat;  ///< the payload
  const int* loc;   ///< key -> offset of the row
  const int* len;   ///< key -> length of the row

  /// The members of hyperedge `id`.
  DYNG_TRIAD_INLINE row_ref row(int id) const {
    return {flat + loc[id + 1], len[id + 1]};
  }
  /// The number of members of hyperedge `id`.
  DYNG_TRIAD_INLINE int size(int id) const {
    return len[id + 1];
  }
};

/// min(x, y) (the device's integer min on the device).
DYNG_TRIAD_INLINE int min_of(int x, int y) {
#if defined(__CUDA_ARCH__)
  return min(x, y);
#else
  return x < y ? x : y;
#endif
}

/// max(x, y) (the device's integer max on the device).
DYNG_TRIAD_INLINE int max_of(int x, int y) {
#if defined(__CUDA_ARCH__)
  return max(x, y);
#else
  return x < y ? y : x;
#endif
}

// ---- Set primitives on sorted, contiguous rows -------------------------

// |A ∩ B|
DYNG_TRIAD_INLINE int intersect_size(const int* a, int na, const int* b, int nb) {
  int x = 0, y = 0, c = 0;
  while (x < na && y < nb) {
    int va = a[x], vb = b[y];
    c += (va == vb);
    x += (va <= vb);
    y += (vb <= va);
  }
  return c;
}

// One merge pass over rows a, b, c: |B ∩ C|, |C ∩ A| and |A ∩ B ∩ C|. The rows end at INT_MAX:
// triad_count's compute() refuses the classifying patterns when a member could be INT_MAX.
DYNG_TRIAD_INLINE void venn3(const int* a, int na, const int* b, int nb, const int* c, int nc,
                             int& cbc, int& cca, int& g) {
  int x = 0, y = 0, z = 0;
  cbc = cca = g = 0;
  while (true) {
    int va = x < na ? a[x] : INT_MAX;
    int vb = y < nb ? b[y] : INT_MAX;
    int vc = z < nc ? c[z] : INT_MAX;
    int m = min_of(va, min_of(vb, vc));
    if (m == INT_MAX) break;
    int ia = va == m, ib = vb == m, ic = vc == m;
    cbc += ib & ic;
    cca += ic & ia;
    g += ia & ib & ic;
    x += ia;
    y += ib;
    z += ic;
  }
}

// Whether rows a, b and c share an element.
DYNG_TRIAD_INLINE bool intersects3(const int* a, int na, const int* b, int nb, const int* c,
                                   int nc) {
  int x = 0, y = 0, z = 0;
  while (x < na && y < nb && z < nc) {
    int m = max_of(a[x], max_of(b[y], c[z]));
    if (a[x] == m && b[y] == m && c[z] == m) return true;
    x += a[x] < m;
    y += b[y] < m;
    z += c[z] < m;
  }
  return false;
}

/// |A ∩ B| of two rows.
DYNG_TRIAD_INLINE int intersect_size(row_ref a, row_ref b) {
  return intersect_size(a.data, a.size, b.data, b.size);
}

}  // namespace dyng::detail::triad
