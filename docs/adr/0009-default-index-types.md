# ADR 0009: Default index types

- **Status:** Accepted (M1b, the `edge_t` benchmark); the headers freeze with it in M3
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

`dyng::graph<vertex_t, edge_t, weight_t>` is compiled for (int32, int32, int32), (int32, int64,
int32) and (int64, int64, int32) (PLAN 4.4.3). The originals of the first releases (MOSP-OpenMP,
MOSP-CUDA, DynLP, ESCHER) use 32-bit edge offsets; CycleEnumeration-GPU already uses 64-bit ones.
PLAN 4.4.2 and decision O6 left the *default* template argument of `edge_t` to a benchmark at the
end of M1b: "if int64 costs more than 3 % on the parity suites, the default is int32 with checked
construction (building a graph past 2^31 edges throws `capacity_error` naming the int64
instantiation)". Until then the provisional default was int64 (risk R8: int64 slower than the
originals' int32).

## The benchmark

`parity/perf_ab.py edge-type` runs `dyng-compat-mosp --edge-type int32` (A) against
`--edge-type int64` (B) A/B/A/B on the inputs, regions and guards of the parity gate (the PLAN
6.4.2 suite: roadNet-PA, roadNet-CA, rgg_n_2_20_s0 and road_usa with K = 3 weights; 50K safe,
50K unsafe and 10K local batches, seed 777), 11 rounds per batch, medians, on both backends
(OpenMP: 28 threads pinned; CUDA: RTX A5000, CUDA 13.1), parity and parity-cuda presets, under the
exclusive perf lock. Both types write byte-identical outputs (checked once per batch) with equal
`invalidated` counters in every round; the golden corpus replays byte-identically with int64
offsets too (`compare.py --configs cuda/int64,sequential/int64,openmp:4/int64`). The records are
`parity/results/M1b-edge-type-{openmp,cuda}-<graph>.json`, measured on the final M1b code
(`0f0fba9`; a first run on `2b39fac` gave the same picture); the tables are in
`parity/results/M1b.md` section 10.

Ratio int64 / int32 of the medians (range over the objectives, or over the batches for apply and
end to end):

| Graph | CUDA SOSP per objective, 50K batches | CUDA, local 10K | OpenMP SOSP per objective, 50K batches | OpenMP, local 10K | apply CUDA / OpenMP | end to end CUDA / OpenMP |
|---|---|---|---|---|---|---|
| roadNet-PA | 1.035-1.044 | 1.013-1.014 | 0.994-1.018 | 1.000-1.012 (spreads 12-15 %) | 0.98-1.04 / 0.90-1.16 (spreads ~20 %) | 0.99-1.01 / 0.98-1.00 |
| roadNet-CA | 1.028-1.036 | 1.005-1.010 | 1.007-1.044 | 1.016-1.062 (spreads 15-24 %) | 0.96-0.99 / 1.01-1.08 | 0.99 / 1.00-1.02 |
| rgg_n_2_20_s0 | 1.038-1.045 | 1.008-1.009 | 1.009-1.032 | 0.913-0.960 (spreads 9-13 %) | 0.99-1.01 / 1.03-1.06 | 1.00 / 1.00-1.01 |
| road_usa | 1.014-1.019 | 1.002-1.004 | 1.013-1.018 | 0.957-0.966 (spreads 8-17 %) | 1.01 / 1.08-1.09 | 1.00 / 1.02-1.03 |

- **CUDA: int64 costs 3-4.5 % on the 50K batches** of roadNet-PA, roadNet-CA and rgg_n_2_20_s0,
  consistently (spreads of 1-3 %), on the gated per-objective region, and 1.5-2 % on road_usa's;
  0.2-1.4 % on the local batches. The persistent kernel reads the row offsets of every relaxed
  vertex twice (begin and end); with int64 they are twice the bytes and the index arithmetic is
  64-bit. The int64 instantiation also needs 60 instead of 59 registers (still 4 blocks of 256
  threads per SM on sm_86).
- **OpenMP: -0.6 % to +4.4 % on the 50K batches** (the largest on roadNet-CA); the local
  batches move by -9 % to +6 % with spreads of 8-24 % between rounds, so they carry no signal
  either way.
- **apply** costs up to 9 % on OpenMP where its medians are stable (road_usa 1.08-1.09x: the CSR's
  offset arrays are twice as large and the host apply is memory-bound); end to end moves by at
  most 3 %, below the noise of the input phase.

## Decision

1. **The default `edge_t` is `std::int32_t`**: `dyng::graph<>` is `graph<int32, int32, int32>`,
   the originals' types and the parity configuration. int64 exceeds the 3 % threshold of PLAN
   4.4.2 on the CUDA backend's gated region (and on OpenMP's roadNet-CA and rgg_n_2_20_s0 50K
   batches and apply).
2. **Construction is checked.** Every path that produces a CSR from an edge count converts it with
   `detail::checked_edge_count<edge_t>()`, which throws `capacity_error` ("... edges do not fit
   the edge offset type; use a graph with 64-bit edge_t (int64)") when the count exceeds
   `numeric_limits<edge_t>::max()`: `graph::from_edges()` (after self-loop handling, undirected
   doubling and multi-edge merging) and `graph::apply()` / every algorithm's `update()` (a batch
   that grows the graph past 2^31 - 1 edges). `graph::from_csr()` receives offsets that already
   have type edge_t; the CSR text reader `io::read_csr_triplet<V, E, W>` rejects an offset outside
   E with an `io_error` naming the file and line, and `io::read_matrix_market` returns an edge
   list, which `from_edges()` checks. The two graph calls document `@throws capacity_error`; the
   conversion is unit-tested at the int32 boundary (a real 2^31-edge graph needs more than 16 GB
   and is not built by the tests).
3. **int64 stays a first-class instantiation** (PLAN 4.4.3): graphs with more than 2^31 - 1 stored
   edges use `graph<int32, int64, int32>` or `graph<int64, int64, int32>`, with the same results
   (byte-identical on the golden corpus and the paper-scale batches).
4. **Vertex ids stay int32 by default** (PLAN 4.4.2; not measured here: every gate graph has
   fewer than 2^31 vertices, and the int64 vertex instantiation exists for larger ones).
   `weight_t` stays int32 for shortest paths, `distance_t` int64.
5. **Re-measure** when the kernels change materially (the operators engine in 0.2, the device
   apply) and for each new algorithm whose original uses 64-bit offsets (`cycle_count`, M2): a
   per-algorithm default is not planned; a change of the library-wide default needs a new ADR.

## Consequences

- `dyng::graph<>` users get the originals' memory footprint (4 bytes less per row offset) and the
  parity speed on CUDA; graphs past 2^31 - 1 edges must name the int64 instantiation, and the
  error says so instead of overflowing.
- The Python bindings (M5) expose the default type and the int64 one; the choice is visible in
  the wheel's type names.
- The provisional int64 default of M1a is gone before the M3 header freeze, so no released API
  changes.
- `dyng-compat-mosp --edge-type` and `perf_ab.py edge-type` stay in the harness for the
  re-measurements of item 5.
