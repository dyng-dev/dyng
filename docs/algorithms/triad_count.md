# triad_count: hypergraph h-motif triad counts

Maturity: **experimental** (until M9b's review and freeze for 0.3.0). Header:
`<dyng/triad_count.hpp>`. Container: hypergraph. Oracle: `compute`. Determinism: `exact_value`.
Backends: sequential, openmp, cuda. Ported from ESCHER-GPU@abcf9b2 (ESCHER, IPDPS 2026; ESCHER+,
TKDE 2026).

:::{note}
In this release the sequential and OpenMP backends **recount** the hypergraph in `update()`
(`stats.fallback_used` is true); the CUDA backend runs ESCHER-GPU's incremental wedge engine. The
incremental host engines follow in M9a step 4.
:::

## 1. Problem

A triad is a set of three live hyperedges every two of which share a vertex (a triangle of the
line graph). `compute()` counts the triads of a hypergraph by their **h-motif** (the 30 patterns of
non-empty regions of their Venn diagram; `pattern::h_motif`, 30 bins, 0-based: ESCHER-GPU's ids
1-30 are bins 0-29, and its open triads 21-26, bins 20-25, are always 0), or one of StatHyper's
three triad statistics (`stathyper_type1`: the sum of C(|e|, 3); `stathyper_type2`: the sum of
C(a, 2) b c over the adjacent pairs that share a >= 2 vertices; `stathyper_type3`: the sum of
x y z, the pairwise-only regions, over the triads). `update()` applies a batch of hyperedge
deletions and insertions and changes the counts by the triads the batch destroys and creates.

## 2. Template mapping

An aggregate-delta problem (PLAN 4.5.1), `ownership::min_member`:

| Step | Hook (stage) | What it does |
|---|---|---|
| 0 | `triad_count.normalize` | the hypergraph's Step 0: the live distinct deletions of G_t, the normalized insertions |
| 1a | `triad_count.count_minus` | the triads (pairs, keys) through the deleted hyperedges on G_t's internal row views, each counted at its smallest deleted member, subtracted |
| apply | `triad_count.commit` | `hypergraph.apply` (once) |
| 1b | `triad_count.identify_affected` | the frontier of the ids the commit ASSIGNED to the insertions (`stats.batch.inserted_ids`; on cuda the applied record's device copy) |
| 2 | `triad_count.count_plus` | the triads through the inserted hyperedges on G_{t+1}, each at its smallest inserted member, added |
| finish | `triad_count.finalize` | both phases applied to the counts; a bin that would go negative is an `internal_error` |

`compute()` (`triad_count.compute`) is the full count: every out-wedge (a, b), b > a, of the line
graph closed into its triangles (a, b, k), k > b.

## 3. API

```cpp
#include <dyng/triad_count.hpp>

dyng::triad_count::options opt;  // kind = pattern::h_motif
auto r = dyng::triad_count::compute(res, h, opt);
dyng::triad_count::stats s = dyng::triad_count::update(res, h, batch.view(), r);
// s.delta[k]: the change of bin k; s.triads_removed / s.triads_added; s.batch.inserted_ids
```

Several patterns are several results updated together by one `dyng::update(res, h, batch, r1,
r2, ...)` (one commit). `h_motif` and types 2 and 3 need the line graph and the
vertex-to-hyperedge rows (`line_graph != none`, `store_v2h`); type 1 needs neither.

## 4. Backends, engines and determinism

cuda: ESCHER-GPU's wedge engine (Tier B, `engine::fused`; `options::cuda_engine =
engine::operators` throws `not_supported_error`): one thread per oriented wedge, block bins in
shared memory flushed once with the phase's sign, a pooled workspace reserved before the commit,
at most 3 host synchronizations per update (1 for type 1). sequential and openmp: the full count
as a loop over the source keys with the same operations (a recount in this release). The counts
are identical on every backend and run.

## 5. Performance notes

The stages map to ESCHER-GPU's timers in `parity/timed_regions/triad_count.toml` (ADR 0048):
`update[b1..b3]` against `triad_count.update`, `full_count` against `triad_count.compute`; the
gates are PLAN 8.6's (1.05, end to end 1.10, device memory 1.05).

## 6. Limitations

The host backends recount in `update()` (M9a step 4 makes them incremental). `h_motif` and type 3
refuse a hypergraph with 2,147,483,647 vertices (the classifying merge supports vertex ids up to
2^31 - 3). The counts are 64-bit (a count past 2^64 - 1 is a `capacity_error` on the host; the
device sums of a phase wrap as ESCHER-GPU's do).

## 7. Differences from the paper

The counts are ESCHER-GPU@abcf9b2's fixed code (its exact frontier delta, not the papers' anchored
delta of commit 6a44c1c). A triangle of no h-motif class raises the device error word
(`internal_error`) where ESCHER-GPU would add to `bins[-1]`.

## 8. Mapping from the original code

| ESCHER-GPU@abcf9b2 | dynG |
|---|---|
| `kernel/wedge.cuh` (countWedgeShape, forEachWedge, TriangleBody) | `cpp/src/algorithms/triad_count/engine/wedge.cuh`, `ownership.hpp`, `sets.hpp` |
| `kernel/motif_utils.cuh` (MotifOp), `include/motif_table.hpp` | `engine/motif_ops.hpp`, `engine/motif_table.hpp` |
| `kernel/count_ops.cuh` (Choose3Op), `src/type2.cu`, `src/type3.cu` (Type2Op, Type3Op) | `engine/count_ops.hpp`, `engine/stathyper_ops.hpp` |
| `src/HMotifCount.cu`, `src/HMotifCountUpdate.cu`, the counters of `src/main.cu`, `src/type1-3.cu` | `engine/counters.{cuh,cu}` (`count_full`, `count_phase`), `cuda.cu` |
| `include/frontier.hpp` (DeviceFrontier), `include/device_bins.hpp` | the workspace of `cuda.cu` (`frontier_kernel`) |
| `src/driver.cu` (subtract, maintain, add) | the problem (`problem.hpp`, `triad_count.cpp`); `dyng-compat-escher run` |

## 9. How to cite

`dyng::citation("triad_count")` returns the library's BibTeX entry and the entries of the
manifest's `cite` keys (`escher2026`, `escherplus2026`).
