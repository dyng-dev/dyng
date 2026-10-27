# triad_count: hypergraph h-motif triad counts

Maturity: **stable** from 0.3.0 (SemVer applies, as for `sssp`, `cycle_count` and `mosp`; the
API review and freeze of {doc}`ADR 0060 <../adr/0060-triad-count-api-review-and-freeze-0-3>`; the
author's decision of 2026-10-22, GOVERNANCE.md). The
sequential, OpenMP and CUDA backends, the Python module `dyng.triad_count`, the command line
`dyng triad_count` and the benchmark suites `benchmarks/paper/ipdps26_escher.yaml` and
`tkde26_escher_plus.yaml` are new in 0.3.0. Header: `<dyng/triad_count.hpp>`. Container:
`hypergraph`. Oracle: `compute`. Determinism: `exact_value`. Parity: exact against
ESCHER-GPU@abcf9b2 on its 300-case golden set (the 30-bin counts and StatHyper's types 1-3, by key
and by content, after every batch; [M9a record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M9a.md),
[M9b record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M9b.md)). Papers: ESCHER
(IPDPS 2026) and ESCHER+ (TKDE 2026). dynG counts the **closed** triads of the fixed code, not
the open, incident-vertex or temporal triads the papers also discuss (section 7).

## 1. Problem

**What it computes.** A **triad** is a set of three live hyperedges {a, b, c} every two of which
share a vertex: a triangle of the line graph (the graph whose vertices are the hyperedges, two of
them adjacent when they share a vertex). The input is a `dyng::hypergraph`; the output is a small
array of 64-bit counts, chosen by `options::kind`:

| `kind` | Bins | What one bin counts |
|---|---:|---|
| `pattern::h_motif` (default) | 30 | the triads by **h-motif**: which of the seven regions of their Venn diagram are non-empty (below) |
| `pattern::stathyper_type1` | 1 | the sum over the live hyperedges of C(\|e\|, 3): the vertex triples inside one hyperedge |
| `pattern::stathyper_type2` | 1 | the sum over the pairs of adjacent hyperedges (e, f) that share a >= 2 vertices of C(a, 2) * b * c, with b = \|e\| - a and c = \|f\| - a |
| `pattern::stathyper_type3` | 1 | the sum over the triads of x * y * z, the sizes of the three pairwise-only regions |

**The 30 bins of `h_motif`.** The seven regions of three hyperedges A, B and C are A only, B only,
C only, A and B only (AB), B and C only (BC), C and A only (CA), and all three (ABC). An h-motif
is a set of non-empty regions up to a permutation of A, B and C (Lee, Ko, Shin, "Hypergraph
Motifs", VLDB 2020; the order of ESCHER-GPU's table and of MoCHy). Bins are **0-based**: bin k is
ESCHER-GPU's h-motif k + 1. The table below is the one the code uses: the test
`TriadCountDocs.TheThirtyBinTableIsTheCodesTable` (`cpp/tests/algorithms/triad_count/triad_count_docs_test.cpp`)
reads this page and checks every row against the counting engine's Venn-id table
(`engine/motif_table.hpp`, ported from ESCHER-GPU's `include/motif_table.hpp`) and the oracle's
representatives (`testing::triad_class_representatives()`). A Venn id has bit 0, 1, 2 for A, B,
C only, bit 3, 4, 5 for AB, BC, CA and bit 6 for ABC; the canonical id of a class is its smallest
id, and the regions listed are those of the canonical id.

| Bin | ESCHER-GPU h-motif | Canonical Venn id | Non-empty regions | Counted |
|---:|---:|---:|---|---|
| 0 | 1 | 64 | ABC | yes: closed, a vertex in all three |
| 1 | 2 | 65 | ABC, A | yes: closed, a vertex in all three |
| 2 | 3 | 67 | ABC, A, B | yes: closed, a vertex in all three |
| 3 | 4 | 71 | ABC, A, B, C | yes: closed, a vertex in all three |
| 4 | 5 | 72 | ABC, AB | yes: closed, a vertex in all three |
| 5 | 6 | 73 | ABC, AB, A | yes: closed, a vertex in all three |
| 6 | 7 | 76 | ABC, AB, C | yes: closed, a vertex in all three |
| 7 | 8 | 75 | ABC, AB, A, B | yes: closed, a vertex in all three |
| 8 | 9 | 77 | ABC, AB, A, C | yes: closed, a vertex in all three |
| 9 | 10 | 79 | ABC, AB, A, B, C | yes: closed, a vertex in all three |
| 10 | 11 | 88 | ABC, AB, BC | yes: closed, a vertex in all three |
| 11 | 12 | 90 | ABC, AB, BC, B | yes: closed, a vertex in all three |
| 12 | 13 | 89 | ABC, AB, BC, A | yes: closed, a vertex in all three |
| 13 | 14 | 91 | ABC, AB, BC, A, B | yes: closed, a vertex in all three |
| 14 | 15 | 93 | ABC, AB, BC, A, C | yes: closed, a vertex in all three |
| 15 | 16 | 95 | ABC, AB, BC, A, B, C | yes: closed, a vertex in all three |
| 16 | 17 | 120 | ABC, AB, BC, CA | yes: closed, a vertex in all three |
| 17 | 18 | 121 | ABC, AB, BC, CA, A | yes: closed, a vertex in all three |
| 18 | 19 | 123 | ABC, AB, BC, CA, A, B | yes: closed, a vertex in all three |
| 19 | 20 | 127 | ABC, AB, BC, CA, A, B, C | yes: closed, a vertex in all three |
| 20 | 21 | 24 | AB, BC | always 0 (open: C and A share no vertex) |
| 21 | 22 | 26 | AB, BC, B | always 0 (open: C and A share no vertex) |
| 22 | 23 | 25 | AB, BC, A | always 0 (open: C and A share no vertex) |
| 23 | 24 | 27 | AB, BC, A, B | always 0 (open: C and A share no vertex) |
| 24 | 25 | 29 | AB, BC, A, C | always 0 (open: C and A share no vertex) |
| 25 | 26 | 31 | AB, BC, A, B, C | always 0 (open: C and A share no vertex) |
| 26 | 27 | 56 | AB, BC, CA | yes: closed, no vertex in all three |
| 27 | 28 | 57 | AB, BC, CA, A | yes: closed, no vertex in all three |
| 28 | 29 | 59 | AB, BC, CA, A, B | yes: closed, no vertex in all three |
| 29 | 30 | 63 | AB, BC, CA, A, B, C | yes: closed, no vertex in all three |

**Why bins 20-25 are always 0.** They are the **open** h-motifs (ESCHER-GPU's 21-26): one pair of
the three hyperedges shares no vertex. Such a triple is not a triangle of the line graph, so the
wedge engine never meets it; ESCHER-GPU's README says the same ("Open triads are not counted ...
this implementation deliberately covers closed triads only"). The bins exist so that the 30 bins
have the paper's and ESCHER-GPU's numbering; every golden of the parity set has 0 there, and so
does every dynG count. Closed triads fill the 24 bins 0-19 and 26-29.

**Update model.** A batch of hyperedge deletions (by id) and insertions (member lists) is applied
once to the hypergraph, and the counts change by the triads the batch destroys and creates,
without recounting the rest. A triad changes only if it contains a changed hyperedge, so the
delta is exact: minus the triads of the hypergraph before the batch (G_t) that contain a deleted
hyperedge, plus the triads of the hypergraph after it (G_{t+1}) that contain an inserted
hyperedge, each counted once, at its changed hyperedge with the smallest id (ESCHER-GPU's
"frontier-incident delta"). Postcondition: `update()` equals `compute()` on the new hypergraph
exactly ("baseline + deltas == recount"), on every backend.

## 2. Template mapping

```
normalize -> prepare -> before_apply -> count(-) on G_t -> commit ->
identify_affected -> count(+) on G_{t+1} -> finalize
```

triad_count is an **aggregate-delta** problem (PLAN 4.5.1; `family::aggregate_delta`, ownership
`ownership::min_member`, invariant I2): `local_pattern_problem` in
`cpp/src/algorithms/triad_count/problem.hpp`, whose template card is at its top, the same driver
`experimental::count_local_patterns<op_t>` runs a user's pattern on (ADR 0058). The cuda backend
is Tier B (ESCHER-GPU's wedge engine, `engine::fused`); the host backends are Tier A (the
incremental host engine, ADR 0054).

| Step | Hook (profiler stage) | What it does | ESCHER-GPU |
|---|---|---|---|
| 0 | `normalize` (`triad_count.normalize`, the hypergraph's Step 0) | the live, distinct deletions of G_t; the insertions with their members sorted and repeats dropped | the driver's batch |
| 1a | `count(-)` (`triad_count.count_minus`) | the frontier of Step 0's deletions; every triad (pair, hyperedge) containing a deleted hyperedge counted on G_t's internal row views, at its smallest deleted member, into the subtract bins; everything both phases need is reserved first (invariant I9) | `update.subtract` |
| apply | `commit` (`triad_count.commit`) | `hypergraph.apply` once: the ESCHER store's delete, best-fit reuse insert, fill / unfill and the line graph's sorted delta merge | `update.maintain` |
| 1b | `identify_affected` (`triad_count.identify_affected`) | the frontier of the ids the commit **assigned** to the insertions (`stats::batch.inserted_ids`; on cuda the applied record's device copy) | the add phase's frontier |
| 2 | `count(+)` (`triad_count.count_plus`) | every triad containing an inserted hyperedge counted on G_{t+1}, at its smallest inserted member, into the add bins | `update.add` |
| finish | `finalize` (`triad_count.finalize`) | the two phases applied to the counts (`count - removed + added`, exact in 64 bits); a bin that would become negative is an `internal_error`, a net change outside `stats::delta`'s signed range a `capacity_error` | the delta print |

On cuda `triad_count.enact_fused` holds identify_affected, count_plus and finalize. `compute()`
runs in `triad_count.compute`, the full count in `triad_count.count` inside it: every out-wedge
(a, b), b > a, of the line graph closed into its triangles (a, b, c), c > b (type 2: every
adjacent pair once; type 1: every live hyperedge). `update()` runs inside `triad_count.update`.

The four Chapter 3 challenges: (i) **affected set**: the hyperedges of the batch, nothing else (a
triad changes only through a changed member); (ii) **propagation scope**: one signed count per
phase over the wedges incident to the frontier, no iteration; (iii) **correctness under
parallelism**: each triad is counted by exactly one thread (the exactly-once filters of the
smallest frontier member), into block-private bins flushed once per block on cuda and per-thread
bins added in thread order on the host, so the counts do not depend on the schedule; (iv) **data
structure**: the ESCHER store's H2V, V2H and H2H trees, read through key-indexed internal row
views built once per version (never the public `h2v()` / `v2h()` / `h2h()` views).

## 3. API

```cpp
#include <dyng/dyng.hpp>

dyng::io::hypergraph_read_options ro;
ro.dedupe_hyperedges = true;  // ESCHER-GPU's coauth preprocessing
const auto hes = dyng::io::read_hypergraph<std::int32_t, dyng::unweighted>("coauth-DBLP.hg", ro);
dyng::hypergraph_properties hp;
hp.line_graph = dyng::line_graph_policy::sorted_merge;
auto hg = dyng::hypergraph<std::int32_t>::from_hyperedges(res, hes.view(), hp);
auto triads = dyng::triad_count::compute(res, hg);  // 30-bin closed h-motifs
dyng::hyperedge_batch<std::int32_t, dyng::unweighted> b;
b.delete_hyperedge(17);
b.delete_hyperedge(42);
b.insert_hyperedge({3, 9, 12});
b.insert_hyperedge({9, 12, 40, 41});
auto st = dyng::triad_count::update(res, hg, b.view(), triads);
const auto& ids = st.batch.inserted_ids;  // the store's choice: read them here
// st.delta[k]: the change of bin k; st.triads_removed / st.triads_added; triads.counts()
```

(PLAN 5.3's example as ADR 0041 section 1 rewrites it against the frozen hypergraph API. The ids of
the inserted hyperedges are the store's choice: read them from `inserted_ids`, never predict them.
On coauth-DBLP they are 17 and 2,467,389: `{3, 9, 12}` fits the slot of hyperedge 17, but
`{9, 12, 40, 41}` does not fit 42's (a slot keeps the size of the row it was made for, here one
member), so it goes after the last live id.)
Several patterns are several results; `dyng::update(res, hg, b.view(), triads, type1, ...)`
updates them all through **one** commit (`examples/cpp/triad_count_update.cpp`; the tutorial
{doc}`../tutorials/hypergraph_triads`).

| Option | Default | Meaning |
|---|---|---|
| `kind` | `pattern::h_motif` | what the result counts (fixed at `compute()`; `set_options()` refuses a change) |
| `cuda_engine` | `engine::automatic` | a tunable: `automatic` and `fused` select the wedge engine; `operators` throws `not_supported_error` on cuda (there is no operators engine); ignored on the host backends |

| `stats` field | Meaning (every counter is deterministic) |
|---|---|
| `batch` | the `hyper_apply_summary` of the commit: `inserted_ids` (the id of every inserted hyperedge, in batch order), the deletion and insertion counts, the skipped operations |
| `deleted_frontier`, `inserted_frontier` | the hyperedges of the subtract and add phases |
| `triads_removed`, `triads_added` | the subtract and add phases summed over the bins (types 1-3: the contributions removed and added) |
| `delta` | `std::array<std::int64_t, 30>`: counts after minus counts before, per bin (0 at and above `num_bins()`) |
| `affected` | the number of bins whose count changed |
| `iterations`, `fallback_used`, `converged` | 0, false, true (no Step 2 loop; every backend is incremental) |
| `frontier_visits` | `deleted_frontier + inserted_frontier` |
| `engine_used` | `engine::fused` on cuda, `engine::operators` on the host backends |

`result` is opaque and move-only: `kind()`, `num_bins()` (30 or 1), `counts()` (host memory on
every backend), `count(bin)`, `total()`, `get_options()` / `set_options()`, `graph_version()`,
`space()` (`memory_space::host`) and `clone(res)`. A result goes stale when its hypergraph
changes without it (`stale_result_error`), and a failed update after the commit poisons it
(see the header's `@guarantee`).

**Python** (`dyng.triad_count`): the same names; `kind` is a string or a
{py:class}`dyng.triad_count.Pattern`, `Result.counts` a read-only uint64 NumPy array,
`Stats.delta` an int64 array of the result's `num_bins` values, and `dyng.update()` updates
several results over one commit:

<!-- snippet: triad-count-python -->
```python
import dyng

hg = dyng.Hypergraph.from_lists([[0, 1, 2], [1, 2, 3], [2, 3, 4], [0, 2, 4]])
tri = dyng.triad_count.compute(hg)  # kind="h_motif": 30 bins
t1 = dyng.triad_count.compute(hg, kind="stathyper_type1")  # one bin: the sum of C(|e|, 3)
print(tri.total, {k: int(c) for k, c in enumerate(tri.counts) if c}, t1.total)
hb = dyng.HyperedgeBatch(delete_hyperedges=[3], insert_hyperedges=[[1, 3, 4]])
st, st1 = dyng.update(hg, hb, tri, t1)  # one commit for both results
print(st.batch.inserted_ids.tolist(), st.triads_removed, st.triads_added, tri.total)
print({k: int(d) for k, d in enumerate(st.delta) if d}, st1.triads_removed, st1.triads_added)
```

<!-- snippet-output: triad-count-python -->
```text
4 {14: 4} 4
[3] 3 3 4
{14: -2, 16: 1, 27: 1} 1 1
```

The four hyperedges pairwise share vertex 2, and every triple has the regions ABC, AB, BC, A and
C (h-motif 15, bin 14). Deleting {0, 2, 4} and inserting {1, 3, 4} (which reuses id 3) removes
three triads and creates three: one stays in bin 14, one is in bin 16 (all four regions shared)
and one has no vertex in all three (bin 27). `examples/python/triad_count.py` is the example
program, and the reference is {doc}`../api/python/index`.

**Command line** ({doc}`../api/cli`): `dyng triad_count compute|update` takes one flag per option
field (`--kind`, `--cuda-engine`), `dyng.io.read_hypergraph`'s keywords as read flags, and
`--batches` (`.dgt` or `.dgb` files). `--format json` (the default) writes the triad-bins JSON
({doc}`../api/file_formats`, `dyng.io.read_triad_bins` reads it back); `--format escher` prints
ESCHER-GPU's own output lines byte for byte (diffed against its unpatched standard output on all
60 make-test cases, `parity/results/M9b.md` section 2.1):

```console
$ dyng triad_count update --input examples/data/triad_count.hg --dedupe-hyperedges \
      --batches examples/data/triad_count.dgt --format escher
Hypergraph: 11 hyperedges, max vertex id 10, 36 pins
Motif counts (30 bins): 0 0 0 2 0 0 0 0 0 9 0 0 0 0 19 6 0 0 0 0 0 0 0 0 0 0 0 0 1 8
Batch 1 (2 deletions, 2 insertions)
Motif delta counts (30 bins): 0 0 0 1 0 0 0 0 0 7 0 0 0 0 -6 -1 0 0 0 0 0 0 0 0 0 0 0 2 0 5
Batch 2 (1 deletions, 1 insertions)
Motif delta counts (30 bins): 0 0 0 2 0 0 7 0 0 -5 0 0 0 0 -3 0 0 0 0 0 0 0 0 0 0 0 0 0 0 4
```

## 4. Backends, engines, requirements and determinism

| Backend | Engine | Full count | Update |
|---|---|---|---|
| cuda | ESCHER-GPU's wedge engine (Tier B, `engine::fused`): one thread per oriented wedge, block-private 64-bit bins in shared memory flushed once per block with the phase's sign, a pooled workspace reserved before the commit | `countWedges` + the pattern's kernel (`triangle_kernel`, `pair_kernel`, `key_kernel`) | the frontier built on the device; at most 3 host synchronizations per update (1 for type 1); 19-22 per batch with the container's apply, against ESCHER-GPU's 90-96 |
| sequential, openmp | the incremental host engine (Tier A, ADR 0054): the same wedges, filters and operations as loops over the sources, run by `operators::count_delta` (a static partition, private bins per thread added in thread order) | the same | the same frontier phases |

**Requirements per pattern** (checked by `compute()`, which names the property; the registry
lists the default pattern's):

| Pattern | Line graph (`line_graph != none`) | Vertex-to-hyperedge rows (`store_v2h`) | Other |
|---|---|---|---|
| `h_motif` | required | required | fewer than 2,147,483,647 vertices (the classifying merge's end marker; vertex ids up to 2^31 - 3) |
| `stathyper_type1` | not read | not read | - |
| `stathyper_type2` | required | required | - |
| `stathyper_type3` | required | required | as `h_motif` |

Every pattern: `hypergraph<std::int32_t, unweighted>` or `hypergraph<std::int32_t, std::int32_t>`
(the weights are ignored; other types fail to compile with a `static_assert` naming these), any
storage, `id_reuse` and capacity policy. Under `line_graph_policy::automatic` `compute()` builds
the line graph once. The batch may not carry incidence edits (`not_supported_error`, 0.4).

**Determinism**: `exact_value`. The counts, the deltas and every counter are identical on every
backend, engine, thread count and run (conformance checks C3 and C4; the OpenMP replays at 1 and 8
threads are identical case by case). The counts are host arrays on every backend.

## 5. Complexity, memory and performance

**Work.** The full count visits every out-wedge (a, b), b > a, of the line graph and intersects
the rows of a and b after b for the closing c (a merge of the two sorted rows), then classifies
each triangle with one three-way merge of the members (h_motif, type 3); type 2 visits each
adjacent pair once, type 1 each hyperedge. An update visits only the wedges incident to the
frontier: 2.57M wedges for a 50K batch on coauth-DBLP (ESCHER-GPU's `CHANGES.md`), against 88M for
the papers' anchored delta. The update's time is dominated by the two phases (on coauth-DBLP 0.49
of 0.55 s per 50K batch in ESCHER-GPU's profile); the container's apply (`triad_count.commit`) is
about 46-55 ms of it.

**Memory: the line graph dominates.** The counting itself keeps a few MiB (the bins, the frontier
flags and the wedge offsets; 2.1-4.2 MiB live during `hypergraph.apply` on the gated inputs). The
hypergraph's ESCHER store holds H2V, V2H and the line graph H2H: on coauth-DBLP (2,467,389
hyperedges, 7.75M pins) H2H has **251.5M entries**, and the store is 1,922,784,204 bytes after
the build (1,833.7 MiB, equal to ESCHER-GPU's `deviceBytes()` at every point). The peak of live
device allocations, 3,023 MiB, is reached in the line graph's sorted delta merge inside
`hypergraph.apply`, when the old and the new H2H payloads coexist (ESCHER-GPU: 3,019 MiB at the
same point). A very dense hypergraph whose line graph does not fit cannot be counted (section 6).
The **public views** `h2v()`, `v2h()` and `h2h()` are copies of the store's rows built on demand
and counted in `storage_stats::view_bytes`, outside the store's bytes: triad_count never builds
them (it reads internal row views of the store), but a caller who reads `h2h()` on coauth-DBLP
adds about 1 GB of 32-bit ids (plus the offsets) to the process. Between batches the
stream-ordered pool keeps what the merge reached (1.63-1.80x ESCHER-GPU's held bytes; reported,
not gated: D1, D11).

**Profiler stages and the paper's timed region.** `parity/timed_regions/triad_count.toml` maps
ESCHER-GPU's timers to the stages (ADR 0048): ESCHER's paper-timed `update` (subtract + maintain +
add) is `triad_count.update` per batch (`update[b1..b3]`), the full count is `triad_count.compute`,
end to end is the driver's total. The gates (PLAN 6.4.6, 8.6) compare dynG with ESCHER-GPU
re-measured in the same session (never with the figures of its `results/README.md`), on GPU 0 at
the base clock lock (SM 1170 / memory 7601 MHz), A/B alternation, 11 rounds on coauth-DBLP and 21
on the synthetic inputs; the device memory as the store's accounting at every gated point and as the
Nsight Systems peak of live allocations (D1).

The gate record (`benchmarks/paper/ipdps26_escher.yaml`, `tkde26_escher_plus.yaml`; M9a's final
reading, `parity/results/M9a.md` section 6, on the library after M9a's review fixes; ratio =
dynG / ESCHER-GPU, medians):

| Gate | coauth-DBLP 50K, seeds 1-3 | syn1M 50K, seeds 1-3 | syn2M 50K, seeds 1-3 | bar |
|---|---|---|---|---|
| update, each of 3 batches | 0.949-0.971 (486-532 ms) | 0.868-1.002 (142-147 ms) | 0.853-1.019 (152-161 ms) | 1.05 |
| full count | 0.998-0.999 (5,286 ms) | 1.002-1.003 (637-642 ms) | 1.002-1.003 (1,277-1,283 ms) | 1.05 |
| end to end (3 batches) | 0.915-0.916 (8.99-9.01 s) | 0.982-0.983 (2.44-2.46 s) | 0.967-0.983 (4.35-4.42 s) | 1.10 |
| store accounting (every gated point) | 1.0000 | 1.0000 | 1.0000 | 1.05 |
| Nsight Systems peak of live allocations | 1.0015 (3,023 MiB) | 1.0021 (1,134 MiB) | 1.0017 (2,215 MiB) | 1.05 |

M9b moved the engine into the installed headers without changing the counting kernels' machine
code, and read every gate again on its own library, the 0.3.0 candidate's, with the same protocol:
at step 3 (`b732aa9`) and again after the M9b review changed the experimental headers (`c83cebf`,
`parity/results/M9b.md` section 5; the counting kernels still identical to M9a's). The latest
reading: update per batch 0.951-0.971 (coauth-DBLP) and 0.862-1.033 (synthetic), full count
0.999-1.003, end to end 0.940-0.992, the store accounting 1.0000 and the peak 1.0015-1.0021; 45 /
45 and 11 / 11 rows within their bars. Recorded, not gated, in the same reading: the default
clocks (update 0.85-1.03), 100K and 200K coauth batches (0.98-1.02), StatHyper's types on coauth
(update 0.74-0.97 of ESCHER-GPU's), and the update with the driver's host model counted (D12: up
to 1.074 on three syn2m rows at the base lock, 1.084 on syn2m_b50k_s1 b2 at default clocks).
The figures of ESCHER-GPU's `results/README.md` (coauth 50K update 0.546 s, full count 5.22 s,
9.1 s end to end, 1.83 GB) are its own default-clock medians of 3 runs; dynG at default clocks
read 0.50-0.55 s, 5.05 s and 8.6 s in M9a's step 5 (`parity/results/M9a.md` section 5.3), and
0.49-0.55 s, 5.04 s and 8.58 s in M9b's latest reading.

## 6. Limitations

- **Closed triads only**: the open h-motifs (bins 20-25) are always 0 (section 1).
- **The line graph is materialized**: a hypergraph whose H2H does not fit in device memory (very
  dense inputs, such as ESCHER-GPU's example "Tags") cannot be counted; there is no implicit line
  graph policy (roadmap, later).
- **Vertex ids** up to 2^31 - 3 for `h_motif` and type 3 (`compute()` refuses 2,147,483,647
  vertices); offsets of the store are int32 (a tree's payload below 2^31 entries;
  `capacity_error` otherwise), as in ESCHER-GPU.
- **64-bit counts**: every count is exact up to 2^64 - 1 on every backend; a count past it, or an
  update that changes one bin by 2^63 or more, throws `capacity_error` (ADR 0057; ESCHER-GPU's
  signed arithmetic goes wrong from 2,097,152 members per hyperedge for type 1).
- **The host backends are not tuned** (no performance gate; D3): the OpenMP backend partitions
  the sources statically, and divergent rows (hubs) are not balanced.
- **No coarse triangles** (ESCHER-GPU's `coarseTriangle`: inner, outer, hybrid and hyperedge
  triangles, which need the vertex-to-vertex graph): planned for a later release (0.5 in the
  {doc}`roadmap <../roadmap>`).
- **Host memory exhaustion** is `out_of_memory_error` on every backend, with the guarantee of
  `update()`'s documentation (the hypergraph poisoned when the failure is inside the commit, the
  result then unchanged): since 0.3.0rc1 also inside the hypergraph container's OpenMP loops,
  which ended the process with `std::terminate` before (D13; {doc}`ADR 0061
  <../adr/0061-exceptions-out-of-openmp-regions>`).

## 7. Differences from the paper

- **Closed triads only.** The papers' h-motif counts include the open h-motifs (one pair of
  hyperedges disjoint); the code, ESCHER-GPU and dynG, enumerates the triangles of the line graph
  only, so bins 20-25 are 0 by design.
- **No open, incident-vertex or temporal triads.** The incident-vertex and temporal triads (thesis
  Chapters 6-7; THyMe+) are in no code of the originals; they are on the roadmap as future
  patterns ("good first issue" material: an Op of `count_local_patterns`).
- **No coarse triangles.** ESCHER-GPU's `coarseTriangle` executable (inner, outer, hybrid and
  hyperedge triangles over the vertex graph) is not ported in 0.3 (PLAN 6.1, C6; planned for a
  later release, 0.5 in the {doc}`roadmap <../roadmap>`).
- **The exact frontier-incident delta.** dynG's update is the fixed code's exact delta (each
  changed triad counted once at its smallest changed member), not the papers' anchored delta,
  which was inexact (section 8, E-C3).
- **Exact 64-bit arithmetic** (ADR 0057): dynG's StatHyper counts are exact where ESCHER-GPU's
  wrap or are undefined (no golden case comes near the limits), and a triangle of no h-motif class
  raises `internal_error` where ESCHER-GPU would add to `bins[-1]`.

## 8. Paper vs fixed code

*This wording, the plan's default (D10), was accepted by the author on 2026-10-22 for 0.3.0.*
The papers' results were produced by ESCHER-GPU's code as published (tag
`baseline-2026-09` = `6a44c1c`). dynG ports the fixed code `abcf9b2` (the `fix/correctness-perf`
branch), whose `README.md`, `CHANGES.md` and `results/README.md` record what changed. In their
words:

| Item | ESCHER-GPU's record | dynG |
|---|---|---|
| **E-C1** CBST position overflow | "`(2*(pos+1-2^k)+1) * 2^log2(n) / 2^k` overflowed int32 for n > 65,535, so the tree lost its BST order: a search found 75% of the keys at n = 65,536, 2.05% at 1M and 0.29% at coauth-DBLP size; the counting kernels silently skipped the rest (**1,430 triangles on coauth instead of 8,354,454,505**)." Fixed with an exact 64-bit shift. | the fixed formula (the store, M8a); every key is found at every size |
| **E-C3** inexact anchored delta | "The anchored update recounted every triangle at each anchor min(f, j, k), with anchor sets computed separately for the deletions (old snapshot) and the insertions (new snapshot); unchanged triangles anchored in A_old \ A_new were subtracted and never added back (coauth, 50K batch: **delta sum -88,969,857 vs the true +1,771,385**)." Replaced by the exact frontier-incident delta. | the exact delta (section 1); "baseline + deltas == recount" after every batch of every golden |
| **E-C7** open h-motifs / type 2 delta | "By design only closed triads (line-graph triangles) are counted; the open-triad bins 21-26 (0-based 20-25) are never enumerated. This is documented in the README. The type2 frontier delta counted only pairs (f, j) with j > f; it now counts every pair containing a changed hyperedge once. type3 had the same defect (triangles only where f was the smallest id) and is fixed the same way." | closed triads only (sections 1, 7); the fixed type 2 and type 3 deltas |

The other fixes (E-C2 the flatten terminator, E-C4 64-bit bins, E-C5 best-fit slot assignment,
E-C6 V2H keys and occupancy, E-C8 one seeded RNG, E-C9 unfill / fill across chained segments,
E-C10 debug output in the timed path) and the performance work (EP1-EP5, H-E1: the wedge-parallel
kernels, the GPU line-graph merge, per-tree capacities; coauth 50K per batch 136.7 s -> 0.546 s
against the correctness-equivalent baseline) are summarized in {doc}`../history/escher_gpu`. As
ESCHER-GPU's `results/README.md` notes, the original code was about as fast per batch only
because E-C1 made it skip nearly all triangles.

## 9. Mapping from the original code

| ESCHER-GPU@abcf9b2 | dynG |
|---|---|
| `DynamicCounter` (count / subtract / add / reference / print), `src/driver.cu` | the problem of `triad_count` (`local_pattern_problem`, `problem.hpp`, `triad_count.cpp`; PLAN's `triad_problem`); `dyng-compat-escher run` (the drop-in driver with ESCHER-GPU's argv, for the gates) |
| `MotifCounter` `subtract()` / `add()` | `triad_count::update()` (`count_minus`, `commit`, `count_plus`, `finalize`) |
| the full count (`HMotifCount.cu`, `init.full_count`) | `triad_count::compute()` |
| `countTriangles<Op>`, `countPairs<Op>`, `countKeys<Op>` (`kernel/wedge.cuh`) | `experimental::count_local_patterns<op_t>` with `shape::triangles`, `pairs`, `keys` (`<dyng/experimental/local_patterns.cuh>`) |
| `countWedgeShape`, `forEachWedge`, `TriangleBody` | `count_wedges_kernel` (`engine/wedge.cuh`, in libdyng), `for_each_wedge` and `triangle_kernel` (`dyng/experimental/detail/local_patterns_kernels.cuh`); on the host `host_source` (`local_patterns_engine.hpp`) |
| `MotifOp` (`kernel/motif_utils.cuh`), `include/motif_table.hpp` | `motif_op` (`engine/motif_ops.hpp`), `engine/motif_table.hpp`; `pattern::h_motif` |
| `build/type1` (`Choose3Op`, `kernel/count_ops.cuh`), `build/type2` (`Type2Op`), `build/type3` (`Type3Op`) | `pattern::stathyper_type1`, `stathyper_type2`, `stathyper_type3` (`choose3_op`, `type2_op`, `type3_op`) |
| `DeviceFrontier` (`include/frontier.hpp`), `include/device_bins.hpp` | the workspace of `cuda.cu` (`frontier_kernel`), the phase bins (`triad_phase_bins`) |
| the row views (`kernel/locator.cu`) | the store's internal row views (`hypergraph_access::views()`) |
| the timers (`timing:update`, `update.subtract`, `update.maintain`, `update.add`, `init.full_count`, `total`) | the stages `triad_count.update`, `count_minus`, `commit`, `count_plus` (+ `identify_affected`, `finalize`), `triad_count.compute`, the driver's total |
| `utils/reference.cpp` (the CPU reference) | `testing::triad_reference` (an independent oracle with its own class map, ADR 0049); `dyng.testing.triad_reference` |
| `--check` (baseline + deltas == full GPU recount) | the oracle `compute` of the conformance kit; `dyng triad_count update --compare-recompute` |
| `build/main` and its output lines | `dyng triad_count compute|update --format escher` |
| `Batch`, `deletedKeys`, `generateHypergraph` | `hyperedge_batch`, `generators::legacy::escher_hypergraph`, `escher_batch_generator` (with `escher_host_model`) |

## 10. How to cite

`dyng::citation("triad_count")` (C++) and `dyng.citation("triad_count")` (Python) return the
library's BibTeX entry and those of the manifest's keys `escher2026` (ESCHER, IPDPS 2026) and
`escherplus2026` (ESCHER+, TKDE 2026) in `docs/references.bib`; {doc}`../citing`.
