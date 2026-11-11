# hyper_sssp: shortest hyperpaths

Maturity: **experimental**, new in 0.4 (a draft: the API is tracked by the API snapshot and
frozen at the end of M11b; whether it is stable at 0.4.0 is the author's decision, HD17;
{doc}`ADR 0090 <../adr/0090-hyper-sssp-draft-api-and-names>`). Header: `<dyng/hyper_sssp.hpp>`.
Container: `hypergraph` (weighted, slack_csr storage). Oracle: `compute`. Determinism:
`bitwise`. Backends: sequential and cuda. Paper: H-SOSP (IA3 2026); code: MOSP_ESCHER@4b86159.
Parity: exact against MOSP_ESCHER on its golden set `hyper_sssp_mosp_escher` (every update and
recompute of 25 cases, three parity modes; [M11a record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M11a.md)).

:::{note}
This page is the M11a draft. M11b adds the shared sssp engine (`engine::operators` on cuda), the
Python module and the command line, and completes the page.
:::

## 1. Problem

**What it computes.** The line graph of a hypergraph has one node per live hyperedge; two nodes
are adjacent when their hyperedges share a vertex. Entering hyperedge h costs its weight w(h), an
integer in [1, 2^28]. From a source vertex s, the distance D(h) of a live hyperedge is the cost
of the cheapest walk h_1, ..., h_k = h in the line graph with s in h_1:
D(h) = w(h_1) + ... + w(h_k). So D(h) = w(h) for every hyperedge that contains s.

**The tree.** The parent of h is -1 when s is in h (D(h) = w(h)); otherwise it is the lowest-id
neighbour p with D(p) + w(h) = D(h) (MOSP_ESCHER's tie rule: the canonical tree, unique for a
hypergraph). Dead and unreachable hyperedges have the distance
`infinite_distance<std::int64_t>()` and the parent -1.

**Vertex distances** (HD11). `distance_to(v)` is the minimum of D(h) over the live hyperedges
that contain v, and 0 for v = s. D(h) already includes w(h), so nothing is added: PLAN 5.3's
"+ w(h)" would count the last hyperedge twice; this rule supersedes it (ADR 0090).

**The update model.** A batch deletes and inserts hyperedges and removes and adds incidences
(`hyperedge_batch::delete_incidence` / `insert_incidence`, slack_csr storage only). `update()`
applies it to the hypergraph once and brings the result up to date; afterwards it equals
`compute()` on the new hypergraph bit for bit.

## 2. Template mapping

| Step | Hook | What it does (MOSP_ESCHER's hsospUpdate) |
|---|---|---|
| 0, 1 | `begin_update` | checks the batch's weights (<= 2^28) before the commit; the commit computes the line-graph pair delta (ADR 0086) |
| 2a | `identify_affected` | the roots: the endpoint b of every deleted pair (a, b) that was a tree edge, every new, recreated or dead id, every hyperedge that lost s while its parent was -1; their subtrees are invalidated (pointer jumping on cuda, at most 64 rounds) |
| 2b | `seed` | every invalidated hyperedge pulls its best (distance, parent) over its neighbours (and w(h) if it holds s); both directions of every inserted pair are relaxed |
| 2c | `loop` | the improved hyperedges push to their neighbours (decrease-only, packed 64-bit atomicMin of (distance << 32 \| parent) on cuda, a priority queue on sequential) until nothing decreases |
| 2d | `recompute` | the budget's fallback (always entered: 0 ms without a fallback) |
| 3 | `finalize` | the counters |

The four challenges: the affected set is the invalidated subtrees (a deletion only lengthens the
paths through its tree edge); the propagation is decrease-only after the invalidation, so its
order does not change the fixed point; correctness under parallelism comes from the packed words
(distance and parent change together) and the lowest-id tie rule; the data structure is the slack
CSR line graph kept by the container (ADR 0086, ADR 0089).

Both engines run through the framework's `update_enactor`: on sequential the hooks above one by
one (Tier A; the loop drains its priority queue in one call and stops at the work budget), on
cuda MOSP_ESCHER's hsospUpdate behind `enact_fused`, which opens the same stages itself (Tier B).
The framework asks the work budget after the loop (its `over_budget` hook, ADR 0092).

## 3. API

```cpp
#include <dyng/hyper_sssp.hpp>

dyng::hypergraph<std::int32_t, std::int32_t> h = /* slack_csr storage, weighted */;
auto r = dyng::hyper_sssp::compute(res, h, /*source vertex=*/0);
dyng::hyper_sssp::stats s = dyng::hyper_sssp::update(res, h, batch.view(), r);
std::int64_t d = r.distance_to(42);
```

`result` holds `hyperedge_distances()` (int64), `hyperedge_parents()` (int32: slack_csr ids are
below 2^31), `distance_to()` and `vertex_distances()` (computed on request from the hypergraph the
result matches; `stale_result_error` once that hypergraph changed or is gone), the options, the
version, `space()` and `clone()`. `dyng::update(res, h, batch, r1, r2, ...)` updates several
results (of different sources, or with other hypergraph algorithms) after one commit.

**Options** (all tunables): `work_budget` (1.0; a multiple of the live line-graph entries),
`max_iterations` (4096 push rounds, cuda), `on_budget` (`budget_action::recompute` or `error`),
`cuda_engine` (`automatic` and `fused`: MOSP_ESCHER's kernels; `operators`: the shared sssp
engine, the same results) and `validate_inputs`.

**Stats**: the inherited `update_stats` fields, `batch`, `line_graph_insertions`,
`line_graph_deletions`, `invalidated`, `relaxations` and `fallback_reason`. Which are
deterministic is stated per field in the header.

## 4. Backends, engines, requirements and determinism

| Backend | Engine | Notes |
|---|---|---|
| sequential | `engine::operators` | MOSP_ESCHER's host emulation (emulateSospRecompute, emulateSospUpdate) on the host line graph; the budget is checked before each expansion; no iteration, width or depth limit |
| cuda | `engine::fused` (the default: `engine::automatic` selects it, by the gates; section 5) | MOSP_ESCHER's kernels as they are (transplanted with `parity/escher_port/transplant.py`, ADR 0091) |
| cuda | `engine::operators` | the shared sssp engine (M11b, O12; ADR 0095): sssp's CUDA operators over the line graph (slack rows, node weights `w[col]` pushing and `w[v]` pulling, one symmetric view, the public mode's base offers, one warp per row), the host loop of sssp's operators engine with MOSP_ESCHER's budget triggers, the static solve on sssp's near-far loop with MOSP's defaultDelta; the same results as `engine::fused`, bit for bit |
| openmp | none | `not_supported_error` naming sequential and cuda (HD14) |

Requirements, checked by `compute()` and before the commit of `update()`: `hypergraph<int32_t,
int32_t>` (other types fail to compile), weights in [1, 2^28], `hypergraph_storage::slack_csr`
with its line graph (`line_graph_policy::slack_csr` or `automatic`) and `store_v2h`. Any
`id_reuse` order.

Determinism: distances, parents, `invalidated` and the line-graph counts are bit-exact across
backends and runs. `fallback_used` and `fallback_reason` are deterministic on sequential and
schedule-dependent on cuda near the budget (the relaxations of a push round depend on the thread
schedule; the results are identical either way; HD7, ADR 0092).

## 5. Complexity, memory and performance

Profiler stages: `hyper_sssp.compute`, `hyper_sssp.identify_affected`, `hyper_sssp.seed`,
`hyper_sssp.loop`, `hyper_sssp.recompute`, `hyper_sssp.finalize`, inside the container's commit
stages. The cuda engine synchronizes the stream once per pointer-jumping round and once per push
round. The compat tool `dyng-compat-hsosp bench` reproduces hsospBench's rows (44-column CSV) and
`parity/hsosp_perf.py run` compares them with MOSP_ESCHER under the gate protocol of ADR 0080.

**The default CUDA engine (O12; HD13, decided by the author on 2026-10-28; ADR 0095).** The
shared sssp engine would become the default only if it gave the same results as the as-is
engine and MOSP_ESCHER and were within every gate against MOSP_ESCHER@4b86159. It gives the same
results (every batch of the golden set, in the three parity modes) and is within the memory
gate (the peak of live device allocations 0.49-0.56 of MOSP_ESCHER's), but in the full gate
reading (16 paper configurations, 21 rounds, base clocks) four `sosp_update` rows of
coauth-DBLP's hyperedge batches are above their 1.05 bar:

| Configuration, batch | MOSP_ESCHER's `t_sosp_update_ms` | `engine::fused` / MOSP_ESCHER | `engine::operators` / MOSP_ESCHER |
|---|---:|---:|---:|
| dblp_hyperedge_100000 b2 | 10.1 ms | 1.024 | 1.051 |
| dblp_hyperedge_200000 b1 | 22.2 ms | 1.002 | 1.056 |
| dblp_hyperedge_200000 b2 | 23.5 ms | 1.001 | 1.061 |
| dblp_hyperedge_200000 b3 | 21.7 ms | 1.004 | 1.059 |

The shared engine's update loses 1.2-1.5 ms there in two places: sssp's pointer jumping runs its
ceil(log2 n) + 1 rounds where MOSP_ESCHER's stops as soon as nothing changes, and the seed pulls
the invalidated nodes and seeds the inserted pairs in two launches. Everywhere else it reads the
same or faster (Geology's updates 0.78-0.99 of MOSP_ESCHER's), and its static solve, sssp's
near-far loop with MOSP's defaultDelta, reads 0.69-0.74 of MOSP_ESCHER's recompute (the as-is
engine's: 0.98-1.04). So `engine::automatic` keeps `engine::fused`; select the shared engine with
`options::cuda_engine = engine::operators` (for example where the static recompute or Geology-like
updates dominate). The numbers: `parity/results/M11b.md` section 2.

## 6. Limitations

No openmp backend; no unweighted hypergraphs; weights at most 2^28; the escher storage is not
supported (it keeps no slack line graph); the Python layer arrives in M11b.

## 7. Differences from the paper and the code

- **No virtual hyperedges.** MOSP_ESCHER adds a source row {s} and a target row {t} of weight 0 and
  runs from the source row; dynG's public API takes a source vertex and has no virtual rows (the
  hyperedges holding s start at their own weight with parent -1). The parity harness runs an
  internal virtual-endpoint mode that reproduces MOSP_ESCHER by id (ADR 0091; no public name).
- **Vertex distances** follow HD11 (above), not PLAN 5.3's "+ w(h)".
- **The budget** is checked by the host engine too (before each expansion), and
  `budget_action::error` throws instead of falling back (ADR 0092).

## 8. Mapping from the original code

| MOSP_ESCHER@4b86159 | dynG |
|---|---|
| `hsosp/src/hsospDevice.cu` (kernels, hsospRecompute, hsospUpdate) | `cpp/src/algorithms/hyper_sssp/kernels.cuh`, `engine.cu` (transplanted), `cuda.cu` (the driver) |
| `hsosp/src/hsospDevice.cu` (the same steps on sssp's engine: `engine::operators`) | `operators.cuh`, `operators.cu` (hyper_sssp's kernels and host loop) over `cpp/src/algorithms/sssp/operators.cuh` (sssp's kernels with the line graph's policies) and `cpp/src/util/slack_rows.cuh` |
| `hsosp/include/hsosp.cuh` (HsospState, UpdateConfig, UpdateStats) | `state.cuh`, `options`, `virtual_endpoints.hpp` |
| `hypergraph/src/HostHypergraph.cpp` (emulateSospRecompute / Update) | `sequential.cpp` |
| `hypergraph/src/HypergraphOracle.cpp` | `dyng::testing::hyper_sssp_reference` (ADR 0082) |

## 9. How to cite

`dyng::citation("hyper_sssp")` returns the library's BibTeX entry and the entry `hsosp2026`.
