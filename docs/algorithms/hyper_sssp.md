# hyper_sssp: shortest hyperpaths

Maturity: **stable** from 0.4.0 on the author's confirmation (HD17, pending; the API is frozen
since the end of M11b, {doc}`ADR 0096 <../adr/0096-hyper-sssp-api-review-and-freeze-0-4>`). The
sequential and CUDA backends, the Python module `dyng.hyper_sssp`, the command line
`dyng hyper_sssp` and the benchmark suite `benchmarks/paper/ia3_hsosp.yaml` are new in 0.4.0.
Header: `<dyng/hyper_sssp.hpp>`. Container: `hypergraph` (weighted, `slack_csr` storage).
Oracle: `compute`. Determinism: `bitwise`. Paper: H-SOSP (IA3 2026); code: MOSP_ESCHER@4b86159.
Parity: exact against MOSP_ESCHER on every update and recompute of its golden set
`hyper_sssp_mosp_escher` (25 cases: its `make test` sessions and the 16 paper configurations;
distances and parents by id, three parity modes, on both CUDA engines; [M11a
record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M11a.md), [M11b
record](https://github.com/dyng-dev/dyng/blob/main/parity/results/M11b.md)). The tutorial is
{doc}`../tutorials/shortest_hyperpaths`.

## 1. Problem

**What it computes.** The line graph of a hypergraph has one node per live hyperedge; two nodes
are adjacent when their hyperedges share a vertex. Entering hyperedge h costs its weight w(h), an
integer in [1, 2^28]. From a source vertex s, the distance D(h) of a live hyperedge is the cost
of the cheapest walk h_1, ..., h_k = h in the line graph with s in h_1:
D(h) = w(h_1) + ... + w(h_k). So D(h) = w(h) for every hyperedge that contains s.

**The tree.** The parent of h is -1 when s is in h (D(h) = w(h)); otherwise it is the lowest-id
neighbour p with D(p) + w(h) = D(h) (MOSP_ESCHER's tie rule: the canonical tree, unique for a
hypergraph). Dead and unreachable hyperedges have the distance
`infinite_distance<std::int64_t>()` (2^61 - 1; Python `dyng.hyper_sssp.INFINITE_DISTANCE`) and
the parent -1.

**Vertex distances** (HD11, decided by the author on 2026-10-28). `distance_to(v)` is the minimum
of D(h) over the live hyperedges that contain v, and 0 for v = s; `vertex_distances()` is that
value for every vertex, computed on request once per hypergraph version. D(h) already includes
w(h), so nothing is added: PLAN 5.3's "+ w(h)" would count the last hyperedge twice, and this
rule supersedes it ({doc}`ADR 0090 <../adr/0090-hyper-sssp-draft-api-and-names>`). A vertex in no
live reachable hyperedge has the infinite distance.

**The update model.** A batch deletes hyperedges (by id), removes vertices from live hyperedges
and adds vertices to them (incidence edits, `slack_csr` storage only), and inserts hyperedges, in
that order whatever the order of its arrays (MOSP_ESCHER's phase order; {doc}`../concepts/hypergraphs`).
`update()` applies it to the hypergraph once and brings the result up to date; afterwards it
equals `compute()` on the new hypergraph bit for bit, distances and parents.

## 2. Template mapping

| Step | Hook | What it does (MOSP_ESCHER's hsospUpdate) |
|---|---|---|
| 0 | `normalize` (stage `hyper_sssp.normalize`) | checks the batch's weights (at most 2^28) and the result's version before the commit |
| commit | `hypergraph.apply` | the ids, the incidence edits, and *translate* inside the commit: the order-free line-graph pair delta computed from the incidence changes (`incidence_delta`, ADR 0086), applied to the slack CSR line graph |
| 1 | `identify_affected` | the roots: the endpoint b of every deleted pair (a, b) that was a tree edge, every new, recreated or dead id, every hyperedge that lost s while its parent was -1; their subtrees of the tree before the batch are invalidated (pointer jumping on cuda) |
| 2a | `seed` | every invalidated hyperedge pulls its best (distance, parent) over its neighbours (and w(h) if it holds s); both directions of every inserted pair are relaxed |
| 2b | `loop` | the improved hyperedges push to their neighbours (decrease-only; on cuda a packed 64-bit atomicMin of (distance, parent); a priority queue on sequential) until nothing decreases |
| 2c | `recompute` | the work budget's fallback (always entered: 0 ms without a fallback) |
| 3 | `finalize` | the counters |

The four challenges: the affected set is the invalidated subtrees (a deletion only lengthens the
paths through its tree edge, an insertion only shortens); the propagation is decrease-only after
the invalidation, so its order does not change the fixed point; correctness under parallelism
comes from the packed words (distance and parent change together) and the lowest-id tie rule;
the data structure is the slack CSR line graph the container keeps (ADRs 0086, 0089).

On sequential the hooks run one by one (Tier A; the loop drains its priority queue in one call and
stops at the work budget). On cuda, `engine::fused` runs MOSP_ESCHER's hsospUpdate behind
`enact_fused`, which opens the same stages itself (Tier B); `engine::operators` runs the hooks
on sssp's CUDA operators (section 4).

## 3. API

```cpp
#include <dyng/hyper_sssp.hpp>

dyng::hypergraph<std::int32_t, std::int32_t> h = /* slack_csr storage, weighted */;
auto r = dyng::hyper_sssp::compute(res, h, /*source vertex=*/0);
dyng::hyper_sssp::stats s = dyng::hyper_sssp::update(res, h, batch.view(), r);
std::int64_t d = r.distance_to(42);
```

```python
import dyng
seq = dyng.Resources("sequential")
hg = dyng.io.read_hypergraph("h.hg", weighted=True, storage="slack_csr", resources=seq)
sp = dyng.hyper_sssp.compute(hg, source=0)
st = dyng.hyper_sssp.update(hg, dyng.HyperedgeBatch(insert_incidences=([3], [7])), sp)
sp.hyperedge_distances, sp.hyperedge_parents, sp.distance_to(42), sp.vertex_distances()
```

`result` holds `hyperedge_distances()` (int64), `hyperedge_parents()` (int32: slack_csr ids are
below 2^31), `distance_to()` and `vertex_distances()` (read from the hypergraph the result matches;
`stale_result_error` once that hypergraph changed or is gone), `get_options()` / `set_options()`,
`graph_version()`, `space()` and `clone()`. `dyng::update(res, h, batch, r1, r2, ...)` updates
several results (of different sources) after one commit; triad_count refuses `slack_csr` storage
in 0.4, so the two algorithms do not share a hypergraph.

**Options** (every field is a tunable: it changes how an update runs, never its result):

| Field | Default | Meaning |
|---|---|---|
| `work_budget` | 1.0 | before each push round (cuda) or expansion (sequential), an update whose relaxations exceed `work_budget` x max(1, the line graph's live entries) takes the `on_budget` action; the work before the first check (the invalidation, the seed's pulls and offers) is never refused, so 0 acts at the first push round or expansion (an update whose work all lies in its seed completes even at 0), +infinity never (MOSP_ESCHER's `workBudget`) |
| `max_iterations` | 4096 | push rounds before the `on_budget` action (cuda; MOSP_ESCHER's `maxIterations`); also the floor of the recompute's round cap |
| `on_budget` | `budget_action::recompute` | `recompute` falls back to a recompute (exact); `error` throws `convergence_error` after the commit (the result is poisoned) |
| `cuda_engine` | `engine::automatic` | `automatic` and `fused`: MOSP_ESCHER's kernels; `operators`: the shared sssp engine (the same results) |
| `validate_inputs` | true | the weights checked against [1, 2^28] (compute: every live hyperedge; update: the insertions, before the commit) |

**Tuning the work budget.** `stats::relaxations` is the work the budget counts (the neighbours the
pulls and pushes read). Run representative batches with `work_budget = +infinity`, time them
against a `compute()`, and set the budget near the relaxations (divided by the live entries) of
the batch size where the update stops being faster; MOSP_ESCHER's default 1.0 means "about one
pass over the line graph" ({doc}`../tutorials/shortest_hyperpaths`, section 7).

**Stats** (each field's determinism is stated in the header): the inherited `update_stats`
fields (`affected` = `invalidated`; `iterations`, the push rounds or expansions;
`frontier_visits`; `fallback_used`; `converged`, always true; `engine_used`), `batch` (the
container's `hyper_apply_summary`: the ids, the applied and skipped edits), `line_graph_insertions`
and `line_graph_deletions` (the pairs the batch added and removed), `invalidated`, `relaxations`
and `fallback_reason` (`none`, `work_budget`, `iterations`, `distance_width`,
`invalidation_depth`).

## 4. Backends, engines, requirements and determinism

| Backend | Engine | Notes |
|---|---|---|
| sequential | `engine::operators` | MOSP_ESCHER's host emulation (emulateSospRecompute, emulateSospUpdate) on the host line graph; the budget is checked before each expansion; no iteration, width or depth limit |
| cuda | `engine::fused` (the default: `engine::automatic` selects it; section 5) | MOSP_ESCHER's kernels as they are (transplanted with `parity/escher_port/transplant.py`, ADR 0091): a 32-bit packed distance, at most 64 pointer-jumping rounds |
| cuda | `engine::operators` | the shared sssp engine (O12; {doc}`ADR 0095 <../adr/0095-hyper-sssp-on-the-shared-sssp-engine>`): sssp's CUDA operators over the line graph with compile-time policies (slack rows, node weights `w[col]` pushing and `w[v]` pulling, one symmetric view, base offers at the source's hyperedges, a work-budget counter), MOSP_ESCHER's budget triggers (its relaxations counted on row lengths as MOSP_ESCHER counts them, but re-queueing a hyperedge only when its distance drops: on equal-distance ties it counts fewer relaxations and push rounds than `engine::fused`, so near the budget its fallback decisions can differ), the static solve on sssp's near-far loop; the same results as `engine::fused`, bit for bit |
| openmp | none | `not_supported_error` naming sequential and cuda and the fix (`dyng::resources::sequential()`, Python `dyng.Resources("sequential")`): no OpenMP backend in 0.4 (HD14, decided by the author on 2026-10-28, jointly with M10's AD19). In Python, pass explicit resources: the CPU wheel's default resources are OpenMP's |

Requirements, checked by `compute()` and before the commit of `update()`: `hypergraph<int32_t,
int32_t>` (other types fail to compile), weights in [1, 2^28], `hypergraph_storage::slack_csr`
with its line graph (`line_graph_policy::slack_csr` or `automatic`) and `store_v2h`. Any
`id_reuse` order (MOSP_ESCHER's ids need `insert_first`; the default is `erase_first`, O16).
On cuda the hypergraph is built with (or moved to, `to_backend()`) the CUDA resources, and the
result's arrays are device memory.

**Determinism.** Distances, parents, `invalidated` and the line-graph counts are bit-exact across
backends, engines and runs. `fallback_used` and `fallback_reason` are **deterministic on
sequential and schedule-dependent on cuda near the budget**: the relaxations of a push round
depend on the thread schedule, so a batch within a few percent of its budget may fall back in one
run and not in another, with identical results (HD7; the per-field sentence of
`<dyng/core/stats.hpp>`). `relaxations`, `iterations` and `frontier_visits` are
schedule-dependent on cuda.

## 5. Complexity, memory and performance

An update costs O(the batch's incidence changes) for the commit's pair delta, O(the invalidated
subtrees and their neighbourhoods) for the pulls, and the pushes until nothing decreases: about
the line-graph rows of the hyperedges whose distance changes. The static solve visits every live
line-graph entry at least once. The line graph of a hypergraph can be much larger than its
incidence: coauth-DBLP's line graph has 250,891,568 entries.

**The stages and MOSP_ESCHER's timers** (`parity/timed_regions/hyper_sssp.toml`):

| MOSP_ESCHER's timer (`hsospBench` CSV) | dynG's stages |
|---|---|
| `t_delta_ms` + `t_csr_apply_ms` (maintain: the host incidence, the GPU line-graph delta, the CSR apply or its rebuild) | `hyper_sssp.normalize` + `hypergraph.apply` (Step 0, the ids, the edits, the pair delta and the line-graph apply) |
| `t_sosp_update_ms` (hsospUpdate, its fallback included) | `hyper_sssp.identify_affected`, `.seed`, `.loop`, `.recompute`, `.finalize` |
| `t_static_ms` (hsospRecompute into a second state) | `hyper_sssp.compute` into a second result |
| `t_delta_ms` + `t_csr_apply_ms` + `t_sosp_update_ms` (dynamic_comparable) | the sum of the two rows above |
| `t_escher_ms` (the CBST maintenance) | none: `slack_csr` storage keeps no CBST (section 7) |
| `[e2e]` load + 3 batches | the driver's end-to-end time (`dyng-compat-hsosp bench`) |

The engines synchronize the stream once per push round (`engine::fused` also once per
pointer-jumping round). The compat tool `dyng-compat-hsosp bench` reproduces hsospBench's rows (44
CSV columns) and `parity/hsosp_perf.py run` compares them with MOSP_ESCHER under the gate protocol
of ADR 0080.

**The gates against MOSP_ESCHER@4b86159** (the as-is engine, the default; the reading of
record on the final 0.4 library, M11b: the unpatched code re-measured in the same session, RTX
A5000 at base clocks, 16 paper configurations x 21 alternating rounds; ratios dynG /
MOSP_ESCHER; `parity/results/M11b.md` section 4, M11a's reading in `parity/results/M11a.md`
section 11.6). Every gated row is within its bar (271 of 271):

| Row (per batch b1-b3) | Bar | coauth-DBLP | Geology |
|---|---|---|---|
| maintain | 1.05 | 0.513-0.998 | 0.478-0.877 |
| sosp_update | 1.05 where MOSP_ESCHER's median >= 10 ms, else 1.10 | 0.998-1.024 | 0.992-1.038 |
| static | 1.05 | 0.980-1.005 | 0.988-1.034 |
| dynamic_comparable | 1.05 | 0.553-1.000 | 0.554-0.888 |
| dynamic (MOSP_ESCHER's total, with its CBST maintenance) | 1.05 | 0.150-0.372 | 0.194-0.460 |
| load_plus_dynamic (end to end) | 1.10 | 0.645-0.716 | 0.682-0.806 |
| memory: the nsys peak of live device allocations (P50 hyperedge and vertex, Geology hyperedge) | 1.05 | 0.536, 0.557 | 0.477 |
| memory: the graph scope (MOSP_ESCHER's graph_mb) after the build and each batch | 1.05 | 1.0000 | 1.0000 |

**The default CUDA engine (O12; HD13, decided by the author on 2026-10-28; ADR 0095).** The
shared sssp engine would become the default only if it gave the same results as the as-is
engine and MOSP_ESCHER and were within every gate against MOSP_ESCHER@4b86159. It gives the same
results (every batch of the golden set, in the three parity modes) and is within the memory
gate (the peak of live device allocations 0.49-0.56 of MOSP_ESCHER's), but in the full gate
reading (16 paper configurations, 21 rounds, base clocks) four `sosp_update` rows of
coauth-DBLP's hyperedge batches are above their 1.05 bar:

| Configuration, batch | MOSP_ESCHER's `t_sosp_update_ms` | `engine::fused` / MOSP_ESCHER (M11a; final 0.4 library) | `engine::operators` / MOSP_ESCHER |
|---|---:|---:|---:|
| dblp_hyperedge_100000 b2 | 10.1 ms | 1.024; 1.024 | 1.051 |
| dblp_hyperedge_200000 b1 | 22.2 ms | 1.002; 1.001 | 1.056 |
| dblp_hyperedge_200000 b2 | 23.5 ms | 1.001; 0.998 | 1.061 |
| dblp_hyperedge_200000 b3 | 21.7 ms | 1.004; 1.004 | 1.059 |

The shared engine's update loses 1.2-1.5 ms there in two places: sssp's pointer jumping runs its
ceil(log2 n) + 1 rounds where MOSP_ESCHER's stops as soon as nothing changes, and the seed pulls
the invalidated nodes and seeds the inserted pairs in two launches. Everywhere else it reads the
same or faster (Geology's updates 0.78-0.99 of MOSP_ESCHER's), and its static solve, sssp's
near-far loop with MOSP's defaultDelta, reads 0.69-0.74 of MOSP_ESCHER's recompute (the as-is
engine's: 0.98-1.04). So `engine::automatic` keeps `engine::fused`; select the shared engine with
`options::cuda_engine = engine::operators` (for example where the static recompute or Geology-like
updates dominate). The numbers: `parity/results/M11b.md` section 2.

## 6. Limitations

- Integer weights in [1, 2^28] (MOSP_ESCHER's limit: every path cost stays below 2^59; the
  container keeps [1, 2^31 - 1], and `validate_inputs = false` accepts heavier weights, which then
  fall back more often on cuda: `fallback_reason::distance_width`); no unweighted hypergraphs.
- int32 vertex ids and hyperedge ids below 2^31 (the parents are int32); the line graph is held
  in memory (on the device on cuda), which bounds the hypergraphs that fit.
- `slack_csr` storage only: the escher storage keeps no slack line graph and refuses incidence
  edits.
- No OpenMP backend in 0.4 (HD14).
- No composition with `triad_count` in 0.4 (it refuses `slack_csr` storage).
- The public parents differ from MOSP_ESCHER's where its parent is a virtual row (section 7).

## 7. Differences from the paper

MOSP_ESCHER's README lists how its fixed implementation differs from the paper (README lines
57-91; in its words):

1. **Unification input.** "The paper derives the line-graph changes on the GPU 'using the same v2h
   and h2v lookups' as ESCHER. Here the GPU keeps a separate slack-row mirror of the incidence
   (`DeviceIncidence`) and derives the net delta from it; the ESCHER h2v / v2h / h2h CBSTs are
   maintained in every batch (and checked by the tests) but nothing on the shortest-path path
   reads them, as in the original code."
2. **Update algorithm.** "The pull-based update with an iteration cap described in the paper does
   not converge after deletions (stale distances rise by at least 1 per iteration). The update
   here first invalidates the pre-batch subtrees of deleted tree edges and of new or dead nodes,
   then pulls and pushes; it is exact without a cap. The budget fallback to the recompute is kept
   and counted in the update time."
3. **Ties.** "Parents are canonical (lowest id among the tight neighbours) in the update, the
   recompute and the oracle."
4. **Weights.** "Integer hyperedge weights in [1, 2^28] (other weights are rejected; the paper's
   weights are positive reals, and the cap keeps every path cost far from 64-bit overflow). Real
   datasets get U[1,100] weights."
5. **Baseline.** "The static baseline is the GPU recompute with the same kernels (as in the
   paper). Those kernels became about 4x faster (P3 in CHANGES.md), which lowers the reported
   speedups."
6. **Scale.** "The CBST payloads are `int`-indexed, and `constructFromRows` caps each at 2.0 x 10^9
   values, so Orkut, AMiner and MAG [...] would be rejected, independently of GPU memory. Only DBLP
   and Geology were measured."
7. **Results.** "On an RTX A5000 the dynamic time per batch (maintenance + unification + CSR apply
   + update, the paper's metric) is below the recompute only for DBLP vertex batches of 25K
   (1.28x); elsewhere it is recompute/dynamic = 0.08-0.85, against 1.3-12.1x reported for DBLP in
   the paper. The update stage alone is 1.4-25x faster than the recompute; ESCHER maintenance and
   unification dominate."

So the paper's speedups do not reproduce on the fixed code: **0.08-1.28 against the paper's
1.3-12.1**, with the update stage alone 1.4-25x faster than the recompute.

dynG's own differences:

- **No CBST.** `slack_csr` storage keeps the incidence and the slack line graph that the
  shortest-path path reads, and no ESCHER h2v / v2h / h2h CBST (MOSP_ESCHER maintains them in every
  batch, item 1, though nothing on the shortest-path path reads them). The paper's dynamic total
  (with the CBST maintenance, `t_escher_ms`) is therefore not comparable stage by stage: dynG's
  `dynamic` row reads 0.15-0.46 of MOSP_ESCHER's, and the gates compare the stages both sides have
  (section 5).
- **No virtual hyperedges.** MOSP_ESCHER adds a source row {s} and a target row {t} of weight 0 and
  runs from the source row; dynG's public API takes a source vertex and has no virtual rows: the
  hyperedges holding s start at their own weight with parent -1. Where MOSP_ESCHER's parent is its
  source row, dynG's is -1; where its parent is its virtual target row (a zero-weight row through
  the target vertex), dynG's is the canonical parent among the real hyperedges. The distances of
  the real hyperedges are the same; the parity harness's `by_id_public` mode counts both kinds of
  parent difference and checks dynG's parents against the oracle (HD2). The harness runs an internal virtual-endpoint mode that reproduces MOSP_ESCHER by id
  (ADR 0091; no public name).
- **Vertex distances** follow HD11 (section 1), not PLAN 5.3's "+ w(h)".
- **The budget** is checked by the host engine too (before each expansion), and
  `budget_action::error` throws instead of falling back (ADR 0092).
- **The gate baseline** is MOSP_ESCHER at `4b86159` re-measured in the same session as dynG, not
  the tables of its `results/README.md` (measured at `07bbceb` with CUDA 12.9, which dynG keeps as
  a paper reference only; HD4).

## 8. Paper vs fixed code

*The plain wording of HD19 (decided by the author on 2026-10-28, jointly with M10's AD17, as
GOVERNANCE D10 was for the 0.3 pages).* The H-SOSP paper was measured on the research code as it
was then (tag `baseline-2026-09` of MOSP_ESCHER, `80cee62`). dynG ports the **fixed** code
(`fix/correctness-perf`, MOSP_ESCHER@4b86159), whose `README.md`, `CHANGES.md` and
`results/README.md` record what changed. dynG is byte-identical to the fixed code, not to the
paper's code. In their words:

| Item | MOSP_ESCHER's record (`CHANGES.md`) | dynG |
|---|---|---|
| **S1** count-to-infinity and the every-batch fallback | "The update re-evaluated seed nodes by pull with no invalidation: after a deletion, nodes that lost their tree path kept stale finite distances that rose by at least 1 per iteration and never converged, so every batch with a disconnection ran 512 capped iterations and then a full recompute (all baseline DBLP batches did; the update was never cheaper than the recompute it is compared with)." Now: roots, pointer-jumping invalidation, a warp-cooperative pull, a push with a packed 64-bit atomicMin, a work budget and an iteration budget (4,096) that fall back to the recompute. "DBLP, 50K hyperedge batches: SOSP stage 5.1-5.8 ms [...] against 473-482 ms for the original." | the fixed update, exact (section 2), with MOSP_ESCHER's budget triggers (HD8) |
| **S3** non-positive weights | "two adjacent zero-weight hyperedges cut off from the source keep a stale finite distance [...] so the update converged to wrong distances without a fallback [...] weights must be >= 1 [...] Weights above 2^28 are rejected as well: with no upper bound a weight near 2^63 wrapped the device distance sums [...] and the following recompute never finished." | weights in [1, 2^28], checked before anything changes (`validate_inputs`) |
| **S4** "targeted" change placement | "Device parents are 0-based node indices but `generateBatch` reads them as 1-based hyperedge ids, so the targeted placement [...] deleted the hyperedge before each SOSP-tree parent." | 0-based hyperedge ids everywhere: the public parents are hyperedge ids, so the mismatch cannot recur |
| **S5** measurement reporting | "On a fallback the `iters` column held only the recompute's rounds (the 512 capped iterations vanished); the CSR overflow rebuild ran after the CSR timer was read, so it was never counted in the dynamic time; `dev_mem_mb` counted other processes' memory without saying so." | `stats` reports the push rounds, `fallback_used` and `fallback_reason` beside each other; the maintain region holds any rebuild; the memory gate reads each process's own allocations (nsys) |
| **P3** warp-cooperative pull kernels | "The recompute (static baseline, initial SSSP and the update's fallback) ran one thread per candidate over its whole row [...] It now runs one warp per candidate [...] DBLP static recompute: 332-334 ms -> 81-83 ms (4.1x). This speeds up the baseline the dynamic time is compared with [...] so the reported speedup becomes smaller." | the ported recompute (`engine::fused`); the shared engine's static solve reads 0.69-0.74 of it |

**The published speedups do not reproduce.** MOSP_ESCHER's README: on an RTX A5000 the dynamic time
per batch is below the recompute only for DBLP vertex batches of 25K (1.28x); elsewhere
recompute / dynamic is **0.08-0.85, against 1.3-12.1x reported for DBLP in the paper** (its
CHANGES.md: P3 moved the ratio "from 0.01-0.16 to 0.17-1.28"). The update stage alone is 1.4-25x
faster than the recompute; ESCHER maintenance and unification dominate. dynG, without the CBST
maintenance, reads 0.15-0.46 of MOSP_ESCHER's dynamic total (section 5); the paper's
end-to-end claim is not dynG's to make. The other fixes (S2 the verification of `hsospBench`, the
operations on the virtual hyperedges, the empty generator pool, the ESCHER core's E1-E9, P1, P2,
P4, P5) are summarized in {doc}`../history/mosp_escher`.

## 9. Mapping from the original code

| MOSP_ESCHER@4b86159 | dynG |
|---|---|
| `hsospUpdate` | `hyper_sssp::update` |
| `hsospRecompute` | `hyper_sssp::compute` |
| `UpdateConfig` (`workBudget`, `maxIterations`, `blockSize`) | `hyper_sssp::options` (`work_budget`, `max_iterations`, `on_budget`; the block size is internal) |
| `DeviceIncidence` (the slack-row incidence mirror) | `hypergraph_storage::slack_csr` (the container's storage, ADRs 0086, 0089) |
| `DeviceH2H` (the slack CSR line graph and its delta) | `line_graph_policy::slack_csr` |
| `hsospDelta` (`deriveDelta`, `applyDeltaToDevice`) | `incidence_delta` (the commit's order-free pair delta, `cpp/src/hypergraph/line_graph/incidence_delta.cu`) and the slack CSR apply |
| `emulateSospRecompute`, `emulateSospUpdate` (`HostHypergraph.cpp`) | the sequential backend (`cpp/src/algorithms/hyper_sssp/sequential.cpp`) |
| `BatchTimes` (the CSV's timers) | `hyper_sssp::stats` and the profiler's stages (section 5) |
| `HsospState` (the packed words, the invalidation and frontier arrays), `hsosp/src/hsospDevice.cu` | `state.cuh`, `kernels.cuh`, `engine.cu` (transplanted, `engine::fused`), `cuda.cu` (the driver); `operators.cuh`, `operators.cu` over `cpp/src/algorithms/sssp/operators.cuh` and `cpp/src/util/slack_rows.cuh` (`engine::operators`) |
| the virtual source and target rows | none in the public API; `virtual_endpoints.hpp` (the parity harness's internal mode) |
| `HostHypergraph::applyBatch`, `HgBatch` | `hypergraph::apply`, `hyperedge_batch` (`delete_incidence`, `insert_incidence`) |
| `HypergraphOracle.cpp` (`rebuildLineGraph`, `lineGraphDelta`, `referenceDistances`, `checkSosp`) | `dyng::testing::hyper_sssp_reference`, `check_hyper_sssp_tree`, `line_graph_delta` (ADR 0082); Python `dyng.testing.hyper_sssp_reference`, `check_hyper_sssp_tree` |
| `hsospBench`, its loader (`--hg`, `--maxcard`, `--seed`) | `dyng-compat-hsosp to-dgb`, `replay`, `bench` (the parity driver); `dyng.io.read_hypergraph(options="hsosp")`; `dyng hyper_sssp --read-options hsosp` |
| `--verify all` | the oracle `compute` of the conformance kit; `dyng hyper_sssp update --compare-recompute` |

## 10. How to cite

`dyng::citation("hyper_sssp")` (C++) and `dyng.citation("hyper_sssp")` (Python) return the
library's BibTeX entry and the entry `hsosp2026` (H-SOSP, IA3 2026) in `docs/references.bib`;
{doc}`../citing`.
