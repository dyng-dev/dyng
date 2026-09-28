# sssp: dynamic single-source shortest paths

Maturity: **experimental** (sequential and OpenMP backends since M1a, the CUDA backend with the
fused engine since M1b; the Python binding arrives in M5 and the operators engine in 0.2).
Header: `<dyng/sssp.hpp>`. Oracle: `compute`. Determinism: `bitwise`. Parity: byte-identical to
MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2 ([M1b certificate](../../parity/results/M1b.md)).

## 1. Problem

Input: a directed (or symmetric) graph with K integer weight columns (weights in
[1, 2^31 - 1]), a source vertex and an objective column. Output: the distance of every vertex
from the source (`infinite_distance<int64_t>()` if unreachable) and the **canonical**
shortest-path tree: the parent of v is the lowest-id in-neighbour u with
dist[u] + w(u,v) == dist[v]; -1 for the source and unreachable vertices.

Update model: a batch of insertions (upserts: an insertion of an existing edge changes its
weights), deletions and weight changes is applied to the graph under its `batch_semantics`
(`graph_properties::mosp_compatible()` reproduces MOSP's `applyChangeBatch()`, parallel edges and
self-loops included), and the tree is brought up to date. Postcondition: the distances equal
`compute()`'s on the new graph and the tree is a shortest-path tree of it; from a canonical input
tree (every result of `compute()`, or `from_arrays(..., canonicalize = true)`) the result equals
`compute()` bit for bit.

**Tie rule** (the rule of MOSP-OpenMP's `sospUpdateCpu` and MOSP-CUDA's `sospUpdateGpu`, on every
backend): a parent changes only when its vertex is re-evaluated. The vertices of the invalidated
subtrees and the insertion heads take their best (distance, lowest parent id) over all
in-neighbours; the out-neighbours x of a vertex a whose distance decreased adopt
(d(a) + w(a,x), a) if that pair is smaller than their own; all other vertices keep their parents.
So a tree adopted with `canonicalize = false` whose tie parents are not the lowest ids (MOSP's
dataset trees are such trees) keeps those parents where the batch does not reach, exactly as the
original `mosp` driver does; the distances are still those of `compute()`. In the distance-only
mode (`stats::packed_parents == false`) every parent is recovered with the lowest-id rule after
the search, so the result equals `compute()` on every input. MOSP-OpenMP's own sequential update
(`sequentialSOSPUpdate`) re-scans all in-neighbours of every candidate and can pick a lower-id
tight parent from a vertex that did not improve; dynG's sequential backend follows the rule
above instead (ADR 0006), so that every backend returns the same tree.

## 2. Template mapping

```
normalize -> translate -> prepare -> [before_apply] -> commit ->
identify_affected -> seed -> { FP: loop until is_converged } -> finalize
```

| Hook (profiler stage) | sssp |
|---|---|
| `sssp.prepare` (G_t) | largest weight (old graph and batch) and weight sum of the objective; the default near-far width; batch weights checked |
| `sssp.commit` | `graph::apply` with the per-edge classification (`apply_delta`): deletions and per-objective weight increases |
| `sssp.identify_affected` | roots = heads v of deleted or weight-increased edges (u,v) with parent[v] == u; their subtrees are invalidated |
| `sssp.seed` | invalidated vertices and insertion heads pull their best (distance, lowest id) over their in-neighbours |
| `sssp.loop` | propagate decreases until no distance changes (near-far worklist on OpenMP; rounds of the same push rule on the sequential backend) |
| `sssp.finalize` | unpack distances and parents (the distance-only mode recovers every parent); count `affected` |

`compute()` is the static enactor: `sssp.reset` -> `sssp.seed` (the source) -> `sssp.loop` ->
`sssp.finalize`, inside `sssp.compute`. `update()` runs inside `sssp.update`.

On the CUDA backend the four hooks from `identify_affected` to `finalize` are one fused engine
(Tier B, PLAN 4.5.4): MOSP-CUDA's persistent cooperative kernel, one launch per objective, profiler
stage `sssp.enact_fused`. The framework still owns everything around it: `sssp.prepare` on G_t,
the commit (`graph.apply` on the host, then `graph.upload`: the updated graph is uploaded and its
in-edges are built on the device, once for all results), `sssp.workspace` and `sssp.changes` (the
objective's change list, built on the host from the commit's classification and uploaded).

The four Chapter 3 challenges: (i) affected set: subtree invalidation below changed tree edges
(this replaces the thesis' "best current in-neighbour" rule, which counts to infinity); (ii)
propagation scope: only vertices whose distance decreased are expanded; distances only
decrease, so there is no iteration cap; (iii) correctness under parallelism: an atomic minimum on
a packed (distance, parent) word gives canonical ties without locks; (iv) data structure: the
compact CSR with stored in-edges and objective-major weight columns.

## 3. API

```cpp
auto res = dyng::resources::openmp(28);              // or resources::sequential(), resources::cuda()
auto g = dyng::graph<std::int32_t, std::int32_t, std::int32_t>::from_csr(
    res, csr.view(), dyng::graph_properties::mosp_compatible());
auto tree = dyng::sssp::compute(res, g, /*source=*/0);
dyng::sssp::stats st = dyng::sssp::update(res, g, batch.view(), tree);
// several objectives on one graph: the batch is applied once
auto [s0, s1] = dyng::update(res, g, batch.view(), tree0, tree1);
```

On a GPU the same calls take CUDA resources and a graph that belongs to them:

```cpp
auto gpu = dyng::resources::cuda(/*device=*/0);      // work ordered on the per-thread stream
gpu.warm_up();                                        // load the kernels before timed work
auto dg = g.clone(gpu);                               // or build the graph with gpu directly
auto dtree = dyng::sssp::compute(gpu, dg, /*source=*/0);   // arrays in device memory
dyng::sssp::update(gpu, dg, batch.view(), dtree);     // the fused engine
std::vector<std::int64_t> dist = dyng::to_vector(gpu, dtree.distances());
```

| Option | Default | Meaning |
|---|---|---|
| `delta` | 0 (automatic) | near-far bucket width: max(1, 32 * average weight / average out-degree) of the graph before the batch |
| `objective` | 0 | the weight column (fixed at compute) |
| `cuda_engine` | automatic | CUDA engine: `automatic` and `fused` run the persistent cooperative kernel and throw `not_supported_error` on a device without cooperative launch (the operators engine that would be the fallback arrives in 0.2); `operators` throws in 0.1; ignored on the CPU |
| `validate_inputs` | true | O(n) checks of trees adopted with `result::from_arrays()` |

Python: planned (M5).

## 4. Backends, engines and determinism

| Backend | Engine | Origin |
|---|---|---|
| sequential | hook-by-hook reference (`engine::operators`) | `sequentialSOSPUpdate` adapted (Step 2 with `sospUpdateCpu`'s push rule) |
| openmp | the ported paper engine (`engine::fused`) | `sospUpdateCpu` / `sospFromScratchCpu` ported straight |
| cuda | the ported paper engine (`engine::fused`): one persistent cooperative kernel | `sospUpdateGpu` / `sospFromScratchGpu` ported verbatim (`cuda.cu`, `fused.cuh`) |

**Engines on CUDA** (`options::cuda_engine`, PLAN 4.5.4; ADR 0017). The choice is checked before
the graph or the result is changed, so a refused call leaves both as they were:

| `cuda_engine` | Device with cooperative launch | Device without it |
|---|---|---|
| `automatic` (default) | the fused engine | `not_supported_error`: there is no other CUDA engine in this release; the message names `resources::openmp()` / `resources::sequential()` |
| `fused` | the fused engine: one persistent cooperative kernel per objective, grid = co-resident blocks of the instantiation (occupancy API), as MOSP-CUDA's `SospWorkspace` sizes it | `not_supported_error` (same message, without the `automatic` note) |
| `operators` | `not_supported_error` (the multi-kernel engine arrives in 0.2, decision O24) | `not_supported_error` |

`stats::engine_used` reports the engine that ran: `fused` on cuda and openmp (the ported paper
engines), `operators` on the sequential reference. The no-cooperative-launch path is tested on the
real device with a forced capability flag (`sssp_cuda_test.cpp`). Every GPU since Pascal
supports cooperative launch; MPS or MIG limits can still make the launch fail, which surfaces as
`cuda_error`.

**Determinism** (`determinism::bitwise`). Distances and parents are bit-identical across runs,
thread counts, backends and edge offset types, and byte-identical to both originals on the golden
corpus. All backends return identical trees, from canonical and non-canonical input trees alike
(the randomized test `NonCanonicalInputTreesAgreeOnEveryBackend` perturbs tie parents; the CUDA test
executable runs it with cuda next to the host backends); `invalidated` and `affected` are
deterministic, `iterations`, `epochs` and `pushes` depend on the schedule. (MOSP-CUDA@e220ee2
reads its `invalidated` count from the candidate-list counter while other threads already append
the insertion heads to it, a data race that its plain runs happen not to expose; the ported
kernel sums per-thread counts instead, ADR 0017 item 1, and the CUDA suite runs under
`compute-sanitizer --tool synccheck` in `ci/gpu_local.sh`.) One counter differs by
origin: right at the packing limit MOSP-CUDA packs (distance, parent) when (n - 1) * max weight
fits next to the parent bits, MOSP-OpenMP only when one more edge fits too, so
`stats::packed_parents` can differ between cuda and the host backends there (the packing-boundary
cases n = 2^17 - 1; the trees are equal).

**The CUDA backend.** A graph belongs to the backend of the resources that built it (PLAN 4.6
rule 5): sssp on CUDA resources needs a graph built with them (or `g.clone(cuda_res)`), and a
host graph with CUDA resources, or the reverse, is an `invalid_argument_error` instead of a
silent copy. Such a graph keeps its CSR in host memory in this release (a batch is applied on the
host, as MOSP-CUDA's `applyChangeBatch` does) and a device copy of the current state (out- and
in-edges, one weight column per objective), uploaded on first use and again inside the commit of
every update, as MOSP-CUDA uploads the updated graph once per batch. Results keep their arrays in
device memory (`r.space() == memory_space::device`; copy them with `dyng::to_vector(res,
r.distances())`), `r.clone(res)` moves a result between host and device, and
`result::from_arrays()` accepts host or device arrays (checked on the host, then uploaded).
`update()` synchronizes the stream once per result (the kernel's control block is read) and once
per batch inside the commit, where the new graph state is uploaded (`graph.upload`); `compute()`
synchronizes once, plus once for the upload if the graph state is not resident yet (ADR 0017 item
7). The
scratch memory is the device workspace of the handle's pool (ADR 0015), the kernel's grid is the
co-resident block count of each kernel instantiation (occupancy API), and `resources::warm_up()`
loads the kernels ahead of timed work. Engine selection, placement and the device graph are
recorded in ADR 0017.

## 5. Performance notes

The paper-timed region of the original, `obj<k>/sosp_update_compute` (sospUpdateCpu), maps to
`sssp.identify_affected + sssp.seed + sssp.loop + sssp.finalize` (`parity/timed_regions/sssp.toml`).

**Scratch memory (ADR 0015).** The engines' workspace (about 38 bytes per vertex: packed words,
stamps, flags and six frontier lists) belongs to the `resources` handle, not to the result.
`compute()`, `from_arrays()`, `clone()` and each result's part of `update()` lease it from the
handle and size it once (profiler stage `sssp.workspace`), so results run through one handle
share it: the K objectives of `dyng::update_each()` use one workspace one after the other, as
MOSP's `mospUpdate()` shares its `SospWorkspace`, and a steady-state update allocates no scratch
memory (on OpenMP up to the per-thread lists, below). Calls that run concurrently on copies of one handle lease distinct workspaces.
`resources::workspace_bytes()` reports the cached bytes, `resources::release_workspaces()` frees
them. As in the original, objective 0 of an update first touches the frontier-list pages it uses.

**The graph around it.** A graph built with `from_csr()` / `from_edges()` does not transpose
itself: the in-edges are built on first use, which for an update is inside the commit, for the
updated graph only (MOSP builds its reverse graph once, in "prepare"). `from_csr()` takes a CSR
that already has the requested form (any CSR under `mosp_compatible()`) as it is, and its rvalue
overload takes over the arrays without a copy. On the OpenMP backend the batch's CSR assembly,
the checks of `from_csr()` and the validation of imported trees (`validate_inputs`) run in
parallel, with results and messages identical to the sequential ones.

**CUDA.** MOSP-CUDA's timer `obj<k>/sosp_update_gpu` is the host time of `sospUpdateGpu()` up
to its final synchronization; dynG's `sssp.enact_fused` has the same scope and also records its
device time with CUDA events (`profiler_options::cuda_events`). The kernel is the original's
code: its int32 instantiation uses 59 registers and 616 bytes of parameters, like the original
(`cuobjdump --dump-resource-usage`), so the co-resident grid that the occupancy API gives the
cooperative launch is the same. The only addition to the kernel is the `affected` count in the
unpack pass of an update, which compares each new pair with the old one and writes only the pairs
that changed (no more bytes than the original's unconditional write).

**OpenMP barriers.** On a small batch the near-far loop runs about a thousand rounds per
objective with a few hundred vertices each, so its time is mostly barrier latency. dynG's rounds
pass three barriers instead of the original's six (the work-sharing loops are `nowait` and the
two per-thread lists of a round are gathered together, `list_gather::gather_pair()`), with the
same lists and outputs: on the 10K local batches the OpenMP update is 0.63-0.89x of
MOSP-OpenMP's. The per-thread lists are kept in the workspace on their own cache lines and keep
their capacity, so a list allocates only when its thread takes a larger share of a round than it
ever took before (the dynamic schedule decides the shares; the growth is geometric, like a
vector's); MOSP creates them in every parallel region.

**Clock state (CUDA).** A short update that follows a long host-only phase runs its kernels
while the GPU is still in a low performance state (on the RTX A5000: P2, SM clock 1.69 GHz and
memory 7.6 GHz instead of P0's 1.92 / 8.0 GHz). Any code sees this; it is why
MOSP-CUDA, whose half-second "upload" stage precedes its first kernel, reads faster on road_usa's
local batch than dynG, which uploads earlier (ADR 0018). Call `resources::warm_up()` and time
repeated updates, or lock the GPU clocks, when a single short update is measured.

**Measured against the originals** (`parity/results/M1b.md` section 13, the re-measurement on
the code fixed after the M1b review; sections 8-9 have the step-4 records and the locked-clock
tables: medians of 21 alternating runs of `dyng-compat-mosp` and the unpatched original, parity
presets, RTX A5000 with CUDA 13.1 / 28 OpenMP threads pinned, with a contamination monitor; 50K
safe, 50K unsafe and 10K local batches on each graph). Ratio dynG / original (below 1 is faster),
the range over the batches and the objectives:

| Graph | CUDA: SOSP region per objective (gate 1.05x; 1.10x under 10 ms) | CUDA: kernel at locked clocks | CUDA: apply / end to end (gate 1.10x) | OpenMP: SOSP region per objective (gate 1.05x; 1.10x under 10 ms) | OpenMP: apply / end to end (gate 1.10x) |
|---|---|---|---|---|---|
| roadNet-PA | 0.98-1.00x | 1.00x | 0.65-0.68x / 0.86-0.89x | 0.63-0.85x | 0.86-1.03x / 0.79-0.90x |
| roadNet-CA | 0.98-1.01x | 0.99-1.01x | 0.80-0.86x / 0.83-0.87x | 0.70-0.93x | 0.85-0.90x / 0.77-0.82x |
| rgg_n_2_20_s0 | 0.99-1.00x | 0.99-1.00x | 0.77-0.79x / 0.86-0.89x | 0.68-0.99x | 0.79-0.81x / 0.79-0.88x |
| road_usa | 1.00-1.01x (50K); **1.05-1.06x (local 10K, clock state)** | 0.99-1.01x | 0.94-0.96x / 0.80-0.87x | 0.79-0.95x | 0.79x / 0.80-0.85x |

Byte-identical outputs and equal `invalidated` counters in every run. The fused kernel uses the
original's 59 registers and runs the same 256 x 256 cooperative grid. The apply ratios of CUDA
include dynG's host copies of the input trees (`sssp.import`), as the original's "upload" does.
The one reading over its gate, road_usa's local batch on CUDA (1.060 / 1.060 / 1.054x; step 4:
1.065 / 1.063 / 1.037x), is not contamination (the monitor saw clean rounds) and at locked clocks
the kernels read 0.99x; the per-round times of both programs fall into the same two modes in
different proportions, which fits the GPU's clock state (ADR 0018). ADR 0018 leaves its verdict to
the author, and until then it is recorded as a gate miss.

Absolute times of the 50K safe batch (ms per objective, medians; the same records):

| Graph | MOSP-CUDA | dynG cuda | MOSP-OpenMP (28 threads) | dynG openmp (28 threads) |
|---|---:|---:|---:|---:|
| roadNet-PA | 4.7-4.8 | 4.6-4.8 | 12.6-16.6 | 10.6-14.1 |
| roadNet-CA | 8.6-8.7 | 8.6-8.8 | 20.2-30.5 | 18.8-24.8 |
| rgg_n_2_20_s0 | 24.7-25.4 | 24.6-25.2 | 40.1-51.8 | 38.0-51.3 |
| road_usa | 100-101 | 100-102 | 268-308 | 253-273 |

The first objective is the slowest on OpenMP on both sides: it first touches the pages of the
frontier lists (section "Scratch memory").

## 6. Limitations

- Integer weights of at least 1 only (zero or negative weights are rejected).
- The graph must store its in-edges (`graph_properties::store_transposed`, the default).
- Distances must fit 62 bits ((n - 1) * max weight); beyond that `compute` and `update` throw.
- A tree imported with `validate_inputs = false` must be a shortest-path tree of the graph. All
  engines still reject a parent cycle that no root of the batch breaks (the CUDA engine sees it
  when pointer jumping is still active after ceil(log2 n) + 1 rounds, which needs at least one
  deletion or weight increase in the batch; the batch is already applied, so the result is then
  poisoned and every later use of it throws `stale_result_error` until it is recomputed), and the
  CUDA engine rejects a distance outside [0, (n - 1) * max weight] (MOSP-CUDA's "the initial tree
  does not belong to this graph"); other corrupt inputs (for example wrong distances) give
  undefined, though memory-safe, results.
- CUDA: the device must support cooperative launch (every GPU since Pascal does); the operators
  engine that runs without it arrives in 0.2. A graph built with CUDA resources keeps a host copy
  of its CSR as well as the device copy (the device apply comes later).

## 7. Differences from the paper

The code follows the corrected MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2, not the thesis
pseudocode: subtree invalidation plus a pull pass replaces Step 1's best-in-neighbour rule (which
counts to infinity, e.g. d(1) = 60 instead of 90 on the n = 6 stress seed), the propagation is a
monotone near-far worklist without an iteration cap or a reachability pass, and ties go to the
lowest parent id. On CUDA the whole update of one objective (roots, invalidation by pointer
jumping, pull, near-far push, unpack) is one persistent cooperative kernel, where the paper's
code ran Step 1 on the host and a loop of kernels with a host round trip per iteration. Weights
are integers (the paper uses real weights). Section 8 lists what the fixes changed.

## 8. Paper vs fixed code

The DynaMOSP papers were measured on the research code as it was then (tag `baseline-2026-09` of
both repositories: MOSP-CUDA `ac29545`, MOSP-OpenMP `7284f50`). The pinned originals that dynG
ports are the **fixed** code (`fix/correctness-perf`: MOSP-CUDA@e220ee2, MOSP-OpenMP@c352151),
whose `CHANGES.md` and `results/README.md` record what changed and by how much. dynG is
byte-identical to the fixed code, not to the paper's code. The papers' own run times were not
re-derived by the fixed code or by dynG; the comparisons below are all on the same machine (RTX
A5000, Xeon Gold 6258R with 28 threads pinned) and the same inputs.

Correctness:

| Paper's code (`baseline-2026-09`) | Fixed code and dynG |
|---|---|
| Step 1 gives an invalidated vertex its best *current* in-neighbour, which can be its own descendant: a stale cycle counts to infinity, cut off at `maxIterations = n` rounds plus a BFS repair; distances of **reachable** vertices can end too small (about 1 in 500 random stress cases; e.g. n = 6, seeds 621705 / 250813: d(1) = 60 instead of 90) | subtree invalidation below the changed tree edges, then a pull pass and a monotone push: no cap, no BFS, exact distances (M-a); the three regression cases are golden cases |
| a batch that disconnects vertices runs the full n rounds (roadNet-CA: 1,971,281 rounds, 95-110 s per objective) | cut-off vertices get infinity and -1 directly; the time equals the connectivity-safe batch's |
| parents of tied distances depend on the schedule; CUDA and OpenMP disagree | lowest-id parent everywhere (M-c): CUDA = OpenMP = Dijkstra's tree, deterministic (dynG: `determinism::bitwise`) |
| Step 1 serial on the host (CUDA) | Step 1 on the GPU, grouped by destination (M-b) |

Performance, SOSP update per objective (K = 3, 50K connectivity-safe batch, seed 777). The first
two columns of each backend are the originals' own records (`results/README.md` of each
repository: the paper's code built with `-O3`, and the fixed code as it measured itself, medians
of 3 runs); the last two are the M1b A/B (`parity/results/M1b.md` sections 8.3 and 9.2, medians
of 21 alternating runs, range over the objectives):

| Graph | CUDA: paper's code | CUDA: fixed (own record) | CUDA: fixed (M1b A/B) | CUDA: dynG | OpenMP: paper's code | OpenMP: fixed (own record) | OpenMP: fixed (M1b A/B) | OpenMP: dynG |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| roadNet-PA | 34.1 ms | 5.21 ms | 4.7-4.8 ms | 4.6-4.8 ms | 159 ms | 14.0 ms | 12.6-16.5 ms | 10.7-14.2 ms |
| roadNet-CA | 73.4 ms | 9.43 ms | 8.6-8.7 ms | 8.6-8.8 ms | 287 ms | 23.1 ms | 20.4-34.0 ms | 18.8-24.6 ms |
| rgg_n_2_20_s0 | 167 ms | 26.5 ms | 24.7-25.4 ms | 24.6-25.2 ms | 371 ms | 50.1 ms | 40.1-51.7 ms | 38.0-51.1 ms |
| road_usa | 1.63 s | 105 ms | 100-101 ms | 101-102 ms | 10.1 s | 308 ms | 271-312 ms | 253-272 ms |

The fixed code is 6-16x faster per objective than the paper's code on CUDA and 7-33x on OpenMP
(the OpenMP road_usa baseline ran under a host load of about 45; its own record calls that ratio
overstated by up to about 30 %). On the batch that disconnects vertices the paper's code needs
minutes per objective (roadNet-PA 39.1 s on CUDA, 106 s on OpenMP) where the fixed code and dynG
take the connectivity-safe batch's time. The MOSP-level "(a) compute" totals of the fixed code
(K = 3 updates plus the combined graph: CUDA 18.5 / 33.7 / 83.0 / 377 ms, OpenMP 63.2 / 110 /
166 / 1,320 ms) are gated when `mosp` is ported (0.2, PLAN 6.4.4).

## 9. Mapping from the original code

| MOSP-OpenMP@c352151 | dynG |
|---|---|
| `sospUpdateCpu` | `sssp::update` on `resources::openmp` (`cpp/src/algorithms/sssp/openmp.cpp`) |
| `sospFromScratchCpu`, `mospPrep init` / `dijkstraCsrGraph` | `sssp::compute`; `testing::dijkstra` |
| `sequentialSOSPUpdate` | `sssp::update` on `resources::sequential` (`sequential.cpp`) |
| `SospWorkspace` (one per `mospUpdate()` call, shared by the objectives) | `detail::sssp_workspace`, leased from the pool of `resources` (one per handle, shared by the results; ADR 0015) |
| `SospStats` | `sssp::stats` (`invalidated`, `iterations`, `epochs`, `pushes`, `packed_parents`) |
| `HostChanges` (changed edges, insert heads), `weightIncreaseMask` | `detail::apply_delta` from the commit |
| `defaultDelta` | `options.delta = 0` |
| `canonicalizeTree`, `--init` files | `sssp::result::from_arrays(..., canonicalize)` |
| `checkSospTree` | `testing::check_sssp_tree` |
| `ListGather`; per-region `std::vector` thread lists | `detail::list_gather` (`cpp/src/util/list_gather.hpp`), plus `gather_pair()` (two lists between one pair of barriers); `thread_list`s kept in the workspace (`util/thread_list.hpp`) |
| `mosp` driver (per-objective part) | `tools/compat` `dyng-compat-mosp` |

| MOSP-CUDA@e220ee2 | dynG |
|---|---|
| `sospUpdateGpu`, `sospPersistentKernel` | `sssp::update` on `resources::cuda` (`cuda.cu`, `fused.cuh`: `sssp_persistent_kernel`) |
| `sospFromScratchGpu` | `sssp::compute` on `resources::cuda` |
| `SospWorkspace` (`gridBlocks` from the occupancy API) | `detail::sssp_cuda_workspace`, leased from the pool of `resources` |
| `DeviceGraph`, `uploadDeviceGraph` (reverse CSR built on the device) | the device copy of a CUDA graph (`graph/device_graph.{hpp,cu}`, stage `graph.upload`) |
| `DeviceChanges` (per-objective change lists, `weightIncreaseMask`) | stage `sssp.changes`: the lists built from `detail::apply_delta` and uploaded |
| `setenv("CUDA_MODULE_LOADING", "EAGER")` | `resources::warm_up()` |
| `ScopedStage` with `cudaDeviceSynchronize` | `profiler` stages; `profiler_options::cuda_events` for device times |
| `mospPrep changes` (`generateChangeBatch`) | `generators::legacy::mosp_changes()`; `dyng-compat-mosp changes` |

## 10. How to cite

`dyng::citation("sssp")`: DynaMOSP (IPDPS 2025) and its journal version (IEEE TPDS 2025), keys
`dynamosp2025` and `dynamosptpds2025` in `docs/references.bib`.
