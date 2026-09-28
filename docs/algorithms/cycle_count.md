# cycle_count: exact k-bounded directed simple-cycle histograms

Maturity: **experimental** (M2a: sequential and OpenMP backends; M2b: the CUDA backend; the
Python binding arrives in M5). Header: `<dyng/cycle_count.hpp>`. Oracle: `compute`. Determinism: `exact_value`.
Ported from CycleEnumeration-GPU@0a976ad, the code of TruCy / DynTruCy (Khanda, Shovan, Satpathy,
Das; submitted to IEEE Transactions on Computers). dynG implements the exact k-bounded
enumeration of that code, **not** the paper's approximate kappa-truncated TruCy search (Section 7).

## 1. Problem

**What it computes.** Input: a directed graph with sorted rows and no parallel edges, and a length
bound k >= 2 (or no bound). Output: the histogram `counts[len]`, the number of directed simple
cycles of each length 2..k. A cycle is a closed path without repeated vertices; a 2-cycle is a
pair of opposite edges; self-loops lie on no cycle. Every cycle is counted once, from its smallest
vertex. The counts are exact: they are what the paper calls the "optimal" (reference) counts.

**Update model.** A batch of edge insertions and deletions is applied to the graph under its
`batch_semantics`, and the histogram changes by the cycles the batch destroys (counted on the
graph before the batch, G_t) and creates (counted on the graph after it, G_{t+1}), without
recounting the rest. Postcondition: `update()` equals `compute()` on the new graph exactly.

### 1.1 Graph requirements

| Requirement | Why | If not met |
|---|---|---|
| `row_order::sorted` | the searches rely on sorted rows (neighbors above the root, the ownership lookups; the CUDA kernels close cycles with one `lower_bound`) | `invalid_argument_error` naming the property and the fix |
| `multi_edges::forbid` | a cycle is a vertex sequence; parallel edges would count it once per edge choice | `invalid_argument_error` |
| weights | ignored: instantiated for `unweighted` and `int32` weights, so a weighted sssp graph with the default properties can be shared (`dyng::update(res, g, batch, tree, hist)`) | – |
| in-edges | not read (`store_transposed` is not needed; the update never transposes the graph) | – |
| vertex type | `int32_t` (the ownership table keys two 32-bit ids); offsets `int32_t` or `int64_t`; weights `unweighted` or `int32_t` | compile error: a `static_assert` in `compute`, `update` and `dyng::update` names the supported types |
| length bound (cuda) | the device searches keep their path in thread-local arrays of at most 64 vertices (`kMaxDeviceCycleLength`): the effective bound max(min(k, n), 2) must be <= 64 (a larger k is accepted while the graph has at most 64 vertices, as the original's; no bound means k = n) | `invalid_argument_error` naming 64 and the bound, before anything changes (for an update: with every vertex the batch names) |

`graph_properties::cycle_enum_compatible()` is the preset of the original: sorted rows, no
parallel edges, no weights and `batch_semantics::set()` (a deletion of a missing edge, an
insertion of an existing edge and a self-loop are no-ops; new vertex ids grow the graph). Every
other batch semantics is accepted as well: Step 0 reduces a batch to the change of the edge set
(weight-only upserts are no-ops for cycle_count). Parity with the original is defined under
`set()`. `batch_semantics::as_sets` cannot be combined with `on_existing_insert = upsert`,
`deletions_first = false`, unsorted rows or parallel edges; a graph built with such properties
throws `not_supported_error` when it is constructed.

## 2. Template mapping

```
normalize -> translate -> prepare -> [before_apply -> (AG: count -)] -> commit ->
identify_affected -> seed -> { AG: count + } -> finalize
```

cycle_count is an **aggregate-delta** problem (template card in
`cpp/src/algorithms/cycle_count/problem.hpp`; tier: custom engine).

| Step | Hook (profiler stage) | What it does | Original |
|---|---|---|---|
| 0 | `normalize` (`cycle_count.normalize`) | the net structural change on G_t: deletions of existing edges, insertions of new (or deleted and re-inserted) edges, each sorted by (source, destination) without repeats; self-loops dropped. Under `batch_semantics::as_sets` the framework computes it once for every result and the commit (ADR 0020) | `prepare_batch` |
| 1a | `before_apply` = count(-) (`cycle_count.count_minus`) | for every deleted edge in id order, the cycles through it on G_t that contain no deleted edge of smaller id (ownership = the smallest id, `changed_edge_index`) | delete phase of `update_static_histogram` |
| apply | commit (`cycle_count.commit` > `graph.apply`) | the batch applied once; no transposition (the searches read the out-edges only). On cuda under set semantics without weight columns the batch is merged into the resident graph on the device | `apply_batch` (CUDA: `build_next_rows_kernel`) |
| 1b | `identify_affected` (`cycle_count.identify_affected`) | the inserted edges and their ownership index | – |
| 2 | count(+) (`cycle_count.count_plus`) | the owned cycles through every inserted edge on G_{t+1} | insert phase |
| finish | `finalize` (`cycle_count.finalize`) | counts += added - removed; a bucket that would become negative throws `internal_error` and poisons the result | `apply_histogram_delta` |

`compute()` is the static enactor: `cycle_count.reset` -> `cycle_count.count` ->
`cycle_count.finalize`.

The four Chapter 3 challenges: (i) the affected set is the set of change edges (a cycle changes
only if it contains a changed edge); (ii) the propagation scope is a depth-bounded search from each
change edge, no iteration; (iii) correctness under parallelism comes from ownership: a cycle
through several change edges of one phase is counted only by the smallest id, so the per-edge
searches are independent and the per-thread histograms are summed; (iv) the data structure is the
compact sorted CSR, whose sorted rows the searches rely on.

## 3. API

```cpp
using graph_t = dyng::graph<std::int32_t, std::int64_t, dyng::unweighted>;
auto edges = dyng::io::read_edge_list<std::int32_t, dyng::unweighted>("DD_A.txt");
graph_t g = graph_t::from_edges(res, edges.view(), dyng::graph_properties::cycle_enum_compatible());
dyng::cycle_count::options opt;
opt.max_length = 4;
auto hist = dyng::cycle_count::compute(res, g, opt);
auto st = dyng::cycle_count::update(res, g, batch.view(), hist);
dyng::io::write_histogram_csv(std::cout, hist.counts());   // "# cycle_size, num_of_cycles" ...
```

Python: planned (M5; PLAN Section 5.5, `dyng.cycle_count.compute(cg, max_length=4)`).

| Option | Default | Meaning |
|---|---|---|
| `max_length` | -1 | longest counted length (>= 2), or -1: no bound. The histogram has min(k, max(n, 2)) + 1 entries (no simple cycle is longer than n), so a bound far above n costs nothing. Without a bound see Section 5 for the cost per backend |
| `method` | `search_method::johnson` | the search (the only one) |
| `mode` | `cycle_mode::simple` | static simple cycles (time-window and temporal modes: 0.4) |
| `cuda_engine` | `engine::automatic` | cuda only: `automatic` and `fused` run the fused kernels (Tier B); `operators` throws `not_supported_error` (no operators engine in 0.1) |
| `scheduler` | `cuda_scheduler::work_queue` | cuda `compute()` only: the work queue, or `naive` (one thread per root; the original's debugging and parity path) |
| `work_items` | `cuda_work_items::automatic` | cuda `compute()` with the work queue: `roots`, `edges` (r -> v1, v1 > r), `two_hop` (r -> v1 -> v2, numbered implicitly), or `automatic` (the original's rule: edges up to k = 3 and at k = 4 below 16 edges per vertex, two-hop otherwise). Every cycle has exactly one prefix of each kind: the counts never depend on it |

`stats`: `batch` (the apply summary), `deletions` / `insertions` (the change edges of the two
phases), `cycles_removed` / `cycles_added`, and the inherited counters with their cycle_count
meaning: `affected` (the number of lengths whose count changed), `frontier_visits` (the change
edges searched, deletions + insertions), `iterations` (always 0: no iterative loop),
`fallback_used` (always false), `converged` (always true), `engine_used` (`engine::operators`
on the host backends, `engine::fused` on cuda). Every counter is deterministic for cycle_count, including the two that
`update_stats` calls schedule-dependent. `result`: `counts()`, `count(len)`, `total()`,
`bound()` (min(k, max(n, 2))), `get_options()`, `clone()`.
`dyng::update(res, g, batch, r1, r2, ...)` updates cycle_count together with other results on one
graph (e.g. sssp on a weighted graph with the default properties; weights are ignored).

The example is `examples/cpp/cycle_count_update.cpp`; the drop-in clone of the original CLI is
`tools/compat/cycle_enum` (`dyng-compat-cycle-enum`).

## 4. Backends and determinism

| Backend | compute | update |
|---|---|---|
| sequential | the original's sequential Johnson: path-membership blocking with a bound, Johnson's blocked lists without one | `update_static_histogram` |
| openmp | the original's OpenMP counter: roots in parallel (`schedule(dynamic)`), one histogram per thread | `update_static_histogram_openmp`: change edges in parallel, one histogram per thread; with one thread the sequential phases |
| cuda | the original's static CUDA counters (`count_simple_cycles_johnson_queue_device`, and `count_simple_cycles_johnson_device` with `scheduler = naive`): the exact pruned DFS (`extend_prefix`, a `lower_bound` closes each cycle), path and cursors in thread-local arrays of a compile-time capacity (4, 8, 16, 32 or 64), one per-thread histogram reduced across the warp and flushed once; the work queue: a resident grid (occupancy limit x SMs, 128 threads) claims prefix items from a global counter | `update_static_histogram_cuda`: the delete phase on the resident G_t, the insert phase on G_{t+1}; work items are (change, first hop) pairs numbered by a prefix sum; path membership instead of a visited array; an `owner[]` array per CSR position gives the ownership id of a change edge |

Determinism: **`exact_value`**. The histograms (and every `stats` counter) are identical on every
backend, thread count and run, because they are sums of integers; the order in which threads find
cycles does not matter. Counts are 64-bit; a sum beyond 2^64 - 1 throws `capacity_error`.

Memory. The per-thread visited marks, search stacks and counters, the ownership table and the
phase histograms live in a workspace leased from the resources handle (ADR 0015); each thread
sizes its own scratch at the start of the parallel region that uses it. The ownership table is a
flat open-addressing table whose arrays are reused (the original's `std::unordered_map` allocates
a node per change edge on every update). So once a first update of a given size has run, the
algorithm phase of an update (after the commit) allocates nothing (invariant I9), except when a
larger batch, a longer cycle than any before or a larger graph grows an array (a histogram of an
unbounded result grows with the vertex count). The per-thread counters grow with the longest
cycle actually found, not with the length bound, and the reduction reads and clears only the
lengths a search reached: an update costs what its searches cost, bounded or not.

**The CUDA backend** is the straight port of CycleEnumeration-GPU's `src/cuda` and
`src/dynamic/update_cuda_kernel.cu` (the kernels templated on the offset type: with `int32_t`
offsets they are the original's 32-bit ones). Two things differ by design:

- **The graph is resident.** A graph built with CUDA resources is uploaded once (the first
  `compute()` or `update()`), not per call as the original does. Under `batch_semantics::set()` (or
  any `as_sets` semantics) without weight columns, `update()` merges the batch into the sorted rows
  on the device (the original's `build_next_rows_kernel`, which it runs and discards), so the next
  update finds G_{t+1} on the device; the insertion ids that kernel writes are the owner array of
  the insert phase. The host copy of the graph is then stale and is downloaded when something reads
  it (`g.view()`, `g.to_csr()`, a host backend, or Step 0 of the next batch, which reads G_t's rows
  on the host as the original's `prepare_batch` does). Other semantics and weighted graphs apply the
  batch on the host and upload the new graph (ADR 0020).
- **Scratch memory is leased.** The item arrays, owner arrays, prefix sums and histograms come from
  the workspace pool of the resources handle (ADR 0015), on its stream; the original allocates them
  with `cudaMalloc` on every call (inside its timed kernel region).

The histograms are copied back to the host at the end of every call (the result is a host array on
every backend).

## 5. Complexity and performance notes

Exact counting is exponential in the length bound: `compute()` explores, from every root, the
simple paths of length < k over vertices above the root; `update()` explores the paths of length
< k through each change edge, on G_t for the deletions and on G_{t+1} for the insertions.

**Without a bound** (`max_length = -1`, the default) the backends differ:

| Search | Algorithm | Time |
|---|---|---|
| `compute()`, sequential | Johnson's algorithm with blocked lists (the original's `JohnsonSearch`) | O((n + m)(c + 1)) for c cycles |
| `compute()`, openmp | the original's OpenMP counter: a path search per root, no blocked lists | grows with the number of simple paths over vertices above each root: exponential even on a graph with few or no cycles (a layered DAG of 28 layers of two vertices, no cycle at all: about 1 s, doubling per layer) |
| `update()`, every backend | a path search through each change edge (the original's `count_cycles_through_edge`) | grows with the number of simple paths from the change edge's head, exponential in the same way |

For an unbounded count of a large sparse graph use the sequential backend, or set a bound. Every
search keeps its path on an explicit stack (the original recurses once per path vertex and
overflows the thread's stack at a path of about 70,000 to 200,000 vertices); a long path costs
memory (24 bytes per level), never the stack. The length bound is clamped to max(n, 2) before
anything is sized.

The timed regions are defined in `parity/timed_regions/cycle_count.toml`:

| Region | dynG | Original | Gate |
|---|---|---|---|
| `static_end_to_end` | process wall time of `dyng-compat-cycle-enum --task count` | process wall time of `cycle-enum --task count` (its RESULTS.md "OpenMP (56 threads), end to end"; the CLI has no CPU count timer) | <= 1.05x |
| `static_count` | stage `cycle_count.compute` | – | reported |
| `update` | `update_ms` = stage `cycle_count.update` (`cycle_count::update()`) | `update_seconds` (around `update_histogram`) | <= 1.05x (>= 10 ms) |
| `update_end_to_end` | process wall time of `--task update` | process wall time | <= 1.10x |

Measured in M2a (`parity/results/M2a.md`: OpenMP, 56 threads, the libgomp defaults on both
sides, the unpatched original, exclusive lock, medians of 11 A/B rounds; COLLAB 5), on the Xeon
Gold 6258R, port `0679ed1` (after the review fixes):

| Case | Region | CycleEnumeration-GPU (ms) | dynG (ms) | Ratio |
|---|---|---:|---:|---:|
| DD k = 3 | static end to end | 256.7 | 238.7 | 0.930 |
| DD k = 4 | static end to end | 251.2 | 229.6 | 0.914 |
| DD k = 5 | static end to end | 293.3 | 262.9 | 0.896 |
| DD k = 6 | static end to end | 514.5 | 373.1 | 0.725 |
| DD k = 7 | static end to end | 1,488.7 | 892.7 | 0.600 |
| GitHub k = 3 | static end to end | 719.2 | 562.4 | 0.782 |
| GitHub k = 4 | static end to end | 2,806.5 | 1,472.0 | 0.525 |
| Twitch k = 3 | static end to end | 1,812.5 | 1,537.2 | 0.848 |
| Twitch k = 4 | static end to end | 3,608.7 | 2,474.1 | 0.686 |
| COLLAB k = 3 | static end to end | 44,565.1 | 12,961.1 | 0.291 |
| DD 25K+25K, k = 4 | update | 26.1 | 16.9 | 0.646 |
| GitHub 25K+25K, k = 4 | update | 294.3 | 130.7 | 0.444 |
| Twitch 25K+25K, k = 4 | update | 157.8 | 76.1 | 0.483 |
| DD 25K+25K, k = 4 | update end to end | 368.1 | 308.5 | 0.838 |
| GitHub 25K+25K, k = 4 | update end to end | 3,283.7 | 1,769.2 | 0.539 |
| Twitch 25K+25K, k = 4 | update end to end | 4,576.7 | 3,329.1 | 0.727 |

Every gate is met, with no contaminated measurement (the harness records the foreign CPU load of
every run). Where the gains come from is measured separately (PLAN 8.6; `parity/results/M2a.md`
Section 3.4, with copies of the original that differ in one change each):

- the straight-ported search is already faster than the original's with the same dense
  histogram (count 0.70-0.94): it scans 4-byte column ids, the original 24-byte adjacency entries;
- the dense per-thread histogram instead of a `std::map` increment per cycle saves about a fifth
  of the original's count at k = 4 and nothing measurable at k = 3;
- the explicit-stack searches of the review fixes scan a row in a tight loop (count 0.76-0.82 of
  the recursive port on the large cases), and the update adds each cycle to the thread's counters
  and looks ownership up in a flat table (update 0.48-0.69 of the port before the fixes).

Where the time goes in the update (DD 25K+25K, ms): normalize 2.5, count_minus 3.6, commit 6.3,
identify_affected 0.3, count_plus 3.7.

**CUDA.** The regions are those of `[reference.cycle_enum_cuda]` in the same file: the static
`kernel_ms` (CUDA events around building the work items and counting: the original's
`--report-timing` and dynG's stage `cycle_count.count`), `memcpy_ms` and `total_ms` (reported),
the update's `update_seconds` / `update_ms` (host clock around the update, as the original), and
both end-to-end times. Each is read in two scopes (`dyng-compat-cycle-enum --scope`): *original*
(the graph uploaded inside the timed call, as the original does per call) and *resident* (the
graph on the device before the call, dynG's model). The gate is PLAN 8.6 at locked GPU clocks
(ADR 0018, `parity/cycle_count_perf.py run --backend cuda`); the measured table is in
`parity/results/M2b.md`. The register counts of the counting kernels of both sides are recorded
with `parity/cycle_count_perf.py kernels`.

The update is not always much faster than a recompute: with the original's fast static kernels,
its own measurements give update-vs-recompute ratios of 1.2x on DD, 5.2x on GitHub, 4.2x on
Twitch and 34x on COLLAB (CUDA, 25K+25K, k = 4). Speedups reported against the paper's slower
kernels do not carry over.

## 6. Limitations

- Exact counting is exponential in the length bound. Without a bound the OpenMP `compute()` and
  every `update()` enumerate simple **paths**, not only cycles (Section 5): their time can be
  exponential on graphs with few or no cycles, where the sequential `compute()` (Johnson) is
  linear per cycle.
- The CUDA backend counts cycles of at most 64 vertices: the effective bound max(min(k, n), 2)
  must be <= 64 (`invalid_argument_error` otherwise). Without a bound (the default options) it
  therefore works only on graphs of at most 64 vertices; set `max_length`.
- On cuda, Step 0 of an update reads the host copy of G_t: after a device apply, the next update
  downloads the graph once (ADR 0020). A device Step 0 is future work.
- The vertex type is `int32_t` (the ownership table keys two 32-bit ids); offsets may be 32 or 64
  bits.
- Graphs with parallel edges or unsorted rows are rejected (`invalid_argument_error`).
- No time-window or temporal modes yet (0.4), no Read-Tarjan or brute-force method, no Python
  binding yet (M5). The original's CUDA time-window and temporal kernels are not ported (0.4), nor
  its environment tuning (`CYCLE_ENUM_CUDA_BLOCK_SIZE`, `CYCLE_ENUM_CUDA_BLOCKS_PER_SM`: the
  defaults, 128 threads and the occupancy limit, are fixed).
- No approximate (kappa-truncated) mode (Section 7).

## 7. Differences from the paper

The paper is TruCy / DynTruCy (A. Khanda, S. M. Shovan, A. Satpathy, S. K. Das: "TruCy:
GPU-Accelerated Cycle Enumeration for Large-Scale Static and Dynamic Networks", **submitted** to
IEEE Transactions on Computers, 2026). The code it was evaluated with is CycleEnumeration-GPU at
tag `baseline-2026-09` (`da2067d`); dynG ports the corrected version `0a976ad`.

**Exact k-bounded enumeration, not the kappa-truncated TruCy.** The paper's TruCy is an
approximate search: a truncated Johnson search with a blocked array of bounded size kappa, ranks
on the truncated blocked path (`rbpath`), iterative unblocking, zero in/out-degree pruning with
relabeling, and roots in decreasing out-degree order. When the blocked array is full, the search
from that root stops, so TruCy can miss cycles (the paper's Figs. 5, 7 and 8). That design is not
in CycleEnumeration-GPU (before or after its fixes) and **not in dynG**. Consequently:

- dynG's counts equal the exact ("optimal") counts the paper compares TruCy against, not TruCy's
  kappa-dependent counts;
- dynG's timings are for exact enumeration and are not directly comparable with kappa-bounded
  TruCy runs;
- the paper's kappa experiments (Figs. 7 and 8) cannot be reproduced with dynG. A future,
  explicitly approximate mode is on the roadmap (PLAN 6.1, row E6), not planned for 0.1.

The update is the paper's DynTruCy scheme: the cycles through the deleted edges are subtracted on
G_t, those through the inserted edges added on G_{t+1}, each attributed to its smallest-id change
edge. dynG counts simple cycles only (the time-window and temporal modes of the original follow
in 0.4).

### Paper vs fixed code

The fixes between the paper's snapshot (`baseline-2026-09`) and `0a976ad`, with the labels of the
original's `CHANGES.md`, and what dynG takes from each:

| Item | Paper's code (`baseline-2026-09`) | Fixed code (`0a976ad`) | dynG |
|---|---|---|---|
| C1 Matrix Market input | symmetric files read in one direction only: lower-triangle storage became a DAG, 0 cycles | `symmetric`, `skew-symmetric`, `hermitian` add both directions; `array` and unknown banners rejected | ported (`io::read_edge_list`; fixture tests for every symmetry) |
| C4 invalid update batches | deleting a missing edge, inserting an existing edge or a self-loop gave wrong histograms; an insertion with an id >= V was dropped on the CPU (out of bounds on the GPU) | `prepare_batch` gives batches set semantics; `apply_batch` grows the graph | ported (`batch_semantics::set()`, Step 0 on G_t); 80 random arbitrary batches and the edge-set recount in CI |
| C6 CPU baselines | the bounded sequential Johnson reset O(V) state per root (O(V^2)); the OpenMP counter allocated a V-byte array per root; the update's prior and recompute always used the sequential Johnson (DD k = 3: 127 s) | bounded search without the reset, unbounded searches reset only touched vertices; per-thread buffers; the recompute uses the update's backend (DD k = 3: 2.5 s) | ported (per-thread marks leased from the workspace, sized by their own thread) |
| H1 parser | single-threaded `istringstream`, `std::set` / `std::map` interning | parallel `from_chars` parser, counting-sort CSR/CSC | ported (`io::read_edge_list`; identical vertex order and CSR) |
| H3 batch application | every touched row rescanned the whole insertion list (335 ms of DD's timed update) | O(E + B log B): both lists sorted once, one merge per touched row | ported (`graph::apply` under `set()`, byte-equal CSR; assembled in parallel blocks) |
| C9 tests | GPU tests could not fail (no device test of the default kernel, tiny graphs, a buggy oracle) | subset-DP oracles, randomized parity suites; two recorded mutations (5-cycles counted twice, weakened ownership) each fail the suite | ported (`dyng::testing` oracles; CTests `cycle_count.mutation.*` require both mutations to fail and a control copy to pass) |
| C10 build | no build type (`-O0`), CUDA architecture default never applied | Release default, sm_86 | dynG's presets; the reference is built Release as its RESULTS.md |
| K1, K2 static CUDA kernels | full-row scans, global-memory paths, one atomic per cycle; roots as work items | exact pruned DFS with a `lower_bound` closure; edge and two-hop prefix work items (3.7x to 714x faster kernels) | ported (M2b: `static_cuda.cu`, `dfs.cuh`; every scheduler and kind of work item, checked against the original's CUDA backend on the fixtures) |
| K3, C5, C7 CUDA update | a V-byte visited array per change (36 GB on GitHub: out of memory), host rebuild of G_{t+1}, context creation inside `update_seconds` | path membership, resident G_t with G_{t+1} built on the device, an `owner[]` array | ported (M2b: `cuda.cu`, `graph/apply_set_device.cu`); dynG also keeps G_{t+1} on the device for the next batch (ADR 0020) |
| C2, C3, time-window / temporal self-loops | bounded time-window Johnson undercounted; time-window Read-Tarjan wrong; self-loop start events | fixed | not ported (modes arrive in 0.4) |

Every fix keeps the counts exact. On valid inputs (the TUDataset graphs, valid batches) the
histograms of `0a976ad` are bit-identical to the paper's code on every dataset the original's
`CHANGES.md` measured, so dynG's parity against `0a976ad` is also parity with the counts the paper
used there; the counts differ only where the paper's code was wrong (C1, C4 and the time-window
modes).

## 8. Mapping from the original code

| CycleEnumeration-GPU@0a976ad | dynG |
|---|---|
| `sequential::count_simple_cycles_johnson` (`JohnsonSearch`) | `cycle_count::compute` on `resources::sequential()` (`static_sequential.cpp`) |
| `openmp::count_simple_cycles_johnson` (`count_root`) | `cycle_count::compute` on `resources::openmp(t)` (`static_openmp.cpp`) |
| `dynamic::update_static_histogram` (`accumulate_phase`) | `cycle_count::update` on `resources::sequential()` (`sequential.cpp`, `cycle_count.cpp`) |
| `dynamic::update_static_histogram_openmp` (`accumulate_phase_parallel`) | `cycle_count::update` on `resources::openmp(t)` (`openmp.cpp`) |
| `dynamic::count_cycles_through_edge`, `ChangedEdgeIndex` | `detail::count_cycles_through_edge`, `detail::changed_edge_index` (`cycles_through_edge.hpp`, `problem.hpp`) |
| `dynamic::apply_histogram_delta` | the finalize hook (`cycle_count.cpp`) |
| `dynamic::prepare_batch`, `apply_batch`, `DirectedGraph` | `detail::compute_structural_change` (Step 0 on G_t), `graph::apply` under `graph_properties::cycle_enum_compatible()` |
| `CycleHistogram` (a map), `to_csv` | `cycle_count::result` (a dense array), `io::write_histogram_csv` |
| `CycleEnumerationOptions` (`max_cycle_length`, `algorithm`, `mode`) | `cycle_count::options` (`max_length`, `method`, `mode`) |
| `cuda::extend_prefix`, `find_edge`, `dispatch_capacity`, `ThreadHistogram`, `CsrView` | `detail::extend_prefix`, `find_edge`, `dispatch_capacity`, `thread_histogram` (`dfs.cuh`), `detail::device_csr` (`util/device_csr.cuh`); the host port `detail::cycle_count_extend_prefix` (`dfs.hpp`) |
| `cuda::count_simple_cycles_johnson[_work_queue]` (`count_roots_kernel`, `count_roots_queue_kernel`, `forward_rows_kernel`, `fill_edge_items_kernel`, `target_degree_kernel`, `count_edge_items_kernel`, `count_two_hop_items_kernel`), `CudaWorkItems`, `resolve_work_items`, `plan_work_queue_launch` | `cycle_count::compute` on `resources::cuda()` (`static_cuda.cu`, same kernel names), `cycle_count::cuda_work_items`, `detail::resolve_work_items`, `detail::plan_work_queue_launch` (`work_queue.hpp`) |
| `dynamic::update_static_histogram_cuda`, `count_update_cycles_device` (`mark_owners_kernel`, `item_counts_kernel`, `count_owned_cycles_kernel`) | `cycle_count::update` on `resources::cuda()` (`cuda.cu`, same kernel names) |
| `change_rows_kernel`, `next_degree_kernel`, `build_next_rows_kernel` (G_{t+1} on the device) | `detail::apply_set_batch_device` (`graph/apply_set_device.cu`), the commit of a resident graph |
| `kMaxDeviceCycleLength` | `detail::cycle_count_max_device_length` (64) |
| `engine::count_histogram`, `engine::update_histogram` | `cycle_count::compute`, `cycle_count::update` |
| `cycle-enum` (CLI) | `dyng-compat-cycle-enum` (`tools/compat/cycle_enum`) |
| `generate_batch`, `read_graph_view` | `generators::legacy::cycle_enum_batch`, `io::read_edge_list` |
| `tests/support/cycle_oracles.hpp`, `count_simple_cycles_bruteforce` | `testing::oracle_simple_cycles`, `testing::brute_force_simple_cycles`, `testing::edge_set_after_batch` |

## 9. How to cite

`dyng::citation("cycle_count")` returns the BibTeX entry `trucy2026` (docs/references.bib; the
paper is submitted, the entry is updated when it is accepted). Please also cite dynG itself
(`CITATION.cff`).
