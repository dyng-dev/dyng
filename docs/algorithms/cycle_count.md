# cycle_count: exact k-bounded directed simple-cycle histograms

Maturity: **experimental** (M2a: sequential and OpenMP backends; CUDA arrives in M2b, the Python
binding in M5). Header: `<dyng/cycle_count.hpp>`. Oracle: `compute`. Determinism: `exact_value`.
Ported from CycleEnumeration-GPU@0a976ad (TruCy / DynTruCy, IEEE Transactions on Computers).

## 1. Problem

Input: a directed graph with sorted rows and no parallel edges, and a length bound k >= 2 (or no
bound). Output: the histogram `counts[len]`, the number of directed simple cycles of each length
2..k. A cycle is a closed path without repeated vertices; a 2-cycle is a pair of opposite edges;
self-loops lie on no cycle. Every cycle is counted once, from its smallest vertex.

Update model: a batch of edge insertions and deletions is applied to the graph under its
`batch_semantics`, and the histogram changes by the cycles the batch destroys (counted on the
graph before the batch, G_t) and creates (counted on the graph after it, G_{t+1}), without
recounting the rest. Postcondition: `update()` equals `compute()` on the new graph exactly. Every
batch semantics is accepted: Step 0 reduces a batch to the change of the edge set (weight-only
upserts are no-ops); parity with the original is defined under `batch_semantics::set()`
(`graph_properties::cycle_enum_compatible()`).

## 2. Template mapping

```
normalize -> translate -> prepare -> [before_apply -> (AG: count -)] -> commit ->
identify_affected -> seed -> { AG: count + } -> finalize
```

cycle_count is an **aggregate-delta** problem.

| Step | Hook (profiler stage) | What it does | Original |
|---|---|---|---|
| 0 | `normalize` (`cycle_count.normalize`) | the net structural change on G_t: deletions of existing edges, insertions of new (or deleted and re-inserted) edges, each sorted by (source, destination) without repeats; self-loops dropped | `prepare_batch` |
| 1a | `before_apply` = count(-) (`cycle_count.count_minus`) | for every deleted edge in id order, the cycles through it on G_t that contain no deleted edge of smaller id (ownership = the smallest id, `changed_edge_index`) | delete phase of `update_static_histogram` |
| apply | commit (`cycle_count.commit` > `graph.apply`) | the batch applied once; no transposition (the searches read the out-edges only) | `apply_batch` |
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
std::cout << dyng::io::format_histogram_csv(hist.counts());   // "# cycle_size, num_of_cycles" ...
```

| Option | Default | Meaning |
|---|---|---|
| `max_length` | -1 | longest counted length (>= 2), or -1: no bound (the histogram then has max(n, 2) + 1 entries) |
| `method` | `search_method::johnson` | the search (the only one) |
| `mode` | `cycle_mode::simple` | static simple cycles (time-window and temporal modes: 0.4) |

`stats`: `batch` (the apply summary), `deletions` / `insertions` (the change edges of the two
phases), `cycles_removed` / `cycles_added`, `affected` (the number of lengths whose count changed);
all deterministic. `result`: `counts()`, `count(len)`, `total()`, `bound()`, `clone()`.
`dyng::update(res, g, batch, r1, r2, ...)` updates cycle_count together with other results on one
graph (e.g. sssp on a weighted graph with the default properties; weights are ignored).

The example is `examples/cpp/cycle_count_update.cpp`; the drop-in clone of the original CLI is
`tools/compat/cycle_enum` (`dyng-compat-cycle-enum`).

## 4. Backends and determinism

| Backend | compute | update |
|---|---|---|
| sequential | the original's sequential Johnson: path-membership blocking with a bound, Johnson's blocked lists without one | `update_static_histogram` |
| openmp | the original's OpenMP counter: roots in parallel (`schedule(dynamic)`), one histogram per thread | `update_static_histogram_openmp`: change edges in parallel, one histogram per thread; with one thread the sequential phases |
| cuda | M2b | M2b |

Histograms are identical on every backend and thread count. Counts are 64-bit; a sum beyond
2^64 - 1 throws `capacity_error`. The steady-state update allocates nothing but its histogram
growth (unbounded results on a growing graph): the per-thread visited marks, per-thread histograms
and the ownership table live in a workspace leased from the resources handle (ADR 0015).

## 5. Performance notes

The timed regions are defined in `parity/timed_regions/cycle_count.toml`: the static count is
gated end to end (the original's CLI prints no CPU timer; RESULTS.md reports process wall times),
the update on the original's `update_seconds` (`cycle_count.update`). The measurements of M2a are
in `parity/results/M2a.md`.

## 6. Limitations

- Exact counting is exponential in the length bound; an unbounded count or update enumerates every
  cycle through the searched vertices.
- The vertex type is `int32_t` (the ownership table keys two 32-bit ids); offsets may be 32 or 64
  bits.
- Graphs with parallel edges or unsorted rows are rejected (`invalid_argument_error`).
- No time-window or temporal modes yet (0.4), no CUDA backend yet (M2b).

## 7. Differences from the paper

**Exact, not kappa-truncated.** The paper's TruCy is an approximate search: a truncated Johnson
search with a blocked array of bounded size kappa, ranks on the truncated blocked path, iterative
unblocking, zero in/out-degree pruning with relabeling and roots in decreasing out-degree order;
when the blocked array is full the search from that root stops, so TruCy can miss cycles (the
paper's Figs. 5, 7 and 8). That design is not in CycleEnumeration-GPU and not in dynG. dynG counts
exactly what the paper uses as its reference ("optimal"); its timings are for exact enumeration and
are not comparable with kappa-bounded runs, and the kappa experiments cannot be reproduced.

**Paper vs fixed code.** The port follows the corrected CycleEnumeration-GPU@0a976ad, not the
paper's code snapshot (tag `baseline-2026-09`). The fixes that concern the CPU path: the update
accepted invalid batches (deleting a missing edge, inserting an existing edge or a self-loop gave
wrong histograms; an insertion naming a new vertex was dropped) until `prepare_batch` gave batches
set semantics and `apply_batch` grew the graph; Matrix Market files were read in one direction
only; the bounded sequential Johnson reset O(V) state per root (O(V^2)) and the OpenMP counter
allocated a V-byte array per root; the update's recompute baseline always used the sequential
Johnson. The update itself is the DynTruCy delete-then-insert scheme with edge-id ownership: the
cycles through the deleted edges are subtracted on G_t, those through the inserted edges added on
G_{t+1}, each attributed to its smallest-id change edge.

**Recorded mutations.** CycleEnumeration-GPU recorded two mutations its suite must detect (double
counting 5-cycles in the static kernel; a weakened ownership rule in the update). dynG builds both
into copies of the library (`DYNG_MUTATION_DOUBLE_COUNT_5`, `DYNG_MUTATION_WEAK_OWNERSHIP`, in the
static counters and in `changed_edge_index`) and requires the randomized suite to fail on each
(CTest `cycle_count.mutation.*`), with a control copy that must pass.

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
| `CycleHistogram` (a map), `to_csv` | `cycle_count::result` (a dense array), `io::format_histogram_csv` |
| `CycleEnumerationOptions` (`max_cycle_length`, `algorithm`, `mode`) | `cycle_count::options` (`max_length`, `method`, `mode`) |
| `cuda::extend_prefix`, `find_edge`, `dispatch_capacity` | `detail::cycle_count_extend_prefix` and friends (`dfs.hpp`, host port, used by the CUDA backend in M2b) |
| `engine::count_histogram`, `engine::update_histogram` | `cycle_count::compute`, `cycle_count::update` |
| `cycle-enum` (CLI) | `dyng-compat-cycle-enum` (`tools/compat/cycle_enum`) |
| `generate_batch`, `read_graph_view` | `generators::legacy::cycle_enum_batch`, `io::read_edge_list` |
| `tests/support/cycle_oracles.hpp`, `count_simple_cycles_bruteforce` | `testing::oracle_simple_cycles`, `testing::brute_force_simple_cycles`, `testing::edge_set_after_batch` |

## 9. How to cite

`dyng::citation("cycle_count")` returns the BibTeX entry `trucy2026` (docs/references.bib).
