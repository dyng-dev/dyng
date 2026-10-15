# ADR 0053: The port of ESCHER-GPU's wedge engine (triad_count on CUDA)

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the CUDA port of M9a step 3 (PLAN 6.3 steps 5-8,
  6.4.6, 4.5.4 Tier B, 4.5.5 I1 / I2 / I7 / I9, 4.6 rule 7, 4.7; the M9a acceptance criteria 6, 7,
  8 (cuda) and 9). It changes no rule the author approved: the parity is exact (PLAN 8.3), the
  gates and their protocol (PLAN 8.6, ADR 0048, D1) are untouched, the frozen headers
  (`hypergraph/*`, `update.hpp`, `core/*`, `graph/*`) are not edited.
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step cuda-port-api-early-gate)
- **Builds on:** ADR 0038 (the transplant method of the CBST store), ADR 0042 (the line-graph port
  and the row views), ADR 0043 (the hypergraph's update path), ADR 0015 (pooled workspaces),
  ADR 0048-0051 (the parity protocol, the oracle, the replays, the driver), ADR 0052 (the API).

## Context

ESCHER-GPU@abcf9b2 (A) counts triads with one engine (`kernel/wedge.cuh`): one thread per oriented
wedge `(a, b)` of the line graph; full counts take the out-wedges (`b > a`) of every key, the
frontier phases every neighbour of each frontier key, skipping the frontier members smaller than
the source (the exactly-once rule: a triad is counted at its smallest frontier member); a wedge is
closed into triangles `(a, b, k)`, `k > b`, by merging the rows of `a` and `b` after `b`; block
bins in shared memory, flushed once per block with the phase's sign. Four operations run on it
(MotifOp: the 30 h-motif classes; Choose3Op, Type2Op, Type3Op: StatHyper's types), driven by four
counter classes (`count()`, `subtract()` before the data-structure update, `add()` after it). A
allocates per call (`cudaMalloc` of the offsets, of a `DeviceFrontier` with a host-built flag
array, of `DeviceBins`), runs on the legacy default stream, synchronizes the device after every
h-motif phase and ends the process on a CUDA error; a triangle of no class would add to
`bins[-1]`.

## Decision

### 1. The transplant and the mechanical classes (PLAN 6.3 step 5; ADR 0038's method)

`parity/escher_port/transplant.py --map parity/escher_port/rename_map_triad.toml --check --rev
645d6dc` regenerates the transplant (`cpp/src/algorithms/triad_count/engine/`): `wedge.cuh`
(verbatim), `count_ops` (Choose3Op), `motif_table.hpp`, `motif_ops` (the class map, the class index,
MotifOp), `stathyper_ops` (Type2Op, Type3Op), `device_bins.hpp`, `device_frontier.cuh`, and
`counters.{cuh,cu}` (computeMotifCounts, runFrontierPhase, the delta accumulator, the four counter
classes without their CPU reference, printing and `main()`); the renames of the map, the namespace
`dyng::detail::triad`, the provenance headers `// Derived from ESCHER-GPU@abcf9b2:<path>`, and the
manual hunks of `manual_hunks_triad.patch` (two `#pragma once`, one include, the shim of the
driver interface). Not ported: the pre-fix anchored kernels (6a44c1c), the coarse triangles
(`coarseTriangle.cu`, TriangleCountOp) and readFlat (already `escher/find.cu`). Then one mechanical
class per commit, each replayed (section 7):

| Commit | Class | What changed |
|---|---|---|
| `645d6dc` c1 | transplant | A's code, run as A's driver runs it by the glue (`cuda.cu`); with it the API, the problem, the host recount, the replay counter, the driver's counting (the infrastructure the replays need, as M8b's line-graph transplant `3b73335`) |
| `64d3b7c` c2 | errors | `exit(-1)` / `cerr` -> `DYNG_CUDA_TRY` (`_NO_THROW` in destructors); a triangle of no class raises the device error word (a 31st slot read with the bins) instead of `bins[-1]`; the read throws `internal_error` |
| `29a02ad` c3 | memory | every array from a pooled workspace (ADR 0015): the offsets and CUB's scan storage (Thrust allocated it per call), the frontier (keys and flags built by a kernel: the deleted ids uploaded once, the inserted ids taken from the applied record's device copy), the bins (subtract and add phases kept apart); the counter classes become `count_full()` / `count_phase()` (without device memory they are their pattern's choice of shape and operation); DeviceBins, DeviceFrontier and the shim are gone |
| `679cec9` c4 | streams | every kernel, copy and memset on the resources' stream; CUB's scan on it; the host waits only at A's read points (the wedge count of a phase, the bins at the end), each counted; A's `cudaDeviceSynchronize` after the h-motif phases is gone; the CBST source rules cover the directory |
| `84f894b` c5 | registration and counters | `DYNG_REGISTER_KERNEL` (one instantiation per operation); every count call reports its outcome (no sources, no wedges, a kernel) and the result keeps the launch counters per mode and shape |
| `c0793d6` c6 | the Op form | the operations M9b will publish: host-and-device structs that see 0-based ids, a rows handle (`hyperedge_rows`, the members of an id; the `+1` stays in the handle) and a bins handle (`block_bins`); the set primitives and the class map host-compilable (`sets.hpp`, a host copy of the map); the frontier filters named (`ownership.hpp`); the phases take `ownership::min_member` explicitly |

**The c6 go/no-go** (`parity/triad_sass.py compare`, the parity-cuda objects): the machine code of
the five counting kernels is identical to c5's (normalized SASS digest, hot loops, registers
14 / 39 / 39 / 38 / 16, no stack, no local memory, the same shared memory: the same occupancy).
**GO.** A bins handle that also tested the upper bound of the bin cost one IADD3 per triangle in
MotifOp's hot loop; the handle tests the sign only (the engine's operations never name a bin at or
above `num_bins`; M9b's user operations may take a handle that tests both). Against A's unpatched
`build/main`, `type1`, `type2`, `type3`: the wedge count, `key_kernel<choose3>`,
`pair_kernel<type2>` and `triangle_kernel<type3>` are machine-code identical; `triangle_kernel<motif>`
has 16 more instructions (552 against 536; the error check of c2), the same 39 registers and
shared memory.

### 2. The problem around the engine (Tier B)

`select_engine`: `engine::fused` on cuda (the engine is the problem's `enact_fused` /
`compute_fused`), `engine::operators` on the host backends. The framework owns Step 0, the commit,
the stages, the budgets and the error words:

- **count(-)** (stage `triad_count.count_minus`, before the commit, on G_t's internal row views):
  lease the workspace, reserve everything both phases need (below), clear the bins, upload Step 0's
  live distinct deletions (int64 ids, one pageable `cudaMemcpyAsync`), build the frontier (keys
  `id + 1`, flags over `views.h2h.max_key + 1`) with a kernel, run the subtract phase with sign -1
  into the subtract bins.
- **commit** (`triad_count.commit`): `hypergraph.apply`.
- **enact_fused** on G_{t+1}: `triad_count.identify_affected` (the frontier of the ASSIGNED
  inserted ids, from `hyper_applied::inserted_ids_device`, by a kernel, no upload),
  `triad_count.count_plus` (the add phase with sign +1 into the add bins), `triad_count.finalize`
  (one read of both phases' bins and the error word; the per-bin delta applied, a bin that would go
  negative an `internal_error` that poisons the result; the stats).
- **compute()** (`triad_count.compute`; inside it `triad_count.reset`, `triad_count.count`,
  `triad_count.finalize`): the full count on the current views, its offsets (one per key: 20 MB
  on coauth-DBLP) from scratch arrays of their own, released before it returns.

### 3. Key ranges

The h2v view covers the keys 1..`hyperedge_id_bound()`, the h2h view 1..`max_bound` (the largest
bound so far: A's `numHyperedgeKeys`; its rows above the bound are empty), and the locator never
shrinks, so h2v entries above the bound are stale. Every h2v-indexed loop is bounded by
`views.h2v.max_key`: type 1's full count runs over the h2v keys (A ran over `numHyperedgeKeys`);
the wedges reach only live keys (their h2h rows). The full counts of the other patterns run over
`views.h2h.max_key` keys (A's), the frontier flags have `views.h2h.max_key + 1` entries and are
reserved before the commit for `max_bound + insertions + 1` (the line graph's keys after the batch
are at most that). A test makes the bound go down and checks the four patterns against the oracle.

### 4. The workspace (I9)

`triad_cuda_workspace` (pooled, ADR 0015): the update's offsets and CUB storage (sized for the
larger phase), compute()'s own offsets and CUB storage (released at its end), the deleted ids, the
frontier keys and flags, 2 x 30 + 1 bins, and a pinned buffer for the read-backs. Everything an
update needs after the commit is reserved in count(-), so the algorithm phase allocates nothing
(conformance check C8 in budget builds: no allocation, host synchronizations within the budget).
The engine's per-call state lives in the problem (no heap allocation per update).

### 5. Host synchronizations

Per update on cuda: one per phase that has sources (its wedge count, A's read point in
`countWedgeShape`) and one for the bins (A's `readTo`): at most 3, and 1 for type 1 (its key kernel
needs no wedge count); `algorithm_budget` and the kit's `host_sync_budget` declare 3 / 1. A's
update (subtract + maintain + add) made 90-96 blocking calls per batch (M9a step 2's API traces);
per phase its counting alone blocks in the flags' and ids' uploads, the allocations and frees, the
scan, the wedge count and a device synchronization, and once more for the bins.

### 6. Deviations from A, each with the same results

1. The frontier flags of type 1 are not built (its key kernel never reads them).
2. Type 1's full count runs over the h2v view's keys (section 3); types 2 and 3 over the line
   graph's keys (A's `numHyperedgeKeys`, equal to the line graph's keys).
3. The subtract and add bins are kept apart (A accumulated both into one array), which gives
   `triads_removed` / `triads_added` and checks the signs of both phases.
4. The deleted ids are uploaded once and turned into keys and flags by a kernel; A built a host
   flag vector of `numHyperedgeKeys + 1` bytes and uploaded it with the ids, per phase.

### 7. Parity of every commit

After every commit of the port, `parity/compare.py triad_count --backend cuda --mode by_key
--counter algorithm` on make-test, syn1m_b50k_s1 and coauth_b50k_s1 (all four patterns; the
reduced Release build `-DDYNG_ALGORITHMS=triad_count` of the parity-cuda preset): 68/68 cases at
c1-c6 (the records and the commit table: `parity/results/M9a.md` section 3).

## Consequences

- The engine's operations are the form M9b publishes (`experimental::count_local_patterns`); the
  host engine of step 4 runs the same operations (the host recount already does, through a host
  bins handle).
- The mutations of step 5 act on code both engines share: the exactly-once predicates
  (`ownership.hpp`), the operations, the bins' sign and the key bound of type 1.
- `triad_sass.py` keeps the SASS and resource readings comparable across commits and against A.
