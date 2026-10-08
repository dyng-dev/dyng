# ADR 0038: The port of ESCHER-GPU's CBST store

- **Status:** Accepted under delegation (2026-10-07; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8a step 3 (PLAN 6.3 step 5, 6.4.5 step 1, 4.2, 4.3, 4.6 rule 7, 4.7,
  Appendix B; the M8a acceptance criteria 9, 10(a), 14, 15 and 16). It changes no rule the author
  approved and weakens no comparison: ESCHER-GPU's traces are compared in the `exact` field set
  of ADR 0037 (more than criterion 10(a) asks), and the one documented divergence is recognized
  by a precise rule, never by a tolerance.
- **Date:** 2026-10-07
- **Deciders:** the AI assistant, on the author's behalf (M8a, step verbatim-port)
- **Number:** 0035 is the 0.2.0 release branch's, 0036 the trace format's, 0037 the host model's.

## Context

PLAN 6.4.5 merges two CBST stores, ESCHER-GPU@abcf9b2 (A) and MOSP_ESCHER@4b86159 (B), in
stages: first A's store is ported verbatim, "with only the mechanical changes of Section 6.3",
and proven by trace replay against A; then B's behaviours are added one at a time (M8a steps 4
and 5). The store is the riskiest code of the project (A's `operations.cu` alone is 830 lines of
host code driving 40 kernels). A reviewer must be able to see that the port is A's code, and
every later commit must be checkable on its own.

## Decision

### 1. Placement and internal API

- `cpp/src/hypergraph/escher/` (PLAN 4.2), namespace `dyng::detail::escher`, in the new OBJECT
  module `dyng_hypergraph` (PLAN 7.2; `cpp/CMakeLists.txt`): the CUDA sources only with
  `DYNG_ENABLE_CUDA`, the host files (`flatten.cpp`, `store_policy.cpp`) always. Nothing is
  public: the hypergraph container of M8b is the only intended caller; the tests include the
  private headers (as the graph module's tests do).
- `cbst_store` (A's `CBSTOperations`, PLAN Appendix B): `cbst_store(res, name, capacity,
  alignment = 4, policy = store_policy::escher_gpu())`, `construct`, `rebuild_from_device`
  (taking the row offsets and the payload as `buffer<int>`), `insert` (returns
  `insert_mapping`), `append` (= fill), `erase`, `remove_values` (= unfill), `find`,
  `device_bytes`, `bin_launches`, `host_syncs`, `poisoned`, `policy`, `context` (the device
  pointers, for the kernels of later modules), and the dynG additions `clone`, `download`,
  `allocated_bytes`. The free functions of A (`construct_cbst`, `insert_cbst`, ...) stay, as in A,
  the bodies of the members.
- `store_policy` (`store_policy.hpp`, host-only): growth (`escher_gpu`: `max(needed + needed/4,
  3 * capacity / 2)` in integers, at most `INT_MAX`; `none`), compaction (`never`), the surplus
  alignment, and `initial_capacity()` (A's `payloadCapacity`: in-place trees `max(requested,
  size + size/8 + 1024)`, merged trees their exact size) for the container. The store grows
  through the policy. The replay passes the capacity of the trace's tree record, as A's
  `DynamicHypergraph` passed its own choice (B's traces need theirs).

### 2. The commit sequence

| commit | class | what changes |
|---|---|---|
| c1 transplant | renames, namespace, headers, CMake | A's files assembled and renamed by `parity/escher_port/transplant.py` from `rename_map.toml` (every renamed identifier), the manual hunks of `manual_hunks.patch` |
| c2 | errors | exit(-1) / exit(2) / printf / verbose() / silent returns |
| c3 | memory | `buffer<T>` on the resources' memory resource, pooled temporaries |
| c4 | streams | the resources' stream, `par_nosync`, A's read points |
| c5 | registration and counters | `DYNG_REGISTER_KERNEL`, profiler stages, per-store bin counters |
| then | one dynG addition each | the empty tree; clone; download; `allocated_bytes` |
| then | policy | `store_policy` and `escher_gpu()` |

**The transplant is reproducible.** `python3 parity/escher_port/transplant.py --check --rev <c1>`
regenerates c1 from `git show abcf9b2:<path>` of the original's clone (read only) and compares
(pytest `test_escher_port.py`). Its steps: assemble each target from parts (whole files or line
ranges: `locator.cu` takes the four row-gathering kernels of `kernel/delta_merge.cu`, `find.cu`
the `readFlat` of `kernel/motif_utils.cuh`), rewrite the includes, apply the context rules (the
store's `fill`/`unfill` members become `append`/`remove_values`; the counters keep their field
names), replace every name of the map as a whole token (also in comments), `#pragma once`, the
namespace, the SPDX and `// Derived from ESCHER-GPU@abcf9b2:<path>` header, clang-format. **The
manual hunks of c1** (`manual_hunks.patch`): `#include <cstddef>`, `<utility>` and the
`#if defined(_OPENMP)` guard of the pragma in `flatten.cpp` (A's build always had OpenMP; dynG's
tsan presets do not, and `-Wunknown-pragmas` is an error there); `flatten.hpp` (A declared
`flatten2DVector` in its `utils.hpp`); `transplant_shims.hpp` with A's `verbose()` (off, as in
every traced run) and `printVector`, removed by c2.

### 3. Errors (c2)

- `checkCuda`'s `exit(-1)` becomes `DYNG_CUDA_TRY` (`DYNG_CUDA_TRY_NO_THROW` in destructors);
  `ensurePayloadCapacity`'s `exit(2)` becomes `capacity_error` (now in
  `store_policy::grown_capacity`); `flatten_rows` throws `capacity_error` when its 64-bit total
  exceeds `INT_MAX` instead of truncating the offsets; the locator's `exit(2)` on chained rows is
  `DYNG_FAIL` (a broken invariant of the caller, the library).
- **A construct with a capacity below its payload throws `invalid_argument_error` before
  anything is allocated.** A printed "Overflow" and kept a tree with records but no nodes (any
  later operation on it was undefined); B throws its `EscherError`. The replay reports it as the
  documented divergence `construct_below_payload` (rule: op `construct`, records > 0, capacity <
  payload, the store threw `dyng::invalid_argument_error`); the tree is not compared after it.
  It occurs once in the golden sets: A's targeted scenario `small_capacity`, and the same in B's
  (where the host model under `escher_gpu()` reports a status mismatch: one class).
- `verbose()` blocks and `printf` become the log sink (trace and debug level; the copies they
  need run only at that level). The printing debug kernels go: `find()` returns the heap position
  of every key's node.
- **Poisoning** (PLAN 4.7.3, basic guarantee): an operation that fails after it began (any
  exception after the validation) marks the store poisoned; every later operation and `clone()`
  throws `invalid_argument_error` until `construct` or `rebuild_from_device`. A tree that was
  never constructed rejects operations the same way.
- No kernel of A detects a contract violation, so the device error word has no user in this
  step; B's bounded chain walker (step 4) will raise it. (Step 4, ADR 0039: the store's control
  block holds the word; broken chains and items on dead or absent keys raise it.)

### 4. Memory (c3) and accounting

Every array of the context is a `buffer<T>` from `res.memory()` on `res.stream()` (the device
pointers of `cbst_context` are views of them); A's `TempBuffer` becomes a pooled temporary (a
buffer from the same resource, the stream-ordered pool by default); the scratch is sized exactly
as A's `reserveScratch`. `device_bytes()` keeps A's formula (`num_records x (56 + 16)` + the
payload capacity + `5 x scratch items + scratch payload` ints x 4), the scope of the memory gate;
`allocated_bytes()` (dynG) sums the buffers in the same scope. A test under a counting memory
resource shows that after every kind of operation the live bytes equal `allocated_bytes()` equal
`device_bytes()` (temporaries are released when an operation returns). `cbst_node` keeps A's
layout (56 bytes), on which the formula depends.

### 5. Streams and host synchronizations (c4)

Every kernel, copy and memset is ordered on `res.stream()` (`DYNG_CHECK_KERNEL` after the
launches), Thrust runs as `thrust::cuda::par_nosync(alloc).on(stream)` with its temporary
storage from the resources' memory resource (`util/thrust_allocator.cuh`: a Thrust allocator over
`memory_resource_ref`; Thrust takes an allocator, not a CCCL memory resource, so the
`cccl_memory_resource.cuh` adapters are not needed), the device of the resources is made current
by every operation, and nothing synchronizes the device. The host waits for the stream only at
A's read points, each counted (`host_syncs()`, and the budget counters):

| operation | read points (dynG synchronizations) | A's blocking calls on the same path |
|---|---|---|
| construct | 0 | 3 (`cudaMemcpy` uploads; plus implicit `cudaMalloc` synchronization) |
| rebuild_from_device | 0 | 7 (6 `cudaFree`, `thrust::sequence`) |
| append (fill) | 1: the total appended | 2 (`inclusive_scan`, the total); +1 with growth (`cudaFree`) |
| erase, remove_values | 0 | 0 |
| insert | 1: D (0 on an empty tree); with reuse +1: the matched count (`count_if`); with matches +1: the matched pairs and slot keys (one synchronization for the three copies); with surplus +1: the surviving count (`copy_if`), +1: the last surviving key | 1 (D) + 6 with reuse (2 `sequence`, 2 `sort_by_key`, `inclusive_scan`, `count_if`) + 3 with matches + 7 with surplus (`copy_if`, the last key, 5 `cudaFree`) |

`CbstStoreSyncs` checks the counts of every path and that they never exceed A's. **The tie rule:**
A's `thrust::sort_by_key` on `int` keys is Thrust's radix sort, which is stable, and the best fit
depends on it; the port says so explicitly with `thrust::stable_sort_by_key`: items of equal size
keep their input order, slots of equal capacity the order of the order-statistic walk (key order).

### 6. Registration and counters (c5)

`DYNG_REGISTER_KERNEL` for every kernel (`resources::warm_up()` loads them); the operations are
profiler stages `hypergraph.cbst.{construct, rebuild, insert, append, erase, remove_values}`
(they nest under the container's `hypergraph.apply.*` of `parity/timed_regions/hypergraph.toml`);
A's process-global `DegreeBinLaunches` becomes per-store counters (`bin_launches()`), which the
replay compares per operation.

### 7. The dynG additions

- **The empty tree:** construct with 0 records allocates the payload; insert reads no D and
  builds the tree from its surplus rows; the level reduction of an empty tree does nothing (A
  launched 0 blocks); append, erase and remove_values change nothing. (Step 4, ADR 0039 section
  5: append and remove_values items name absent keys there, and throw `internal_error`.)
- **Clone:** an independent deep copy on any resources (event-ordered with the source's stream
  both ways); the nodes' absolute pointers are re-wired (`rewire_tree_kernel`).
- **Download:** the state on the host, the pointers as heap positions; the replay adapter reads
  the store through it.
- **`allocated_bytes()`** (section 4).

### 8. The replay

The device store is a store of the comparator of ADR 0037 (`cbst_device_adapter`, library
`dyng_cbst_device`): every tree a `cbst_store` under `escher_gpu()`; after every operation its
state is downloaded and described exactly as the recorder describes the originals' (an in-order
walk of the child pointers from the root, the rows along their chains), and the store's
integrity checker (A's `checkTreeRows`, ported) runs on the rows as read: the comparator compares
its verdict with the trace's `check.rows` (after the state, so that a first divergence names the
state field). `dyng_cbst_replay --store device`, `parity/compare.py cbst_trace --store device`
(on GPU `--gpu`, default 1).

- **A's sets** under `escher_gpu()` must match in the `exact` field set: InsertMapping, rows in
  order, avail, subtree_avail, length, size, offsets (value, tail base), occupancy, tail
  capacity, segments, the bump pointer, the capacity, the record count, the scratch capacities,
  D, `device_bytes()` = A's `deviceBytes()`, the per-op degree-bin launches, the exception, the
  integrity verdict, the state before every op.
- **B's sets** run under `escher_gpu()` (the store has no B preset yet): every tree must diverge
  at exactly the op and in exactly the field where the host model under `escher_gpu()` diverges
  (the behaviour differences of ADR 0037: B's caller counts, compaction, validation and
  `compact()`), and nowhere else (`first_divergences` in `parity/cbst_trace_goldens.py`; the
  gtest `MospEscherTracesDivergeWhereTheModelDoes` for the committed traces).
- `parity/cbst_merge_steps.sh <commit>...` builds any commit's replay targets in a temporary
  worktree and re-runs its replays; its rows are the merge-step table of parity/results/M8a.md.

### 9. Robustness

compute-sanitizer memcheck (with leak check), racecheck, synccheck and initcheck are clean on
`dyng_cbst_store_tests`, `dyng_cbst_device_tests` (one small replay of each original) and
`dyng_cbst_coverage_tests`; `ci/gpu_local.sh` runs them (step `cbst`) with the device replays of
the small sets. Racecheck sees shared memory only; the global-memory hazard of the originals
(two items naming one row in one fill or unfill) is undefined in both and in the host model
(ADR 0037) and occurs in no trace. Two stores on two non-blocking streams, interleaved without
host synchronization, equal a sequential run; a replay run twice gives identical digests.

## Later changes

M8a step 4 (ADR 0039) merges MOSP_ESCHER's behaviours on top of this port, one per commit, and
lists every deviation from B's literal code there (section 2): the construct validation, the fill
clamps (dynG clamps the occupancy, B the room), the caller row counts (never B's `nullptr`), the
chain walk bounded by the used payload instead of B's 1,024 hops, the integrity checks, `rows()`,
the scratch rule, `compact()` (growth or `capacity_error` instead of `EscherError`), the storage
counters (no process-global mempool setting) and the rule for dead or absent keys. A's traces
still match in the `exact` field set after each of them.

## Consequences

- The store is A's code, checkable line by line against the original; every later commit is one
  class of change, and the merge-step table shows the replays after each.
- A's traces match in every recorded field; B's traces diverge only where the host model says the
  originals' behaviours differ. Steps 4 and 5 add B's behaviours on top, with B's preset.
- The memory gate of M8 (store accounting) is met by construction and measured on coauth-DBLP
  (parity/results/M8a.md, section 11).
- `host_syncs()` and `bin_launches()` are per store; nothing of the store is process-global.
