# ADR 0039: The staged merge of MOSP_ESCHER's CBST behaviours (part 1: chain bound, integrity, scratch growth, compaction)

- **Status:** Accepted under delegation (2026-10-07; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out M8a step 4 (PLAN 6.4.5 step 2, 4.7.3, 8.4; the M8a acceptance criteria
  10, 11 and parts of 13 and 16). It changes no rule the author approved and weakens no
  comparison: ESCHER-GPU's traces are still compared in the `exact` field set at every commit,
  and MOSP_ESCHER's traces are judged tree by tree against the host model under the same preset.
  Step 5 (automatic compaction, `mosp_escher()`, dynG's default) extends this ADR.
- **Date:** 2026-10-07
- **Deciders:** the AI assistant, on the author's behalf (M8a, step merge-invariant)
- **Number:** 0035 is the 0.2.0 release branch's, 0036-0038 this milestone's.

## Context

PLAN 6.4.5 step 2 adds MOSP_ESCHER@4b86159's (B) behaviours to the port of ESCHER-GPU@abcf9b2's
(A) store (ADR 0038) one at a time, replaying both originals' traces after each. This first part
merges the behaviours that change no trace of A: construct validation, the fill clamps, the
caller's row counts, the bounded chain walk, the integrity checks, the `find.cu` and
`delete_avail.cu` differences, scratch growth, an explicit `compact()`, the accounting, and a rule
for dead or absent keys. B's automatic compaction (the trigger points inside fill and insert),
its preset and dynG's default preset are step 5.

## Decision

### 1. One behaviour per commit, judged by a staging preset

- **The staging preset `store_policy::mosp_escher_staged()`** (`store_policy.hpp`) is B's
  behaviours merged so far: B's fixed capacity (`growth_rule::none`: where B compacts, the store
  throws `capacity_error`), no compaction of its own, alignment 4, from (g) on B's scratch growth.
  It exists for the merge steps; step 5 replaces it with `mosp_escher()`.
- **The comparator** resolves the same name to the host-model policy with the same behaviours
  (`staged_host_policy()` in `cpp/tests/parity/escher/cbst_compare.cpp`: A's behaviour plus,
  commit by commit, B's validation, clamps, caller counts, scratch growth and `compact()`, with
  dynG's exceptions). B's traces replayed under the staging preset must diverge in exactly the
  trees, ops and fields where the host model under it does (ADR 0037's first-divergence rule;
  `compare.py cbst_trace --store device --policy mosp_escher_staged`, the gtest
  `MospEscherTracesDivergeWhereTheModelDoes`), so every remaining divergence is caused by a
  behaviour the store has not merged. `parity/cbst_merge_steps.sh` runs B's sets under both
  `escher_gpu()` and the staging preset from the commit that adds it, and lists each set's first
  divergences by field.
- **A's traces** stay under `escher_gpu()` and must match in the `exact` field set (offsets,
  occupancy, tail capacity, `device_bytes()` = A's `deviceBytes()`, the per-op degree-bin
  launches, the integrity verdicts) at every commit; the one documented divergence of ADR 0038
  (`construct_below_payload`) is unchanged.

At the end of this part B's small sets under the staging preset diverge in 18 trees of
`mosp_escher_test` (11 `test_cbst_smoke` constructs without row counts, `key.occupancy`: B's
legacy occupancy 0, which dynG never takes over; 7 trees at their first automatic compaction,
`status`: step 5) and 4 trees of `mosp_escher_targeted` (the two automatic compactions, B's
capacity throw after its compaction, the construct below its payload); before the merge they
diverged in 50 and 7 (34 + 4 more at their first scratch growth). The table of every commit is
in `parity/results/M8a.md`, section 12.

### 2. The behaviours and every deviation from B's literal code

| | behaviour | B's source | dynG | deviation from B's literal code |
|---|---|---|---|---|
| (a) | construct validation | `escher/structure/operations.cu:196-211` | `validate_construct` before anything is allocated or changed (the store is not poisoned) | `invalid_argument_error` / `capacity_error` instead of `escher::EscherError`; dynG adds a `capacity_error` for a record count whose operation scratch (3 ints per record) exceeds the int32 indexing; B's device synchronizations after the construct kernels (`:259, 263, 274`) are not taken over |
| (b) | fill clamps | `escher/kernel/payload.cu:123, 178, 241` | the three fill kernels use the occupancy clamped to [0, tail capacity] | B clamps only the room left (`max(0, tailCapacity - occupancy)`), which keeps the write position (and an overflow's chain pointer) past the segment; dynG clamps the occupancy, so both stay in the segment; the warp and block unfill kernels clamp it the same way (A and B read `occupancy` slots there unclamped). B's unused `tailBase`/`tailCap` locals of the warp kernel are not ported |
| (c) | caller row counts | `escher/structure/operations.cu:123-132, 268-275`, `include/structure.hpp` | `construct(..., row_counts)`, a fast path next to the scan (`init_tail_metadata_kernel`), which stays the default | B's `nullptr` (occupancy 0, its "legacy behaviour") is not taken over: the overload rejects a null array; every count must fit its row (`invalid_argument_error`); a Debug build checks the counts against the scan (`internal_error`; an extra synchronization in Debug builds only); B's synchronization after `setInitialOccupancy` is dropped. Every construct of every replayed trace is built both ways and the node arrays are compared byte for byte (the device adapter's twin check; `twin constructs` in the records) |
| (d) | the bounded chain walk | `escher/kernel/find.cu:9-20` (`readFlat`) | `chain.cuh`: `walk_chain` (host and device) for the row locator, the integrity checker, the row reads of `find` and the unfill writer `unfill_row_serial`; `warp_walk_chain` for the compaction | **the bound** (section 3) and the **error**: B returned `INT_MIN` (the row silently ended); dynG raises the store's device error word (section 4) |
| (e) | integrity | `escher/structure/integrity.cu`, `include/integrity.hpp` | `check_tree_rows` (A's call sites: the device replay, the tests) returns `long long`; order-insensitive rows sorted on both sides (`:131-134`); checked copies (`DYNG_CUDA_TRY` since c2); `check_subtree_avail` (`:218-243`) | new in neither original: `check_tree_index` (avail flags 0/1, the keys array strictly increasing and holding each rank's key, the row offsets array each node's first segment, the heap layout of the pointers). The replay compares `check.rows` with the original's verdict, `check.subtree_avail` with B's recorded one (0 for A's states) and `check.index` with 0, after every op |
| (f) | `find.cu`, `delete_avail.cu` | `escher/kernel/find.cu` (`findContents`), `escher/kernel/delete_avail.cu:33-36` (a comment) | `cbst_store::rows(keys)`: the rows of keys, read through `chain.cuh`; B's comment | `findContents` printed the rows; `rows()` returns them (absent and deleted keys: none) and throws `internal_error` for a broken chain |
| (g) | scratch growth | `escher/structure/operations.cu:139-165` (`ensureScratchCapacity`) | `store_policy::scratch_rule`: `exact` (A's `reserveScratch`) under `escher_gpu()`, `geometric` (B's +n/2+1 steps from 1 item / 4 values) under the staging preset | computed in 64 bits and capped (B's `int` loop overflowed near `INT_MAX`); only the arrays that grow are reallocated (B reallocated all four). **The rule for dynG's default (step 5): geometric** (amortized reallocation in long update streams; the scratch is small next to the payload, and the memory gate is read under `escher_gpu()`) |
| (h) | explicit `compact()` | `escher/structure/operations.cu:289-339` (`compactCBST`), `escher/kernel/payload.cu:338-409` (`warpWalkRow`, `compactSizesKernel`, `compactCopyKernel`) | `compact.cu`, `compact_cbst`, `cbst_store::compact()` | the warp walk is bounded (section 3); compacted rows that need more than the capacity (a live row of L values takes (L+4)/4*4 slots, more than a surplus row's) grow the payload by the policy, or throw `capacity_error` before anything changed (the store is not poisoned; B threw `EscherError`); the copy goes through a pooled temporary of the store's memory resource (`compaction_temp_bytes()`), not B's process-global pool; its one host synchronization reads the compacted size. Under `escher_gpu()` the replay reports a traced `compact()` as unsupported (A had none) |
| (i) | accounting | `hypergraph/src/DynamicHypergraph.cpp:145-156` (`escherDeviceBytes`); `escher/structure/operations.cu:57-69` (`TempBuffer`) | `device_bytes()` keeps A's scope (the memory gate's); `stats()`: the payload capacity and use, growths, compactions, the largest compaction temporary, B's scope (`mosp_escher_bytes`: the payload capacity and the per-record arrays, without the operation scratch), the scratch bytes, host synchronizations, the degree-bin paths, the device path counters | B's process-global mempool release threshold is not taken over (a test checks the default pool's attribute is unchanged) |
| (j) | dead or absent keys | (no rule in either original) | section 5 | dynG's |

### 3. The chain bound

B bounded `readFlat` at a literal 1,024 hops and then treated the row as ended. dynG does not port
the literal bound:

- Under `escher_gpu()` a V2H row gains a segment for every fill that overflows its tail and is
  never compacted, so a legitimate row can have more than 1,024 hops; B's walk would truncate it
  silently (the test `TwoThousandOverflowingAppendsReadBackCompletely`: 2,000 hops, read back
  completely by the integrity checker, the row locator, `rows()` and the unfill writer).
- **The bound is the used payload:** a well-formed row reads each of the `initial_payload_size`
  used slots at most once, so a walk that reads more slots than that, or leaves [0, used), is a
  broken (cyclic or dangling) chain. It never hangs and never returns a truncated row.
- **Evidence that B's bound never triggers in B's traces:** the longest chain of every trace of
  both originals is 98 hops (`escher_trace.py summary`, `max_chain_hops`: `mosp_escher_test` 98,
  `mosp_escher_targeted` 4, the DBLP runs 3; A's sets 3-4; `parity/results/M8a.md`, section 5).
  So the literal bound is not observable in B's own traces (the stop condition of the step does
  not apply) and replacing it changes no trace.

### 4. The device error word (the control block)

PLAN 4.7.3's sticky device error word gets its first users in the store: a few words on the device
(`cbst_context::d_control`: the `device_error` bits, row-walk overruns, rejected items, a path
counter of serial unfills) and their pinned host mirror. Every read point of the store queues the
copy of the words before its synchronization and throws after it if a kernel raised an error
(`internal_error`, with the counts): the error of a walk or an item surfaces at the operation's
own read point (append, insert, compact, rows) or, for the operations without one (erase,
remove_values), at the next operation's or at the new `check_errors()` (one synchronization; the
container's end of an update). The store is then poisoned. `download()` does not check. The words
are outside A's accounting scope: `device_bytes()` and `allocated_bytes()` do not count them.

### 5. Appends and removals on dead or absent keys

Neither original defined it: an item naming an absent key was skipped silently, one naming a
deleted key (which keeps its node) wrote to or compacted its row like a live one. **dynG skips the
item, counts it and raises the device error word** (`lookup.cuh`); the next read point throws
`internal_error` (the hypergraph container never names such a key: it is a broken invariant of
the caller), and the store is poisoned. Erase keeps both originals' behaviour (absent keys
ignored; a deleted key erased again stays deleted). An empty tree's append and remove_values name
absent keys and therefore throw now (ADR 0038 had them change nothing). No item of the four small
sets names a dead or absent key (0 of 1.87 M fill, 1.84 M unfill and 0.29 M erase items).

### 6. The other differing files (behaviour (k))

| file | difference | disposition |
|---|---|---|
| `include/structure.hpp` | B: `rowOccupancy`, `compactCBST`, `scratchKeysCap`/`scratchPayloadCap`; no `rebuildFromDevice`, `deviceBytes`, `DegreeBinLaunches` | merged into `cbst_store.hpp`: the counts overload (c), `compact()` (h); A's members kept; the scratch fields keep A's names |
| `include/integrity.hpp` | B: `long long`, `checkSubtreeAvail`, Doxygen | merged (e) |
| `utils/flatten.cpp` | B: sequential, `int` totals | not needed: A's parallel 64-bit version with the 2^31 check (PLAN 6.4.5 step 3, ADR 0038) |
| `kernel/device_utils.cuh` | B: `atomicAdd_sll` (a CAS loop for signed 64-bit atomics), comment wording | **not ported**: `atomicAdd_sll` is defined but called nowhere in B's tree (`git grep` at 4b86159), and CUDA has a native `atomicAdd(unsigned long long*)`; the rank formula is identical |
| `kernel/kernels.cuh` | B: no `initTailMetadata`, the compaction kernels | union: A's declarations kept, the compaction kernels added (h) |
| `kernel/build_tree.cu` | B: no `initTailMetadata`, a comment, `placeNodeRecords` moved | A's kept (the scan is the default); B's `setInitialOccupancy` added here (c) |
| `utils/binning.cpp` | `using std::size_t` | not needed: dead code in both (`binByDegree` is called nowhere; ADR 0038) |
| `include/printUtils.hpp`, `utils/printUtils.cpp` | A: `setVerbose`/`verbose()`; B: without them | not needed: debug printing, replaced by the log sink (ADR 0038, c2) |
| `include/escher_errors.hpp` | B only: `EscherError`, `ESCHER_CHECK_CUDA` | not needed: dynG's error classes and `DYNG_CUDA_TRY` (PLAN 4.7.3) |
| `kernel/insert_reuse.cu`, `kernel/unfill.cu`, `include/binning.hpp` | identical | nothing to merge |

## Consequences

- Every behaviour of B that does not change A's traces is in the store, each in its own commit
  with its replays (`parity/results/M8a.md`, section 12); A's traces match in every recorded
  field at every commit; B's traces diverge only at their first automatic compaction (step 5) and
  at the constructs without row counts.
- **Handoff to step 5:** `mosp_escher()` replaces the staging preset (B's trigger points: a fill
  whose overflow segments do not fit compacts and refills the rest once; an insert whose surplus
  rows do not fit compacts first; then `capacity_error`); B's `escher::EscherError` must be
  mapped to the dynG exception of each throw site in the comparator; the 11 `test_cbst_smoke`
  constructs with `rowOccupancy = nullptr` need a documented divergence (B's legacy occupancy 0 is
  never taken over), or criterion 10(b)'s "no divergence" cannot hold for them.
