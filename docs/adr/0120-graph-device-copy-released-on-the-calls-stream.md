# ADR 0120: A mutating call releases the graph's device copy on its own stream

- **Status:** Accepted under delegation (2026-11-13; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. A correctness fix of 0.1-0.3 code, found by M10b step 4
  (compute-sanitizer memcheck on a `parity-cuda` build). It applies the rule of ADR 0016 item 10
  (arrays replaced through a handle on another stream are moved to the updating stream before
  they are released) to the graph's resident device copy. Its reading follows HD16, **decided by
  the author on 2026-10-28** (GOVERNANCE.md, approvals log, "The measurement protocol and gates of
  0.4 (sheet B1-B6)": ADR 0024 at 1.02; a region above the bar goes to the author for
  acknowledgement on the D14 path, never a silent pass). It changes no rule the author approved:
  no public signature, no documented guarantee, no gate, protocol or parity rule. **Amended on
  2026-11-18 on the merged code** (after M10b merged): the same rule for the slack rows' device
  growth (ADR 0103), under the same delegation ("The merged code" below). The `end_to_end`
  excursions of both readings were **acknowledged by the author on 2026-11-18** on the D14 path
  (GOVERNANCE.md, approvals log, "D14-path acknowledgement of ADR 0120's ADR 0024 reading").
- **Date:** 2026-11-13 (amended 2026-11-18)
- **Deciders:** the AI assistant, on the author's behalf
- **Builds on:** ADR 0016 (CUDA execution resources, item 10), ADR 0020 (the resident device graph
  and one Step 0), PLAN Section 4.7.4 (threading: the caller orders calls on different streams),
  ADR 0024 (dynG against dynG), the M8b review's hypergraph fix (the test
  `HypergraphCuda.ApplyOnAnotherStreamIsOrderedAfterThatStreamsWork`).
- **Number:** 0120, outside ADR 0063's blocks of 0.4 (M10 0063-0079 and 0100-0109, M11
  0080-0099, the release line 0110-0119), so that this fix, on its own branch from `ae5fc47`,
  clashes with no track.

## Context

**The finding.** M10b step 4 (2026-11-13) ran `compute-sanitizer --tool memcheck` on the
`parity-cuda` build (Release, `DYNG_CUDA_DEBUG_SYNC` off) and got 20 errors "Use-after-free on
allocation of size 800 bytes ... potentially accessed after it is free'd" in
`dyng_cycle_count_conformance_cuda_tests --gtest_filter='*C12*'` (C12: a device chain whose
graph and result are built with resources on stream a, then `update()` alternating resources on
streams s and a). The same 20 errors occur on `ae5fc47` (origin/main before M10b), so the defect
predates M10. The `dev-cuda` and `sanitize-cuda` builds hide it, and `ci/gpu_local.sh`'s memcheck
step runs on `dev-cuda`.

**The buffers and the ordering.** Reproduced on `ae5fc47`
(`$DYNG_SCRATCH/runs/streamfree/repro-ae5fc47.log`): 18 errors in `v32_e32_w32`, 2 in
`v32_e64_w32`, none in the unweighted cases; the arrays are the row offsets (244 bytes, 61
`int32`; 488 bytes, 61 `int64`) and the neighbours (800 bytes, 200 `int32`) of the kit's medium
graphs. Both backtraces name the parts:

1. A graph state's device copy (`device_graph`) holds `buffer`s allocated on the stream of the
   call that built them: the upload (`graph_impl::device_edges()`), the in-edges, a device merge.
   `buffer::release_storage()` frees on the buffer's own stream (`cudaFreeAsync` through
   `cuda_async_memory_resource::deallocate`).
2. cycle_count's CUDA update counts the cycles through the deletions on G_t in `before_apply`
   (`count_minus_cuda()` -> `cycle_count_cuda_phase()` -> `run_phase()`): the item counts, a scan,
   a read-back that synchronizes, then `count_owned_cycles_kernel` **launched asynchronously** on
   the stream of the call, reading G_t's `out_row_ptr` and `out_col_ind`.
3. The commit, `graph_access::apply()`. A graph with weight columns takes the host apply
   (`apply_batch_host`), then `drop_device_edges()` releases G_t's copy, each array on the stream
   it was allocated on. Nothing orders that free after the kernel of step 2 when the streams
   differ. In C12 the first update runs on s while the copy was uploaded on a (by `compute()`);
   the next update runs on a while G_{t+1}'s copy was uploaded on s (by the insert phase of the
   update before), and so on.
4. A graph without weight columns under set semantics takes the device merge
   (`apply_set_batch_device`), which synchronizes the stream of the call at its end, before
   `replace_device_edges()` releases G_t's copy: ordered, no error.

With `DYNG_CUDA_DEBUG_SYNC` the phase's `DYNG_CHECK_KERNEL` synchronizes the stream after the
kernel, so it completes before the commit and nothing is reported.

**What it could do.** The default memory resource is a stream-ordered pool: memory freed on a
stream is available at once to the next allocation on that stream (and, by the pool's
opportunistic reuse, to other streams once the free has run). A program that updates a graph
through handles on two streams can thus have cycle_count's kernel read neighbours another
allocation has overwritten: wrong counts, without an error. A program on one stream (every gate,
every parity run, the Python package's default handle) is unaffected: there the copy's stream is
the call's.

**The other release sites (the audit).** memcheck over C12 of every CUDA conformance suite of
the `parity-cuda` build (cycle_count, dynamic_bfs, hyper_sssp, label_propagation, mosp, sssp,
triad_count, triangle_delta, and the hypergraph kit) reported errors in cycle_count only
(`audit-base/`). sssp, mosp and the others read G_t's copy only after the commit (their engines
run on G_{t+1}). By review:

- the slotted rows' device copy (ADR 0074) is changed in place by the device apply;
  `drop_slotted_device()` runs only on the host vertex-batch path, where the call queues no
  device work, and its batch arrays are resized only by Step 0 at the start of a call
  (`device_slotted::batch_current`). This holds for `ae5fc47`; the slack rows that M10b added
  replace their arrays inside a call ("The merged code" below);
- the device apply's and the normalized batch's scratch, and `scratch_buffer`, live in pooled
  workspaces whose leases are fenced across streams (ADR 0016 item 10), and are reallocated
  before their first use in a call;
- results already move their arrays to the updating stream on vertex growth (sssp, mosp,
  dynamic_bfs: `buffer::set_stream()`, ADR 0016 item 10);
- the hypergraph's store works on its own stream, ordered after the call's stream (the M8b
  review's fix).

## Decision

1. **`graph_impl::drop_device_edges(res)` and `replace_device_edges(res, next)` take the
   resources of the mutating call** (`graph_access::apply()` passes its own). When `res` is a CUDA
   handle of the copy's device, every array of the copy that is dropped or replaced is moved to
   `res.stream()` first (`device_graph::release_on()`, which calls `buffer::set_stream()` on its
   seven arrays), so its stream-ordered release follows the work the call queued on that stream,
   a participant's kernel on G_t among it. Work of earlier calls on other streams is ordered
   before this call by the caller (PLAN 4.7.4), so the call's stream is the one the release has to
   follow. This is ADR 0016 item 10's rule for results, applied to the graph.
2. **Not an event and not a synchronization.** Recording an event on the call's stream and making
   the copy's stream wait for it would enqueue work on a stream the call does not own (the
   library orders work on the call's stream only, ADR 0016; the caller may have destroyed the
   older stream), and a synchronization of the call's stream before the drop would add a host
   synchronization to every CUDA update of a weighted graph (sssp, mosp; their C8 budgets). A
   general change of `buffer`'s release (rmm::device_uvector's semantics, documented in its
   header) is not needed: the owner that releases knows the call.
3. **On one stream nothing changes**: `set_stream()` assigns the stream the arrays already have.
   No kernel, launch, synchronization or allocation is added, and no result changes.
4. **The test.** `GraphStreamOrder.HostApplyReleasesGtsCopyAfterTheCallsStream` and
   `GraphStreamOrder.DeviceMergeReleasesGtsCopyAfterTheCallsStream`
   (`cpp/tests/graph/stream_order_cuda_test.cpp`, in `dyng_graph_cuda_tests`, label gpu) queue a
   delayed reader of G_t's resident neighbours on s (the hypergraph test's
   `queue_delayed_copy`: about half a second of spinning, then a copy), apply a batch through a
   handle on s to a graph whose copy was built on a, upload G_{t+1}'s copy on a, and compare what
   the reader copied with G_t's neighbours. The host apply launches no library kernel between the
   reader and the free, so the test does not depend on `DYNG_CUDA_DEBUG_SYNC`. Before the fix
   (`ae5fc47` with the test), on `parity-cuda` and on a `parity-cuda` build with
   `DYNG_CUDA_DEBUG_SYNC=ON` alike: the host apply case read 39,918 of 40,000 entries wrongly (the
   upload on a reused the memory) and memcheck reported one use after free (160,000 bytes, G_t's
   `out_col_ind`); the device merge case passed (it synchronizes). After it: both pass, memcheck
   reports no error. `ci/gpu_local.sh` runs memcheck on every gpu executable of `dev-cuda`, so
   the race is now caught there too, which C12 alone could not.
5. **Docs.** `graph::apply()`'s reference entry and `docs/concepts/backends_and_resources.md`
   say that a call releases the memory it replaces in the order of its stream.

## Verification

On `3a0a7aa` (the fix), `parity-cuda` with tests, GPU 1 (shared lock, after `locklog.py
quiet-ok`):

- memcheck over C12 of the nine CUDA conformance suites above: **0 errors** in each
  (cycle_count: 20 before);
- the full CUDA conformance suites of cycle_count, dynamic_bfs, hyper_sssp, label_propagation,
  mosp, sssp, triad_count and triangle_delta, the hypergraph kit and `dyng_graph_cuda_tests`
  pass; on the CPU, `dyng_graph_tests`, `dyng_cycle_count_tests`,
  `dyng_cycle_count_conformance_tests` and `dyng_framework_tests` pass;
- the golden corpora on cuda, as `ci/gpu_local.sh` runs them: sssp with `--configs
  cuda,cuda-operators` ("ALL EQUAL: 495 cases x 2 configurations") and cycle_count's
  `cycle_count_cuda` set with `--configs cuda,cuda:resident` ("ALL EQUAL: 48 replays (24 cases x
  2 configurations, 0 skipped)");
- the portability pre-checks: the Clang 18 syntax pass over the 52 translation units of
  `cpp/src/graph`, `cpp/src/framework`, `cpp/src/algorithms/cycle_count` and the graph CUDA tests,
  and the fortified (`-D_FORTIFY_SOURCE=3`) `cpu-only` build of the library, clean;
- `ci/docs.sh --no-linkcheck` (Doxygen coverage, the API snapshot: 1436 declarations unchanged,
  the Sphinx site with warnings as errors).

The logs are under `$DYNG_SCRATCH/runs/streamfree/`.

## The merged code (2026-11-18)

**The merge.** M10b merged into main on 2026-11-18 (`c67dd40`, dyng-dev/dyng#25). Under the
author's merge order of that day, this branch merged main (`ab24b06`, both sides' rows kept in
GOVERNANCE.md and the ADR index). A review of what main added found one more instance of this
ADR's class, in M10b's slack rows (ADR 0103).

**The instance.** In `apply_vertex_batch_slack_device()`, the call grows two kinds of array.
`grow_device_slack()` (vertex growth) replaces the four per-vertex arrays: the row starts, row
ends, row limits and states, each by a `grown_copy()`. When the radix sort's key bits change, it
also replaces the sort's scratch. `ensure_arena()` (a move to the tail of a full arena) replaces
the slots and their weights. Each new array is filled by a copy of the old one, queued on the
call's stream, and the old array is then released at once on the stream it was allocated on.
Take rows uploaded on a and updated through a handle on s: the old arrays went back to the pool
while those copies were still waiting on s, along with earlier work of s that reads the rows (a
label_propagation engine's kernels, for example).

**The test.** `GraphStreamOrder.SlackDeviceApplyReleasesReplacedRowsAfterTheCallsStream` is in
the same file. It sets up the case step by step:

1. it builds the rows, then uploads and applies them on a;
2. it queues a delayed reader of G_t's row starts and slots on s;
3. it applies a vertex batch through a handle on s: 100 new vertices, and 200 more entries for
   vertex 0, whose row of 3 slots moves to the tail of the full arena;
4. it allocates on a after the apply.

The apply runs as `dyng::update()` runs it: `graph_access::apply()`, with the report settled at
the commit.

On the merged code without the fix, memcheck reports **6 use-after-free errors**, one for each
array replaced:

- the row starts, 80,004 bytes;
- the row ends, limits and states, 80,000 bytes each;
- the slots and the weights, 240,000 bytes each.

Without the sanitizer the test passes. The moves synchronize the call's stream (for their
pageable sources) before the call returns, so the window lies inside the call. For this case
memcheck is the detector, and `ci/gpu_local.sh`'s memcheck step runs the executable.

**The fix.** `replace_in_call(res, old, next)` in `apply_slotted_device.cu` moves the old array
to the stream of the call (`buffer::set_stream()`) before it is replaced. This is Decision item
1's rule, applied to the six arrays and the scratch. On one stream nothing changes (item 3): no
kernel, launch, synchronization or allocation is added.

**Not changed, by review:**

- the slack compaction (`compact_device_rows()`) synchronizes the call's stream before it
  replaces the rows;
- `ensure()` and `reserve_device_slotted()` replace the batch arrays, the edge scratch and the
  staging at the start of a call, and every device apply has synchronized its stream before its
  call returns (`graph_access::finish_device_apply()`, at the commit);
- label_propagation's device state is made in a new object per `compute()` or `clone()` and
  never resized, and its engines' scratch is held in fenced workspaces (`scratch_buffer`);
- M10b's other changes (the kernel groups, the batch reader) release no array a call may still
  read.

**Verification on the merged code.** On `6567ce2`, a `parity-cuda` build with tests, on GPU 1
(shared lock, after `locklog.py quiet-ok`):

- the three `GraphStreamOrder` tests pass, both plain and under memcheck (0 errors), and
  `SlottedCuda.*:SlackCuda.*` pass under memcheck (0 errors);
- memcheck over C12 of the nine CUDA conformance suites reports **0 errors** in each;
- every `cpp/tests/dyng_*cuda_tests` executable passes, and so do label_propagation's CUDA, checker,
  operators, slack and large slack suites;
- memcheck of `dyng-compat-dynlp --layout slack --streaming-tau` at 50k, as `ci/gpu_local.sh` runs
  it, reports 0 errors for both rules (damped, inplace);
- the golden corpora on cuda: sssp "ALL EQUAL: 495 cases x 2 configurations", cycle_count "ALL
  EQUAL: 48 replays (24 cases x 2 configurations, 0 skipped)", and label_propagation's trace
  replay of P (`compare.py label_propagation trace --backend cuda`, the nine cases of
  `ci/gpu_local.sh`) "PASSED", every case exact;
- on the CPU, `dyng_graph_tests`, `dyng_cycle_count_tests`,
  `dyng_cycle_count_conformance_tests` and `dyng_framework_tests` pass.

The logs are under `$DYNG_SCRATCH/runs/streamfree/merged/`. The ADR 0024 reading of the merged
code is in "The reading on the merged code" below.

## The reading (ADR 0024, HD16)

`ae5fc47` (A, the parent) against `3a0a7aa` (B, the fix): the stable algorithms whose library
code changed (`graph.cpp`, compiled into the library that sssp, mosp and cycle_count link) read
with M10a step 5's command forms (`parity/results/M10a.md` section 12): sssp `perf_ab.py run
--backend {cuda,openmp} --layouts 17` (51 rounds, 34 for road_usa; the OpenMP local10k updates
over 85 layouts and 170 rounds), mosp `perf_ab.py mosp` with the same options, cycle_count
`perf_ab.py cycle_count run --backend cuda --baseline-kind port` (34 rounds), and rule 3's longer
readings (cycle_count's Twitch count and GitHub update at 102 rounds, as M10a, and the Twitch
update, whose resident interval was 5.9 % wide; the COLLAB update, which cannot hold the boost
lock under the power cap, at the base lock, the M3 precedent). Both sides are `parity-cuda` builds
of `git archive` trees at paths of equal length (`$DYNG_SCRATCH/runs/streamfree/ab/{A,B}`); CUDA at
the boost lock on GPU 0 with `CUDA_MODULE_LOADING=EAGER`, the contamination monitor on, rejected
rounds repeated; the deterministic counters equal in every sample. One exclusive window per
reading (`tools/locklog.py run [--reset-gpu 0] EXCL M10b fix0120-adr0024 ...`: the calendar takes
only the 0.4 tracks' labels; the step key marks the windows as this fix's, and M10b's agents were
told so), 52 windows from 2026-11-17 20:08 to 11-18 05:53 UTC: 44 readings (the 21 of the reading, 4 of
the parts, 16 with FB, 3 of rule 3; the first cycle_count reading ended rc=1 only because its
COLLAB update could not hold the boost lock, its other cases complete, as in M10a) and 8 that
ended at once (the first A/B trees were configured with `DYNG_BUILD_PARITY_TESTS` off, which
`perf_ab.py` refuses as not a parity build; both sides were rebuilt with the preset as it is);
every GPU-0 reset `ok`. Every region is read by `parity/ab_modes.py --bar 1.02`; the tables, every
region's reading, the parts and the windows are in `parity/results/ADR0120.md` (the samples stay
in `$DYNG_SCRATCH/runs/streamfree/ab/`).

1. **Within the bar: every region of the updates.** sssp `sosp_update` (CUDA 0.999-1.001; OpenMP
   0.972-1.012, the local10k regions by their 170-round readings: within-mode ratios
   0.989-1.006 with equal mode fractions, where one region of roadNet-PA read 1.069 at 17
   layouts) and `apply` (CUDA 0.985-1.018, OpenMP 0.970-1.012); mosp `compute` (CUDA 0.999-1.000,
   OpenMP 0.960-1.003) and `combined` (CUDA 0.999-1.001, OpenMP 0.968-1.012); cycle_count
   `static_kernel` 0.990-1.012, `update` 0.985-1.001, `update_end_to_end` 0.997-1.019 and
   `static_end_to_end` 1.008-1.020 (Twitch 1.016 at 102 rounds), the Twitch update at 102 rounds `update` 0.994-0.996 and `update_end_to_end` 1.015, the COLLAB update at the
   base lock 1.000-1.004. The commit's own change (the device merge's `replace_device_edges()`
   on cycle_count's set batches; the host apply's `drop_device_edges()` on sssp's and mosp's) is
   in `apply`, `update` and `update_end_to_end`.
2. **Outside the bar: `end_to_end` of `dyng-compat-mosp`'s sssp and mosp updates**, on both
   backends, every graph and every batch: sssp CUDA 1.030-1.092, sssp OpenMP 1.054-1.101, mosp CUDA
   1.032-1.081, mosp OpenMP 1.037-1.084, with B in the slow mode in nearly every round. mosp's
   `path_costs` (`gate = "none"` in `parity/timed_regions/mosp.toml`: reported, not gated; a noisy
   region whose code did not change) reads 0.971-1.236.
3. **Located: code placement in the graph reader, code the fix did not change** (the method of
   `parity/results/M11b.md`, section 1.7):
   - Of the build's 132 library objects only `graph.cpp.o`'s code differs (its `.text` section
     2,214 -> 2,385 bytes; `libdyng.so`'s `.text` 3,520 bytes longer with the instantiations of the new members;
     the `.cu` objects differ only in nvcc's path-hashed symbol names). The graph reader's
     objects (`dyng_io`) are identical.
   - The parts of the tool's own report (`e2e_parts_ab.py`, rgg safe50k, 21 rounds): `read
     graph` is the whole end-to-end difference, OpenMP +90.1 ms of +87.7 ms (1.180), CUDA +89.6 ms
     of +86.0 ms (1.174); the update's parts within 1.2 % (commit 0.999 / 1.004, SOSP 1.000 /
     1.000).
   - The reader's nine functions moved by 16 bytes mod 64 in `libdyng.so`, and the branches the
     JCC erratum affects (`jcc_scan.py`; this Xeon Gold 6258R is Cascade Lake) moved with them:
     251 in A and 240 in B, differently placed (`parse_rows` 47 -> 56, `read_values<int>` 60 ->
     68); 6 and 6 with R020's flag set FB on both sides.
   - With FB on both sides, the parts read `read graph` 0.997 / 1.002 and `end_to_end` 1.000 /
     1.005, and the 16 FB readings of sssp and mosp put `end_to_end` at 0.989-1.008 on every graph
     and batch, every update region within the bar, but one batch: mosp OpenMP road_usa local10k,
     where B landed in a slow process mode in 13-16 of 34 rounds and A in none (`combined` 1.026,
     `end_to_end` 1.024; fast-mode ratios 1.007-1.008). The same batch of the reading itself has
     `compute` 0.994 and `combined` 0.996 with equal mode fractions.
4. **Acknowledged by the author on 2026-11-18 (HD16, the D14 path):** the `end_to_end` excursions
   of item 2, and those of the reading on the merged code below (GOVERNANCE.md, approvals log,
   "D14-path acknowledgement of ADR 0120's ADR 0024 reading"). They are the class acknowledged for
   M10b step 4 on 2026-11-13 (the graph reader's placement), on this fix's own readings. No change
   to code, flags, bar or round counts; a region above 1.02 not on that list goes to the author
   again. The merge order the author set on 2026-11-18 (after M10b's pull request, before R040;
   GOVERNANCE.md, approvals log) had this branch merge main and take the reading again on the
   merged code, whose placement differs.

### The reading on the merged code (2026-11-18)

`c67dd40` (A, main after M10b) against `6567ce2` (B, the merged fix: this ADR's commits, the merge
and the slack rows' fix), with the same command forms, rule 3's same longer readings, the same
protocol and the same builds (`git archive` trees at `$DYNG_SCRATCH/runs/streamfree/ab2/{A,B}`,
`parity-cuda`). 28 exclusive windows, step key `fix0120-merged`, from 2026-11-18 16:29 to 22:52
UTC: the 21 readings, 3 of rule 3 and 4 of the parts (plain and FB). Every window ended rc=0 but
the first cycle_count reading, which ended rc=1 only because its COLLAB update could not hold the
boost lock (its other cases complete), as in the first reading and M10a; every GPU-0 reset `ok`.
`parity/results/ADR0120-merged.md` has the tables, every region's reading, the parts, the objects,
the reader's placement and the windows (the samples stay in `$DYNG_SCRATCH/runs/streamfree/ab2/`).

1. **Within the bar: every region of the updates.**
   - sssp `sosp_update`: CUDA 0.998-1.001, OpenMP 0.989-1.013 (local10k by its 170-round
     readings; 0.984-1.019 at 17 layouts). sssp `apply`: CUDA 0.992-1.017, OpenMP 0.984-1.012
     (rgg local10k read 1.023 [0.986, 1.032] at 17 layouts, 1.002 at 170 rounds).
   - mosp `compute`: CUDA 1.000, OpenMP 0.946-1.005; `combined`: CUDA 0.998-1.000, OpenMP
     0.949-1.004.
   - cycle_count `static_kernel` 0.997-1.012, `static_end_to_end` 0.993-1.007, `update`
     0.988-1.003 and `update_end_to_end` 0.993-1.012, but one region: the Twitch update's
     resident `update` read 1.034 [0.998, 1.042] at 34 rounds, bimodal. At 102 rounds (rule 3) it
     reads 0.970, with within-mode ratios 0.999 and 0.995 and equal mode fractions (p 0.21). The
     Twitch count and the GitHub update at 102 rounds read 0.999-1.005, and the COLLAB update at
     the base lock 1.000-1.005.
2. **Outside the bar: `end_to_end` of `dyng-compat-mosp`'s sssp and mosp updates, again**, on
   both backends and every graph: sssp CUDA 1.050-1.089, sssp OpenMP 1.064-1.096, mosp CUDA
   1.020-1.070, mosp OpenMP 1.020-1.071, with B in the slow mode in nearly every round; every
   batch is outside the bar but mosp OpenMP road_usa local10k (1.020 [1.014, 1.030], not flagged).
   mosp's `path_costs` (not gated) reads 0.783-0.984.
3. **Located as in the first reading: code placement in the graph reader.**
   - Of the build's 144 library objects only two differ in code (by their disassembly, nvcc's
     path-hashed names normalized): `graph.cpp.o` and `apply_slotted_device.cu.o`, the fix's two
     files. `libdyng.so`'s `.text` is 4,720 bytes longer. The graph reader's objects are
     identical.
   - The parts (rgg safe50k, 21 rounds): `read graph` is the whole end-to-end difference, OpenMP
     +82.3 ms of +81.8 ms (1.164), CUDA +82.3 ms of +87.7 ms (1.161); the commit reads 0.998 /
     0.999.
   - The reader's nine functions keep their sizes and moved by 16 or 32 bytes mod 64; the
     branches the JCC erratum affects moved with them: 236 in A and 234 in B, differently placed
     (`parse_rows` 41 -> 52, `read_values<int>` 60 -> 68); 9 and 9 with FB on both sides.
   - With FB on both sides, the parts read `read graph` 0.997 / 0.998 and `end_to_end` 1.001 /
     0.998.

   The 16 FB readings of sssp and mosp were not taken again on the merged code: the objects, the
   parts with and without FB, and the first reading's FB readings already locate the class.

## Consequences

- A call that releases a library-owned device array it may have used on its stream releases it
  on that stream (`buffer::set_stream()` first, when the array may have been allocated on another
  one). The sites listed in "The other release sites" hold this, and so do the slack rows'
  growth since the merge ("The merged code"). A new site that replaces an array inside a call
  follows the same rule.
- C12 alone does not detect this class on debug-sync builds; the graph-level test does, and
  memcheck on a non-debug-sync build (as M10b step 4 ran it) is the stronger check of C12.
