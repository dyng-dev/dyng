# ADR 0120: A mutating call releases the graph's device copy on its own stream

- **Status:** Accepted under delegation (2026-11-13; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. A correctness fix of 0.1-0.3 code, found by M10b step 4
  (compute-sanitizer memcheck on a `parity-cuda` build). It applies the rule of ADR 0016 item 10
  (arrays replaced through a handle on another stream are moved to the updating stream before
  they are released) to the graph's resident device copy. Its reading follows HD16, **decided by
  the author on 2026-10-28** (GOVERNANCE.md, approvals log, "The measurement protocol and gates of
  0.4 (sheet B1-B6)": ADR 0024 at 1.02; a region above the bar goes to the author for
  acknowledgement on the D14 path, never a silent pass). It changes no rule the author approved:
  no public signature, no documented guarantee, no gate, protocol or parity rule.
- **Date:** 2026-11-13
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
  (`device_slotted::batch_current`);
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

## The reading (ADR 0024, HD16)

Pending: the stable algorithms whose library objects changed (`graph.cpp`) read against `ae5fc47`
with M10a step 5's command forms (`parity/results/M10a.md` section 12).

## Consequences

- A call that releases a library-owned device array it may have used on its stream releases it
  on that stream (`buffer::set_stream()` first, when the array may have been allocated on another
  one). The sites listed in "The other release sites" hold this today.
- C12 alone does not detect this class on debug-sync builds; the graph-level test does, and
  memcheck on a non-debug-sync build (as M10b step 4 ran it) is the stronger check of C12.
