# ADR 0094: sssp's OpenMP regions carry allocation failures out (carried item (a) of 0.4)

- **Status:** Accepted under delegation (2026-11-04; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The fix is the carried item (a) of 0.4 (the three items the author
  asked on 2026-10-23 to be fixed "with their own gate readings"); M11 as its single owner and its
  place as M11b step 1, before any engine change of step 2, are the orchestrator's delegated
  reconciliation of 2026-10-27. Its reading follows HD16, **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log, "The measurement protocol and gates of 0.4 (sheet B1-B6)": ADR
  0024 at 1.02; a region above the bar goes to the author for acknowledgement on the D14 path of
  2026-10-23, never a silent pass). It changes no rule the author approved: sssp's and mosp's
  headers, their documented guarantees, the gates and their protocol (ADRs 0018, 0021, 0024) and
  the parity rules are untouched.
- **Date:** 2026-11-04
- **Deciders:** the AI assistant, on the author's behalf (M11b, step sssp-d13-carried)
- **Builds on:** ADR 0061 (D13: an exception must not leave an OpenMP region; `region_exceptions`;
  its "Consequences" list this region set as found in passing), ADR 0062 (the longer reading),
  ADR 0083 (the carried item (b), the same rule for `openmp_exec::for_each`), ADR 0015 (the
  pooled workspace and its per-thread lists), ADR 0024 (dynG against dynG).

## Context

sssp's OpenMP engine (`cpp/src/algorithms/sssp/openmp.cpp`, MOSP-OpenMP@c352151's
`sospUpdateCpu`) collects the vertices each thread finds in a per-thread list inside a parallel
region and concatenates the lists with `list_gather` (`util/list_gather.hpp`). The lists are
`thread_list`s kept in the pooled workspace (`util/thread_list.hpp`, cache-line blocks): they keep
their capacity between rounds and runs and allocate only when a thread takes a larger share of a
round than ever before, which the dynamic schedule decides. Such a growth, a `push_back` inside the
region, and the gather's resize of its output inside `#pragma omp single`, can throw
`std::bad_alloc`; an exception that leaves an OpenMP region or a worksharing iteration ends the
process (`terminate called after throwing an instance of 'std::bad_alloc'`, exit status 134), the
D13 class of ADR 0061. sssp.hpp documents `out_of_memory_error` with its guarantee instead. The
regions: identify_affected (the invalidated vertices, the insertion heads), seed (the pull pass's
frontier), loop (the first near/far split of the frontier, every push iteration, every far
re-split) and their gathers. mosp's OpenMP compute and update run this engine once per objective.

ADR 0061 listed the defect as found in passing (R030, outside D13's scope). No commit on `main`
since 0.3.0 and no `m10*` branch touches `sssp/openmp.cpp`, `util/thread_list.hpp` or
`util/list_gather.hpp` (checked at the start of the step, `git log`).

## Decision

1. **`region_push_back(list, value, errors)`** (`util/thread_list.hpp`) replaces every
   `push_back` of a per-thread list inside the engine's regions. Its fast path is the list's own
   capacity compare (`size() != capacity()`, which the compilers fold to push_back's
   `finish != end_of_storage`) and push_back's store; a full list calls the cold, `noexcept`
   `region_grow_push_back()`, which grows it under `region_exceptions::run()`: a `std::bad_alloc`
   is kept in the region's carrier (the first one; the others are dropped), and after a failure of
   any thread nothing more is appended or allocated. The thread continues to every barrier and
   `single` of the region. The element is passed by reference (`const value_t&`) down to the cold
   path, as `push_back` and its `_M_realloc_insert` take it (`8e2f3df`, after the first reading:
   by value, GCC 12 allocated the registers of the push iteration's per-edge loop differently and
   that loop read 2-4 % slower; section "The reading").
2. **`list_gather::gather(local, errors)` and `gather_pair(..., errors)`** take the region's
   carrier: the output's resize inside `omp single` runs through it, both barriers are passed by
   every thread, and after a failure (the resize's or an earlier push's) no thread copies its list
   (the output may be too short). The signatures without a carrier are gone, so no caller can
   forget it; the engine is their only caller.
3. **One carrier per hook** (identify_affected, seed, loop), **rethrown after each region**: the
   hook stops at the first failed region, the exception (the original `std::bad_alloc`) reaches
   sssp.cpp's entry points, and `DYNG_TRANSLATE_ALLOCATION_FAILURE` makes it
   `out_of_memory_error` with sssp.hpp's guarantee (an update's engine runs after the commit, so
   `g` holds the new version and `r` is poisoned; compute is strong). The failed run's workspace
   is discarded by its lease (ADR 0015). Nothing changes when nothing fails: the same lists, in
   the same order, between the same barriers. The fused CUDA engine, the operators engine,
   `sequential.cpp` and every header are untouched.
4. **For M10** (coordination): the carrier exists; a new OpenMP region that appends to a
   `thread_list` or gathers lists uses `region_push_back()` and passes its `region_exceptions` to
   the gather, as here.

5. **Found by the tests, fixed in its own commit** (`e80c24c`, outside the D13 class but against
   the same documented guarantee): `framework::run_update()` builds what the engines read of
   G_{t+1} (the host in-edges, or the device copy) inside the commit, after `graph::apply()` has
   put the new version in place. A failure there (a host allocation of the in-edge build; the
   sweeps reached it at 56 threads) left every result unpoisoned: stale for `update()` but still
   readable, where the headers say "`r` is poisoned". The results are now poisoned before the
   exception leaves the commit, as for a failing `after_apply`.

## The tests

`cpp/tests/support/failing_allocator.hpp` gains, behind `DYNG_TEST_FAIL_ALIGNED_NEW` (the
executables of ADRs 0060, 0061 and 0083 do not opt in and are unchanged), the aligned
`operator new` (the cache-line lists allocate with it, so the earlier executables could not fail
them), counts by kind, `failure::from_then_on` (every allocation of every thread fails once the
countdown is reached: memory exhausted for the whole team) and `within_seconds()`, a watchdog
that ends the process with a message when a call does not return (a thread that missed a
barrier). Two new executables:

- `dyng_sssp_openmp_allocation_failure_tests`:
  - **the 18 allocation sites**, driving the engine's hooks directly: identify_affected's
    invalidated-vertex list, insertion-head list and their two gathers; seed's frontier list and
    gather; the loop's first split (near list, far list, both gathers), a push iteration (near,
    far, both gathers) and a far re-split (near list, kept list, both gathers). Each input is
    shaped (list capacities, output capacities, deltas, a chain, a star or a path of far groups)
    so that at one thread the hook's first allocation is at the site; the test checks its kind
    (aligned for a list, plain for a gather), that no other kind allocates in the hook, and, in
    the loop, the near-far counters at the failure (iterations, epochs) that name the region;
  - **every allocation of each hook failed in turn**, at **1, 8 and 56 threads**, of the calling
    thread or of any thread, **once or from then on**, each sweep under the watchdog: every
    reached failure is a `std::bad_alloc` out of the hook (none is lost), and the hook without a
    failure gives the reference output (the gathered lists, the packed words);
  - **sssp::compute and sssp::update on openmp** (1, 8, 56 threads; the calling thread's and any
    thread's allocations, each call through a handle of its own so that the lists grow inside
    the call): `out_of_memory_error`, compute strong, update with both outcomes of sssp.hpp
    reached (before the commit `g` and `r` unchanged and the update then succeeds; after it `g`
    updated and `r` poisoned).
- `dyng_mosp_openmp_allocation_failure_tests`: mosp::compute and mosp::update on openmp likewise,
  with mosp.hpp's guarantee.
- `list_gather_test.cpp`: after a failure kept in the carrier by any thread (1, 3, 8 threads;
  `gather` and `gather_pair`), every thread passes both barriers, nothing is resized or copied,
  and the failure is rethrown after the region.

**On the parent** (`64a5123`, the new tests compiled against it; `parity-cuda` preset) **each of
the six new tests ended the process** (`terminate called after throwing an instance of
'std::bad_alloc'`; `$DYNG_SCRATCH/runs/m11b-step1/before-fix.txt`); on the fix they pass in the
Debug (`dev`) and the Release (`parity-cuda`) builds. The sanitizer presets skip them, as ADR
0060's.

**The fast path** (`parity/thread_list_fast_path.py`, tested by
`parity/tests/test_thread_list_fast_path.py`): a probe compiled with the library's flags holds
`push_back` and `region_push_back` as functions; the fast path of each (the shortest path from
the entry to a `ret` without a call) has the same accesses to the list's header (load of `finish`,
compare with `end_of_storage`, store of `finish`), one element store, one conditional branch and
no call, for 32- and 64-bit elements, at -O2 and -O3, with GCC 12.2, GCC 13.4 and Clang 18 (with
the libstdc++ of GCC 12 and 13); a planted fast path that also reads the carrier is caught. In the
engine's object, the capacity compares per function are the same before and after (section "The
reading").

## The reading (ADR 0024, HD16)

`parity/perf_ab.py run --backend openmp --baseline-exe <the parent 64a5123's dyng-compat-mosp>
--exe <the fix's>` on roadNet-PA, roadNet-CA, rgg_n_2_20_s0 and road_usa (safe50k and unsafe50k:
17 layouts x 51 rounds, 34 on road_usa; local10k: 85 layouts x 170 rounds; read again per ADR
0024 rule 3 over 85 layouts x 170 or 340 rounds), mosp's OpenMP update with `perf_ab.py mosp
--baseline-exe --layouts 17` (and 85, rule 3), read with `parity/ab_modes.py --bar 1.02`,
every reading in its own exclusive window; the tables, the records and the windows are in
`parity/results/M11b.md` section 1.5-1.9 (`M11b-d13-*.json`).

1. **The first version (v1: `e0d6129` + `e80c24c`) read sosp_update 1.023-1.039 on roadNet-PA,
   roadNet-CA and rgg** (road_usa 1.011-1.018) and end_to_end 1.040-1.063 on every graph. Located
   (ADR 0024 rule 3) with the stage reading (`parity/experiments/sssp_stage_ab.py`: the
   difference is `sssp.loop`, +2-4 %; identify_affected, seed and finalize unchanged), the parts
   of the tool's own report (the end-to-end difference is the graph reader, `read graph`, whose
   objects the fix did not change), both sides rebuilt with R020's flag set FB (the JCC-erratum
   mitigation: end_to_end 0.996-1.004, so code placement; sosp_update still 1.013-1.032 on
   roadNet-CA and rgg, so not placement) and the instructions of the push iteration's outlined
   function: **a real cost of the fix**: passed by value to the noinline cold path, the element
   stayed in a register and the per-edge relaxation loop spilled the vertex instead.
2. **`8e2f3df` (v2, the final library) passes the element by reference**, as `push_back` does;
   that loop's instructions and registers are the parent's again. v2 reads **sosp_update within
   the bar** on every graph at 17 layouts (0.987-1.018, one region of roadNet-PA at 1.029 with a
   5.7 % interval; read again over 85 layouts x 170 rounds, `parity/results/M11b.md` section
   1.5), the update stages within 1.2 % in the parts (loop 1.012), and with FB on both sides
   **every region within the bar on all four graphs** (sosp_update 0.992-1.018, end_to_end
   0.994-1.008).
3. **ADR 0024 rule 3's longer readings** (85 layouts x 170 rounds for the 50K batches, 340 for
   local10k) put **every `sosp_update` and `apply` region of sssp's OpenMP update within the
   bar** on all four graphs and three batches (by the median, or by the within-mode ratios, at
   most 1.017, where the rounds are bimodal), and mosp's update (`sosp_update`, `combined`,
   `compute`) within it on every 50K batch.
4. **What stays above the bar on v2 is listed for the author** (HD16, the D14 path; GOVERNANCE.md,
   Open decisions, M11 (HD)):
   - **end_to_end 1.071-1.100** of sssp's OpenMP update (mosp's 1.042-1.087), code placement in
     the graph reader, code the fix did not change: in the parts on rgg +88 ms of +90 ms is
     `read graph` (1.176; with FB 0.989); 6 of its 9 functions moved by 16 bytes mod 64 in
     `libdyng.so`, with other branches split by the JCC erratum; the FB builds put end_to_end at
     0.994-1.008 (mosp 0.990-1.002);
   - mosp's **path_costs** (a byte-identical object; a noisy region, 1.004-1.164 with intervals up
     to 28 % wide, road_usa 1.019-1.047), placement;
   - mosp's roadNet-CA local10k **`sosp_update obj1` 1.022** at 340 rounds: each mode as fast as
     the parent's (within-mode 1.011 / 1.004), the slow mode more frequent (Fisher p 0.02),
     +0.16 ms of 7.2 ms.

   The one-layout PLAN 8.6 gates of the OpenMP rows of `ipdps25_dynamosp_sosp` and `_mosp`
   against MOSP-OpenMP@c352151 hold on v2 (section 1.7 of the record: 84 gated cells, none above
   its gate; end_to_end 0.73-0.86, sosp_update at most 0.95), so none of this is a STOP (S7).

## Consequences

- A host allocation failure inside sssp's OpenMP regions, and so inside mosp's OpenMP compute and
  update, is the documented `out_of_memory_error`; no process ends. With ADRs 0061 and 0083 the
  rule of ADR 0061 holds for every OpenMP region of `cpp/src` that allocates.
