# Retrospective: M2b (merge of M2a, `cycle_count` on CUDA)

Status: **complete (2026-09-28)** on the branch `m2b-cycle-cuda` (from `main` at `eda8b8b`,
INT1), pending the independent review and the orchestrator's pull request into `main`. M2b
merges the accepted M2a branch `m2-cycle` (`cycle_count` on the sequential and OpenMP backends)
and ports `cycle_count` to CUDA against CycleEnumeration-GPU@0a976ad. Steps 1-3 each appended
their section; step 4 (close-out) added the milestone summary, the M2 summary, the acceptance
record and the re-estimate of the roadmap.

## Step 1: merge of M2a (merge-m2a)

| Commit | What |
|---|---|
| `c8d5d5d` | `git merge --no-ff m2-cycle` ("Merge branch m2-cycle: M2a cycle_count on the CPU backends"). Six textual conflicts, resolved by combining both sides (below) |
| `174707d` | the API page of the `cycle_count` Doxygen group (the M4 site fails with warnings as errors on a group without a page, which M2a predates) |
| `3aedcb8` | `cycle_count` status in the README, landing page, algorithm table, roadmap and short plan (sequential and OpenMP working, CUDA in progress); the ADR table notes M2a's amendment of ADR 0010 |
| `fbeb58a` | the repository state of 2026-09-28: the `main` ruleset exists (17 required checks, strict; the Repository admin role bypasses only through pull requests); `DCO` is not required yet (the author's organization membership is private). Repository settings guide, `GOVERNANCE.md` (text and approvals log), `CONTRIBUTING.md` |
| `06e3e72` | the re-verification records (`parity/results/M2b.md`, section 1), CHANGELOG, this page |

**Conflict resolution.**

- `cpp/include/dyng/graph/graph.hpp`: the class comment lists M2a's `unweighted` instantiations
  and M1b's int32 default `edge_t` with checked construction (ADR 0009); `from_edges()` and
  `from_csr()` document both sides' errors (M1b's copy-policy and `capacity_error`, M2a's
  unusable batch semantics).
- `cpp/src/graph/apply_host.cpp`: both sides had changed the validation at the top of
  `apply_batch_host()`. M1b had replaced the per-array host checks with `expect_host_batch()`;
  M2a had moved all checks into `validate_batch_shape()` (`apply_common.hpp`), shared with the
  set apply, which also checks host accessibility. `validate_batch_shape()` stays (it runs
  before the set/multigraph dispatch, so both applies are checked); the checks are the same.
  M1b's file-local `expect_host` is dropped in favour of M2a's `detail::expect_host` (two
  definitions would be ambiguous).
- `checked_edge_count`: M1b moved it from `apply_host.cpp` to `apply_host.hpp`, M2a to
  `apply_common.hpp`; both headers are included together, so it now exists once
  (`apply_host.hpp`, with M1b's ADR 0009 comment). This was not a textual conflict; the build
  would have failed with a redefinition.
- `cpp/src/graph/graph.cpp`: both includes (`core/staging.hpp`, `graph/apply_common.hpp`).
  `graph::apply` stages device batches on the host (M1b) and then reaches the same host apply,
  which dispatches to the set apply under `batch_semantics::as_sets` (M2a).
- `cpp/tests/CMakeLists.txt`: M1b's framework tests (`thread_list`, `list_gather`) and M2a's
  generator and CycleEnum parity tests.
- `parity/export_goldens.py` (`global REFERENCE, COMMIT` first, then the `cycle_count`
  dispatch), `parity/tests/test_harness.py` (both test sets), `README.md` (both rows and
  paragraphs).
- Auto-merged but checked: `device_graph.cu` instantiates the unweighted device graph (M2a), the
  mutation-test libraries link `CUDA::cudart` in a CUDA build (M2a anticipated it), and the M1b
  placement checks reject a CUDA graph in `cycle_count` (`CycleCountBackend.PlacementIsChecked`).

**ADR numbering.** M2a wrote no ADR of its own: its decisions are an amendment of ADR 0010 (set
semantics and the unweighted graph), which keeps its number. Nothing had to be renumbered from
0020; the next new ADR of M2b takes 0020.

**Re-verification** (`parity/results/M2b.md`, section 1): `ci/check.sh --parity`,
`ci/gpu_local.sh` and `ci/docs.sh` pass; `sssp` 495 / 495 byte-identical on sequential, OpenMP
(1-28 threads, int32 and int64 edge offsets) and CUDA (both edge types); `cycle_count` 24 cases x
3 configurations equal to the goldens; the `sssp` OpenMP and CUDA gates (locked clocks) on
roadNet-CA and road_usa and the `cycle_count` OpenMP update gate on DD and GitHub read as before
the merge. No regression was found, so no fix was needed.

**Deviation.** The `sssp` performance spot check on road_usa ran 11 rounds per batch instead of
M1b's 21 (every gated region there is >= 10 ms, where PLAN 8.6 asks for >= 5 runs); roadNet-CA,
with regions under 10 ms, ran 21. The performance records carry `+dirty` because `CHANGELOG.md`
was edited during the runs (documentation only; the binaries were built from `fbeb58a`).

## Step 2: `cycle_count` on CUDA (cycle-cuda)

| Commit | What |
|---|---|
| `8fdec6e` | graph and framework: Step 0 of set semantics once per update (`normalized_batch`, `update_participant::use_normalized`), the device set apply (`graph/apply_set_device.cu`, CycleEnumeration-GPU's `build_next_rows_kernel` and friends), the lazily downloaded host copy of a device-updated graph, the device in-edges on first use (ADR 0020) |
| `d1bab68` | `cycle_count` CUDA backend: the static counters (`static_cuda.cu`, `dfs.cuh`, `work_queue.hpp`) and the update phases (`cuda.cu`); options `cuda_engine`, `scheduler`, `work_items`; the 64-vertex bound |
| `f5475cc` | the exporter's CUDA build and the CUDA fixtures (`cases/*.cuda`, `counts/*.cuda`), each checked equal to the original's sequential backend when written |
| `9ac4520` | the shared `cycle_count` suites on cuda and the CUDA-only cases |
| `40a0000`, `7d3e116` | `dyng-compat-cycle-enum --backend cuda` and its CLI test against 18 runs of the original on cuda |
| `fdc0f3a` | the golden set `cycle_count_cuda`, its replay, the CUDA mode of the performance harness, the CUDA timed regions, `ci/gpu_local.sh` |
| `6123e1d` | ADR 0020, the algorithm page, statuses, CHANGELOG, `parity/results/M2b.md` section 2, this section |
| `69540a5` | `dyng-compat-cycle-enum` reads only the timed call's stages (the resident scope's profiler had also recorded the untimed calls: a false 1.34x on DD) |
| `bba3017` | `parity/results/M2b.md` section 3: the CUDA harness on DD, the OpenMP update re-check, the sanitizers |

**What was ported, and how straight.** The kernels are CycleEnumeration-GPU's, line for line:
`count_roots_kernel`, `count_roots_queue_kernel`, `forward_rows_kernel`, `fill_edge_items_kernel`,
`target_degree_kernel`, `count_edge_items_kernel`, `count_two_hop_items_kernel`,
`extend_prefix` / `find_edge` / `lower_bound_u32` / `ThreadHistogram` / `dispatch_capacity`,
`mark_owners_kernel`, `item_counts_kernel`, `count_owned_cycles_kernel` / `extend_owned_path` /
`closes_owned`, `change_rows_kernel`, `next_degree_kernel`, `build_next_rows_kernel`, and the host
logic around them (`resolve_work_items` with its `kDenseAverageDegree`, `plan_work_queue_launch`,
`effective_length`, the grid rules). The mechanical changes: names and namespaces, the row offsets
as a template parameter (`std::make_unsigned_t<edge_t>`: with `int32_t` offsets the kernels are the
original's 32-bit ones; `cuobjdump` gives the same register counts and stack sizes, e.g. 19
registers for `count_edge_items_kernel<4>` and 22 for `count_owned_cycles_kernel<4>` on both
sides), dyng buffers leased from the workspace pool on the stream of `resources` instead of
`cudaMalloc` on the legacy stream, scalar read-backs through pinned memory, errors as dyng
exceptions.

**Design decisions** (ADR 0020 where they touch the graph):

- The graph is resident. The original uploads G_t in every update and builds G_{t+1} on the
  device only for its insert phase; dynG must also produce G_{t+1} (the update applies the
  batch), and the M2a host set apply alone costs more than the original's whole DD update (6 ms
  against 4.4 ms). So `graph_access::apply` merges the batch on the device for a resident graph
  under `as_sets` without weight columns, and the host CSR becomes a lazily downloaded copy.
  `build_next_rows_kernel` lives in the graph module (it is the container's device apply, PLAN
  6.4.1), and its `next_owner` array is kept with the device state (`insertion_ids`) for the
  cycle_count insert phase. The delete-phase owner array is computed by cycle_count and, a second
  time, by the device apply for its deletion marks (a memset and one binary search per deletion:
  tens of microseconds on DD), so the container does not depend on the algorithm.
- Step 0 once. Under set semantics the framework normalizes the batch before the participants
  and hands the result to the commit; cycle_count copies its change lists from it. This removes
  M2a's second normalization inside the commit on every backend.
- The 64-vertex bound follows the original: the *effective* bound max(min(k, n), 2) must be
  <= 64, so a large k (or no bound) is accepted on a graph of at most 64 vertices; an update
  checks the bound with every vertex its batch names before anything changes.
- `engine::operators` is rejected with `not_supported_error` on cuda, as for sssp in 0.1; the
  original's `--cuda-scheduler` and `--cuda-work-items` are options (`scheduler`, `work_items`);
  its environment tuning variables are not ported (the defaults are fixed).
- `dyng-compat-cycle-enum` uses `int32_t` edge offsets on cuda by default (the original's 32-bit
  device CSR; `--edge-type int64` is replayed as well) and has `--scope original|resident`: the
  original scope moves the graph to a fresh copy without a device copy before the timed update
  (`g.clone(res)` keeps the state, so the result still matches), so the update uploads G_t inside
  the timed call as the original does.

**Deviations from the plan, recorded here.**

- PLAN 6.4.3 names `work_queue.cuh`; the launch planning and the choice of work items are
  host-only code, so the file is `work_queue.hpp` (unit tested in the CPU suite as the original
  tests them without a device). `framework/work_items.cuh` and `operators/expand_work_items` are
  not created: the work items have one user (rule of two, PLAN 4.5.3); M3 extracts them.
- The COLLAB update (k = 4) is part of the CUDA golden set but replayed on the cuda
  configurations only: its prior is the k = 4 static count of COLLAB (199.7 billion cycles; 6.5 s
  on the original's CUDA backend), far too long for a replay on the host backends.
- The CUDA fixtures and goldens are new files and a new set next to the CPU ones (the M2a corpus
  and fixtures are unchanged, byte for byte); every CUDA histogram of the original was checked
  equal to its sequential or OpenMP histogram of the same case when it was written.

**Found on the way.** A first version of the randomized CUDA tests drew graphs from the shapes
of the original's CPU tests (up to 60 vertices at edge probability 0.7, bounds up to 8): the
sequential reference recount of such a graph runs for hours. The CUDA suites use sparse shapes
(or at most 30 vertices) and bounds up to 7; dense graphs stay in the small-graph suites.

**Open items after this step.** The performance gates of criterion 4 (the harness is in place:
`parity/cycle_count_perf.py run --backend cuda`), the fresh-clone runs of the three gate
scripts, and a device Step 0 (sorting and classifying a batch on the device, so a chain of
updates never downloads G_t) as future work beyond the port.

## Step 3: parity and performance of the CUDA backend (cuda-parity-perf)

| Commit | What |
|---|---|
| `8e4d453` | graph: Step 0 of set semantics sorts with `std::sort` by (source, target, position) and skips lists already in order (was `std::stable_sort`); the DD 100K+100K update 12.2 -> 8.0 ms |
| `c0c100e` | tools: the resident count scope uploads the graph with an untimed 2-cycle count instead of a full count |
| `778ebf5` | parity: `cycle_count_perf.py kernels` covers every kernel with the sm_86 occupancy and pairs both sides; a case with more rejected rounds than `--runs` is recorded as incomplete instead of ending the run; the JSON is written after every case |
| `fe753d2` | parity: the CUDA record keeps the original's monitor window apart from the port's (they were merged under one key) and adds a per-side GPU clock summary |
| `94523c5` | cycle_count, graph: the update returns the static-count work items when it begins, and the deletion marks of G_t are computed once per update and shared with the device apply (ADR 0020, point 6) |
| `11a9b2f` | parity: `cycle_count_perf.py memory` (the peak of live device allocations of both sides, from Nsight Systems' memory trace) |
| `98a0dd3` | `parity/results/M2b.md` sections 4-6, ADR 0018 update, the algorithm page's CUDA performance table, statuses, CHANGELOG, this section |
| `08865bf` | the certificate names the commit the gate scripts passed on |

**Done.** The CUDA gate of acceptance criterion 4 in full: ten cases, both scopes, the clocks
locked for the whole A/B, 21 rounds (11 for the COLLAB update, a region of about 200 ms), plus the
default-clock readings (ungated), the register, stack and occupancy table of all 33 kernels, and
the device memory of every gate case. Parity replayed on the final code: the CUDA set on cuda,
cuda:resident and cuda:int64 (72 / 72), on openmp:56 (23 / 23) and the CPU corpus with `--full`
(72 / 72). Every gated reading is within its gate (the COLLAB update's at the base lock, which
since the review counts only once the author accepts ADR 0021); the summary is
`parity/results/M2b.md` section 6.

**What the first gate campaign found**, and what changed (numbers: `parity/results/M2b.md`
section 4.4):

1. *The resident static kernel read 1.06-1.09x* on GitHub, Twitch and COLLAB while the original
   scope read 0.97-1.00x. The kernel is the same; the compat tool's residency call was a full
   count right before the timed one, and a counting kernel that follows another at once runs
   5-12 % slower even at locked clocks (Nsight Systems: 56.0 then 61.3 ms in one process; a one-
   or two-second pause removes it). The original's process never has that state before its
   kernel. The residency call is now a 2-cycle count (about a millisecond). No library change.
2. *DD 100K+100K update read 1.13x* (original scope). Step 0 on the host took 8.8 of the 12.2 ms:
   `std::stable_sort` of 16-byte changes, 1.5x the original's `std::sort`. The unstable sort by
   (source, target, position) is equivalent (positions are unique); a sortedness check first
   skips the sort for lists already in order. 0.74x now; all replays unchanged.
3. *The COLLAB update could not be measured at the boost lock*: its 6.5 s prior runs under the
   GPU's power cap, the SM clock drops to 1350-1680 MHz in every process of both programs, and
   the monitor rejected 22 rounds in a row; the harness then exited without writing any record.
   The harness now records such a case as incomplete and continues; the case is read at the base
   lock, where the clocks hold (first written as an update of ADR 0018; since the review ADR 0021,
   Proposed, pending the author).
4. *The record lost the original's monitor window*: in the CUDA mode the round record merged
   `{"original": window}` with the port's scope windows, and the scope "original" overwrote it.
   The rejection itself read the right window; only the JSON was wrong. Fixed before the campaign
   whose records are committed.
5. *Device memory*: see below; fixed in `94523c5`, after which the whole campaign was measured
   again on the final code (the committed records; the `fe753d2` campaign read the same within
   0.02 in every ratio).

**Device memory** (PLAN 8.6 lists device memory <= 1.05x; M1b did not measure it). A first
measurement with nvidia-smi's per-process totals showed the CUDA update far above the original
(COLLAB 25K+25K: 1040 against 592 MiB; Nsight Systems' trace of the live allocations then gave
762 against 383 MiB). The causes were the prior's static-count work items (about
300 MB of two-hop items on COLLAB k = 4), kept in the workspace through an update that never reads
them, and a second m-int array of deletion marks (cycle_count's delete phase and the device apply
each marked the deleted positions). Both are fixed (`94523c5`). Nsight Systems' memory trace (the
high-water mark of live `cudaMalloc` / `cudaMallocAsync` allocations) now gives the same peak for
both programs on all ten cases; nvidia-smi totals still differ by a fixed amount (the stream-ordered
pool reserves in 32 MB granules, and dynG's process loads the kernels of every algorithm of the
library: about 34 MB on a three-edge graph).

**Deviations from the plan, recorded here.**

- The COLLAB update is read at the base clock lock (1170 MHz), not at boost (ADR 0021, Proposed;
  written here as an ADR 0018 update and moved to ADR 0021 by the review):
  the GPU cannot hold the boost lock through its prior. Its default-clock reading (0.989x /
  0.937x) is recorded as for every case.
- The resident scope of the count task makes the graph resident with a 2-cycle count, not with a
  full count (`parity/timed_regions/cycle_count.toml`).
- Device memory is compared as the peak of live allocations (Nsight Systems) and reported next to
  the process totals; PLAN 8.6 does not say which. It is within 1.05x on every case; M2b's
  acceptance criteria do not list it.

**Lessons.**

- On a GPU near its power cap, what ran just before a timed kernel matters even at locked clocks.
  A scope that adds GPU work before the timed region (here, to make the graph resident) must keep
  the GPU as idle as the original's process leaves it.
- A harness that gives up on one case must still write what it measured: one unmeasurable case
  cost a whole campaign. On a shared machine, a record per case also lets one case be repeated
  alone: another user's GPU job hit two cases of the final campaign, which were repeated
  (`M2b-cuda-perf-cycle_count-repeat.json`).
- Retained workspaces are right for steady-state throughput and wrong for peak memory when an
  operation keeps another operation's scratch. Measuring device memory per gate case, not only
  time, found it.

**Open items after this step.** The fresh-clone runs of `ci/check.sh --parity`, `ci/gpu_local.sh`
and `ci/docs.sh` (criterion 1); the M2 summary and the re-estimate that close this page; the
author's view on reading the COLLAB update at the base lock; the device Step 0 (ADR 0020) and the
one-time pinned-buffer allocation in the first static call (0.85 ms of DD's reported
`static_total`) as future work.

## Step 4: close-out (finish)

| Commit | What |
|---|---|
| `6894e2a` | the algorithm page: the CUDA work items of every scheduler and phase, determinism on cuda (claim order, unchecked device sums), the default-clock table, DD's reported static total; the manifest lists the cuda backend and the original's CUDA sources; the short plan marks M2 done |
| (this commit) | the final verification from a fresh clone, the gates on the final code (`parity/results/M2b.md` section 7), this section, the M2 summary, the re-estimate (here and in the short plan), CHANGELOG |

**Done.** The documents listed by acceptance criterion 5 were checked against the code and the
records: `docs/algorithms/cycle_count.md` (work items, the 64-vertex bound, the resident graph
and the device apply, determinism, the performance tables in both scopes at locked and at
default clocks), the README, landing page and algorithm tables (sequential, OpenMP and CUDA), the
roadmap, the short plan, the CHANGELOG. The manifest of `cycle_count` still listed only the CPU
backends ("cuda: M2b"); it now lists cuda and the original's CUDA sources (`scripts/regen.py`
reads it in M3). The three gate scripts ran from a fresh clone of the branch (below).

**Deviations.** None new.

**Final verification** (fresh `git clone` of the branch into the scratch area, heavy steps under
the shared perf lock, niced; the performance re-check under the exclusive lock):

| Command | Commit | Result |
|---|---|---|
| `ci/check.sh --parity` (14 min 46 s) | `6894e2a` | every step OK: clang-format; `cpu-only` 401 / 401 and `dev` 414 / 414 (`ctest -L cpu`, `-Werror`); clang-tidy naming; REUSE; provenance (116 files); harness tests; Doxygen (116 compounds) and the site; pre-commit; `ctest -L parity` 4 / 4 on the `parity` preset |
| `ci/gpu_local.sh` (dev-cuda, GPU 1, 27 min) | `6894e2a` | all ten steps passed: build, gpu 155 / 155, cpu 414 / 414, sssp CUDA goldens 495 / 495, cycle_count CUDA goldens 48 / 48, memcheck, synccheck (sssp and cycle_count), racecheck, clang-tidy on the CUDA branches |
| `ci/docs.sh` | `6894e2a` | OK |
| the sssp CUDA gate (locked clocks) and OpenMP gate on roadNet-CA (21 rounds) and road_usa (11); the cycle_count OpenMP update gate on DD and GitHub (11) | `6894e2a` (the clone's `parity` and `parity-cuda` builds) | every gated reading within its gate: sssp CUDA per objective 0.976-1.001x, OpenMP 0.68-0.99x; cycle_count update DD 0.516x, GitHub 0.383x (`parity/results/M2b.md` section 7) |

The commit after `6894e2a` (this one) changes only documents and adds the performance records;
`ci/docs.sh` and the quick steps of `ci/check.sh` (format, REUSE, provenance, harness,
pre-commit) passed on it.

## Step 5: the independent review and its fixes (review-fix)

An independent review of `79a323e` confirmed 19 findings (each verified by a second reviewer).
All are fixed on the branch; none is deferred. Commits in order:

| Commit | Findings fixed |
|---|---|
| `58f00fb` | **Self-loops under set semantics with `self_loop::keep`** (high): the normalized lists kept a self-loop as a change edge, and every backend counted a spurious 2-cycle through it (a later deletion drove a bucket negative; CUDA and sequential disagreed on longer spurious cycles). The phases now skip self-loop change edges (`count_cycles_through_edge()`, `item_counts_kernel`) without renumbering; "set, self-loops kept" variants (directed and undirected) in `EverySemanticsMatchesCompute` and `SetSemanticsWithKeptSelfLoopsMatchesCompute` on all three backends, both failing without the fix. ADR 0020 point 9 |
| `ed2ef4f` | **The recorded mutations were only in the host code**: `DYNG_MUTATION_CUDA_DOUBLE_COUNT_5` (the closing increment of `extend_prefix`) and `DYNG_MUTATION_CUDA_WEAK_OWNERSHIP` (the owner test of the update kernel) mutate the device kernels, where the original recorded them; `cycle_count.mutation.cuda.*` (labels `gpu;mutation;cycle_count`, run by `ci/gpu_local.sh`) links the gpu suite against them: killed by 22 and 19 failing tests, the control passes |
| `37c0c03` | **Host-only kernel tests in the gpu executable**: `cycle_count_kernels_test.cpp` runs in the cpu executable only (14 tests were counted twice) |
| `d5a702c` | **Staging errors named `dyng::update`**: the name comes from the commit stage (`cycle_count.commit` gives `cycle_count::update`) |
| `17e12f0` | **`result::set_options()`** for the tunables (PLAN 5.1, ADR 0006 point 2; the test no longer writes the detail state); **poisoned `bound()` and `get_options()`** throw `stale_result_error` (ADR 0006 point 5); **the edgeless graph on cuda** returns the zero histogram before the 64-vertex check, as the original; **the 32-bit change-id limit** raises `capacity_error` and is documented; **the `capacity_error` contract** says that the device sums wrap on cuda (two findings); the copy-policy test asserts the function name |
| `fc78fdb` | **The original's large-graph device test** (`CudaCountersMatchBruteForceOnLargerGraphs`): its seven shapes with every scheduler and kind of work item, the 300,000-vertex graph at k = 5, and a 20K + 20K update with far more items than resident threads, in the gpu suite the sanitizers run |
| `1dcb684` | **The DD 50K and 100K gates passed only through the sortedness skip** (high): Step 0 now sorts with a least-significant-digit radix sort (stable on the batch order, so the lists are unchanged; its cost does not depend on the order of the batch) and the skip is gone. 100K records: shuffled 1.39 ms against the original's `std::sort` 5.78 ms, sorted 1.39 against 1.18 ms. `SetSemantics.StepZeroListsEqualAStableSortOnLargeBatches` |
| `5463ad5` | **Chained updates downloaded G_t every time** (high): Step 0 of a graph whose host copy is stale runs against the device copy (the membership of each requested change in G_t tested on the device, the sort on the host); `DeviceStepZeroEqualsTheHostStepZero`, `ChainedUpdatesNeverDownloadTheGraph`. ADR 0020 point 7 |
| `f30a58b` | **The download used the legacy default stream and pageable memory**: it is ordered on the stream of the state's buffers; a growing vector is released first and reserved with room (no regrowth copy). ADR 0020 point 8 |
| `632528d` | **No steady-state allocation test for the CUDA update**; the device apply's scratch (change rows, degrees, scan scratch) is pooled; `SteadyStateUpdatesAllocateOnlyTheNextGraph` pins that an update allocates exactly the three arrays of G_{t+1}. ADR 0020 point 10 |
| `41d6de4`, `297ece6` | **The resident scope was measured on the first update only**: `dyng-compat-cycle-enum --chain n` and the regions `update_chain_steady` / `update_chain_worst` (gated as the resident scope); **the short updates claimed CUDA events**: dynG's CUDA-event time of the update is recorded next to the host clocks (region `update_device`, a separate process), and the deviation is recorded below; **the CPU-load threshold** is recorded in the protocol and in every case's monitor record |
| `ace78d5` | **The public graph documentation** describes both apply paths |
| `6dc3187` | **ADR 0020 claimed an acceptance nobody gave** and **ADR 0018 was amended after its acceptance**: ADR 0020 is Proposed (with the review's points 7-10), ADR 0018 is back to its accepted text, and the base-lock rule is the new ADR 0021 (Proposed) |
| `91d02ca` | **README**: the M2b certificate and the set-semantics graph row |
| `f120131` | **The radix sort still missed DD 100K** (1.055x in the first campaign after the review, `b38de7e`'s intermediate record): Step 0 sorts with a bucket sort (0.70x of the original's `std::sort` on sorted input, 0.29x on shuffled), and the DD 100K update reads 0.930x |
| `b38de7e`, `c9e0347`, `34f3fca` | the gate records after the review (`parity/results/M2b.md` section 8.2-8.3) |
| `843f6bb`, `3d11c26` | the corpora replayed on the review-fixed code (72 / 72, 23 / 23, 72 / 72); the algorithm page, CHANGELOG, ADR 0020's sort wording, `parity/results/M2b.md` section 8 and the certificate, this section |
| `65ac05a`, `f7c9d00` | two defects the first fresh-clone run found (a docstring over 100 columns; a seed-count assertion that failed under `DYNG_TEST_SEEDS=2`); the second fresh clone of `f7c9d00` passes `ci/check.sh --parity`, `ci/gpu_local.sh` and `ci/docs.sh` (`parity/results/M2b.md` section 8.4) |

**Deviations recorded by this step:**

1. **The short update regions are host clocks on both sides** (PLAN 8.6 asks for CUDA events or
   nsys kernel sums over >= 20 runs below 10 ms). The original's update region is host-side by
   construction (its `update_seconds` is a `steady_clock` around `update_histogram()`, which runs
   `prepare_batch` and `apply_histogram_delta` on the host) and has no device timer, so the gate
   compares the same host clock on both sides over 21 rounds, as M1b did for sssp (M1b
   retrospective). dynG's CUDA-event time of the whole update is recorded next to it for the
   cases under 10 ms (`update_device`, `parity/results/M2b.md` section 8.2): it reads 0.14-0.20 ms
   above dynG's host clock of the same case (its stage events synchronize the stream), so the host
   clock hides no device time. The certificate row no longer
   claims CUDA events for these regions.
2. **Step 0's membership test runs on the device** for a graph that a device apply produced
   (not in the original, which re-uploads G_t per call and tests on the host); the lists are the
   host path's. The sort stays on the host.
3. **The COLLAB update at the base lock is pending the author** (ADR 0021): its reading is
   recorded, and acceptance criterion 4 is met for it only once the author accepts the rule.

**Lesson.** A performance fix must be judged on inputs the gate does not use. The sortedness
skip made the gate pass on the one input order the generator produces; the radix sort passes on
every order. The same goes for scopes: a resident scope measured once after the upload says
nothing about the resident graph across batches, which is what it claims.

## Step 6: the acceptance verification and its fixes (acceptance-fix 1)

The independent acceptance verifier (fresh clone at `450a1c6`) passed criteria 2, 3 and 5 on its
own evidence: its own archive build of 0a976ad, byte parity of the CUDA CLI on every required
dataset and delta, about 11,400 randomized checks with no failure, the sssp gates re-run on the
final code, and its own boost-lock A/B of GitHub static and the DD and GitHub 25K+25K updates
within the gates. It failed two things:

1. **The status documents overclaimed** (criterion 1): the short plan (status line and M2 row)
   said every CUDA gate was met, the roadmap said the CUDA backend is within the gates, and the
   algorithm page said "every gate is met in both scopes" right after its COLLAB caveat, while
   README and the certificate named the COLLAB update as pending. The certificate (section 4.5)
   and this retrospective (steps 3 and 4) still cited an "ADR 0018 update" that the review had
   moved to ADR 0021. Fixed: every status page names the exception, the short plan lists ADR 0021
   among the open decisions, and the stale references point at ADR 0021.
2. **The COLLAB 25K+25K update has no gate reading under the accepted protocol** (criterion 4).
   The verifier's own boost-lock run rejected every round (SM 1410-1620 MHz under the 230 W power
   cap during the 6.5 s prior), as ours did; its base-lock reading is 0.991x (original scope) /
   0.946x (resident), ours 0.989x / 0.945x. This is not something the implementation can fix:
   ADR 0018's rule (every busy sample of both programs at the boost lock) cannot be met by an
   unpatched original on this GPU (a different power limit, alternative 1 of ADR 0021, needs root
   and, as step 7 found, is not possible at all), and changing the rule, or reading only part of
   the process, would weaken an accepted check.
   It stays open for the author's decision on ADR 0021; once accepted, the committed record
   `parity/results/M2b-review-cuda-perf-cycle_count-collab-update-base.json` is the gate reading.

## Step 7: the second acceptance verification (acceptance-fix 2)

The second acceptance verifier (fresh clone at `6efc49f`) passed criteria 1, 2, 3 and 5 on its own
evidence (`ci/check.sh --parity`, `ci/docs.sh` and `ci/gpu_local.sh` in the clone; sssp 495 / 495
on ten configurations and the sssp gates at the boost lock; cycle_count CPU parity 72 / 72 and the
OpenMP update gate; its own 0a976ad CUDA build byte-identical to dynG on 14 dataset cases; the
CUDA goldens 72 / 72; 4,193 randomized CUDA / OpenMP / sequential / oracle checks with no failure;
GitHub static and the DD and GitHub 25K+25K updates within the gates at the boost lock; 33 / 33
kernels with the original's registers). It failed criterion 4 on one case only, the same one as
step 6:

1. **The COLLAB 25K+25K update has no reading under ADR 0018's accepted protocol.** Its boost-lock
   run rejected every round (SM 1365-1605 MHz under the 230 W power cap during the 6.5 s prior, on
   both programs); its base-lock reading is 0.988x (original scope) / 0.945x (resident), end to end
   0.992x. No code change can fix this: GPU 0's power limit is already its maximum (`nvidia-smi -q
   -d POWER`: current, default and maximum limit all 230 W), so even with root the limit cannot be
   raised; `ncu --clock-control` offers no lock between base and boost; and the original computes
   the prior with its CUDA counter in the same process as the timed update (`run_update()` of
   `src/cli/cycle_enum_main.cpp`), so the prior cannot be moved out of the monitored process
   without patching the original. Accepting ADR 0021 ourselves, or checking the lock only in part
   of the process, would change an accepted rule without the author. **The case stays pending the
   author's decision on ADR 0021**; nothing in the code or the records was changed for it.

It also recorded a non-blocking observation: in its first `ci/gpu_local.sh` run, the racecheck
step failed once (`compute-sanitizer racecheck` on `dyng_cycle_count_cuda_tests`: "process didn't
terminate successfully", 0 hazards, no gtest failure printed; the process ended during the
`CycleCountFixtures` tests, the last output being the skip of
`CycleCountRandom.UnboundedUpdateCostsAboutTheBoundedOne/cuda`). It ran next to a CPU-heavy
`compare.py cycle_count --full`, and did not reproduce in the verifier's six reruns or in its
second complete `ci/gpu_local.sh`. This step tried again at `6efc49f` (GPU 1, `DYNG_TEST_SEEDS=2`,
as `ci/gpu_local.sh` runs it): the whole suite once (58 tests, 0 hazards, peak RSS 0.87 GB, so host
memory pressure is an unlikely cause on a 124 GB machine), and the `Fixtures` and `Random` suites
four times, the first two while the whole `dev` `ctest` suite ran next to them (-j 40, all passed)
and the last two while `parity/compare.py cycle_count --full` did (ALL EQUAL, 72 / 72). All five runs exited
0 with 0 hazards. The cause is not known; the kernel log of the shared machine is not readable
without the `adm` group, so an OOM kill or a signal could not be confirmed. It is carried forward
as open item 8.

## Milestone summary (M2b)

M2b was done in four steps on 2026-09-28 on the branch `m2b-cycle-cuda` (from `main` at
`eda8b8b`): the merge of M2a and its re-verification (step 1), the CUDA port (step 2), parity and
the CUDA gates (step 3) and this close-out (step 4). 26 commits after the merge commit, about nine
hours from the merge to the close-out, several of them under the exclusive perf lock (the sssp
and cycle_count spot checks of the merge, the DD smoke run, the three CUDA campaigns and their
repeats, the final re-checks). Nothing is pushed (the orchestrator opens the pull request). Size of
the branch against the merge commit `c8d5d5d` (added lines, tracked files):

| Area | Lines added |
|---|---:|
| Library (`cpp/include`, `cpp/src`): the CUDA counters and update phases, the device set apply, the normalized batch, the resident graph | 2,880 |
| Tests (`cpp/tests`, without the fixture data) | 780 |
| Parity harness, compat driver, CI (`parity`, `tools`, `ci`, without records and fixtures) | 1,560 |
| Documentation (Markdown pages, ADR 0020, CHANGELOG) and the parity certificate `parity/results/M2b.md` | 1,170 |
| Committed CUDA fixtures (`cpp/tests/data/cycle_enum`, 140 files, 109 KB; all tracked test data: 464 KB) | 4,620 |
| Parity and performance records (`parity/results/M2b-*.json`, 1.5 MB) | 67,760 |

### What was built

- `cycle_count` on CUDA behind the same `compute()` / `update()` contract as the host backends:
  the original's static counters (the naive counter, the work queue with root, edge and implicit
  two-hop items, its automatic choice) and its update phases (`mark_owners`, `item_counts`,
  `count_owned_cycles`), the kernels templated on the offset type (with `int32_t` offsets they are
  the original's 32-bit kernels, register for register). Options `cuda_engine`, `scheduler`,
  `work_items`; the effective bound limited to 64 with `invalid_argument_error`.
- The resident device graph under set semantics (ADR 0020): the batch merged on the device (the
  original's `build_next_rows_kernel`, which it runs and discards), a lazily downloaded host
  copy, device in-edges on first use, Step 0 once per update for every result and the commit, the
  deletion marks of G_t shared by the delete phase and the device apply.
- `dyng-compat-cycle-enum --backend cuda` with the two scopes; the CUDA golden set
  `cycle_count_cuda` (24 cases) and its replay (CTest `parity.cycle_count.cycle_enum_cuda_0a976ad`,
  `ci/gpu_local.sh`); the CUDA mode of the performance harness with the clock lock and the machine
  monitor, the kernel table (registers, stack, occupancy) and the device-memory command.

### Acceptance record

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | `--no-ff` merge of `m2-cycle`; no conflict markers; ADRs renumbered from 0020; docs consistent (tables, roadmap, short plan, repository settings with the `main` ruleset); a fresh clone passes `ci/check.sh --parity`, `ci/gpu_local.sh`, `ci/docs.sh` | `c8d5d5d` (parents `eda8b8b`, `0876d28`); `git grep` for conflict markers is empty; M2a wrote no ADR (its decisions amend ADR 0010), so the first M2b ADR is 0020; the documents of steps 1 and 4; the fresh-clone runs above | met |
| 2 | No regression from the merge: sssp 495 / 495 on sequential, OpenMP, CUDA; the sssp OpenMP and CUDA gates on roadNet-CA and road_usa; cycle_count CPU parity unchanged; the OpenMP update gate on DD and GitHub | step 1 (`parity/results/M2b.md` section 1); the CPU corpus again after every change of the update (72 / 72 at `6123e1d` and `94523c5`, sections 2 and 5); the gates again on the final code (section 7) | met |
| 3 | CUDA histograms bit-identical to the original's CUDA backend (fixtures, DD k = 3..7, GitHub / Twitch k = 4, COLLAB k = 3, the seed-1 update deltas); CUDA = OpenMP = sequential; k <= 64 with a clear error; memcheck, racecheck, synccheck clean | `parity/results/M2b.md` sections 2, 5 and 6: 72 / 72 CUDA replays (cuda, cuda:resident, cuda:int64), 23 / 23 on openmp:56, 72 / 72 CPU replays, the CUDA fixtures and randomized chains; the bound tests; the sanitizers of `ci/gpu_local.sh` | met |
| 4 | CUDA gates in both scopes at locked clocks, >= 20 runs for regions < 10 ms; default clocks recorded; registers and occupancy recorded | section 4, and on the review-fixed code section 8: every gated reading within its gate at the boost lock (static kernels 0.674-0.998x, updates 0.386-0.930x, chained updates 0.226-0.885x), end to end 0.90-0.99x; the short updates on host clocks on both sides with dynG's CUDA-event time next to them (deviation 1 of step 5); default clocks recorded; 33 / 33 kernels with the original's registers, stack and occupancy; device memory 1.000-1.001x | met, except the COLLAB update: read at the base lock (0.989x original scope, 0.945x resident; `parity/results/M2b.md` section 8.2), which is within the gate only once the author accepts ADR 0021 |
| 5 | The algorithm page covers the CUDA backend; CHANGELOG; this retrospective with the M2 summary and a re-estimate | steps 2-4 | met |

### Deviations, consolidated

Every deviation is in the table or list of the step that made it; the ones that matter beyond
M2b:

1. **The graph module owns the device merge** (ADR 0020): `build_next_rows_kernel` and its
   helpers are the graph's device apply (PLAN 6.4.1), used for every resident graph under
   `as_sets` without weight columns and with 32-bit vertex ids; other graphs keep M1b's host
   apply and re-upload. G_{t+1} is built even for a batch without insertions.
2. **The host copy is lazy.** After a device apply the host CSR is downloaded when read, on the
   stream of the state; Step 0 of the next update tests the changes against the device copy
   (step 5), so a chain of updates never downloads G_t. A fully device Step 0 (the sort on the
   device) is future work.
3. **Step 0 once per update**, through `update_participant::use_normalized()`, on every backend
   (also a CPU speed-up), with a bucket sort (step 5) instead of `std::stable_sort`.
4. **The 64-vertex bound** applies to the effective bound max(min(k, n), 2), as in the original,
   and an edgeless graph counts before the check, as in the original.
5. **Files and harness**: `work_queue.hpp` (host-only) instead of `work_queue.cuh`; no
   `framework/work_items.cuh` or `expand_work_items` operator yet (one user; M3); the original's
   environment tuning variables are not ported; the CUDA fixtures and goldens are a new set next
   to the unchanged M2a data; the COLLAB update is replayed on cuda only; `dyng-compat-cycle-enum`
   uses `int32_t` offsets on cuda by default.
6. **Measurement**: the COLLAB update is read at the base clock lock (ADR 0021, Proposed, pending
   the author); the resident count scope uploads with a 2-cycle count; device memory is compared
   as the peak of live allocations; two cases of the main campaign were repeated after another
   user's GPU job; the short update regions are host clocks on both sides, with dynG's CUDA-event
   time recorded next to them (step 5).

### Lessons

1. **A straight port of kernels is the easy half; the container is the hard half.** The kernels
   had the original's registers on the first build, and every histogram matched. The time went
   into where the graph lives: the original's update never produces G_{t+1} on the host, and a
   host apply alone would have missed the DD gate. Decide where the data lives (ADR 0020) before
   porting the code that reads it.
2. **Name the scopes before measuring.** The original uploads per call; dynG keeps the graph
   resident. Reading every region in both scopes, defined in `timed_regions/cycle_count.toml`
   first, keeps the comparison honest both ways: the resident scope is not credited for the
   original's upload, and the original scope must meet the gate on its own.
3. **What runs before a timed kernel is part of the measurement** (step 3): a warm-up that was a
   full count slowed the next kernel by 5-12 % at locked clocks. A warm-up must be as small as
   what the original's process does before its region.
4. **Measure memory per gate case, not only time.** The update held the prior's scratch and a
   duplicate array (1.5-2x the original's peak); timing alone never showed it.
5. **A harness must write what it measured, per case**, and survive a case it cannot measure; on
   a shared GPU the per-case record also lets one case be repeated alone.
6. **Stable sorts are not free, and neither are shortcuts.** `std::stable_sort` of 16-byte
   records cost 1.5x the original's `std::sort`; skipping the sort of sorted lists made the gate
   pass only on the generator's input order (step 5). A sort must be judged on every input order.

## M2 summary (M2a and M2b)

M2 ported CycleEnumeration-GPU@0a976ad's exact k-bounded cycle counting and its DynTruCy-style
update to all three backends in two calendar days (2026-09-27 and 2026-09-28). M2a (the CPU
backends, the parser, the set semantics, the generator, the oracles, the golden corpus, the
OpenMP gates, an independent review with 16 confirmed findings, all fixed) took about eleven
hours on the branch `m2-cycle`, 51 commits; M2b (the merge, the CUDA backend, the resident graph,
the CUDA gates) about nine hours, 27 commits including the merge. The plan estimated M2 at 2-3
working weeks, the M1a re-estimate at 4-6 days of focused work.

| | M2a (CPU) | M2b (CUDA) |
|---|---|---|
| Parity | 72 / 72 replays of the 24-case corpus of the original's OpenMP backend (sequential, OpenMP 4 and 56); the parser, CSR, generated batches and normalized batches byte-equal first | 72 / 72 replays of the 24-case corpus of the original's CUDA backend (cuda, cuda:resident, cuda:int64), 23 / 23 on OpenMP; CUDA = OpenMP = sequential |
| Performance | OpenMP 56 threads: static end to end 0.29-0.93x, update 0.44-0.65x | CUDA at locked clocks, both scopes (review-fixed code): static kernels 0.67-1.00x, updates 0.39-0.93x, chained updates 0.23-0.89x, end to end 0.90-0.99x; device memory 1.00x |
| Correctness tooling | subset-DP, brute-force and edge-set-recount oracles; the recorded mutations fail the suite | the shared suites on cuda; the recorded mutations in the CUDA kernels fail the gpu suite; memcheck, racecheck and synccheck clean |
| Decisions | ADR 0010 amendment (set semantics, the unweighted graph) | ADR 0020 (the resident graph, Step 0 once) and ADR 0021 (the base lock), both Proposed |

What M2 means for M3: `cycle_count` is the aggregate-delta counterpart of `sssp`'s fixed point.
Both run through `run_update` with a normalize stage, participants and one commit. After M2 the
pieces with two users are the resident device graph, the device workspace with pinned
read-backs, and the batch normalization; the persistent grids differ (sssp's cooperative kernel,
cycle_count's claim counter). The work items and a device Step 0 have one user and stay out of
the framework until a second algorithm needs them (rule of two).

## Re-estimate of the roadmap

**Measured.** The M1b retrospective re-estimated M2b at 3-5 days of focused work after the M1b
merge; it took about nine hours, with its independent review still to come. Over M1a, M1b, M2a
and M2b, a port of a good original took about a day of wall-clock time per milestone including
its gates, and each review found real defects (M1a 25, M1b 19, M2a 16) and took about half a day
more. What dominates now is (i) measurement time on one shared machine (the exclusive lock, other
users' jobs, the GPU's power cap), (ii) reviews and their fix steps, (iii) the author's decisions
and account actions, and (iv) merges of parallel branches.

**What this changes.** The remaining milestones are less mechanical than the ports so far: M3
refactors two algorithms under unchanged parity and performance (every refactor commit re-runs
the replays, about 40 minutes, and a full GPU and OpenMP campaign takes three to four hours of
lock time), M8 merges two diverged copies of a data structure, M10 needs calibrated float
tolerances. So the speed-up of the ports is applied only in part:

| Milestone | Plan (working weeks) | M1b re-estimate (focused) | Now: focused effort, incl. review and fix | Calendar incl. author gates | Main risk |
|---|---|---|---|---|---|
| M2 `cycle_count` | 2-3 | M2b 3-5 days | done (M2a about 11 h with its review, M2b about 9 h); M2b's review and fix step 0.5-1 day | the pull request | review findings in the device apply |
| M3 framework + conformance kit + 0.1 API freeze | 2-3 | 4-7 days | 2-4 days | 1-2 weeks | parity and gates re-run per refactor commit (lock time); the author's API sign-off |
| M5 Python CPU wheel, CLI, docs | 2-3 | 4-7 days | 2-4 days | 1-2 weeks | nanobind + scikit-build-core, stubs, the TestPyPI release candidate on a clean machine |
| **0.1.0** | **12-17** (from M0) | about 3-4 weeks | **about 1-2 weeks** from now | **about 3-5 weeks** | the API review and the release approval; the first hosted runs of every workflow |
| 0.1.x (M6) | 4-6 | 1-2 weeks | 4-7 days | 2-3 weeks | the GPU runner decision (O11); CUDA wheel sizes; checkpoint A4 |
| 0.2.0 (M7-M9) | 8-12 | 2.5-3.5 weeks | 1.5-2.5 weeks | 4-6 weeks | the operators engine within 1.05x of the fused one; the CBST merge of two diverged copies; memory <= 1.05x on the 2M-hyperedge suites |
| 0.3.0 (M10-M11) | 8-12 | 2.5-3.5 weeks | 1.5-2.5 weeks | 4-6 weeks | DynLP's calibrated tolerances and slotted layouts; H-SOSP's seven mutations; conda-forge review |
| **Up to 0.3.0** | **about 32-47** | about 9-13 weeks | **about 5-8.5 weeks** | **about 13-20 weeks** | |

**Recommended order now:** an independent review of M2b and its fix step; the pull request of
`m2b-cycle-cuda` into `main`; then M3, which starts from both algorithms on all three backends.
M3's measurement campaigns should be batched (the replays per refactor commit, one full campaign
per group of commits) to keep the lock time within the estimate.

## Open items carried forward

For **the author**:

1. Decide ADR 0021 (Proposed): whether the COLLAB update may be read at the base clock lock (the
   GPU cannot hold the boost lock through its 6.5 s prior under its 230 W power cap); and accept
   (or amend) ADR 0020 (Proposed), recording the acceptances in GOVERNANCE.md.
2. The open decisions of the short plan (O3, the NOTICE line); DCO becomes a required check once
   the organization membership is public.

For **M2b's review, M3 and later**:

3. A fully device Step 0 (the sort on the device as well); the membership test already runs
   there, so a chain of CUDA updates never downloads G_t (ADR 0020, point 7).
4. The one-time pinned-buffer allocation of the first static call (about 0.85 ms of DD's
   reported static total in the original scope); `warm_up()` could pre-allocate it.
5. The clock monitor checks only busy 50 ms samples; non-busy samples below the lock were seen on
   the long static kernels (`parity/results/M2b.md` section 4.3). A stricter rule or a finer
   sampler is a harness improvement for M3's campaigns.
6. The work items (`work_queue.hpp`) and the device merge become framework pieces when a second
   algorithm uses them (M3, M9).
7. The original's 9,000-graph fuzz campaign (PLAN 6.4.3: nightly) and sanitizer jobs in CI (M6).
8. One unexplained exit of `dyng_cycle_count_cuda_tests` under `compute-sanitizer racecheck` in
   the second acceptance verification (step 7; 0 hazards, not reproduced in 12 later runs). If it
   recurs, `ci/gpu_local.sh` should keep the exit status or signal of the sanitized process
   (and a core file) so that a crash can be told from a sanitizer failure.
