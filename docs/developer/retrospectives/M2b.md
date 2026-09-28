# Retrospective: M2b (merge of M2a, `cycle_count` on CUDA)

Status: **in progress** on the branch `m2b-cycle-cuda` (from `main` at `eda8b8b`, INT1). M2b
merges the accepted M2a branch `m2-cycle` (`cycle_count` on the sequential and OpenMP backends)
and ports `cycle_count` to CUDA against CycleEnumeration-GPU@0a976ad. This page grows step by
step; the M2 summary and the re-estimate close it.

## Step 1: merge of M2a (merge-m2a)

| Commit | What |
|---|---|
| `c8d5d5d` | `git merge --no-ff m2-cycle` ("Merge branch m2-cycle: M2a cycle_count on the CPU backends"). Six textual conflicts, resolved by combining both sides (below) |
| `174707d` | the API page of the `cycle_count` Doxygen group (the M4 site fails with warnings as errors on a group without a page, which M2a predates) |
| `3aedcb8` | `cycle_count` status in the README, landing page, algorithm table, roadmap and short plan (sequential and OpenMP working, CUDA in progress); the ADR table notes M2a's amendment of ADR 0010 |
| `fbeb58a` | the repository state of 2026-09-28: the `main` ruleset exists (17 required checks, strict; the Repository admin role bypasses only through pull requests); `DCO` is not required yet (the author's organization membership is private). Repository settings guide, `GOVERNANCE.md` (text and approvals log), `CONTRIBUTING.md` |
| (this commit) | the re-verification records (`parity/results/M2b.md`, section 1), CHANGELOG, this page |

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
| (this commit) | ADR 0020, the algorithm page, statuses, CHANGELOG, `parity/results/M2b.md` section 2, this section |

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
