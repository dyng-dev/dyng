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
