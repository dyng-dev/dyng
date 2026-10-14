# Retrospective: M8b (the hypergraph container)

Status: **in progress** on the branch `m8b-hypergraph`, created by the orchestrator from the head
of the accepted M8a branch `m8a-escher-store` at `b45e685` (M8a's pull request #14 is open and the
author merges it; `origin/main` `30c3eef`, 0.2.0 released and 0.3.0.dev0, is merged into M8a at
`889953a`, so nothing was merged at the start of M8b). M8b is the second half of PLAN 11.3's M8
row: the public `hypergraph` on the ESCHER store of M8a, its line graph, update path, I/O,
generators and Python, and the freeze of the hypergraph headers; it runs in five steps, and each
step adds its section here. Release 0.3.0 = M8a + M8b + M9 (PLAN Appendix F).

**The author's decisions for M8 (2026-10-07; GOVERNANCE.md approvals log, already recorded on
this branch by M8a).** (1) The memory gate is read as store accounting: dynG's device bytes
against ESCHER-GPU@abcf9b2's own `DynamicHypergraph::deviceBytes()` on coauth-DBLP after the build
and after each of 3 x 50K batches (seed 1); the Nsight Systems peak of both processes is recorded,
not gated, in M8 (it is gated in M9). (2) O16: the hypergraph `id_reuse` default is
`erase_first`; the H-SOSP parity runs set `insert_first`. Until the author decides at the 0.3.0
release, the plan's default holds: the hypergraph headers are frozen at the end of M8b and stable
from 0.3.0 (PLAN 11.3), the `.hg` and `.dgb` formats at version 1 in the stable tier; this awaits
the author's confirmation at the release.

## Step 1: the container on the ESCHER store (container-core)

### What was built

| Commit | What |
|---|---|
| `81196ef` | ADR 0041 (before any code): PLAN 5.2 and the sketch reconciled item by item, the sketch's three open questions answered (dead rows, `inserted_ids`, weights), placement per backend, the host id allocator, `capacity_policy`, overlap sizes, the batch order per `id_reuse`, skipped ops, the guarantees, the applied record, the container-level parity; the base of the branch in `parity/results/M8b.md` |
| `21e07ab` | `cbst_store::set_used_payload()`, a dynG addition: the bump pointer of a tree restored from another backend's state |
| `07e00d8` | the six public headers under `dyng/hypergraph/` (host-only, in `dyng.hpp`, `tracked` in the API baseline), `to_string` of the three enumerations, the CHANGELOG api-change entry, `@guarantee` required on the hypergraph, batch and list classes |
| `f995b78` | `capacity.{hpp,cpp}` (the policy checks, the mapping to the store's presets, the initial payload), `id_allocator.{hpp,cpp}` (the host best-fit allocator), `store.{hpp,cpp}` (the internal store interface, `group_by_vertex`); tests against the host model and the store presets |
| `5b64c23` | `hypergraph.cpp`, `hypergraph_impl.hpp` (`hypergraph_access`, the applied record), `incidence_model.{hpp,cpp}` (the host store), `escher_store.{hpp,cu}` (the device store), `instantiate.hpp`; the container's tests on every backend |
| `6d8a773` | `dyng_hypergraph_ids` (`cpp/tests/parity/escher/hypergraph_ids*`), the batch extracts of `parity/escher_trace.py extract-batches` and two committed extracts, the id parity tests (cpu and gpu) and the kernel-path test |
| `2bb19e9` | `parity/compare.py hypergraph_ids` (`parity/hypergraph_ids.py`) |
| `c730e8d` | the mutations `hypergraph.mutation.{h2v_not_deduplicated, allocator_ties_reversed}` (CTests of every build) and the suite `hypergraph` of `parity/mutate.py` |
| `55a0292`, `df9c52b`, the docs/provenance commit | reduced sizes under compute-sanitizer; the profiler-stage test; A's store bytes in the replay report |
| (the records commit) | `parity/results/M8b.md` section 1 and its JSON records, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **Deletion by content cannot reproduce A's ids where A's deleted set has duplicates.** The
   step asks that "deletion by id and deletion by content give the same state". When a deleted
   member set is held by several live hyperedges (A's generator samples copies of live
   hyperedges, and its random rows of one vertex repeat), no rule that sees only the content can
   know which of them A deleted, and best fit then gives later inserts other ids. The harness
   resolves each deleted set to the smallest live id with that set not yet taken in the batch (a
   documented, deterministic rule), and requires (a) the same live member-set multiset as A's
   model after every batch (the state seen by content), and (b) A's exact ids and rows by id for
   every batch up to the first ambiguous resolution. The ambiguous resolutions are counted per set
   (results, section 1.3). Deletion by id is compared with A in full.
2. **MOSP_ESCHER's Caps factor is not always 1.3.** `mosp_escher(extra)` reproduces B's default
   `headroomFactor` 1.3; its tests set 2.0, 1.25 and 1.5 (`test_h2h_delta`,
   `test_hsosp_matches_dijkstra`, `test_hsosp_scale`, `hsospStress`). The harness derives the
   factor and the extra on which a session's trees agree and replays under
   `mosp_escher(extra)` with `initial_headroom = factor - 1` (exact for these factors; tested).
   The preset itself is unchanged.
3. **`skipped_ops` is broken down.** dynG counts the repeated members it drops in `skipped_ops`
   (the step's criterion 3), MOSP_ESCHER deduplicates them silently, so "skipped_ops plus the
   harness-filtered virtual-id ops == skippedOps" cannot hold literally on a batch with repeated
   members (B's `hsospBench` real hyperedge batches have 841 over 3 batches, its DBLP 50K
   hyperedge batches 22,703).
   `hyper_apply_summary` carries `ignored_deletions` and `dropped_members` (`skipped_ops` is their
   sum); the harness requires `ignored_deletions + filtered == skippedOps` and `dropped_members ==`
   B's deduplicated count (its `he_insert` values minus the payload of its H2V insert).
4. **B's replayed hyperedge-kind batches have no skipped deletions.** In every set the
   `skippedOps` of the batches before a session's first vertex-kind batch are 0 (B's skipped ops
   are in its vertex-kind batches, M11's), and no replayed batch names a virtual hyperedge. The
   comparison runs (0 == 0 on the 130 replayed B batches of the two sets); B's skip rules are covered by the container's unit
   tests (`SkippedOpsCountDeadOutOfRangeRepeatedAndDuplicates`, the `limits` mix of the random
   streams) instead.
5. **A B session stops at its first vertex-kind batch.** B's tests interleave hyperedge-kind and
   vertex-kind batches on one hypergraph; once a vertex-kind batch (incidence edits, 0.4) has run,
   B's state cannot be reproduced. Each session is replayed up to that batch, and the batches not
   replayed are counted (`vertex_batches`).
6. **The batch extracts are a new committed artifact.** Hosted CI needs B's batch records and
   M8a's committed subset has none (its B traces are `test_cbst_ops` runs); a whole trace with
   batches is several MB. `parity/escher_trace.py extract-batches` writes a format-1 trace without
   digests (batch and tree records, the construct and insert sizes); two extracts (A's `016-main`,
   83 KB; the first 9 sessions of B's `014-test_h2h_delta`, 158 KB, under the 256 KB file limit of
   pre-commit) live in `cpp/tests/data/goldens/hypergraph_batches` with their SHA-256 and command
   in `SUBSET.txt`. They are kept out of `cbst_trace/` because M8a's tests replay every file there.
   The repository's test data stays under 1 MB.
7. **Rule 3 in the allocator.** The allocator implements the overflow segments of an append (rule
   3, with B's fill trigger point) although no hyperedge batch of 0.3 appends to H2V (incidence
   inserts are 0.4): the random allocator test exercises it against the host model, and M11's
   `slack_csr` reuses it. The removal of values from H2V rows (incidence deletions, 0.4) is not
   modelled: it changes no length or bump pointer, only tail occupancies.
8. **V2H has no payload limit on the host backends.** The host store keeps V2H as sorted vectors
   (ADR 0041, section 5): no id depends on its layout. Under a fixed-capacity policy
   (`mosp_escher(extra)`) the CUDA store's V2H can therefore raise `capacity_error` where the host
   store does not; no parity set comes near it, and the random streams use small fixed payloads
   only with low-churn mixes.
9. **A's initial capacity above 2^31 - 1** ends A's process; dynG clamps the initial capacity to
   `max_payload` (ADR 0041, section 6). No dataset of A or B comes near it.
10. **`hyperedge_id_bound()` is the store's largest key, not A's host vector size.** A's
    `numHyperedgeKeys` never shrinks; dynG's bound goes down after a surplus rebuild that drops
    dead slots at the top (documented in `hypergraph.hpp`). The harness compares rows up to the
    larger of the two (the extra ones are empty).
11. **`line_graph_policy::automatic` throws `not_supported_error` from `h2h()`** in this step
    (the line graph is step 2's), and `sorted_merge` is refused at construction until then.
12. **The internal applied record is a member of the container state**, read through
    `hypergraph_access::applied()`; `dyng::update()` (step 4) will move it into its transaction.

### Measurements

- **Container-level id parity** (`parity/results/M8b.md`, section 1.3): every case of A's
  `make test`, `coauth50k` and `syn1m` sets and of B's `make test` and `dblp50k_hyperedge` sets on
  `cuda`, `sequential` and `openmp` with 1 and 8 threads: 0 mismatches.
- **Memory, early reading** (not the gate, which is step 2's with the line graph): in the scope
  the container has now (H2V and V2H trees and the H2V row view), the container's
  `store_stats().device_bytes` equals ESCHER-GPU's bytes of the same trees and view after the build
  and every batch: 271 of 271 points, ratio 1.0000 (section 1.4).
- **Tests**: `dyng_hypergraph_tests` 47 (host), `dyng_hypergraph_cuda_tests` 24,
  `dyng_hypergraph_ids_tests` 2 and `dyng_hypergraph_ids_cuda_tests` 2; compute-sanitizer memcheck
  clean on both CUDA executables (reduced sizes); the two mutations killed, the control passing;
  M8a's store replays of the small sets unchanged (results, section 1.6).
- **Checks**: `ci/check.sh` passed after the docs/provenance commit (its first run found the
  missing API page, two provenance lines and the harness's mutation-config test); Clang 18 syntax
  and the fortified build clean; every commit of the step builds and passes its tests.

### Notes for step 2 (sorted_merge and the memory gate)

- The device store keeps the H2V row view (`row_locator::build`, keys `1..bound`) per commit; the
  H2H view and the delta merge go next to it in `escher_store.cu` (`finish_apply()` is the place
  for `hypergraph.apply.line_graph` before `hypergraph.apply.views`).
- `store_stats().device_bytes` equals A's in the current scope at every point of `016-main` and
  `syn1m` and `coauth50k` (section 1.4); adding A's H2H tree and view should give A's whole
  `DynamicHypergraph::deviceBytes()`.
- A's merged trees (H2H) are sized exactly, B's by its Caps factor; `initial_payload()` has no
  "merged tree" variant yet.
- The id replay tool already reads the update records' `device_bytes` (A's whole scope) into its
  report.

## Step 2: the line graph, the views, the memory gate (line-graph-views-memory)

### What was built

| Commit | What |
|---|---|
| `208350a` | ADR 0042 (before any code): the policies, the staged port, the line graph's key count, the internal and public views, the order of an apply, the oracle, the parity method |
| `3b73335` | the transplant of ESCHER-GPU's H2H merge (`line_graph/sorted_merge.{cuh,cu}`, `device_frontier.cuh`; `parity/escher_port/rename_map_line_graph.toml`, `manual_hunks_line_graph.patch`, `transplant.py --map`), wired into the CUDA store; the host form of the merge and the port of `flatHyperedgeAdjacency`; the internal row views (`row_views.hpp`) and the decoded `h2h()` view; `automatic` and `sorted_merge`; `testing::rebuild_line_graph` and its isolation test; `hypergraph_line_graph_test`; the random streams keep the line graph; `compare.py hypergraph_ids --line-graph` |
| `fbfa931`, `8823491`, `0d9d97f`, `cdae5c6` | the mechanical classes of the port: errors, buffers, streams, registration (the last removes the source rules' staged exemption) |
| `42983d1`, `a9c59c3` | `parity/perf_ab.py hypergraph memory` and `maintain` (`parity/hypergraph_perf.py`); `dyng_hypergraph_ids --timing / --sync-stages / --nvtx` |
| `9c66eee`, `cd9cc9e` | `HypergraphCuda.TwoContainersOnTwoNonBlockingStreams` (C12), the step `hypergraph` of `ci/gpu_local.sh` |
| `f561043` | the mutation `hypergraph.mutation.mirror_additions_dropped` / `hypergraph_mirror_additions_dropped` |
| `dac2a72` | the API page, the store README, the CHANGELOG |
| (the records commit) | `parity/results/M8b.md` section 2 and its records, this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The transplant commit also carries the infrastructure.** The step asks to compare the line
   graph with A's after every commit of the port, which needs the container wired from the first
   one; so `3b73335` holds the transplant together with the host stores, the views, the oracle and
   the replay's comparison. The transplanted files themselves are only the generated transplant
   (`transplant.py --map ... --check --rev 3b73335`: identical), and the mechanical classes are one
   commit each. Until the last of them the source rules of `cpp/src/hypergraph` skipped the two
   transplanted files (a staged exemption, removed in `cdae5c6`).
2. **The line graph's key count is the largest bound so far, not the bound.** A's H2H has
   `numHyperedgeKeys` keys (its host vector never shrinks), dynG's bound can go down after a
   surplus rebuild (ADR 0041, section 2). Keeping A's count (`max_bound`) gives A's layout, A's
   memory and frontier flags that cover the deleted ids; the rows above the bound are empty and the
   public view stops at the bound (ADR 0042, section 3).
3. **syn1m is compared by its layout hashes only.** The step asks for per-key hashes and the full
   rows of touched keys after every batch of make-test, coauth50k and syn1m. M8a's syn1m set is a
   `batches` trace (no state digests), so it is compared by the hashes of A's H2H rebuild inputs
   (the whole payload and the row offsets), which is stronger than per-key hashes but does not
   localize a difference; coauth50k gives per-key row hashes (its H2H rows are recorded as hashes
   only), make-test the full rows. No new golden set was exported.
4. **The merge's errors raise the V2H tree's device error word.** The H2H tree's own word is reset
   by its rebuild; the merge reads V2H and its errors (a broken V2H chain, a merged position outside
   its row) are read with V2H's word, once at the end of the apply.
5. **The H2H tree is sized exactly under every preset** (A's rule for merged trees, also under
   `mosp_escher()` and the default); its limit is the policy's payload limit.
6. **The `larger_than` mix of the random streams is capped.** It tripled the hypergraph every
   batch (43K hyperedges on 1,200 vertices after 6 batches); with the line graph checked against
   the oracle and in `check_integrity()` after every batch the two random tests took 92 s each in
   Debug. A batch inserts more hyperedges than the hypergraph has while it has fewer than 2,000
   (`m / 4` after): 11 s. The test `SpecialBatches` has 1,100 rows on one hub (rows above 1,024
   entries).
7. **`h2h().weights` is host memory on every backend** (the weights by id are a host array; an
   `array_view` carries its own memory space), as `weights()`.
8. **An empty batch still rewrites the line graph** (A's merge copies every row); kept for
   parity of the timing and memory, a hypergraph that never had a hyperedge skips the merge (no
   zero-block launch; tested).
9. **The CTest entry of the new mutant came one commit early.** `9c66eee` added the CMake entry of
   `hypergraph.mutation.mirror_additions_dropped` (the file was staged with the C12 test), its hook
   came with `f561043`; at `9c66eee` alone that CTest fails (the mutant equals the control).
10. **parity/mutate.py did not count the new mutation at first**: the C12 test let an exception
    of a driving thread terminate the executable, and mutate.py counts only failed tests, never a
    crash; `cd9cc9e` reports the exception as a failure. 3 / 3 hypergraph mutations detected
    after it.

### Measurements

- **H2H parity** (results, section 2.2): every state of A's make-test (255), coauth50k (4) and
  syn1m seeds 1-3 (12) equal to A's on cuda after every commit of the port, and on sequential and
  openmp.
- **Memory gate: GO** (results, section 2.3): byte for byte A's `DynamicHypergraph::deviceBytes()`
  after the build and every batch of coauth50k; nsys peaks 3,018.9 MiB (A, with counting) and
  3,019.1 MiB (dynG), recorded.
- **Maintain reading** (results, section 2.4; reported, not gated; at the base clock lock, ADR
  0021, because A's process runs under the power cap at boost): `hypergraph.apply` 0.854x of A's
  `update.maintain` on coauth50k and 0.859x on syn1m (sums of the 3 batches; worst batch 1.026x);
  insert 0.68x / 0.87x, line graph 0.83x / 0.81x, views 0.30x / 0.22x; delete 1.72x / 2.96x (the
  host grouping of the deleted incidences is timed in dynG's delete stage, in A's
  `maintain.prepare`); host syncs per batch 21 (A: 76). A CPU profile of the first reading found
  the applied record's sort of the touched vertices (about 16 ms per syn1m batch); `16898f4`
  makes it linear (the first reading: 0.893x and 1.120x).
- **Sanitizers**: memcheck (leak check) clean on `dyng_hypergraph_cuda_tests` and
  `dyng_hypergraph_ids_cuda_tests`; racecheck and synccheck clean without the long random streams
  (results, section 2.6).
- **Mutations**: `parity/mutate.py run --suites hypergraph`: control passes, 3 / 3 detected
  (the new `hypergraph_mirror_additions_dropped` by 10 host and 8 CUDA tests).

### M9 risks

- **The delete stage's host grouping.** `hypergraph.apply.delete` is 1.7-3.0x A's delete timers
  because it times the single-threaded grouping of the deleted incidences by vertex (A times it in
  `maintain.prepare`); `maintain` as a whole is 0.85x. If M9's `update` reading needs room, the
  grouping (and the insert's) can run in parallel or on the device.
- **The Nsight Systems peak** of the container (3,019.1 MiB on coauth50k) is the merge's: the new
  H2H payload exists next to the old one until the rebuild, as in A (3,018.9 MiB with its
  counting); M9 gates the peak with the counting included, where dynG's counting temporaries add
  to it.
- **The boost clock lock does not hold** for A's process on coauth-DBLP (power cap during its full
  count); M9's `update` gate is read at the base lock (ADR 0021) unless that changes.

### Notes for M9 (triad_count)

- `detail::hypergraph_access::require_line_graph(h)` then `views(h)`: A's `RowView`s of H2V and
  H2H (`out` set) in the store's memory space, keys = ids + 1, values + 1; the H2H view covers keys
  `1..max_bound` (empty rows above the bound). Count(-) must read them before `apply()` (the merge
  replaces the H2H payload, the H2V insert reuses dead slots); count(+) after.
- The frontier of A (`device_frontier.cuh`, flags over `num_keys + 1`) is ported in
  `line_graph/`; the inserted ids are on the device in the applied record.
- `hypergraph.apply` has the stages of the toml; A's `update` region adds subtract and add around
  it.

### Open items

- Steps 3-5 of M8b.
