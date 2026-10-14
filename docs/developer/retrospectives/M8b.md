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

## Step 3: the update path, the kit's hypergraph axis, the scaffold (update-kit-scaffold)

### What was built

| Commit | What |
|---|---|
| `09c9158` | ADR 0043 (the update path and the kit's container axis) and ADR 0044 (the hypergraph scaffold and the requirements of a hypergraph manifest), before any code; GOVERNANCE and index rows |
| `a8305e9` | the host store's lazy views and `automatic` line graph counted as the container's work, as on the device store (a gap the kit's C8 exposes; ADR 0043 finding 1) |
| `a01f0eb` | `detail::hyper_transaction` (`transaction.hpp`): begin with Step 0 once (`hyper_staged_batch`), `before()`, `commit()`, `after()`, `applied()`; `hypergraph::apply()` is a transaction without participants (behaviour, messages and stages unchanged) |
| `5aed474` | `participant_of<hypergraph<V, W>>`, `hyper_update_participant`, the hypergraph's `run_update()` (`hyper_update.cpp`), `is_hypergraph_v`; the framework's composition for both containers (`hyper_requested_batch`, `hyper_applied_batch`, `has_batch_summary`, `state_identity`); the API baseline |
| `672d3bc` | the aggregate-delta fakes on the hypergraph (`fake_hyper_problems.hpp`, the device counts in `fake_hyper_kernels.cu`), `hypergraph_update_test.cpp` on every backend, the compile-fail tests `hypergraph.update.static_assert.*` |
| `7cf3d5b` | the kit's container axis: `containers.hpp` (`container_traits`, the graph's moved verbatim), `hypergraph_generators.hpp` (the hypergraph's presets, model, twelve mixes), `container_checks` (C7, C9) |
| `f60aadd` | `dyng_hypergraph_kit_tests` / `..._cuda_tests` (the fakes through C0-C12), the planted bug (CTest `conformance.mutation.frontier_requested_ids`, suite `kit` of `parity/mutate.py`) |
| `e0b4025`, `22a09ab` | `new_algorithm.py --container hypergraph` (`_template_hypergraph`), `regen.py`'s hypergraph requirements, the scaffold check's hypergraph probes |
| `e80900f` | `conformance.md`, `framework.md`, the API page, the CHANGELOG |
| `524d6c9` | a test fix found by memcheck (a hand-run participant's workspace outlived its resources handle) |
| (the records commit) | `ci/gpu_local.sh`: the kit's CUDA executable under racecheck and synccheck; `parity/results/M8b.md` section 3 and `M8b-mutate-kit.json`; this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The applied record stays container state** (ADR 0041 deviation 12 said `dyng::update` would
   move it into its transaction). The transaction exposes the record of its commit through
   `applied()`, which is the hypergraph's last record; moving it out would take it from
   `hypergraph_access::applied()`, which the parity tools and M11's incidence-delta hook read
   after a plain `apply()` (ADR 0043, section 1).
2. **"Prepares the row views and h2h for participants that read them"** is the line graph only:
   the internal row views are rebuilt by every commit already (ADR 0042), and the public views
   must not be built on the apply path. `reads_prepared_graph()` of a hypergraph participant means
   "reads the line graph" (default false); `run_update()` makes sure of the line graph before the
   before-apply phase (on G_t, so count(-) reads it), not inside the commit.
3. **The planted bug's "requested ids"** are the ids a batch would get without reuse,
   `id_bound(G_t) + i` (an inserted hyperedge has no requested id; these are the ids a caller who
   predicts ids would use). The kit catches it in C2, C5 and C7 (and C8 in budgets builds) on
   every backend.
4. **The test-only fakes are not registered algorithms.** The kit gained two optional traits
   members, `registry_entry()` (C0 and the backends read the entry the traits give) and
   `test_partners` (C10's partners besides the registered algorithms), so the hypergraph axis runs
   on every backend before the first hypergraph algorithm (M9's `triad_count`) exists. The fakes'
   own suite is one test executable per backend kind (`dyng_hypergraph_kit_tests`,
   `..._cuda_tests`), not a `<name>_conformance_tests` of the registry.
5. **The kit's graph axis keeps `graph_types`**; a hypergraph algorithm's traits may say
   `container_types` instead (both are read). The graph suites' test lists are unchanged
   (compared with the binaries built before the change, every graph conformance executable, host
   and CUDA, and the registry test).
6. **Two template folders** instead of `//@@ graph` / `//@@ hypergraph` blocks in one (ADR 0044):
   the two versions differ in almost every line.
7. **`graph_requirements` keeps its name** in a hypergraph manifest and names the requirements on
   `hypergraph_properties` (ADR 0044); a graph manifest's table is unchanged.
8. **Recorded, not fixable in this step:** the frozen `update.hpp`'s static_assert text for a wrong
   batch names the graph's `edge_batch_view` also for a hypergraph, and a graph result passed with
   a hypergraph stops in sssp.hpp's frozen `update_traits` (`no type named 'edge_type'`); both
   headers must stay byte-identical (ADR 0043 findings 3 and 4). Compile-fail tests pin both.
9. **Two stale strings of criterion 21 fixed early:** `update_dispatch_test`'s "(the hypergraph of
   0.2)" and `new_algorithm.py`'s refusal message (gone with the refusal).
10. **The hypergraph stages of an update:** Step 0 runs under `update.normalize` (or
    `<algo>.normalize`) and the commit under `update.commit` around `hypergraph.apply`, so inside
    `dyng::update()` the `hypergraph.apply` stage covers the commit only; a plain `apply()` keeps
    `hypergraph.apply` around validation and commit (the stages of `parity/timed_regions/
    hypergraph.toml` are unchanged).

### Measurements

- **The kit's hypergraph axis:** `dyng_hypergraph_kit_tests` 28 tests (26 checks over the two
  hypergraph types, C4 and C12 skipped on the host: one engine, no CUDA) and the generator checks,
  53 s in Debug; `dyng_hypergraph_kit_cuda_tests` 28 tests (C4 skipped), 29 s on GPU 1. C8 holds
  with no allocation in the steady state and 0 (host) / 2 (CUDA) host synchronizations.
- **The planted bug:** in the budgets build (dev) the mutant fails 8 tests (C2, C5, C7, C8 of
  both types; the first C2 failure is the `mixed` mix under erase_first, where reuse makes the
  predicted ids wrong); `parity/mutate.py run --suites kit` (parity-cuda, Release: no C8) detects
  it by 6 host and 6 CUDA tests (C2, C5, C7), control passing.
- **Sanitizers:** memcheck clean on the kit's CUDA executable and the hypergraph CUDA tests after
  a test fix (`524d6c9`: a hand-run participant's workspace outlived its resources handle);
  racecheck and synccheck clean on the kit's C1, C3, C11 and C12 and the update tests. The kit's
  racecheck took 760 s, so `ci/gpu_local.sh` runs C1, C11 and C12 under racecheck and synccheck
  (memcheck covers every check).
- **The CPU and GPU gates:** `ci/check.sh` (after the ruff fix), `ctest -L gpu` (670) and
  `ctest -L cpu` (1,055) on dev-cuda, the Clang 18 syntax pass (328 + 710 files) and the fortified
  cpu-only build: clean.
- **The scaffold check:** 54 tests of the two hypergraph probes (13 checks x 2 types x 2 probes,
  two hand cases) passed, next to the graph probes (80) and `my_bfs` (41).

### Notes for M9 (triad_count)

- A hypergraph algorithm is a problem with `container_type = hypergraph<V, W>`; its `requested`
  is `framework::hyper_requested_batch` (`normalized->deletions`: the live deleted ids of G_t) and
  its `applied` is `framework::hyper_applied_batch` (`summary.inserted_ids`;
  `record.inserted_ids_device` on CUDA; `record.deleted_members`). `framework/fake_hyper_problems.hpp`
  is the worked example of count(-) on the old row views and count(+) on the new ones, on every
  backend.
- `reads_prepared_graph()` returning true keeps the line graph (`automatic` builds it on G_t).
- The traits: `container_types`, `hypergraph_shape`, `require(hypergraph_properties&)`; the
  kit's presets add both `id_reuse` orders and the three capacity presets. `scripts/new_algorithm.py
  triad_count --container hypergraph --family aggregate_delta` gives the green start (its
  `planned.toml` entry is removed by the scaffold).

### Open items

- **Hosted CI has not run on this branch** (nothing is pushed).
- Steps 4 and 5 of M8b (I/O, generators, Python; the freeze). The kit's hypergraph mixes come from
  a kit-local generator; step 4 may switch them to the public hypergraph generators.
- The author's confirmations at the 0.3.0 release (the frozen hypergraph headers, the stable
  `.hg` / `.dgb` formats) are still pending; the plan's default applies until then.

## Step 4: hypergraph files, batch files and generators (io-generators)

### What was built

| Commit | What |
|---|---|
| `b4e5c09` | ADR 0045 (`.hg`, the hypergraph batches of `.dgt`, `.dgb` version 1, the dump decoder and resolver, the generators), before any code; GOVERNANCE and index rows |
| `055fb18` | `io/hypergraph_io.hpp` (`read_hypergraph`, `write_hypergraph`, `hypergraph_read_options::escher_gpu()` / `hsosp()`); `read_hyperedge_batches` / `write_hyperedge_batches` (`+h`, `-h`; one container kind per file); the `.dgb` format (`batch_writer`, `batch_reader` with pinned staging on CUDA, the whole-file helpers, CRC32 in `util/crc32`), the shared writer checks (`io/batch_checks.hpp`); the fuzz targets `fuzz_hg` and `fuzz_batch_binary`, hypergraph batches in `fuzz_batch_text`, their corpora and reproducers (`make_dgb_seeds.py`), `fuzz_regression_test`'s allocation bounds; the io tests (host and CUDA) |
| `12b0358` | `generators/hypergraphs.hpp` (`uniform_hypergraph`, `random_hyperedge_batch`, thirteen mixes, library-owned draws, golden hashes); `generators::legacy::escher_hypergraph`, `escher_batch_generator`, `standard_library_matches_legacy_streams`; the kit's hypergraph models and mixes from the public generators; the API baseline |
| `129bea6` | the parity side: `escher_dump` (A's batch records, A's command line, the by-content resolver), `cbst_trace::trace_source` moved into the dynG-free trace library, `dyng_hypergraph_ids --legacy-generator` and `--hg FILE`, `compare.py hypergraph_ids` with both; the committed traces replayed with the legacy generator in hosted CI; `ci/gpu_local.sh` |
| `581dc5c` | `tools/compat/escher` (`dyng-compat-escher to-dgb` / `replay`, CTests on the committed traces), `datasets/preprocess/benson_to_hg.py` (both variants), the registry's `dyng_converter` and its check in `cbst_trace_goldens.py datasets` |
| `b7b6a38` | `file_formats.md` (`.hg`, `.dgt` hyperedge operations and the kind rule, `.dgb` with its format version), the io and generators API pages, `robustness.md`, the CHANGELOG |
| (the records commit) | `parity/results/M8b.md` section 4 and the records; this section |

### Deviations from the plan and the step's text (each with the same intent)

1. **The originals' preprocessing are option sets** (`hypergraph_read_options::escher_gpu()`,
   `hsosp(seed, max_cardinality)`), static factories like the capacity presets, on top of PLAN
   5.7's options. Two options were added: `index_base` (both originals' files are 1-based; ids
   must not be guessed) and `num_vertices` (ESCHER-GPU's synthetic runs have isolated vertices
   above the largest id that a `.hg` file cannot show; the converter records the count in its
   `.txt`). PLAN 5.7 names `relabel = dense_sorted` for ESCHER: ESCHER-GPU's **loader** keeps the
   ids (`none`); `dense_sorted` is its **converter's** numbering, so `escher_gpu()` uses `none`,
   which gives the same rows on coauth-DBLP (whose ids are already 1..V) and the right vertex count
   everywhere.
2. **`weight_spec::hsosp_target_draw`.** H-SOSP's weights are drawn after one draw for its virtual
   target vertex; the virtual hyperedges are M11's (`hyper_sssp` adds them), but the draw changes
   every weight, so the option set reproduces the draw (and discards it). Documented in the
   header and file_formats.md.
3. **Weighted `.hg` files** end their lines with `; w` (as `.dgt`'s `+h`), an extension neither
   original writes; without it `write_hypergraph` could not write a weighted list.
4. **Tokens.** The `.hg` reader takes ESCHER-GPU's line rules (comments at the first character
   only; commas separate) and dynG's tokens: MOSP_ESCHER's loader also accepts a `+` sign and reads
   `1-2` as two ids; dynG rejects both (ADR 0010's strict readers). No registered file has them.
5. **`.dgb` carries annotations.** The flags `deleted_members` and `expected_ids` (and
   `insert_first`) are the "semantics flags" of PLAN 5.7, chosen so that M9 replays A's batches
   from `.dgb` files by content and checks the ids by key without the traces.
6. **The "ESCHER dump reader"** is a decoder of the batch records of M8a's CBST traces (the dumps
   of ADR 0036), on the parity side as the step allows (no public API; step 5's review decides
   whether a resolver belongs in `dyng::testing`). `dyng_hypergraph_ids` now uses it and its
   resolver for the by-content replay (a refactor: the replays of steps 1-3 give the same
   records); `trace_source` moved from `cbst_compare` into the trace library, which uses the
   standard library only, so that `dyng-compat-escher` can open traces without the test
   libraries. The converter compiles the two parity sources into the tool.
7. **"Test that by-content and by-key replay of coauth50k give the same rows"** is the
   `--mode both` replay of `compare.py hypergraph_ids` (A's live member sets after every batch in
   both containers; A's ids until the first ambiguous resolution, 824 on coauth50k) and
   `dyng-compat-escher replay` of the converted files (the live member sets of the by-id and the
   by-content containers equal after every batch).
8. **The kit's generator switch changes the kit's batches** (the same twelve mixes and their
   promises, other draws). `heavy_delete` is a mix of the public generator that the kit does
   not use (no kit check names it); the kit's twelve mixes map one to one. Test counts are
   unchanged.
9. **`write_batches()` now rejects negative ids** (shared writer checks; such files did not read
   back): a behaviour change of a tracked function, in the CHANGELOG's "Changed".
10. **The fuzz workflow's dispatch limit** went from 1,000 to 750 s per target: eight targets
    must fit the job's 120 minutes.
11. **Reproducers without findings.** The fuzzers found nothing in 30 minutes each; the
    `regressions/` of the two new targets hold hand-made inputs of the class M6b found (size
    fields that lie), so that the class is replayed by every test build and bounded by
    `fuzz_regression_test`.
12. **`hyperedge_batch_mix::limits` deletes `INT64_MIN`**, which no batch file can hold (the
    writers reject negative ids): batches of that mix are applied, not written.

### Measurements

- **Parity** (`parity/results/M8b.md` section 4): `.hg` with `escher_gpu()` equals A's 2,467,389
  loaded hyperedges, with `hsosp()` B's 2,466,792 (numbering and weights bit for bit);
  `benson_to_hg.py` writes the pinned bytes in both variants, and `write_hypergraph` of A's loaded
  hypergraph writes them too; the legacy generators give 20/20 synthetic hypergraphs and 180/180
  batches of `make test`, 3/3 and 9/9 of syn1M (openmp and cuda), 3/3 coauth50k batches; by key
  and by content give the same member sets on coauth50k (824 ambiguous resolutions);
  `dyng-compat-escher` replays its `.dgb` files of coauth50k and syn1M on openmp and cuda with the
  expected ids.
- **Fuzzing:** `fuzz_hg` 5.9 M, `fuzz_batch_binary` 15.6 M and `fuzz_batch_text` 11.2 M
  executions in 31 minutes each, no finding; peak RSS 126 / 450 / 124 MB.
- **`.dgb` size:** A's coauth50k batches (3 x 50K operations, with the deleted member sets and the
  expected ids) take 4.3 MB; the reader allocates the largest record twice (pinned, CUDA) plus its
  device copy.
- **Gates:** `ci/check.sh` passed (cpu-only 1,062 and dev 1,095 tests); `ctest -L gpu` of the
  step's executables on dev-cuda 153/153; the Clang 18 syntax pass over 355 + 746 files and the
  fortified cpu-only build (250 tests of the step's labels) clean; the intermediate commit
  `12b0358` built and passed in a worktree; compute-sanitizer memcheck, racecheck and synccheck of
  `dyng_io_cuda_tests` clean.

### Notes for M9 (triad_count)

- **Replaying A's batches.** `dyng-compat-escher to-dgb <trace> <prefix>` writes `<prefix>.hg`,
  `.dgb` and `.txt` (A's command line, vertex count, H2V payload capacity); `replay` shows the two
  ways: by id (the file's deletions; `expected_ids` must equal `inserted_ids`) and by content
  (`deleted_members` through `escher_dump::by_content_resolver`). M9's parity reads the `.dgb`
  with `io::batch_reader(res, path)` (device views on CUDA) and the counts need no ids.
- **Regenerating A's runs** without the traces: `generators::legacy::escher_hypergraph` and
  `escher_batch_generator` with one `std::mt19937_64(seed)`, the hypergraph under {erase_first,
  escher_gpu(capacity)}; `escher_options::parse()` turns A's command line into the options.
- **Datasets:** `read_hypergraph(coauth.hg, escher_gpu())` is A's coauth-DBLP;
  `benson_to_hg.py --variant escher` re-derives the file.

### Open items

- **Hosted CI has not run on this branch** (nothing is pushed); the fuzz workflow's 60 s runs and
  the new CTests (`fuzz.*.hg`, `fuzz.*.batch_binary`, `compat_escher.*`) run there first.
- Python bindings of the step's functions (`dyng.io.read_hypergraph`, `.dgb`, the generators and
  the legacy generators, Hypothesis strategies): step 5.
- The author's confirmations at the 0.3.0 release (the frozen hypergraph headers, `.hg` and
  `.dgb` stable at version 1) are still pending; the plan's default applies until then.
