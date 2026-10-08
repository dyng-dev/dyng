# Retrospective: M8a (the ESCHER CBST store)

Status: **in progress** on the branch `m8a-escher-store` (from `main` at `8b9067e`, which
contains M7 and M6b). M8a is the first half of PLAN 11.3's M8 row: the internal ESCHER store,
proven against both originals by CBST trace replay. M8b (the hypergraph container, its I/O and
Python, the header freeze) follows on the accepted store; both are in 0.3.0 with M9. Each step
adds its section; the milestone's acceptance table and the open items close the record.

**Planning change (not an author decision; reported to the author on 2026-10-07).** M8 runs as
M8a then M8b, both in 0.3.0 (0.3.0 = M8a + M8b + M9; PLAN Appendix F unchanged in content), as
M1a/M1b, M2a/M2b and M6a/M6b did: the store port and the staged merge are internal and fully
checkable by trace replay, and the public hypergraph headers are frozen only after the
container, the line graph, the update path, the I/O and Python exist. `docs/developer/plan.md`
and `docs/roadmap.md` show the split; together M8a and M8b carry every item of PLAN 11.3's M8
row (the old single row omitted the row views, the `sorted_merge` line graph, the hypergraph
generators, the CBST host model and the header freeze).

**The author's decisions for M8 (2026-10-07; GOVERNANCE.md approvals log).** (1) The memory gate
is read as store accounting: dynG's store device bytes against ESCHER-GPU's own
`DynamicHypergraph::deviceBytes()` on coauth-DBLP after the build and after each of 3 x 50K
batches (seed 1); the Nsight Systems peak of live allocations of both processes is recorded,
not gated, in M8 (it is gated in M9). (2) O16: the hypergraph `id_reuse` default is
`erase_first`; the H-SOSP parity runs set `insert_first`.

## Step 1: references, recorders, traces (refs-traces)

### What was built

| Commit | What |
|---|---|
| `28de27b` | `parity/references.toml` pins ESCHER-GPU@abcf9b2 and MOSP_ESCHER@4b86159; `build_reference.sh` keeps the original's HEAD and `git status` before and after every build as files and requires them unchanged, checks them around `--test` too, and reports whether the build needs a gitlink |
| `71ca715` | the CBST trace recorders as link-time twins (`parity/exporters/escher`, the export patches of both originals): the originals' unchanged objects relinked with `-Xlinker --wrap`; the nm census; the twin output check |
| `f3089fa` | the readers of trace format 1: `cpp/tests/parity/escher/cbst_trace.{hpp,cpp}` (C++, host-only) and `parity/escher_trace.py` |
| `81a8897` | `parity/timed_regions/hypergraph.toml` and its strict-loader tests, before any port commit |
| `1a0f4cc` | smaller traces: integrity reports only on failure, `batches` mode hashes the ops' inputs |
| `cdd2ed7` | the cross-check drivers of the two cores (`cbst_crossreplay`), the targeted scenarios (`cbst_targeted`), the fast verifier |
| `f0d8920` | `datasets/registry.toml`: the coauth-DBLP archive and both originals' variants |
| `c418064`, `a040c20` | the fixes of the resume (below): output-option files are not inputs; the summary skips hash-only inputs; the comparator sees compactions |
| `40cc358` | the golden sets (`export_goldens.py cbst_trace`), their manifests in `goldens.toml`, the committed subset and the reader, census, dispatch and comparator tests |
| `f320cd6` | ADR 0036, `parity/results/M8a.md` with the census, twin and cross-check records, this retrospective, the M8a / M8b split in `docs/developer/plan.md` and `docs/roadmap.md`, GOVERNANCE.md, CHANGELOG |
| (the certify commit) | `certify.py` lists the `cbst_trace` sets without gating them before the store's replay records exist (deviation 7) |

### Deviations from the plan and the step's text (each with the same intent)

1. **MOSP_ESCHER's `bin/main` is not in `cbst_trace/mosp_escher_test`.** It draws its graph from
   `std::random_device` (`mosp/src/generateGraph.cu:89-90`), so two runs of the unpatched copy
   print different numbers and record different traces; no set that contains it can be exported
   twice with identical manifests. It is recorded in every export and listed in the set's
   `info.json` under `excluded` with its op counts; its CBST calls are `DynamicGraph`'s, which
   the seeded `stressTest`, `parallelStressTest` and `test_dynamicgraph_roundtrip` make too (all
   in the set). The twin output check compares it on its exit status and verdict lines.
2. **Section names.** The sets are named `cbst_trace/<set>` in the milestone text and live in
   `$DYNG_SCRATCH/goldens/cbst_trace/<set>`; their `goldens.toml` sections are
   `[sets.cbst_trace_<set>]`, because the goldens.toml writers split sections by a name without
   `/`. The writers now accept digits in set names (`coauth50k`).
3. **Digest modes.** "Full mode for small runs" is applied per tree: in the default `auto` mode a
   tree with more than 4,096 keys or row values gets delta digests (the changed and named keys in
   full, the 64-bit hash of every other key's record). Delta digests are lossless (both readers
   rebuild every state and check every hash); the test suites' 100K-hyperedge runs would
   otherwise need several GB per trace.
4. **Targeted sets.** `cbst_trace/escher_gpu_targeted` and `cbst_trace/mosp_escher_targeted`
   (`parity/exporters/escher/cbst_targeted.cpp`) drive each original's unchanged
   `CBSTOperations` API through the paths the suites miss or that are too large to commit (the
   coverage table of `parity/results/M8a.md`).
5. **ADR number.** The trace format and recorder ADR is 0036: the 0.2.0 release branch took 0035.
6. **The cross-check of A's line graph.** A rebuilds its H2H (and V2V) from the device every
   batch (`rebuildFromDevice`), which B's store has no call for: A's H2H and V2V traces are
   compared through B's store up to their first rebuild (`rebuild_only_in_a`); the step's
   minimum (H2V and V2H of A, and B's h2h through A) is fully replayed. A's coauth H2H inputs are
   hash-only (the 10 GB budget), so they could not be replayed anyway.
7. **The trace sets in the release certificate.** `certify.py` lists the `cbst_trace` sets
   (kind `cbst_trace`) but keeps a set without a replay record out of the golden verdict: the
   store that replays them does not exist before step 2. A replay record of one is checked as
   usual; the store's replay writes them later (`parity/results/M8a.md`, section 8).
8. **Subset size.** The committed subset is 307,170 bytes ("at most about 300 KB"); the
   repository's test data stays at 470 KB, under 1 MB.

### The interruption and the resume (2026-10-07)

The step was interrupted by a usage limit after its first full chain of checks. Two exports of that
chain had failed and were fixed on resume, each by a recorder- or reader-side change, never by
relaxing a comparison: `cbst_trace_mosp_escher_test` differed between its two exports because the
recorder took the value of `hsospBench --out` (a CSV an earlier run had appended to) as an input
file; `cbst_trace_escher_gpu_syn1m` failed in the summary, which read hash-only inputs. Both sets
were exported twice again with identical manifests; the sets that passed were not re-exported
(no other invocation names an existing output file, so their traces are unchanged by the
recorder fix). The cross-checks that read the changed set, and A through A, were re-run with the
final comparator: 0 unexplained divergences.

### Measurements

The goldens take 6.75 GB of zstd data (limit 10 GB; largest: B's DBLP hyperedge run, 2.75 GB);
the full chain (fresh builds and suites, census, twin checks, datasets, 8 double exports,
cross-checks) runs in about 80 minutes on GPU 1. See `parity/results/M8a.md` (the originals' suites, the census, the twin check, the golden sets
and their sizes, the dataset pins, the cross-check, the coverage table).

### Notes for step 2 (the host model)

- Read the traces through `dyng::test::cbst_trace::reader` and `state_tracker`
  (`cpp/tests/parity/escher/cbst_trace.hpp`): the tracker already rebuilds every state and
  checks the recorded hashes; the comparator compares a store's state with the tracker's.
- Large sets are zstd files: decompress to a temporary file (`zstd -dc --long=31`) or pipe into
  the reader's `std::istream` constructor; the side file (`trace.blob.zst`) is only needed for
  construct and rebuild inputs.
- Hash-only arrays (`DYNG_CBST_TRACE_HASH_ONLY`; listed per set in `goldens.toml`) cannot be
  replayed: a tree whose rebuild or construct input is hash-only is checked by its hashes only.

## Step 2: the CBST host model and the replay comparator (host-model)

### What was built

| Commit | What |
|---|---|
| `200b36c` | `dyng::testing::cbst_host_model` (`dyng/testing/cbst_host_model.hpp`, `cpp/src/testing/cbst_host_model.cpp`, its own object library `dyng_testing_cbst_model` inside `dyng_testing`), `cbst_host_policy` with the presets `escher_gpu()`, `mosp_escher()`, `dyng_default()`, the layout helpers `cbst_rank_of_position` / `check_cbst_layout`; its tests (the cbst_test sizes, the documented rules); the isolation check `cbst_model_isolation`; `public_api.txt`, the CBST identifiers in `ci/provenance_check.py`, CHANGELOG |
| `52912ad` | `cpp/tests/parity/escher/cbst_compare.{hpp,cpp}` (the adapter, the host model's adapter, the replay with three field sets, JSON reports, `trace_source` for `.cbt.zst`), the tool `dyng_cbst_replay`, the replay of the committed subset and the perturbed traces (cpu, parity, cbst), the large-set replay (parity, cbst, slow; skipped without the sets), `parity/compare.py cbst_trace --store host_model` and its tests |
| (this commit) | ADR 0037, `parity/results/M8a.md` section 10 and the records `M8a-host-model-<set>-exact.json`, this section, GOVERNANCE.md |

### Deviations from the step's text (each with the same intent)

9. **The model keeps the flat payload**, not only the per-key segments and rows the step lists: the
   originals' observable rows depend on payload bytes a row-level model loses (dead rows keep their
   values, a reuse clears only the first segment, the unfill kernels differ by degree bin). The
   per-key segments and rows are derived from it (`records()`); ADR 0037, section 1.
10. **Three field sets instead of "criterion 8 plus offsets when asked".** `semantic` is exactly
    criterion 8, `offsets` adds the offsets (for step 3), and `exact` compares every recorded
    field; the model is held to `exact`, the strictest. Nothing is compared less.
11. **Unreplayable operations are reported, not compared.** 3 ops of `escher_gpu_coauth50k` (the H2H
    rebuilds, hash-only inputs) and the 54 ops of the batches-mode set `escher_gpu_syn1m` cannot be
    replayed by any store; the replay reports them with their reason and fails on any other. The
    acceptance criterion's "100% agreement on every set" holds for every replayable op.
12. **The public-header rule.** `dyng/testing/cbst_host_model.hpp` exposes CBST keys, offsets and the
    layout, the documented exception to criterion 9's rule (which is about the hypergraph API).
13. **Model records are not release-certificate records.** The replay records of this step are
    `parity/results/M8a-host-model-<set>-<fields>.json`; `certify.py` keeps waiting for the store's
    `parity-cbst_trace_<set>-*.json` records (step 3 on), so the host model cannot stand in for
    the store in the 0.3.0 certificate.
14. **`dyng_default()` is provisional.** It composes B's compaction triggers with A's growth where B
    throws; no trace exists for it. Step 4 aligns it with the store's dyng preset.

### Measurements

The Release-built replay of all 8 sets through the model takes 4 min 45 s with 8 jobs, the
manifest checks included (the largest, B's DBLP hyperedge run: 2 min, about 63 million key states
compared). The layout check of the 17 large sizes takes 20 s in the Debug build.

### Notes for step 3 (the verbatim port)

- Drive the store through `store_adapter` (`cpp/tests/parity/escher/cbst_compare.hpp`): implement
  `create_tree`, `destroy_tree`, `run` (the seven ops; `InsertMapping` in `op_outcome::mapping`,
  degree-bin launches in `bins` with `has_bins`), `scalars` and `digest` (read back the device
  state: keys in BST order, `subtree_avail` by heap position, per key the record and the row
  walked as the recorder walks it), then `replay(reader, adapter, {field_set::offsets or exact})`.
  The model's adapter (`host_model_adapter`) is the reference implementation of the interface.
- `parity/compare.py cbst_trace --store device` is reserved (it exits 2 now): point it at a device
  replay tool, and write the certificate records `parity-cbst_trace_<set>-*.json` there.
- Under `escher_gpu`, A's sets must pass `exact` (offsets, occupancy, `deviceBytes()`, bin launches
  are all in it); the scenarios of step 4 can mirror every op on `cbst_host_model` and compare
  `records()` with the store's digest.
- The 3 coauth H2H rebuilds stay unreplayable until M8b computes the line-graph merge: their inputs'
  recorded hashes can then check dynG's merge.

## Step 3: the verbatim port of ESCHER-GPU's store (verbatim-port)

### What was built

| Commit | Class | What |
|---|---|---|
| `477d592` | c1 transplant | ESCHER-GPU@abcf9b2's store in `cpp/src/hypergraph/escher` (`dyng::detail::escher`, the OBJECT module `dyng_hypergraph`), regenerated from the original by `parity/escher_port/transplant.py` (the rename map, the manual hunks); the device adapter of the comparator, the device replays of the committed traces, `dyng_cbst_replay --store device` |
| `3634128` | c2 errors | `DYNG_CUDA_TRY`, `capacity_error` at the offset limit and in `flatten_rows`, the construct below its payload rejected, the log sink, `find()`, poisoning; the comparator's documented divergences |
| `528b41a` | c3 memory | `buffer<T>` on the resources' memory resource, pooled temporaries |
| `9220f9c` | c4 streams | the resources' stream, `par_nosync`, A's read points only, stable sorts |
| `f2c48df` | c5 registration | `DYNG_REGISTER_KERNEL`, profiler stages, per-store bin counters |
| `69cac6d` | harness | `compare.py cbst_trace --store device` (MOSP_ESCHER's sets judged against the host model under the same preset) |
| `c89a9b8`, `e22cd71`, `9e54f3a`, `cef82b5` | dynG additions | the empty tree; clone; download; `allocated_bytes()` |
| `d2be78a` | tests | the cbst_test layout sizes, kernel-path coverage, errors, two streams, host syncs per op, the source rules |
| `f67eee1` | policy | `store_policy` and `escher_gpu()` |
| `3716bbe` | test | the early memory reading on coauth-DBLP, `perf_ab.py cbst memory` |
| `762522e`, `f16c453` | harness, CI | `parity/cbst_merge_steps.sh`, the transplant check, the certificate records; `ci/gpu_local.sh` step `cbst`, the provenance symbols |
| (the docs commit) | docs | ADR 0038, the store's README, `parity/results/M8a.md` section 11 with the merge-step table, this section, GOVERNANCE.md, CHANGELOG |

### Deviations from the step's text (each with the same intent)

15. **c1 carries the replay harness.** The step asks for the replay after every commit, so the
    device adapter, its tests and `--store device` of the replay tool are in c1 next to the
    transplant; the store files of c1 are the pure transplant (`transplant.py --check`).
16. **The exception of a construct below its payload is `invalid_argument_error`** (the caller's
    capacity contradicts its own payload; `capacity_error` is reserved for the offset limit and
    full payloads). A kept a tree without nodes there, so the replay reports it as the documented
    divergence `construct_below_payload` (one op of the golden sets, A's targeted scenario
    `small_capacity`), by a precise rule and not as a tolerated mismatch.
17. **The device error word has no user yet.** No kernel of A detects a contract violation;
    B's bounded chain walker (step 4) is its first user.
18. **The comparator learned three things** (test code, used by every later step): the store's own
    integrity verdict is compared with the trace's `check.rows` (after the state, so that a first
    divergence names a state field), documented divergences are reported apart from mismatches,
    and `compare.py` judges MOSP_ESCHER's sets under `escher_gpu()` by their first divergences per
    tree against the host model under the same preset (the step's "up to their first divergence,
    listed with its cause": the causes are the behaviour classes of ADR 0037).
19. **`store_policy` is its own commit after the additions**, not part of the transplant: c1-c5
    keep A's growth inline as A had it, and the policy struct then takes it over with the same
    numbers (the replays are unchanged).
20. **Thrust's temporary storage** comes from a Thrust allocator over `memory_resource_ref`
    (`util/thrust_allocator.cuh`): Thrust takes an allocator, not a CCCL memory resource, so the
    adapters of `cccl_memory_resource.cuh` are not needed for it.
21. **The layout test calls two host launchers of the library** (`launch_build_layout`,
    `launch_extract_keys_from_positions`): nvcc keeps the host stubs of a library's kernels
    hidden, so a test cannot launch them directly as A's `cbst_test` did.
22. **`find.cu`.** A's debug lookups printed from the device; `find()` returns the heap position
    of every key, and `findContents` (which printed rows) is gone (rows are read with `download()`).
23. **Untested at full scale:** `flatten_rows`' 2^31 check and the growth beyond `INT_MAX` need more
    than 8 GB of host or device memory; the policy's numbers (`grown_capacity` at and beyond
    `INT_MAX`) are unit-tested on the host, `flatten_rows`' check is code-reviewed only.
24. **The memory reading replays the coauth H2H rebuilds by their sizes.** Their inputs are
    hash-only (A's line-graph merge); the accounting depends on the row count, the used payload
    and the capacity only, which are recorded in full.
25. **The merge-step script builds a reduced library** (`DYNG_ALGORITHMS=sssp`, minimal index
    types, no tools or tutorials, Release): only the replay targets are needed, and every commit
    builds in a few minutes.
26. **ADR number.** 0038 (0035 is the release branch's, 0036-0037 this milestone's).
27. **Names.** Every camelCase identifier of the ported files became its snake_case spelling
    (locals included), the kernels `<verb>_<object>_kernel`; all listed in `rename_map.toml`.
    The store's stages are `hypergraph.cbst.<op>`, nested under the container's
    `hypergraph.apply.*` of the timed-region map.

### Measurements

- The replay of the small sets (escher_gpu_test, escher_gpu_targeted, mosp_escher_test,
  mosp_escher_targeted) on the device takes about 4 minutes with a Release tool and 6 jobs on one
  GPU; a commit of the merge-step table takes about 8 minutes (the reduced build included). The
  large sets take about 25 minutes (coauth-DBLP's 33 million key states, B's DBLP runs with 63 and
  48 million).
- The store accounting on coauth-DBLP equals A's `deviceBytes()` at every op (ratio 1.0000);
  the nsys peaks: A's whole program 3,018.9 MiB, the store replay 2,772.6 MiB
  (`parity/results/M8a.md`, section 11).
- Sanitizers: racecheck over the store tests takes 85 s (the 2^24-key layout), the other tools a
  few seconds each.

- `ci/gpu_local.sh` (dev-cuda, GPU 1, at `7e397a2`): all 14 steps passed, the new step `cbst`
  included (about 2 hours; memcheck over every GPU executable is most of it).

### Notes for step 4 (MOSP_ESCHER's behaviours)

- The store takes a `store_policy`; add `mosp_escher()` and grow the struct (B's fixed capacity
  `flat * 1.3 + extra`, compaction triggers, construct validation, the fill clamps, scratch
  growth, caller counts) one behaviour per commit; `parity/cbst_merge_steps.sh` replays any list
  of commits.
- The device adapter accepts `escher_gpu` only (`device_store_adapter`); `compare.py cbst_trace
  --store device --policy` rejects the others. Map B's `escher::EscherError` to the dynG
  exception of each throw site in the comparator when the B preset arrives (the exception type is
  compared).
- MOSP_ESCHER's traces now diverge under `escher_gpu()` at three classes only (caller counts,
  scratch growth, compaction / capacity / validation): each commit of step 4 should move the
  first divergence of exactly the trees of its class, which `first_divergences` per tree shows.
- `cbst_store.hpp` still includes `<cuda_runtime.h>` (as A's `structure.hpp`); the container of
  M8b, compiled by the host compiler in CPU builds, needs a host-only facade or a forward-declared
  context.

## Step 4: MOSP_ESCHER's behaviours, part 1 (merge-invariant)

### What was built

- Ten behaviours of MOSP_ESCHER@4b86159 merged into the store, one per commit (ADR 0039): (a)
  construct validation; (b) the fill clamps; (c) caller row counts as a fast path of construct;
  (d) one bounded chain walk, `chain.cuh`, for every reader and the unfill writer; (e) B's
  integrity checks and dynG's index checks; (f) `rows()` (B's `find.cu`) and B's `delete_avail.cu`
  comment; (g) the scratch growth rule; (h) an explicit `compact()` (`compact.cu`); (i) the storage
  counters (`stats()`, B's accounting scope) without B's process-global mempool setting; (j) a rule
  for appends and removals on dead or absent keys. (k), the disposition of every other differing
  file, is ADR 0039 section 6 (`atomicAdd_sll` verified unused and not ported).
- The store's control block: PLAN 4.7.3's device error word, its counters and a pinned mirror
  read back with every read point; `check_errors()`.
- The harness: the staging preset `mosp_escher_staged()` and its host-model twin
  (`staged_host_policy()`), so that B's traces are judged after every commit; the merge-step rows
  list the first divergences by field; every construct of every replay built both ways (twin
  constructs); the replay compares three integrity checks; `decode_op` shared by the replay and the
  new compact-after-every-op lockstep (`dyng_cbst_compact_tests`, `dyng_cbst_compact_large_tests`).
- 23 new tests (17 in `cbst_merge_test.cpp`; the twin constructs, the lockstep (2), the validation, the staging preset and the scratch rules), the merge-step table (`parity/results/M8a.md`,
  section 12), ADR 0039, the store README and ADR 0038 updated.

### Deviations from the step's text (each with the same intent)

28. **A staging preset.** The step asks for B's sets "under the most complete B preset of that
    commit"; there was no B preset before step 5, so `store_policy::mosp_escher_staged()` (B's
    behaviours merged so far, B's fixed capacity, no automatic compaction) is that preset, and the
    comparator's `staged_host_policy()` gives the host model the same behaviours. Each commit then
    shows B's divergences moving: 50 + 7 trees before, 18 + 4 after (only automatic compaction and
    the constructs without row counts remain). Step 5 replaces the preset by `mosp_escher()`.
29. **Harness commits in the table.** `2bc0e24` (the staging preset), `1e01ad8` (the rows by
    field) and `4402fcf` (the provenance identifiers) change no store behaviour; they have rows of
    their own.
30. **The chain bound** is the used payload, not B's 1,024 hops (as the step asks; ADR 0039
    section 3): the longest chain of any trace is 98 hops, so no trace changes, and the stop
    condition (B's bound observable in its own traces) does not apply.
31. **The clamps go further than B's literal code:** dynG clamps the occupancy (the write position
    and an overflow's chain pointer stay in the segment), B only the room; the warp and block unfill
    kernels clamp it too. Only inconsistent metadata is affected; no trace state has it.
32. **The device error word is a control block** of 4 words outside ESCHER-GPU's accounting scope:
    `device_bytes()` must stay A's `deviceBytes()` (criterion 10(a)) and `allocated_bytes()` keeps
    its scope; the accounting test counts the block separately.
33. **Dead or absent keys:** the rule chosen of the two the step offers is "count them and surface
    them as internal_error at the next existing sync point"; the items are also skipped (nothing is
    written). An empty tree's append and remove_values now throw (ADR 0038 had them change
    nothing); the test was changed accordingly.
34. **Caller counts** are also range-checked on the host (dynG; B trusted them), and the Debug check
    against the scan costs one synchronization in Debug builds only. A's traces have no counts:
    their twin constructs use counts computed on the host.
35. **`compact()` grows or throws `capacity_error`** where B threw `EscherError`, before anything
    changed (not poisoning); a traced `compact()` is unsupported under `escher_gpu()` (A had none),
    except in the lockstep, which enables it.
36. **The host model's `compact()`** already existed (step 2 wrote it from B's `compactCBST`,
    independently of the store); the step uses it as the oracle of the lockstep instead of adding a
    second one.
37. **Validation:** a capacity below the payload is now rejected for a tree of 0 records too (B's
    rule; step 3 allowed it), and a record count above `INT_MAX / 3` is a `capacity_error` (dynG:
    the scratch's int32 indexing).
38. **The integrity checks in the replay:** `check.subtree_avail` is compared with B's recorded
    verdict and with 0 for A's states (A had no such check); `check.index` (dynG's) with 0.
39. **Per-commit checks:** before each commit the CUDA build, `ctest -L cbst -LE slow`, memcheck on
    the touched tests (and racecheck after (d) and (h)) and the fast `ci/check.sh` steps (format,
    reuse, provenance, pre-commit); the replays right after each commit (a temporary worktree
    needs the commit); the full `ci/check.sh`, the portability checks and the large sets at the
    end of the step.
40. **The 11 `test_cbst_smoke` constructs without row counts** stay listed divergences: B's
    `nullptr` path (occupancy 0) is never merged; step 5 must document them for criterion 10(b).
41. **ADR number** 0039. The scratch rule of dynG's default preset (step 5) is chosen here:
    geometric.

### Measurements

- A commit of the merge-step table takes about 8-10 minutes (the reduced Release build and the
  small sets under both presets; GPU 1, shared with the 0.2.0 release's test runs).
- The compact-after-every-op lockstep over the four small sets takes 211 s (Release, GPU 1):
  15,385 ops, 15,383 compactions.
- Racecheck over `dyng_cbst_store_tests` takes about 170 s (the 2^24-key layout and the new
  2,000-segment row).

### Notes for step 5 (automatic compaction and the presets)

- Replace `mosp_escher_staged()` by `mosp_escher()` (`store_policy::compaction_rule`: B's trigger
  points in `fillImpl` and `insertCBST`, ADR 0039 section 2 (h) gives the kernels) and its
  host-model twin by the model's `mosp_escher()`; `compare.py`'s `against_model` then stops
  judging B's sets against the model.
- Map `escher::EscherError` to the dynG exception of each throw site in the comparator (construct
  validation: `invalid_argument_error`; fill, insert, compact capacity: `capacity_error`).
- Document the 11 `rowOccupancy = nullptr` constructs of `test_cbst_smoke` (B's legacy occupancy
  0, never merged) as a divergence class, or criterion 10(b) cannot hold for them.
- dynG's default: compact first, then grow (A's rule), the geometric scratch, `capacity_error` only
  at the int32 offsets or on OOM; `cbst_host_policy::dyng_default()` must be aligned with it.
- The mutations of criterion 17 that touch this part (compaction padding, surplus padding, the
  tie order, requested keys) can use `dyng_cbst_compact_tests` and the replays: a changed
  compaction padding fails the lockstep.
