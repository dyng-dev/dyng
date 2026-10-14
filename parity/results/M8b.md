# M8b parity record: the hypergraph container

Date: 2026-10-13 onwards. M8b puts the public `hypergraph<vertex_t, weight_t>` on the ESCHER CBST
store of M8a and proves its ids against ESCHER-GPU@abcf9b2 (A) and MOSP_ESCHER@4b86159 (B) at
the level of the container (PLAN 6.4.5, 8.3, 11.3; ADR 0041). Each step adds its section.

| Item | Value |
|---|---|
| Branch | `m8b-hypergraph`, created by the orchestrator from the head of the accepted M8a branch `m8a-escher-store` at `b45e685` (M8a's pull request #14 is open: `git merge-base --is-ancestor m8a-escher-store origin/main` fails, so the base is M8a's head, not `origin/main`; `origin/main` `30c3eef` (0.2.0 released, 0.3.0.dev0) is already merged into it at `889953a`). `origin/main` is merged again once the author merges M8a. |
| References | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A), MOSP_ESCHER `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (B, the committed tree only) |
| Inputs | M8a's CBST trace sets (`parity/goldens.toml`, `[sets.cbst_trace_*]`, files under `$DYNG_SCRATCH/goldens/cbst_trace`): their batch records (A: `escher_gpu.build` / `escher_gpu.update`; B: `mosp_escher.bulk_load` / `begin_batch` / `finish_batch`) are the dumps of the container-level replay |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86 |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1; GPU 1 for every untimed run |

## 1. Step container-core: the container and its id parity

### 1.1 What is compared (criterion 4; ADR 0041, section 11)

`parity/compare.py hypergraph_ids --set <set> --backend <b> [--threads N] --record parity/results`
replays the batch records of M8a's CBST trace sets through `dyng::hypergraph` with
`dyng_hypergraph_ids` (built in the Release tree `build/parity-cuda`), after verifying each set's
manifest:

- **A (ESCHER-GPU@abcf9b2)**, sets `cbst_trace_escher_gpu_test` (every `make test` invocation),
  `cbst_trace_escher_gpu_coauth50k` (coauth-DBLP, 3 x 50K, seed 1) and
  `cbst_trace_escher_gpu_syn1m` (synthetic 1M, 3 x 50K, seeds 1-3): the loaded rows of
  `DynamicHypergraph::build` (A's vertex `v` = dynG vertex `v - 1`, key `k` = id `k - 1`) under
  `{erase_first, escher_gpu(min_initial_payload = A's --payload-capacity when the trace's H2V
  capacity shows one)}`; per batch `inserted_ids == insertedKeys - 1`, the h2v rows by id equal
  A's host model (`h2v_` rebuilt from the batch records), v2h equals the transpose; a second
  container deletes by content (deviation 1 of the retrospective: the member sets after every
  batch, the ids up to the first ambiguous resolution).
- **B (MOSP_ESCHER@4b86159)**, sets `cbst_trace_mosp_escher_test` (every `make test` invocation;
  the sessions of the hsosp tests and benches) and `cbst_trace_mosp_escher_dblp50k_hyperedge`:
  each `bulkLoad` session under `{insert_first, mosp_escher(extra)}` with the Caps factor and extra
  its trees agree on (deviation 2), B's rows as they are (vertices 0-based; B id `i` = dynG id
  `i - 1`; the virtual source and target are ordinary rows, weight 0 loaded as 1), ops on a
  virtual id filtered and counted, each session up to its first vertex-kind batch; per batch
  `inserted_ids == finalIds - 1`, `ignored_deletions + filtered == skippedOps`,
  `dropped_members ==` B's deduplicated members (its `he_insert` values minus its H2V insert's
  payload), the weights of the new ids equal `final_weights`, h2v and v2h equal B's model.
- Every batch also runs `check_integrity()` and compares the container with two
  `testing::cbst_host_model`s (H2V, V2H) driven with the container's operations (criterion 6),
  and on CUDA records `store_stats().device_bytes` next to A's bytes.
- The set `committed` is the committed `001-main.cbt` and the two batch extracts
  (`cpp/tests/data/goldens/hypergraph_batches`), which `dyng_hypergraph_ids_tests` also replays in
  hosted CI on the CPU backends.

### 1.2 Replay records

Run at `9de082d` (the tool of `build/parity-cuda`, Release), GPU 1 for `cuda`; one record per set
and configuration, `parity/results/M8b-hypergraph-ids-<set>-<backend>[-t<threads>].json`, with
every case's report (its counts, the first mismatches, the store's kernel paths and device bytes).
The records of A's sets on `cuda` are of the re-run at `9de082d` that also reads A's per-tree bytes
(section 1.4); every other record is of `2bb19e9`'s tool (the same comparison).

| set | backend | traces | sessions | batches | vertex-kind (not replayed) | ids | rows | B batches with skipped compared | B skippedOps | dropped members | virtual ops filtered | ambiguous by content | mismatches | verdict | record |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| committed | cuda | 3 | 11 | 14 | 8 | 291 | 25186 | 10 | 0 | 0 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-committed-cuda.json` |
| committed | openmp (1 threads) | 3 | 11 | 14 | 8 | 291 | 25186 | 10 | 0 | 0 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-committed-openmp-t1.json` |
| committed | openmp (8 threads) | 3 | 11 | 14 | 8 | 291 | 25186 | 10 | 0 | 0 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-committed-openmp-t8.json` |
| committed | sequential | 3 | 11 | 14 | 8 | 291 | 25186 | 10 | 0 | 0 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-committed-sequential.json` |
| cbst_trace_escher_gpu_coauth50k | cuda | 1 | 1 | 3 | 0 | 75000 | 12337324 | 0 | 0 | 0 | 0 | 824 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_coauth50k-cuda.json` |
| cbst_trace_escher_gpu_coauth50k | openmp (1 threads) | 1 | 1 | 3 | 0 | 75000 | 12337324 | 0 | 0 | 0 | 0 | 824 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_coauth50k-openmp-t1.json` |
| cbst_trace_escher_gpu_coauth50k | openmp (8 threads) | 1 | 1 | 3 | 0 | 75000 | 12337324 | 0 | 0 | 0 | 0 | 824 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_coauth50k-openmp-t8.json` |
| cbst_trace_escher_gpu_coauth50k | sequential | 1 | 1 | 3 | 0 | 75000 | 12337324 | 0 | 0 | 0 | 0 | 824 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_coauth50k-sequential.json` |
| cbst_trace_escher_gpu_syn1m | cuda | 3 | 3 | 9 | 0 | 225000 | 12001327 | 0 | 0 | 0 | 0 | 3159 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_syn1m-cuda.json` |
| cbst_trace_escher_gpu_syn1m | openmp (1 threads) | 3 | 3 | 9 | 0 | 225000 | 12001327 | 0 | 0 | 0 | 0 | 3159 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_syn1m-openmp-t1.json` |
| cbst_trace_escher_gpu_syn1m | openmp (8 threads) | 3 | 3 | 9 | 0 | 225000 | 12001327 | 0 | 0 | 0 | 0 | 3159 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_syn1m-openmp-t8.json` |
| cbst_trace_escher_gpu_syn1m | sequential | 3 | 3 | 9 | 0 | 225000 | 12001327 | 0 | 0 | 0 | 0 | 3159 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_syn1m-sequential.json` |
| cbst_trace_escher_gpu_test | cuda | 75 | 75 | 180 | 0 | 114965 | 3196275 | 0 | 0 | 0 | 0 | 21330 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_test-cuda.json` |
| cbst_trace_escher_gpu_test | openmp (1 threads) | 75 | 75 | 180 | 0 | 114965 | 3196275 | 0 | 0 | 0 | 0 | 21330 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_test-openmp-t1.json` |
| cbst_trace_escher_gpu_test | openmp (8 threads) | 75 | 75 | 180 | 0 | 114965 | 3196275 | 0 | 0 | 0 | 0 | 21330 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_test-openmp-t8.json` |
| cbst_trace_escher_gpu_test | sequential | 75 | 75 | 180 | 0 | 114965 | 3196275 | 0 | 0 | 0 | 0 | 21330 | 0 | ok | `M8b-hypergraph-ids-escher_gpu_test-sequential.json` |
| cbst_trace_mosp_escher_dblp50k_hyperedge | cuda | 1 | 1 | 3 | 0 | 75000 | 9942176 | 3 | 0 | 22703 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_dblp50k_hyperedge-cuda.json` |
| cbst_trace_mosp_escher_dblp50k_hyperedge | openmp (1 threads) | 1 | 1 | 3 | 0 | 75000 | 9942176 | 3 | 0 | 22703 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_dblp50k_hyperedge-openmp-t1.json` |
| cbst_trace_mosp_escher_dblp50k_hyperedge | openmp (8 threads) | 1 | 1 | 3 | 0 | 75000 | 9942176 | 3 | 0 | 22703 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_dblp50k_hyperedge-openmp-t8.json` |
| cbst_trace_mosp_escher_dblp50k_hyperedge | sequential | 1 | 1 | 3 | 0 | 75000 | 9942176 | 3 | 0 | 22703 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_dblp50k_hyperedge-sequential.json` |
| cbst_trace_mosp_escher_test | cuda | 24 | 153 | 129 | 96 | 34597 | 1154269 | 127 | 0 | 841 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_test-cuda.json` |
| cbst_trace_mosp_escher_test | openmp (1 threads) | 24 | 153 | 129 | 96 | 34597 | 1154269 | 127 | 0 | 841 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_test-openmp-t1.json` |
| cbst_trace_mosp_escher_test | openmp (8 threads) | 24 | 153 | 129 | 96 | 34597 | 1154269 | 127 | 0 | 841 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_test-openmp-t8.json` |
| cbst_trace_mosp_escher_test | sequential | 24 | 153 | 129 | 96 | 34597 | 1154269 | 127 | 0 | 841 | 0 | 0 | 0 | ok | `M8b-hypergraph-ids-mosp_escher_test-sequential.json` |

**Every case of every set, on every backend, gives the original's ids, rows and skipped counts:
0 mismatches.** Notes:

- *ids*: inserted ids compared (A: `insertedKeys - 1`; B: `finalIds - 1`); *rows*: h2v rows by id
  compared after the build and every batch (v2h, the integrity check and the host-model lockstep
  run at the same points).
- *ambiguous by content* (A): deleted member sets that several live hyperedges hold, where the
  smallest id is not the one A deleted (retrospective, deviation 1); the member-set multiset of
  the by-content container equals A's after every batch in every case, and its ids equal A's up
  to the first ambiguous resolution.
- B's sessions stop at their first vertex-kind batch (*vertex-kind (not replayed)*, M11's). B's
  replayed batches have no skipped deletion and name no virtual hyperedge (*B skippedOps* 0,
  *virtual ops filtered* 0, retrospective deviation 4); *dropped members* are B's silently
  deduplicated members (`hsospBench` real hyperedge batches: 841 in `020-hsospBench`), counted by
  dynG in `dropped_members`.
- `mosp_escher_test`: of its 24 traces, the 15 without a `bulkLoad` (`test_cbst_smoke`, `test_cbst_ops`, the
  `DynamicGraph` tests and the MOSP stress tests on the graph-on-CBST) contribute no batch; B threw in 5 of its hooks: 3 bulk loads (their sessions are not replayed) and 2 batches
  rejected in B's validation, compared as rejected (dynG throws `invalid_argument_error`, the state
  unchanged). In `dblp50k_hyperedge` B deduplicated 22,703 repeated members over the 3 batches; dynG
  dropped the same number.

### 1.3 Backend agreement (criterion 5)

The same records on `sequential`, `openmp` with 1 and 8 threads and `cuda`: each backend equals
the original in every compared field, so the backends agree with each other in inserted ids,
rows and summaries. The host backends' ids come from the independent allocator
(`id_allocator.cpp`), the CUDA backend's from the store. Beyond the replays:
`HypergraphRandom.*` (random streams of nine mixes under four policies and both orders, every
backend in lockstep with the host model and with each other, openmp at 1 and 8 threads),
`CloneAndToBackendKeepIdsAndLaterIds` (ids, rows and payload accounting survive clone and
to_backend in every direction, and a later batch gives the same ids on the copy), and
`HypergraphCuda.RepeatedRunsGiveIdenticalIds` (10 runs of a 6,000-hyperedge stream on CUDA, both
orders, equal to each other and to sequential). Hosted CI runs `dyng_hypergraph_ids_tests`
(the committed set) on the CPU backends.

### 1.4 Store bytes (early reading; the gate is step 2's)

In the scope the container has in this step (the H2V and V2H trees and the H2V row view; no line
graph), `store_stats().device_bytes` on CUDA equals ESCHER-GPU's bytes of the same objects (its
trees' `deviceBytes()` from their op records plus its H2V `RowLocator`, `(numHyperedgeKeys + 1) x
12` bytes) at every point: 255 of 255 points of `escher_gpu_test`, 12 of 12 of `syn1m`, 4 of 4 of
`coauth50k` (ratio 1.0000). On coauth-DBLP: 605,045,612 bytes after the build, 605,046,632 /
605,046,644 / 605,047,088 after the three batches; A's whole `DynamicHypergraph::deviceBytes()`
(1,922,784,204 ... 1,924,170,844) adds its H2H tree and view, which step 2 adds to the container.

### 1.5 Kernel paths (criterion 12)

`HypergraphIds.KernelPathsOfTheSkewedAndMediumTraces` (CUDA, `dyng_hypergraph_ids_cuda_tests`):
A's skewed (`042-main`, 2 x 2,800) and medium (`043-main`, 3 x 640) traces replayed through the
container, `store_stats()` counts fill thread 5 / warp 3 / block 2, reuse 5 / 3 / 2, unfill
5 / 3 / 2 (summed over both): every path ran.

### 1.6 Checks of the step

| Check | Result |
|---|---|
| `ci/check.sh` (format, the cpu-only and dev presets with `ctest -L cpu`, consumer, tidy, reuse, provenance, regen, harness, docs, python, api, scaffold, pre-commit) | OK (the first run failed provenance, the harness's mutation-config test, the docs' missing hypergraph page and, through it, scaffold and pre-commit; fixed in the docs/provenance commit and re-run: OK) |
| `ctest -L 'hypergraph\|cbst\|api' -LE slow` on `dev-cuda` (GPU 1) | all passed (2 skipped: a second device, the CPU copy of the CUDA-only path test) |
| tests | `dyng_hypergraph_tests` 47, `dyng_hypergraph_cuda_tests` 24, `dyng_hypergraph_ids_tests` 2, `dyng_hypergraph_ids_cuda_tests` 2 |
| compute-sanitizer memcheck (`DYNG_CBST_REDUCED=1`) | `dyng_hypergraph_cuda_tests`, `dyng_hypergraph_ids_cuda_tests`: 0 errors, 0 leaks |
| mutations (`ctest -R hypergraph.mutation`) | control passes; `h2v_not_deduplicated` killed by 5 tests, `allocator_ties_reversed` by 4 |
| M8a's store replays (`compare.py cbst_trace --store device`, the original's preset, `exact`) | `escher_gpu_targeted`, `mosp_escher_targeted`, `escher_gpu_test`, `mosp_escher_test`: OK, 0 mismatches, the pinned documented divergences |
| header self-containment, `DYNG_INDEX_TYPES=minimal` | built and passed (the cpu-only tree) |
| portability: Clang 18 `-fsyntax-only` over the dev database; cpu-only with `-D_FORTIFY_SOURCE=3` | 0 failures over 286 files; built, `dyng_hypergraph_tests` passed |
| every commit of the step built (cpu-only, a temporary worktree) | `07e00d8`, `f995b78`, `5b64c23`, `6d8a773`: built, their hypergraph tests passed |

## 2. Step line-graph-views-memory: the line graph, the views, the memory gate

### 2.1 What was built (ADR 0042)

- `line_graph_policy::sorted_merge` and `automatic` (built by the first `h2h()` or
  `hypergraph_access::require_line_graph()`, then kept); `none` throws, `slack_csr` raises
  `not_supported_error` naming 0.4. The stage `hypergraph.apply.line_graph` (A's
  `maintain.h2h_merge`) is emitted when the line graph is kept.
- The port of ESCHER-GPU's sorted delta merge (`cpp/src/hypergraph/line_graph/sorted_merge.cu`,
  `sorted_merge.cuh`, `device_frontier.cuh`), staged: the transplant (`3b73335`, reproducible with
  `parity/escher_port/transplant.py --map parity/escher_port/rename_map_line_graph.toml --check
  --rev 3b73335`: 3 files identical), then errors (`fbfa931`), buffers (`8823491`), streams
  (`0d9d97f`) and registration (`cdae5c6`). The initial line graph is the port of
  `flatHyperedgeAdjacency` (`line_graph/flat_adjacency.cpp`); the CPU backends keep the line graph
  with the host form of the merge (`line_graph/sorted_merge_host.cpp`), in A's layout.
- The internal row views (`cpp/src/hypergraph/row_views.hpp`: A's `RowView` of H2V and of H2H with
  `out`) rebuilt once per commit on every backend; the decoded public views per version.
- `dyng::testing::rebuild_line_graph` (`dyng/testing/line_graph_oracle.hpp`; MOSP_ESCHER's
  `rebuildLineGraph`), isolated (CTest `line_graph_oracle_isolation`: no project include but its
  header, no undefined dynG symbol).

### 2.2 The line graph against ESCHER-GPU (criterion 7)

`parity/compare.py hypergraph_ids --line-graph --mode by_id` replays A's batch records with the
line graph kept and compares, after the build and every batch, the container's H2H layout (row
offsets and padded payload, ESCHER-GPU's construct layout) with the input of A's H2H construct or
rebuild at the same point, by the FNV-1a hashes the trace records (every set records them); where
the trace has state digests, every key's row hash (coauth: the rows are recorded as hashes only)
and the rows the digests record in full (`make test`). Equal layouts are equal rows by id,
`out` offsets and padding included.

Replayed after every commit of the port (`3b73335` ... `cdae5c6`; the Release tool of
`build/parity-cuda`, GPU 1), each time with the same result:

| set | states (build + batches) | layouts equal | payload entries compared | key hashes | full rows | mismatches |
|---|---:|---:|---:|---:|---:|---:|
| `escher_gpu_test` (75 traces) | 255 | 255 | 274,227,060 | 3,095,365 | 3,095,365 | 0 |
| `escher_gpu_coauth50k` | 4 | 4 | 1,032,222,412 | 9,869,850 | 0 (hash-only rows) | 0 |
| `escher_gpu_syn1m` (seeds 1-3) | 12 | 12 | 1,118,210,756 | 0 (`batches` set: no digests) | 0 | 0 |

The same three sets on `sequential` and `openmp` (8 threads) give the same result (the host form of
the merge; during the port and again at `16898f4`). Records:
`parity/results/M8b-h2h-<set>-<backend>[-t8].json` (the final runs, section 2.7).

### 2.3 The memory gate (criterion 8): GO

`parity/perf_ab.py hypergraph memory --gpu 0` (exclusive perf lock; record
`parity/results/M8b-memory.json`): the container under `{erase_first, escher_gpu(),
sorted_merge}` loaded from the construct rows of `escher_gpu_coauth50k` and replaying its 3 x 50K
batches (seed 1); `store_stats().device_bytes` (A's `DynamicHypergraph::deviceBytes()` scope: the
H2V, V2H and H2H trees and the H2V and H2H row views) against A's recorded value at each point:

| point | ESCHER-GPU `deviceBytes()` | dynG `device_bytes` | ratio (gate <= 1.05) |
|---|---:|---:|---:|
| after the build | 1,922,784,204 | 1,922,784,204 | 1.0000 |
| after batch 1 | 1,923,300,764 | 1,923,300,764 | 1.0000 |
| after batch 2 | 1,923,713,052 | 1,923,713,052 | 1.0000 |
| after batch 3 | 1,924,170,844 | 1,924,170,844 | 1.0000 |

Byte for byte equal at every point (the same trees, sized the same way, and the same row
views; the line graph's key count is A's `numHyperedgeKeys`, ADR 0042 section 3). The unpatched
copy's own print points (`src/driver.cu:196` / `:239`) print `[memory] ESCHER device allocations:
1833.7 MB after init, 1835.0 MB after the last batch`, which are the recorded first and last values
(1,922,784,204 and 1,924,170,844 bytes in MiB). The replays of section 2.2 check the same ratio at
every point of `escher_gpu_test` and `syn1m` (255 + 12 points): all 1.0000.

**Nsight Systems peaks** (recorded, not gated in M8; the author's decision of 2026-10-07): the
peak of live device allocations is **3,018.9 MiB** for A's whole `build/main --input coauth.hg
--batches 3 --batch-size 50000 --seed 1 --check` (the store, its row views and the motif counting)
and **3,019.1 MiB** for the container's replay (the stores and views, 1,834 MiB, plus the merge's
temporaries: the new H2H payload and row offsets exist next to the old ones until the rebuild takes
them over, as in A; the stream-ordered pool reached 3,104 MiB). M9 gates this peak with the
counting included.

### 2.4 The maintain reading (criterion 9; reported, not gated)

`parity/perf_ab.py hypergraph maintain --gpu 0 --runs 5 --lock-clocks base` (record
`parity/results/M8b-maintain-base.json`): A/B/A/B, 5 accepted rounds per side and dataset, the
exclusive perf lock, GPU 0, one heap layout (a fixed `--timing` file name), the contamination
monitor on every round. A side is the UNPATCHED `build/main` of the golden case (coauth-DBLP 50K
seed 1; synthetic 1M `1000000 10 1 333333`, seed 1; without `--check`), timed by its ScopedTimers
(device-synchronized at both ends); the other is `dyng_hypergraph_ids --apply-only --line-graph
--sync-stages --timing` replaying the same run's batch records (the stream synchronized at the
stage boundaries). Regions from `parity/timed_regions/hypergraph.toml`.

**The clock lock.** The first reading at `--lock-clocks boost` (SM 1695 MHz) rejected 11 rounds in
a row on coauth: A's process (the full motif count of 5 s precedes the batches) runs under the
power cap at 1455-1680 MHz, and in some rounds dynG's as well. As ADR 0021 decides for such a
case, the reading is taken at the base lock (SM 1170 MHz, memory 7601 MHz), applied to both
programs and checked in every busy sample; no round was rejected there.

Medians of 5 rounds (ms; the sum of the 3 batches and the worst batch):

| region (A's timers -> dynG's stages) | coauth50k A | coauth50k dynG | ratio | syn1m A | syn1m dynG | ratio | worst batch |
|---|---:|---:|---:|---:|---:|---:|---|
| maintain (`update.maintain` -> `hypergraph.apply`) | 207.78 | 177.37 | **0.854** | 165.83 | 142.37 | **0.859** | syn1m batch 2: 1.026 |
| maintain.delete (`h2v_delete + v2h_unfill` -> `.apply.delete`) | 11.11 | 19.07 | 1.716 | 9.77 | 28.88 | 2.957 | syn1m batch 2: 4.006 |
| maintain.insert (`h2v_insert + v2h_fill` -> `.apply.insert`) | 57.72 | 39.28 | 0.681 | 43.53 | 37.95 | 0.872 | syn1m batch 2: 1.003 |
| maintain.line_graph (`h2h_merge` -> `.apply.line_graph`) | 96.77 | 80.27 | 0.829 | 59.83 | 48.50 | 0.811 | coauth batch 3: 0.840 |
| maintain.views (`row_views` -> `.apply.views`) | 22.65 | 6.77 | 0.299 | 15.26 | 3.38 | 0.221 | syn1m batch 2: 0.981 |

**Host synchronizations per batch** (one Nsight Systems CUDA API trace per side, the blocking
calls inside each batch's maintain region: A's NVTX range `update.maintain`, dynG's
`hypergraph.apply` without stage synchronization): A 76 (13 `cudaDeviceSynchronize` of its
ScopedTimers, 18 `cudaMemcpy`, 23 `cudaStreamSynchronize`, 22 `cudaMemset`; 70 in syn1m's batch
2), dynG 21 `cudaStreamSynchronize` (18 in syn1m's batch 2), on both datasets.

**Stages above 1.05x and their profile.** One region is above 1.05x: `maintain.delete`. A CPU-
sampled Nsight Systems profile of the port on syn1m (`nsys profile -t cuda,nvtx -s cpu`, NVTX
stage ranges; about 1 sample per 0.85 ms) puts 31 of the delete stage's 36 samples in
`group_by_vertex` of the deleted rows (the host sort of their incidences by vertex). A does the
same grouping in `maintain.prepare` (host, outside its delete timers, inside `update.maintain`);
its delete timers hold only the erase and unfill kernels. The difference is where the grouping is
timed, not more work: `maintain` as a whole is 0.85x on both datasets. The same profile found the
largest host cost of dynG's apply outside the stages in the applied record (the sort of the
touched vertices, about 16 ms per syn1m batch), fixed in `16898f4` before this reading (the
reading before it: maintain 0.893x on coauth50k and 1.120x on syn1m). **M9 risk** (recorded in
the retrospective): `update` = subtract + maintain + add is gated in M9; maintain itself is below
A, and the delete stage's host grouping (about 9 ms per syn1m batch, single-threaded) is the
first candidate if the M9 reading needs room (a parallel or device grouping).

### 2.5 Views, profiler stages, C12 and sanitizers (criteria 7, 19, 11 C12)

- Views: `HypergraphLineGraph.ViewsAndHowOftenTheyAreBuilt` (every backend): the row views
  rebuilt once per commit (two applies, two rebuilds), no public view built by `apply()`, `v2h()`
  and `h2h()` built at most once per version; the internal views decoded equal the public ones,
  `out` the first entry above the key; views in the hypergraph's memory space.
- Profiler stages (criterion 19): `hypergraph.build`, `.upload`, `.apply`, `.apply.delete`,
  `.apply.insert`, `.apply.line_graph`, `.apply.views` (tests `ProfilerStages` and
  `HypergraphLineGraph.ProfilerStage`), the names of `parity/timed_regions/hypergraph.toml`.
- C12: `HypergraphCuda.TwoContainersOnTwoNonBlockingStreams`: two containers with the line graph
  on two non-blocking streams, driven by two host threads, equal the sequential runs (ids and the
  line graph's layout), both `id_reuse` orders.

### 2.6 Checks of the step

| Check | Result |
|---|---|
| `ci/check.sh` (format, the cpu-only and dev presets with `ctest -L cpu`, consumer, tidy, reuse, provenance, regen, harness, python, docs, api, scaffold, pre-commit) | OK after `3f5a272` (the first run found a missing provenance line of `host_line_graph.hpp` and three spellings codespell reads as typos); `format provenance precommit` re-run: OK |
| `ctest -L hypergraph -LE slow` on `dev-cuda` (GPU 1) | 110 passed (at `16898f4`); `ctest -L 'hypergraph\|cbst\|api' -LE slow`: all passed at `3b73335` (272 tests) |
| `dyng_hypergraph_tests` / `dyng_hypergraph_cuda_tests` | 66 / 35 tests (step 1: 47 / 24; the line-graph suite on every backend, C12) |
| compute-sanitizer memcheck (`--leak-check full`, `DYNG_CBST_REDUCED=1`) | `dyng_hypergraph_cuda_tests` (35 tests), `dyng_hypergraph_ids_cuda_tests`: 0 errors, 0 leaks |
| compute-sanitizer racecheck and synccheck (`DYNG_CBST_REDUCED=1`, without the long random streams, as `ci/gpu_local.sh` runs them) | `dyng_hypergraph_cuda_tests` (31 tests: the line graph on cuda, C12, the container suites) and `dyng_hypergraph_ids_cuda_tests`: 0 hazards, 0 errors. A full racecheck of the random streams was stopped after 42 minutes (it takes hours) |
| mutations (`parity/mutate.py run --suites hypergraph`, at `cd9cc9e`) | control passes; `hypergraph_h2v_not_deduplicated`, `hypergraph_allocator_ties_reversed` and `hypergraph_mirror_additions_dropped` detected (the last by 10 host and 8 CUDA tests); the CTest `hypergraph.mutation.mirror_additions_dropped` passes (the mutant fails) |
| the id replays (no line graph) on cuda at `16898f4` | `escher_gpu_test`, `mosp_escher_test`, `committed`: OK, 0 mismatches (as in section 1) |
| M8a's small store replays | not affected (no change in `cpp/src/hypergraph/escher` but the move of `struct row_view`); `ctest -L cbst` passed at `3b73335` |
| portability: Clang 18 `-fsyntax-only` over the dev database; cpu-only with `-D_FORTIFY_SOURCE=3` | 0 failures over 335 files; built, `ctest -L hypergraph` passed (73 tests) |
| `transplant.py --map parity/escher_port/rename_map_line_graph.toml --check --rev 3b73335` | 3 files identical; M8a's `transplant.py --check --rev 477d592` still identical |

### 2.7 Records

- `M8b-h2h-<set>-<backend>[-t8].json`: `escher_gpu_test`, `escher_gpu_coauth50k`, `escher_gpu_syn1m` on
  cuda, openmp (8 threads) and sequential, at `16898f4` (the Release tool of `build/parity-cuda`):
  every case passed, every H2H state equal to A's, `memory_ratio_max` exactly 1 on cuda.
- `M8b-memory.json` (criterion 8; at `16898f4`, the working tree differing only in these records
  and the retrospective): passed.
- `M8b-maintain-base.json` (criterion 9; at `16898f4`): the reading of section 2.4.

## 3. Step update-kit-scaffold: the update path, the kit's hypergraph axis, the scaffold

### 3.1 What was built (ADRs 0043, 0044)

- The hypergraph's two-phase transaction (`cpp/src/hypergraph/transaction.hpp`: begin with Step 0
  once, `before()` with the old view valid until the commit, `commit()`, `after()` with the
  applied record); `hypergraph::apply()` is a transaction without participants.
- `dyng::update()` / `update_each()` over hypergraphs: `detail::participant_of<hypergraph<V,
  W>>`, `hyper_update_participant`, the hypergraph's `run_update()` (one Step 0, one commit, every
  before-apply on G_t, every after-apply on G_{t+1}, poisoning of a failing result while the
  others complete); `update.hpp` and `cpp/include/dyng/graph/*` are byte-identical to `main`
  (`git diff main HEAD -- cpp/include/dyng/update.hpp cpp/include/dyng/graph/`: empty).
- The framework's adapter for problems on hypergraphs; the conformance kit's container axis
  (`container_traits`, `container_checks`); the test-only aggregate-delta fakes
  (`test_hyper_pairs`, `test_hyper_squares`) through C0-C12 on every backend; the planted bug.
- `scripts/new_algorithm.py --container hypergraph` and `regen.py`'s hypergraph requirements.

Nothing in this step changes what the container computes: the id replays, the line graph and the
memory accounting of sections 1-2 are untouched (the transaction runs the same store phase in the
same order; `hypergraph_test`, `hypergraph_line_graph_test`, the random streams and the committed
id replays pass on every backend).

### 3.2 The update path (criterion 10)

| Property | Test |
|---|---|
| begin, `before()`, commit, `after()`, `applied()`; `before()` after the commit and a second commit throw | `HypergraphUpdate.TheTransactionRunsBeginBeforeCommitAfter` (every backend) |
| an old view used after the commit throws (Debug builds; I1) | `HypergraphUpdate.AnOldViewUsedAfterTheCommitThrowsInDebugBuilds`, `...AProblemThatKeepsItsOldViewPastTheCommitFailsInDebugBuilds` (the result is poisoned, the hypergraph at version + 1); skipped with NDEBUG |
| begin rejects before anything changes (version, state, rows); a commit after another apply is refused | `...TheBeginRejectsAnInvalidBatchBeforeAnythingChanges`, `...ACommitAfterAnotherApplyIsRefused` |
| one Step 0 (the same staged object for every participant) and one commit; the order use_normalized, before@t, after@t+1; the applied record's host and device inserted ids | `...OneStepZeroAndOneCommitForEveryResult` (update and update_each) |
| duplicate targets; a before-apply rejection changes nothing; an invalid batch is rejected before any participant runs | `...AResultPassedTwiceIsRejectedBeforeAnythingChanges`, `...ARejectionBeforeTheCommitChangesNothing` |
| a failing participant poisoned, the others complete, the first error rethrown | `...AFailingResultIsPoisonedAndTheOthersComplete`, `...AFailingProblemIsPoisonedWhileItsPartnerCompletes` |
| the line graph prepared for its readers (built on G_t under `automatic`; refused under `none`) | `...TheLineGraphIsPreparedForItsReaders` |
| stale results by version and state identity (another hypergraph, a clone that diverged) | `...AStaleResultIsDetected` |
| stages `update.normalize`, `update.commit`, `hypergraph.apply` and sub-stages | `...TheStagesOfAnUpdate` |
| aggregate-delta fakes through `dyng::update()` equal to compute() under both orders | `...TheFakesStayEqualToComputeThroughDyngUpdate`, `...WeightedHypergraphsTakePartToo` |
| misuse does not compile | `hypergraph.update.static_assert.{control, owning_batch, edge_batch_with_hypergraph, hyperedge_batch_with_graph, hypergraph_result_with_graph, graph_result_with_hypergraph}` |

### 3.3 The kit's hypergraph axis (criterion 11)

- `dyng_hypergraph_kit_tests` (sequential, openmp; 28 tests: C0-C12 over (int32, unweighted) and
  (int32, int32), the generator checks) and `dyng_hypergraph_kit_cuda_tests` (cuda; C3 compares
  it with both host backends; C12 on two non-blocking streams): every applicable check passes; C4
  is skipped (one engine). C8 in the budgets builds: no allocation in the steady state, 0 (host) and
  2 (CUDA) host synchronizations, the problem's declared budget.
- Presets {erase_first, insert_first} x {escher_gpu(), mosp_escher(), default} plus
  `sorted_merge` as the traits' extra; twelve mixes; `HypergraphKitGenerators.*` checks that each
  makes what its name promises (reuse under erase_first takes the freed ids and insert_first does
  not; a hyperedge of more than 1,024 members on the medium shape; INT64_MIN / INT64_MAX and the
  bound deleted; members 0 and n - 1; ...).
- **The planted bug** (`kit_frontier_requested_ids`, frontier = id_bound(G_t) + i instead of
  `inserted_ids`): `parity/mutate.py run --suites kit` at `22a09ab` (parity-cuda preset, Release):
  control passed; the mutation detected by 6 tests of `dyng_hypergraph_kit_tests` and 6 of
  `dyng_hypergraph_kit_cuda_tests` (C2, C5 and C7 of both types; record
  `M8b-mutate-kit.json`). In the budgets build (`dev`) the CTest
  `conformance.mutation.frontier_requested_ids` passes (the mutant fails C2, C5, C7 and C8).
- **The graph suites are unchanged:** the `--gtest_list_tests` output of every graph conformance
  executable (sssp, cycle_count, mosp, dynamic_bfs, triangle_delta; host and CUDA) and of the
  registry test equals that of the binaries built before the change; all pass.

### 3.4 The scaffold (criterion 11)

`ci/scaffold_check.sh` scaffolds `scaffold_probe_hg_ad` (aggregate delta, sequential and OpenMP)
and `scaffold_probe_hg_fp` (fixed point, sequential) with `--container hypergraph`, builds the
subset with warnings as errors and budgets, and runs their kits and hand cases: 54 tests passed
(next to 80 of the graph probes and 41 of `my_bfs`); Doxygen, the coverage check, the API baseline
(only the probes' headers added) and Sphinx pass on the result; `--remove` leaves the tree as it
was.

### 3.5 Checks of the step

| Check | Result |
|---|---|
| `ci/check.sh` (format, cpu-only and dev with `ctest -L cpu`, consumer, tidy, reuse, provenance, regen, harness, python, api, scaffold, docs, pre-commit) | every step passed at `e80900f` but `precommit` (ruff format of two Python files; fixed in `22a09ab`, `pre-commit run` then clean); cpu-only 1,026 and dev 1,055 tests passed; `api_snapshot --against origin/main`: no breaking change of the frozen API; griffe green |
| `ctest -L gpu` on `dev-cuda` (GPU 1) | 670 / 670 passed (the hypergraph kit on cuda among them) |
| `ctest -L cpu` on `dev-cuda` | 1,055 / 1,055 passed |
| the compile-fail tests | `hypergraph.update.static_assert.*` (6), `update.static_assert.*`, `framework.conformance.*`, `conformance.registration.*`: passed (in `ctest -L cpu`) |
| `ci/scaffold_check.sh` | passed inside `ci/check.sh` (section 3.4) |
| `ci/python.sh`, `ci/docs.sh` | passed inside `ci/check.sh` (the regen outputs unchanged: no new manifest) |
| compute-sanitizer memcheck (`--leak-check full`) | `dyng_hypergraph_kit_cuda_tests` (28 tests) and `dyng_hypergraph_cuda_tests` (50 tests, `DYNG_CBST_REDUCED=1`): 0 errors, 0 leaks, after `524d6c9` (the first run found the device buffers of a hand-run participant's workspace that outlived its resources handle in one test reported as leaked; a test fix) |
| compute-sanitizer racecheck and synccheck | `dyng_hypergraph_kit_cuda_tests` C1, C3, C11, C12 (8 tests): 0 hazards, 0 errors (racecheck 760 s, so `ci/gpu_local.sh` runs C1, C11 and C12); the update tests of `dyng_hypergraph_cuda_tests`: 0 hazards, 0 errors |
| `parity/mutate.py run --suites kit` (at `22a09ab`) | control passed; `kit_frontier_requested_ids` detected (6 + 6 tests) |
| portability: Clang 18 `-fsyntax-only` over the dev and dev-cuda databases; cpu-only with `-D_FORTIFY_SOURCE=3` | 0 failures over 328 + 710 files; built, `ctest -L 'hypergraph\|conformance\|core\|framework'`: 543 passed |
| intermediate commit `672d3bc` in a worktree (dev preset) | built; `ctest -L hypergraph`: 105 passed |
| `git diff main HEAD -- cpp/include/dyng/update.hpp cpp/include/dyng/graph/` | empty |

### 3.6 Records

- `M8b-mutate-kit.json`: the `parity/mutate.py run --suites kit` record at `22a09ab`.
