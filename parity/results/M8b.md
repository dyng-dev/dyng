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
