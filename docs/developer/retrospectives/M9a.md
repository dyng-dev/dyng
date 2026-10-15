# Retrospective: M9a (triad_count against ESCHER-GPU)

Status: **in progress** on the branch `m9a-triad-count`. M9a is the first half of PLAN 11.3's
M9 row: `triad_count` exact against ESCHER-GPU@abcf9b2 on the sequential, OpenMP and CUDA
backends, the independent oracle, the conformance kit and the mutations, and the paper gates
(the Nsight Systems peak included); M9b follows with `experimental::count_local_patterns`,
Python, the CLI, the docs and the freeze. It runs in five steps, and each step adds its section
here. Release 0.3.0 = M8a + M8b + M9a + M9b (PLAN Appendix F; the split is the author's decision
D2).

**The author's decisions for M9 (2026-10-15; recorded in GOVERNANCE.md's approvals log by step
1).** D1: the gated Nsight Systems peak is the peak of live device allocations (`cudaMalloc` +
`cudaMallocAsync`, kind Device) over the whole timed process, on coauth_b50k_s1, syn1m_b50k_s1 and
syn2m_b50k_s1, port / A <= 1.05; the footprint is reported, not gated. D2: the M9a / M9b split.
D3: the host backends are incremental host engines (an untuned transliteration of A's wedge
engine), `testing::triad_reference` the independent oracle. D5: the committed test-data budget is
1.25 MB, at most about 16 KB of make-test counts. D4 and D6-D10 are taken on the plan's defaults,
pending the author's confirmation at the 0.3.0 release.

## Step 1: the reference side (regions-goldens-crosscheck)

### What was built

| Commit | What |
|---|---|
| `2f54bce` | **First commit of the branch**: `parity/timed_regions/triad_count.toml` (the region map: `update[b1..b3]`, `full_count`, `end_to_end` gated; the sub-rows reported; the scope rules), the per-batch rows and the refusal of unknown keys in `perf_ab.load_regions` and `bench_suite.region_map`, `ci/tests/test_triad_regions.py`, ADR 0048 and its GOVERNANCE line |
| `23d0c03` | the author's M9 decisions in GOVERNANCE.md; `plan.md` (the M9 row split into M9a / M9b) and the roadmap (D3: the 0.5 OpenMP item becomes tuning) |
| `903b3aa` | `parity/triad_goldens.py` (export, verify, crosscheck, list), registered in `export_goldens.py` and `compare.py`; `parity/tests/test_triad_goldens.py` |
| `f91b226` | the set `triad_count_escher_gpu` in `parity/goldens.toml` (with the inputs and the coverage table), named in `references.toml` |
| `7eb7e61` | `certify.py` lists a `triad_count` set without replays without a verdict (deviation 13) |
| `07ee804` | `parity/results/M9a.md` section 1, `parity/results/M9a-crosscheck-escher_gpu.json`, this section, ADR 0048's figures, plan.md, CHANGELOG |

Results: the set exported twice from two fresh patched copies with byte-identical manifests
(300 cases, 367.7 MB); every case's `--check` run agrees; `generators::legacy` reproduces all 12
synthetic inputs; A agrees with its own CPU reference on all 90 cross-check runs, whose counts
equal the goldens' (`parity/results/M9a.md` section 1). Nothing was found against the original.

### Deviations from the plan and the step's text (each with the same intent)

1. **The branch starts at `origin/main` `9cde147`**, not at `m8b-hypergraph` `65a62c9`: the
   author had merged PR #15 (M8b) when M9a started, so the step's own rule applies ("if
   origin/main already contains PR #15's merge, branch from origin/main instead"). The frozen-path
   diff of the acceptance (`git diff 65a62c9 -- cpp/include/dyng/{hypergraph,update.hpp,core,
   graph}`) is empty on `9cde147` too.
2. **The unpatched counts of an export come from its own fresh copy.** The export patch keeps A's
   Makefile-built executables unchanged as `parity_export/orig/<exe>` before it installs the
   recording twins (ADR 0036); each of the two exports counts with its copy's originals, so the
   double export compares two independent builds of A end to end. The persistent unpatched copy
   (the performance baseline) is the one the cross-check runs, and its counts equal the goldens'
   on all 90 runs, so both builds print the same counts.
3. **`update.add` includes `identify_affected`.** The step names `update.add = count_plus +
   finalize`, but `update` contains `identify_affected` (A builds the inserted keys' frontier
   inside `update.add`), and the three sub-rows should add up to the gated row on both sides.
4. **`full_count` maps to `triad_count.compute`**, the stage the framework emits for the whole
   `compute()` call (`hook_stages::compute`); the sketch's `triad_count.count` is the aggregate
   count inside it. A's `init.full_count` includes allocating and reading back its bins, so the
   whole call is the matching scope (the step asked to use the framework's name and record it).
5. **One `.dgb` per batch.** `to-dgb` writes one file with all batches; the acceptance asks for the
   SHA-256 of each batch's file, so the exporter splits the file into valid one-record `.dgb`
   files (`batch<N>.dgb`, header count 1, CRC recomputed; dyng's `BatchReader` reads them). The
   SHA-256 in the records is that of the uncompressed file; the stored files are zstd-compressed.
6. **Case names use the size labels of the gate cases** (`coauth_b50k_s1`, `syn1m_b5k_s3`,
   `syn2m_b200k_s2`, `syn100k_b20_s1`), as M8a's sets and the acceptance's gate rows name them;
   `info.json` maps the labels to the sizes. make-test's groups are `test/<NN>-<label>`.
7. **make-test's seven input-rejection invocations are not cases** (they print no counts). The
   60 counting invocations are checked against M8a's set, invocation by invocation.
8. **The `--check` run of a case is its count run plus `--check`.** The suite's own
   `--reference` is exercised by the cross-check, which runs every make-test case with
   `--check --reference` (also the cases the suite runs with `--reference` only).
9. **The region loader is stricter**: it refuses an unknown key of a region (a misspelt
   `per_bach` would otherwise be silently ignored) and a repeated region name; a region with only
   a `port_report` (the process wall time) counts as naming its port side. All existing maps
   load unchanged.
10. **No batches-only recorder mode was added**: the recorder's existing batches mode records
    coauth-DBLP in about 15 s per run (alone), and its exit record already carries `inserted_keys`.
11. **The legacy-generator proof runs through the Python binding** (`dyng.generators.legacy.
    escher_hypergraph_arrays` + `dyng.io.write_hypergraph`), which calls the C++ generator; no new
    C++ tool was needed, so this step changed no C++ code.
12. **`bench_suite.py` knows triad_count's region map** (`REGION_MAPS`, `MAP_KEYS`) but not yet a
    harness tool (`TOOLS`): `perf_ab.py triad_count` arrives in step 2, and a suite file naming it
    before then would not validate.
13. **`certify.py` lists the triad set without a verdict until it has replays.** A golden set
    without replay records fails every certificate (`golden_parity`), and the triad replays
    arrive with the port; until M9a step 5 makes them required from the 0.3 series, a
    `triad_count` set without replays is listed with `passed = None` (shown as "listed"), and a
    replay record, once there, is judged as usual. The 0.1 / 0.2 certificate tests pass
    unchanged.
14. **The author's M9 decisions were recorded in this step** (GOVERNANCE.md, plan.md, roadmap), as
    the first step of the milestone.

### Machine time

| What | Time |
|---|---|
| Double export (two copies built, two exports at once, 3 groups each, GPU 1, shared lock) | 62 min wall (06:28-07:30); 10,569 s / 10,472 s of group time; coauth 61%, syn2m 23% |
| Cross-check (90 runs, 2 at a time) | about 4 min wall, 401.7 s of run time; coauth main's CPU reference 143.5 s (time box 2 h) |
| Development exports (subsets, the existing patched copy) and the first coauth reference run | about 10 min |
| `verify` | 16 s |

### Notes for step 2 (oracle, replay harness, driver, pre-reads)

- **Reading the set** (`$DYNG_SCRATCH/goldens/triad_count/escher_gpu`; `triad_goldens.py`):
  `cases/<group>/group.json` has the input name, the vertex count (`vertices`, A's
  maxVertexId: pass it as `hypergraph_read_options::num_vertices`), the H2V `payload_capacity`
  for `capacity_policy::escher_gpu(capacity)` (0 everywhere but `test/01-smoke`: 4096), and the
  batch files `batch<N>.dgb.zst` (decompress with `zstd -dc`; flags `deleted_members` and
  `expected_ids`, ids = A's keys - 1). `cases/<group>/<exe>.json` has `full_count`, and per batch
  `delta`, `counts_after`, `deletions`, `insertions` and `batch_sha256`. Inputs:
  `inputs/<name>.json` + `inputs/<name>.hg.zst` (A's loaded rows, `index_base` 1), coauth-DBLP
  from `$DYNG_SCRATCH/datasets/hyper/coauth.hg` (identical bytes).
- **Coverage**: `triad_goldens.coverage_cases(backend)` (and `compare.py triad_count --list
  --backend B`) gives the cases per backend: cuda 300, sequential 120, openmp 142.
- **`compare.py triad_count`** today verifies (`--verify`) and lists (`--list`); step 2 adds the
  replay modes and counters in `triad_goldens.compare_main`. Do not change the export functions
  without exporting again (`export --twice`): the committed manifest was produced by `903b3aa`'s
  exporter.
- **Driver names the map expects**: `update_ms`, `full_count_ms`, `total_ms`, `load_ms`,
  `generate_ms` (driver timers, A's `--timing` JSON format), stages `triad_count.update`,
  `triad_count.compute`, `driver.total`, `driver.batch.generate`, `io.read_hypergraph`,
  `hypergraph.build`, `hypergraph.upload`.
- **Memory points for store_memory**: `group.json` `device_bytes` holds A's `deviceBytes()` after
  the build and after every batch for every group (coauth_b50k_s1: 1,922,784,204 after the build,
  as M8a's trace).
- **For step 5**: `certify.golden_suites` lists `triad_count` sets without replays as "not yet
  required" (deviation 13); the step-5 rule (required from (0,3), all backends per the coverage
  table, both modes) replaces that branch. `bench_suite.TOOLS` needs `triad_count` once
  `perf_ab.py triad_count` exists.
- **make-test stdout** (`cases/test/*/<exe>.stdout`) is A's unpatched stdout, for M9b's CLI;
  the hosted-CI exact-parity test of D5 may commit at most about 16 KB of these counts (tracked
  `cpp/tests/data` is 998,934 bytes; the 1.25 MB budget and its check in `ci/check.sh` are not in
  place yet).
