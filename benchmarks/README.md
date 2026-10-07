# Benchmark suites and release certificates

This directory holds the benchmark-suite records of the papers whose code dynG ports (PLAN
Section 8.5) and, per release, the committed results: the performance gates against the pinned
originals (PLAN Section 8.6) and the parity certificate (PLAN Section 8.3). The scripts that use
them live in `parity/`: `parity/bench_suite.py` runs a suite through the A/B harness and
`parity/certify.py` writes the certificate.

```text
benchmarks/
  paper/<suite>.yaml          one machine-readable suite per paper and algorithm
  results/<version>/          what a release commits: the suite summaries, the compacted harness
                              records, the golden replays, the checks, parity.json, README.md
```

## The suites

| Suite | Algorithm | Paper | Readings |
|---|---|---|---|
| `paper/ipdps25_dynamosp_sosp.yaml` | `sssp` | DynaMOSP (IPDPS 2025, TPDS 2025): the per-objective SOSP update | roadNet-PA, roadNet-CA, rgg_n_2_20_s0, road_usa; K = 3, U[1, 100] seed 12345, source 0; 50K safe / 50K unsafe / 10K local batches (seed 777); OpenMP (28 threads pinned) and CUDA (boost clock lock) gated; CUDA at default clocks recorded; device memory |
| `paper/ieee_tc_dyntrucy.yaml` | `cycle_count` | TruCy / DynTruCy (IEEE TC) | DD, GitHub, Twitch, COLLAB; static k = 3..7, updates 25K+25K (DD also 50K and 100K), seed 1; OpenMP (56 threads) and CUDA (boost lock; the COLLAB update at the base lock, ADR 0021) gated; CUDA at default clocks recorded; device memory |
| `paper/ipdps25_dynamosp_mosp.yaml` (0.2) | `mosp` | DynaMOSP (IPDPS 2025): the MOSP update, the K per-objective updates plus the combined graph and its SOSP | the same four graphs, inputs and batches as the sssp suite (K = 3, default preferences); "(a) compute" (<= 1.05x) and "(b) end to end" (<= 1.10x, every output file written by both sides, as the originals' `bench/run.sh`) against the unpatched `bin/mosp` of MOSP-OpenMP@c352151 (28 threads pinned) and MOSP-CUDA@e220ee2 (boost clock lock); CUDA at default clocks recorded; device memory (`perf_ab.py memory --mosp`, <= 1.05x); the per-objective updates, the combined step and the path costs reported |

The mosp suite reads the gates M7 took (`parity/results/M7.md` sections 6 and 8) and is required
in every certificate from 0.2.0 on; the per-objective update alone stays in the sssp suite. The K
sweep of the TPDS 2025 extension (roadNet-CA widened to four objectives, `tpds25_dynamosp`) is
reported, not gated (M7.md section 5), and listed under `deferred` in the mosp suite; the temporal
datasets of DynTruCy (CollegeMsg, email-Eu-core, time-window mode) come with the temporal modes in
0.4 (listed under `deferred` in the suite). The ESCHER, DynLP and H-SOSP suites follow their
ports.

## The suite format

A suite is YAML (schema 1). `parity/bench_suite.py validate` checks every key below, and checks
the file against the harness it drives, so the suite cannot silently describe something else than
what is measured.

| Key | Content |
|---|---|
| `suite`, `title`, `algorithm`, `paper` | the name (equal to the file name), what is measured, the paper and its scope |
| `harness` | the harness entry (`perf_ab.sssp`: `parity/perf_ab.py run` or `memory`; `perf_ab.mosp`: `parity/perf_ab.py mosp` or `memory --mosp`; `perf_ab.cycle_count`: `parity/perf_ab.py cycle_count run` or `memory`), the input root, the compat executable, the build presets (`parity`, `parity-cuda`), the region map |
| `graph` | sssp, mosp: K (`num_weights`), the weights' distribution and seed, the source; mosp: the preferences (`default`, all 1, the only ones the harness runs) |
| `datasets` | name, `source` (collection, id, URL), `recipe` (how the inputs are made from the source), `sha256` (one digest, or a map file -> digest relative to the input root) |
| `batches` | name, generator, sizes, `seed` (and the harness's directory for sssp and mosp) |
| `cases` | cycle_count: task, dataset, k, batch; the harness names a case `<dataset>_k<k>[_<D>_<I>_s<seed>]` |
| `backends` | threads, GPU, scopes, environment, and the baseline each backend is compared with; mosp: the CUDA engine (`engine`, `perf_ab.py mosp --cuda-engine`; `automatic` is the fused engine on the RTX A5000) |
| `runs` | sssp, mosp: alternating rounds per dataset (`default`, or per dataset); cycle_count: per reading |
| `metrics` | each gated region of `parity/timed_regions/<algorithm>.toml`, per backend: the original's timer and the dynG profiler stages it sums, and its gate kind (`compute`, `end_to_end`, `memory`, `none`) |
| `baselines` | the pinned originals (commit equal to `parity/references.toml`, variant `unpatched`) |
| `tolerance` | PLAN 8.6: `compute` 1.05, `compute_short` 1.10 below `short_region_ms` 10, `end_to_end` 1.10, `memory` 1.05 |
| `readings` | one harness invocation per dataset (sssp, mosp) or per case list (cycle_count): `kind` run or memory, backend, `clocks` (CUDA: `boost`, `base`, or `none` = default clocks), runs, `gated` (a default-clock reading is recorded and never gated, ADR 0018) |
| `paper_reference` | informational: the paper's published numbers |

`validate` requires, among others: every gated region of the region map is a metric with the same
gate and the same profiler stages; the baselines are the pinned unpatched commits; the
tolerances are the ones the harness applies; every case of a reading is a case of the harness and
of the suite; a default-clock reading is not gated.

## Running a suite

The protocol is the one of PLAN 8.6 and `parity/README.md`: parity presets; the originals built
from the unpatched `git archive` copies; every timing under the exclusive lock
`$DYNG_SCRATCH/perf.lock` (the harness takes it itself); medians of alternating rounds; the
contamination monitor repeats disturbed rounds; outputs compared in every round; CUDA on GPU 0
with the clocks locked for the whole A/B (ADRs 0018 and 0021), the default-clock reading
recorded next to it.

```bash
source scripts/dev_env.sh
parity/build_reference.sh                       # the pinned originals (unpatched + patched)
for g in roadNet-PA roadNet-CA rgg road_usa_g; do parity/perf_ab.py prepare --graph $g; done
cmake --preset parity && cmake --build --preset parity
cmake --preset parity-cuda && cmake --build --preset parity-cuda
parity/bench_suite.py validate benchmarks/paper/*.yaml
parity/bench_suite.py plan benchmarks/paper/ieee_tc_dyntrucy.yaml        # the commands, not run
parity/bench_suite.py run benchmarks/paper/ipdps25_dynamosp_sosp.yaml --version 0.1.0rc1
parity/bench_suite.py run benchmarks/paper/ieee_tc_dyntrucy.yaml --version 0.1.0rc1
parity/bench_suite.py run benchmarks/paper/ipdps25_dynamosp_mosp.yaml --version 0.2.0rc1
```

`run` first checks the SHA-256 of every input against the suite (`--no-verify-inputs` skips it),
then runs each planned command; `--skip-existing` resumes an interrupted execution, and
`--build-root` points at another checkout's build trees (for example a clean clone of the release
commit). The full records go to `$DYNG_SCRATCH/runs/bench/<version>/`, with the log of every
command (`<suite>.log`); `run` never replaces a record that exists (`--skip-existing` keeps it,
`--force` replaces it; to take a reading again, move its record to `superseded/` first).
`--readings`, `--datasets` and (sssp, mosp) `--batches` select a part and `--runs` overrides the
rounds (at least 5): such a **narrowed** execution is summarized as `<suite>.partial.json` and only into an `--out` outside
`benchmarks/results/` (a release's summary is the whole suite's; after taking readings again, run
`summarize` for the whole suite). The harness records name the NVIDIA driver and the CUDA driver
API version, and the inputs as measured: the sssp harness hashes every input file when a run
starts, the cycle_count harness hashes the datasets in the measuring process. `summarize` (run at
the end of `run`, or on its own) writes to `benchmarks/results/<version>/`:

- `<suite>-<reading>[-<dataset>].json`: each harness record, compacted (the per-round
  machine-monitor windows are dropped; the samples, verdicts, rejected rounds and clock checks are
  kept; the record names its full copy);
- `<suite>.json`: every reading with its regions (original and port medians, ratio, gate,
  verdict), the commits that were measured, the driver each record names, when the execution
  started, the inputs check and the verdict.

The summary re-derives every gate from the suite's tolerances and fails if a record's gate,
reference commit or variant, input digests, clock lock or build preset differ from the suite, if
a record is of another algorithm or other batches than the suite's (a mosp record must say
`mosp`: an sssp record of `perf_ab.py run` or `memory` without `--mosp` has the same shape), if a
mosp record has another K, other preferences or another CUDA engine, did not write the output
files, or has `invalidated` counts that differ between the two sides in a round, if
a gated reading is missing or incomplete, if a gated region exceeds its gate or is provisional
(a region under 10 ms read with fewer than 20 rounds), or if the inputs of a record are not
verified. A record's inputs are verified by the digests the harness took while measuring; a
record without such digests (the sssp records of 0.1.0rc1, whose digests are the ones
`perf_ab.py prepare` wrote; the memory records of 0.1.0rc1) is verified only if every input file
hashes to the suite's digest now **and** has not been written or replaced (modification and
status-change times) since the execution started, the first command of `<suite>.log`. It lists, without failing (PLAN 8.6:
flagged, not failed), the runs the OpenMP cycle_count harness flagged for foreign CPU load above
its threshold (the other harnesses repeat such rounds); a reading with flagged runs is read again
on a quiet machine before a release.

## The release certificate

PLAN 10.3 steps 1-3 (`docs/developer/release.md`) end in `benchmarks/results/<version>/`:

1. **The golden replays** (`parity/compare.py ... --json benchmarks/results/<version>/parity-<set>-<backend>.json`):
   every golden set of `parity/goldens.toml` on every backend it covers (sssp: sequential, OpenMP
   and CUDA; the cycle_count CPU corpus `--full`: sequential and OpenMP; the cycle_count CUDA
   corpus: cuda, cuda:resident, cuda:int64).
2. **The performance gates**: `parity/bench_suite.py run` for every suite of the release's
   algorithms. `certify.py` requires the suites of `REQUIRED_SUITES` by release series:
   `ipdps25_dynamosp_sosp` (sssp) and `ieee_tc_dyntrucy` (cycle_count) from 0.1,
   `ipdps25_dynamosp_mosp` (mosp) from 0.2 (a 0.2.0rc1 or 0.2.0 certificate without the mosp
   gates fails); each summary must hold a gated reading of every gated metric of its suite file
   on every backend it is gated on (mosp: "(a) compute" and "(b) end to end" on OpenMP and CUDA,
   the device memory on CUDA).
3. **The checks**: the `asan` and `tsan` presets' test suites, `ci/gpu_local.sh` (the CUDA
   tests, the CUDA goldens and compute-sanitizer memcheck, synccheck and racecheck), the mutation
   CTests (`ctest -L mutation`) and the golden mutations (`parity/mutate.py run --json
   benchmarks/results/<version>/mutation-sssp.json`), one `parity/certify.py check` each, for
   example

   ```bash
   parity/certify.py check --version 0.1.0rc1 --name asan --command "ctest --preset asan" \
       --ctest-log asan-ctest.log --commit <the commit that was built>
   parity/certify.py check --version 0.1.0rc1 --name gpu_local --command ci/gpu_local.sh \
       --scope repo --gpu-summary build/dev-cuda/gpu_local_summary.md \
       --require-steps memcheck,synccheck,racecheck --tests-log gpu_local.log
   parity/certify.py check --version 0.1.0rc1 --name check-parity --scope repo \
       --command "ci/check.sh --parity" --result passed --evidence check-parity.log \
       --tests-log check-parity.log
   ```

   (`--ctest-log` reads ctest's summary line, `--gpu-summary` the step table of
   `ci/gpu_local.sh`, `--json-verdict` a record with a top-level `passed` such as
   `mutation-sssp.json`, `--result passed|failed --details ...` anything else); they collect in
   `checks.json`.

   Every check names the commit that was built and its **scope**, the paths that must be the
   release's (below). A ctest log (`--ctest-log`, or `--tests-log` for another log with ctest
   output, such as `ci/gpu_local.sh`'s) also gives the per-test results the committed fixtures
   need. Every log a check names is recorded with its SHA-256 and size, and an excerpt (its step
   headers and result lines) is committed under `benchmarks/results/<version>/checks/`; a check
   recorded with `--result` alone must give its log with `--evidence`. A `--gpu-summary` check
   records the steps it required (`--require-steps`; a skipped step passes only if it was not
   required).

Then, on the committed release tree, `parity/certify.py write --version <version>` writes
`parity.json` and the generated part of `README.md` (between the `certify` markers; the text
around it is written by hand). The certificate records:

- the release commit and the commits whose builds were measured, each of which must have the
  release commit's sources in the paths of its scope (`git diff <measured> <release> --
  <paths>` is empty):

  | Scope | Paths | For |
  |---|---|---|
  | `library` | `cpp/include cpp/src cpp/CMakeLists.txt tools cmake CMakeLists.txt CMakePresets.json docs/references.bib` | the gates, the golden replays, `mutate.py` (`check --scope library`) |
  | `tests` | the library paths, `cpp/tests`, `examples` and `README.md` (its C++ quickstart is a test) | a C++ test suite: the sanitizer presets, `ctest -L mutation` (the default of `check`) |
  | `packaging` | the test paths, `python`, `pyproject.toml`, `VERSION`, `CHANGELOG.md`, `CITATION.cff`, the licence files, `ci/wheels.sh`, `ci/wheel_check.py`, `release.yml`, `wheels.yml` | the distributions (`select`, `ci/wheels.sh`, `twine check`, `wheel_check`, the fresh venvs) |
  | `repo` | every tracked file but `benchmarks/results/` | a check of the whole tree: `ci/check.sh`, `ci/gpu_local.sh` |

  `VERSION` is not a library path: it only names the build, so a release candidate's gates
  certify the final release (step 9 of `docs/developer/release.md`).

  **The one exception: generated metadata.** A measured commit whose library differs from the
  release's **only** in `METADATA_PATHS` of `certify.py` (the manifests
  `cpp/src/algorithms/{sssp,cycle_count}/manifest.toml`, which no build reads, and
  `cpp/src/core/registry_table.inc`, the registry table `scripts/regen.py` generates from them,
  read only by `dyng::algorithms()`) still certifies the gates, the golden replays and the golden
  mutations, in the library scope only and never a test-suite check (whose tests read the
  registry), if `equivalence.json` holds a passed pair for it whose release side has the
  release's library. The pair is written by

  ```bash
  parity/certify.py equivalence --version <version> --measured <sha> --clone <a clean clone>
  ```

  which refuses a difference outside `METADATA_PATHS`, then builds, in that one clone and with
  the measured commit's `VERSION` and build system, the presets `parity` and `parity-cuda`
  (targets `dyng-compat-mosp`, `dyng-compat-cycle-enum`) twice: at the measured commit, and with
  the release's library paths checked out over it. It compares `libdyng.so` and both compat
  tools: each must be byte-identical, or equivalent, meaning every section and symbol at the same
  address and every differing byte inside the listed functions' code (`dyng::algorithms()`) or in
  the build id, unwind and symbol tables. The 0.1.0rc1 certificate uses it (measured at
  `d13d393`, before the maturity change);
- the hardware, the CUDA and compiler versions, and the NVIDIA driver **of the measurements**:
  the one the records name (all equal, and equal to the driver seen when the certificate is
  written), or, for CUDA records that predate the per-record field (0.1.0rc1), the driver seen
  when the certificate is written, accepted only if it is shown unchanged since before the first
  measurement (the loaded kernel module, the boot time, the module file and `libcuda.so.1`
  installed before it);
- the pinned originals (commit, the local `baseline-2026-09` SHA, upstream);
- every golden set of `parity/goldens.toml` with its manifest SHA-256, the SHA-256 of every
  case and every replay's case x configuration matrix and tolerance;
- every **committed fixture set** of `cpp/tests/data` (`parity/fixtures/fixtures.toml`: the
  generator script, the original and its commit, the CTests that compare the set) with the
  SHA-256 of its files and the result of each of its tests in the release checks;
- the suites the version requires (`required_suites`, from `REQUIRED_SUITES`), the
  performance-gate table of every suite (gated, and recorded-only readings separately) and how
  each reading's inputs were verified;
- the checks, with their scopes and evidence.

It exits 1, and says why in `verdict.problems`, if any part failed or is missing (a required
suite, or a gated metric of a suite without a gated reading), if a replayed
manifest is not the one of `goldens.toml`, if a measured tree was dirty, if a suite summary is
not of the committed suite file (SHA-256), is partial, or misses, repeats or has an incomplete
reading of the suite's plan, if a summary's inputs are not verified, if the records' drivers
disagree, if a directory of `cpp/tests/data` is not in `fixtures.toml`, or if a fixture test
pattern was not passed by any release check (or failed in one).
