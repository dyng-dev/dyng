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

## The suites of 0.1

| Suite | Algorithm | Paper | Readings |
|---|---|---|---|
| `paper/ipdps25_dynamosp_sosp.yaml` | `sssp` | DynaMOSP (IPDPS 2025, TPDS 2025): the per-objective SOSP update | roadNet-PA, roadNet-CA, rgg_n_2_20_s0, road_usa; K = 3, U[1, 100] seed 12345, source 0; 50K safe / 50K unsafe / 10K local batches (seed 777); OpenMP (28 threads pinned) and CUDA (boost clock lock) gated; CUDA at default clocks recorded; device memory |
| `paper/ieee_tc_dyntrucy.yaml` | `cycle_count` | TruCy / DynTruCy (IEEE TC) | DD, GitHub, Twitch, COLLAB; static k = 3..7, updates 25K+25K (DD also 50K and 100K), seed 1; OpenMP (56 threads) and CUDA (boost lock; the COLLAB update at the base lock, ADR 0021) gated; CUDA at default clocks recorded; device memory |

The mosp-level DynaMOSP suites (`ipdps25_dynamosp.yaml`, `tpds25_dynamosp.yaml`: the K updates
plus the combined graph, the papers' "(a) compute" totals) come with `mosp` in 0.2; the temporal
datasets of DynTruCy (CollegeMsg, email-Eu-core, time-window mode) with the temporal modes in 0.4
(listed under `deferred` in the suite). The ESCHER, DynLP and H-SOSP suites follow their ports.

## The suite format

A suite is YAML (schema 1). `parity/bench_suite.py validate` checks every key below, and checks
the file against the harness it drives, so the suite cannot silently describe something else than
what is measured.

| Key | Content |
|---|---|
| `suite`, `title`, `algorithm`, `paper` | the name (equal to the file name), what is measured, the paper and its scope |
| `harness` | the harness entry (`perf_ab.sssp`: `parity/perf_ab.py run` or `memory`; `perf_ab.cycle_count`: `parity/perf_ab.py cycle_count run` or `memory`), the input root, the compat executable, the build presets (`parity`, `parity-cuda`), the region map |
| `datasets` | name, `source` (collection, id, URL), `recipe` (how the inputs are made from the source), `sha256` (one digest, or a map file -> digest relative to the input root) |
| `batches` | name, generator, sizes, `seed` (and the harness's directory for sssp) |
| `cases` | cycle_count: task, dataset, k, batch; the harness names a case `<dataset>_k<k>[_<D>_<I>_s<seed>]` |
| `backends` | threads, GPU, scopes, environment, and the baseline each backend is compared with |
| `runs` | sssp: alternating rounds per dataset (`default`, or per dataset); cycle_count: per reading |
| `metrics` | each gated region of `parity/timed_regions/<algorithm>.toml`, per backend: the original's timer and the dynG profiler stages it sums, and its gate kind (`compute`, `end_to_end`, `memory`, `none`) |
| `baselines` | the pinned originals (commit equal to `parity/references.toml`, variant `unpatched`) |
| `tolerance` | PLAN 8.6: `compute` 1.05, `compute_short` 1.10 below `short_region_ms` 10, `end_to_end` 1.10, `memory` 1.05 |
| `readings` | one harness invocation per dataset (sssp) or per case list (cycle_count): `kind` run or memory, backend, `clocks` (CUDA: `boost`, `base`, or `none` = default clocks), runs, `gated` (a default-clock reading is recorded and never gated, ADR 0018) |
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
```

`run` first checks the SHA-256 of every input against the suite (`--no-verify-inputs` skips it),
then runs each planned command; `--readings` and `--datasets` select a part, `--runs` overrides
the rounds (at least 5), `--skip-existing` resumes an interrupted execution, and `--build-root`
points at another checkout's build trees (for example a clean clone of the release commit). The
full records go to `$DYNG_SCRATCH/runs/bench/<version>/`. `summarize` (run at the end of `run`, or
on its own) writes to `benchmarks/results/<version>/`:

- `<suite>-<reading>[-<dataset>].json`: each harness record, compacted (the per-round
  machine-monitor windows are dropped; the samples, verdicts, rejected rounds and clock checks are
  kept; the record names its full copy);
- `<suite>.json`: every reading with its regions (original and port medians, ratio, gate,
  verdict), the commits that were measured and the verdict.

The summary re-derives every gate from the suite's tolerances and fails if a record's gate,
reference commit or variant, input digests, clock lock or build preset differ from the suite, if
a gated reading is missing or incomplete, or if a gated region exceeds its gate or is provisional
(a region under 10 ms read with fewer than 20 rounds).

## The release certificate

PLAN 10.3 steps 1-3 (`docs/developer/release.md`) end in `benchmarks/results/<version>/`:

1. **The golden replays** (`parity/compare.py ... --json benchmarks/results/<version>/parity-<set>-<backend>.json`):
   every golden set of `parity/goldens.toml` on every backend it covers (sssp: sequential, OpenMP
   and CUDA; the cycle_count CPU corpus `--full`: sequential and OpenMP; the cycle_count CUDA
   corpus: cuda, cuda:resident, cuda:int64).
2. **The performance gates**: `parity/bench_suite.py run` for every suite of the release's
   algorithms.
3. **The checks**: the `asan` and `tsan` presets' test suites, `ci/gpu_local.sh` (the CUDA
   tests, the CUDA goldens and compute-sanitizer memcheck, synccheck and racecheck), the mutation
   CTests (`ctest -L mutation`) and the golden mutations (`parity/mutate.py run --json
   benchmarks/results/<version>/mutation-sssp.json`), one `parity/certify.py check` each, for
   example

   ```bash
   parity/certify.py check --version 0.1.0rc1 --name asan --command "ctest --preset asan" \
       --ctest-log asan-ctest.log --commit <the commit that was built>
   parity/certify.py check --version 0.1.0rc1 --name gpu_local --command ci/gpu_local.sh \
       --gpu-summary build/dev-cuda/gpu_local_summary.md \
       --require-steps memcheck,synccheck,racecheck
   ```

   (`--ctest-log` reads ctest's summary line, `--gpu-summary` the step table of
   `ci/gpu_local.sh`, `--json-verdict` a record with a top-level `passed` such as
   `mutation-sssp.json`, `--result passed|failed --details ...` anything else); they collect in
   `checks.json`.

Then, on the committed release tree, `parity/certify.py write --version <version>` writes
`parity.json` and the generated part of `README.md` (between the `certify` markers; the text
around it is written by hand). The certificate records the release commit; the commits whose
builds were measured, each of which must have the release commit's library and tool sources
(`git diff <measured> <release> -- cpp tools cmake CMakeLists.txt CMakePresets.json` is empty);
the hardware, the NVIDIA driver, the CUDA and compiler versions; the pinned originals (commit,
the local `baseline-2026-09` SHA, upstream); every golden set with its manifest SHA-256, the
SHA-256 of every case and every replay's case x configuration matrix and tolerance; the
performance-gate table of every suite (gated, and recorded-only readings separately); and the
checks. It exits 1, and says why in `verdict.problems`, if any part failed or is missing, if a
replayed manifest is not the one of `goldens.toml`, or if a measured tree was dirty.
