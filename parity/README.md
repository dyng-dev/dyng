# Parity harness

The parity harness proves that each dynG port computes exactly what its original computes, and
records how fast it does it (PLAN Sections 6.3 and 8.3; the decisions are in
[ADR 0013](../docs/adr/0013-parity-and-goldens.md)). **The original repositories are never
modified, built in place or checked out**: they are only read with `git archive`.

All generated artifacts live in the persistent work area `$DYNG_SCRATCH` (default
`~/Projects/dyng-work`, set by `scripts/dev_env.sh`), never in the repository:

```text
$DYNG_SCRATCH/
  ref/<name>@<commit7>/{unpatched,patched}/   scratch copies (build_reference.sh)
  ref/<name>@<commit7>/archive-check.txt      what the archive needs from outside
  goldens/<set>/                              golden cases + MANIFEST.sha256 (export_goldens.py)
  bench/mosp/<graph>/                         benchmark inputs (perf_ab.py prepare)
  perf.lock                                   exclusive lock of every performance measurement
```

## Files

| File | Purpose |
|---|---|
| `references.toml` | the pinned originals: commit, `baseline-2026-09` SHA, build and test commands, run environment, external inputs, toolchain |
| `build_reference.sh` | `git archive` the pinned commit into `ref/<name>@<commit7>/{unpatched,patched}`, verify the copy against the archive, build it, apply the additive export patch (patched only), write the archive-build check; `--test` runs the original's own tests |
| `export_patches/<repo>/build.sh` | the additive export patch: builds the exporters of `exporters/` against the copy's unchanged sources |
| `exporters/mosp/` | `export_graph_io` (generators, `applyChangeBatch`) and `export_sssp` (file-based SOSP updates) |
| `exporters/cycle_enum/` | `export_cycle_enum` (CycleEnumeration-GPU: parser, `build_directed_graph`, `prepare_batch`, `apply_batch`, `generate_batch`, the tests' subset-DP oracle and brute force) |
| `mosp_scale_goldens.py` | the paper-scale `mosp` goldens (M7): `export` records the SHA-256 of every output file of MOSP-OpenMP@c352151's `mosp` on the benchmark inputs (the four gate graphs x three batches, Pref {4, 1, 4} on roadNet-CA, the K sweep on roadNet-CA widened to 4 objectives) after MOSP-CUDA@e220ee2 wrote the same bytes, into the `[sets.mosp_scale]` section of `goldens.toml`; `compare` replays them through `dyng-compat-mosp --mosp`; reached as `export_goldens.py mosp_scale` and `compare.py mosp_scale` |
| `cycle_count_goldens.py` | the `cycle_count` corpus of CycleEnumeration-GPU@0a976ad: `export` (histograms of the original `cycle-enum`, OpenMP 56 threads, DD k = 3..7, GitHub and Twitch k = 3, 4, COLLAB k = 3; the seed-1 updates 1K/25K/50K on DD, GitHub, Twitch with prior, delta and the generated batch; a DD locality and bound sweep; cross-checked against the original's sequential backend, the exporter's counts and the plan's totals) and `compare` (replay through `dyng-compat-cycle-enum`); reached as `export_goldens.py cycle_count` and `compare.py cycle_count` |
| `cycle_count_perf.py` | the OpenMP A/B of `cycle_count` against the unpatched CycleEnumeration-GPU (`perf_ab.py cycle_count run`; regions from `timed_regions/cycle_count.toml`) |
| `export_goldens.py` | writes the `sssp` golden set from MOSP-OpenMP@c352151 (495 cases, including 107 with non-canonical initial trees) after cross-checking the original implementations; `--twice` re-exports from a fresh copy and compares |
| `goldens.toml` | GENERATED manifest: the SHA-256 of each golden set's `MANIFEST.sha256` and one digest per case |
| `compare.py` | verifies the goldens, then replays every case through `tools/compat` (dynG; `--configs sequential,openmp:<t>,cuda[:<device>]`) or through another original's copy, byte for byte |
| `timed_regions/<algo>.toml` | the original's timers -> dynG profiler stages (written before each port); `perf_ab.py` loads it |
| `perf_ab.py` | `prepare` the benchmark inputs with the original's own tool; `run` the A/B/A/B comparison under the perf lock (`--backend openmp` against MOSP-OpenMP, `--backend cuda` against MOSP-CUDA; `--baseline-exe` against an earlier dynG build instead, for refactors: reported, not gated; with `--layouts N` the rounds cycle through N heap layouts, ADR 0024) |
| `bench_suite.py` | the benchmark suites of `benchmarks/paper/` (PLAN 8.5): `validate` a suite against this harness, `plan` and `run` it through `perf_ab.py` / `cycle_count_perf.py` (every reading of the suite, gated or recorded), and `summarize` the records into `benchmarks/results/<version>/` with every gate re-derived; `benchmarks/README.md` |
| `mutate.py` | the mutation checks of PLAN 8.4 against the golden suites, each on the sequential, OpenMP and CUDA backends: sssp (the skipped subtree invalidation), cycle_count (CycleEnumeration-GPU's double-counted 5-cycles and weakened ownership rule, host and device) and mosp (a combined-edge weight from one tree only; path costs from one objective's weights). Each mutation is applied to a `git archive` copy, built (`parity-cuda`) and replayed through `compare.py`; the control must pass and every mutation must fail (`mutate.py list`; [robustness checks](../docs/developer/robustness.md)). The CTests `*.mutation.*` check the cycle_count mutations against the unit suites as well |
| `certify.py` | the release certificate (PLAN 8.3): `check` records a release check (sanitizers, mutation CTests, `ci/gpu_local.sh`, `ci/check.sh`, the distributions) in `checks.json` with its scope and evidence; `equivalence` compares the builds of a measured commit and of the release when they differ only in generated metadata; `write` assembles `benchmarks/results/<version>/parity.json` from the golden replays, the committed fixtures (`fixtures/fixtures.toml`), the suite summaries and the checks; `benchmarks/README.md` |
| `ab_modes.py` | reads the records of a dynG-against-dynG A/B by ADR 0024: every gated region's ratio of medians with its bootstrap interval, and a bimodal region by its within-mode ratios and the Fisher test of equal mode fractions; exit 1 if a region is outside the bar |
| `experiments/` | one-off diagnostics, not gates: `sssp_stage_ab.py` and `cycle_count_stage_ab.py` (two builds stage by stage), `sssp_layout_scan.py` (two builds over heap layouts), the cycle_count variants of M2 |
| `clock_lock/clock_holder.cu` | the idle helper through which `perf_ab.py run --backend cuda --lock-clocks boost` (the default; or `base`) keeps the GPU clocks locked for the whole A/B with Nsight Compute, without root (ADR 0018) |
| `tests/` | smoke tests of the scripts (pytest; run by `ci/check.sh` and `lint.yml`) |
| `fixtures/` | scripts that regenerate the small committed test fixtures (`cpp/tests/data`) from the same scratch copies |
| `results/` | the committed parity and performance records, one per milestone (a release's certificate is `benchmarks/results/<version>/parity.json`) |

## Typical session

The harness reads each original from a clone in `$DYNG_ORIGINALS_DIR/<name>` (default
`~/Projects/<name>`); clone the ones you need once from their `upstream` in `references.toml`,
for example `git clone https://github.com/SMShovan/MOSP-OpenMP.git ~/Projects/MOSP-OpenMP`.

```bash
source scripts/dev_env.sh
parity/build_reference.sh --test                  # build (and test) every pinned original
parity/export_goldens.py --twice                  # write the goldens; prove the export is reproducible
cmake --preset parity && cmake --build --preset parity
ctest --preset parity -L parity                   # golden replay (skipped without goldens)
parity/compare.py --exe build/parity/tools/compat/dyng-compat-mosp --json out.json
parity/compare.py --driver original --ref "$DYNG_SCRATCH/ref/MOSP-CUDA@e220ee2/patched"
cmake --preset parity-cuda && cmake --build --preset parity-cuda
parity/compare.py --exe build/parity-cuda/tools/compat/dyng-compat-mosp --configs cuda
parity/perf_ab.py prepare --graph roadNet-CA      # inputs, as the original's bench/prepare.sh
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py run --graph roadNet-CA \
    --exe build/parity/tools/compat/dyng-compat-mosp --runs 21 --json out.json
# graphs of the sssp gate (PLAN 6.4.2): roadNet-PA, roadNet-CA, rgg, road_usa_g
# cycle_count (M2a): goldens, replay, OpenMP gate against CycleEnumeration-GPU@0a976ad
parity/export_goldens.py cycle_count --twice
parity/compare.py cycle_count --exe build/parity/tools/compat/dyng-compat-cycle-enum
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py cycle_count run \
    --exe build/parity/tools/compat/dyng-compat-cycle-enum --runs 11 --json out.json
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py run --backend cuda --gpu 0 --graph roadNet-CA \
    --exe build/parity-cuda/tools/compat/dyng-compat-mosp --runs 21 --json out.json
# mosp (M7): the paper-scale goldens, the K sweep input, the gates (timed_regions/mosp.toml)
parity/perf_ab.py prepare --graph roadNet-CA-K4 --widen roadNet-CA:4
parity/export_goldens.py mosp_scale
parity/compare.py mosp_scale --exe build/parity/tools/compat/dyng-compat-mosp --configs openmp:28,sequential
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py mosp --backend cuda --gpu 0 --graph roadNet-CA \
    --exe build/parity-cuda/tools/compat/dyng-compat-mosp --runs 21 --json out.json
```

`perf_ab.py run` takes `$DYNG_SCRATCH/perf.lock` itself; under the machine's convention
`flock "$DYNG_SCRATCH/perf.lock" <command>` it sees in `/proc/locks` that its `flock(1)` ancestor
holds the lock and runs under it (`DYNG_PERF_LOCK_HELD=1` or `--no-lock` say so explicitly). A
lock held by another measurement is waited for at most `--lock-timeout` seconds. It times only
a `parity`-preset build (`parity-cuda` with `--backend cuda`; `--allow-non-parity-build` marks
an experiment) and exits non-zero if the deterministic counters differ in any round. The CUDA A/B
runs both sides on `--gpu` (default 0, the performance GPU). Replays and tests of CUDA code run on
GPU 1 (`CUDA_VISIBLE_DEVICES`) unless set.

## Rules

- A comparison is never loosened to make a port pass; a mismatch is fixed in dynG (never in the
  original) or, if the original is wrong, documented on the algorithm page as "Paper vs fixed code".
- Export patches are additive only: new files, no change to any tracked file of the original
  (`build_reference.sh` checks it after every build).
- Performance baselines come only from the unpatched copy.
- Results that matter are committed under `results/`, not left in logs.
