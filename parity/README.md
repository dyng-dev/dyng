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
| `export_goldens.py` | writes the `sssp` golden set from MOSP-OpenMP@c352151 (495 cases, including 107 with non-canonical initial trees) after cross-checking the original implementations; `--twice` re-exports from a fresh copy and compares |
| `goldens.toml` | GENERATED manifest: the SHA-256 of each golden set's `MANIFEST.sha256` and one digest per case |
| `compare.py` | verifies the goldens, then replays every case through `tools/compat` (dynG; `--configs sequential,openmp:<t>,cuda[:<device>]`) or through another original's copy, byte for byte |
| `timed_regions/<algo>.toml` | the original's timers -> dynG profiler stages (written before each port); `perf_ab.py` loads it |
| `perf_ab.py` | `prepare` the benchmark inputs with the original's own tool; `run` the A/B/A/B comparison under the perf lock (`--backend openmp` against MOSP-OpenMP, `--backend cuda` against MOSP-CUDA) |
| `clock_lock/clock_holder.cu` | the idle helper through which `perf_ab.py run --backend cuda --lock-clocks boost` (the default; or `base`) keeps the GPU clocks locked for the whole A/B with Nsight Compute, without root (ADR 0018) |
| `tests/` | smoke tests of the scripts (pytest; run by `ci/check.sh` and `lint.yml`) |
| `fixtures/` | scripts that regenerate the small committed test fixtures (`cpp/tests/data`) from the same scratch copies |
| `results/` | the committed parity certificates and performance records, one per milestone |

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
flock "$DYNG_SCRATCH/perf.lock" parity/perf_ab.py run --backend cuda --gpu 0 --graph roadNet-CA \
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
