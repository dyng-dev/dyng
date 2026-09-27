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
| `export_goldens.py` | writes the `sssp` golden set from MOSP-OpenMP@c352151 (388 cases) after cross-checking the original implementations; `--twice` re-exports from a fresh copy and compares |
| `goldens.toml` | GENERATED manifest: the SHA-256 of each golden set's `MANIFEST.sha256` and one digest per case |
| `compare.py` | verifies the goldens, then replays every case through `tools/compat` (dynG) or through another original's copy, byte for byte |
| `timed_regions/<algo>.toml` | the original's timers -> dynG profiler stages (written before each port) |
| `perf_ab.py` | `prepare` the benchmark inputs with the original's own tool; `run` the A/B/A/B comparison under the perf lock |
| `fixtures/` | scripts that regenerate the small committed test fixtures (`cpp/tests/data`) from the same scratch copies |
| `results/` | the committed parity certificates and performance records, one per milestone |

## Typical session

```bash
source scripts/dev_env.sh
parity/build_reference.sh --test                  # build (and test) every pinned original
parity/export_goldens.py --twice                  # write the goldens; prove the export is reproducible
cmake --preset parity && cmake --build --preset parity
ctest --preset parity -L parity                   # golden replay (skipped without goldens)
parity/compare.py --exe build/parity/tools/compat/dyng-compat-mosp --json out.json
parity/compare.py --driver original --ref "$DYNG_SCRATCH/ref/MOSP-CUDA@e220ee2/patched"
parity/perf_ab.py prepare                         # roadNet-CA inputs, as the original's bench/prepare.sh
parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp --runs 9 --json out.json
```

`perf_ab.py run` takes `$DYNG_SCRATCH/perf.lock` itself (flock(2)); do not wrap it in
`flock(1)` on the same file. CUDA references run on GPU 1 (`CUDA_VISIBLE_DEVICES`) unless set.

## Rules

- A comparison is never loosened to make a port pass; a mismatch is fixed in dynG (never in the
  original) or, if the original is wrong, documented on the algorithm page as "Paper vs fixed code".
- Export patches are additive only: new files, no change to any tracked file of the original
  (`build_reference.sh` checks it after every build).
- Performance baselines come only from the unpatched copy.
- Results that matter are committed under `results/`, not left in logs.
