# Run the parity check

Every port is proved against its pinned original: the originals are built from `git archive`
copies in a scratch directory, their outputs are exported as golden files, and dynG replays
them byte for byte on every backend (ADR 0013).

```bash
source scripts/dev_env.sh                 # sets DYNG_SCRATCH (default ~/Projects/dyng-work)
git clone https://github.com/SMShovan/MOSP-OpenMP.git ~/Projects/MOSP-OpenMP   # once
parity/build_reference.sh MOSP-OpenMP     # build the pinned original in $DYNG_SCRATCH/ref
parity/export_goldens.py                  # export the sssp golden corpus to $DYNG_SCRATCH/goldens
ci/check.sh --parity                      # the local gate plus `ctest --preset parity -L parity`
```

The harness only reads the original (with `git archive`); it looks for the clone in
`$DYNG_ORIGINALS_DIR/<name>` (default `~/Projects/<name>`), so set `DYNG_ORIGINALS_DIR` if your
clones live elsewhere. `DYNG_SCRATCH` can point anywhere with a few GB free.

The goldens (about 185 MB for `sssp`) stay outside the repository. The CUDA backend is replayed
against the same corpus, and cross-checked against the CUDA original, with a CUDA toolkit and a
GPU:

```bash
git clone https://github.com/SMShovan/MOSP-CUDA.git ~/Projects/MOSP-CUDA       # once
parity/build_reference.sh MOSP-CUDA       # the CUDA original (needs nvcc)
cmake --preset parity-cuda && cmake --build --preset parity-cuda
parity/compare.py --exe build/parity-cuda/tools/compat/dyng-compat-mosp --configs cuda
ci/gpu_local.sh                           # the GPU gate also replays the corpus on `cuda`
```

Performance is compared with the unpatched originals by `parity/perf_ab.py` (alternating A/B
runs under an exclusive lock; the parity guide below has the details).

Each milestone commits its record, the **parity certificate**: correctness, the performance
gates and the raw measurements of the run.

- [M1a certificate](https://github.com/dyng-dev/dyng/blob/main/parity/results/M1a.md): `sssp`
  on the sequential and OpenMP backends against MOSP-OpenMP@c352151.
- [M1b certificate](https://github.com/dyng-dev/dyng/blob/main/parity/results/M1b.md): `sssp`
  on the CUDA backend against MOSP-CUDA@e220ee2, cross-backend equality, the CUDA and OpenMP
  performance gates and the `edge_t` benchmark (ADR 0009).

The full guide, with the file layout and the rules, is {doc}`../developer/parity`.
