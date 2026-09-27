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

The goldens (about 185 MB for `sssp`) stay outside the repository; the committed record of the
last run is the parity certificate under `parity/results/`. The full guide, with the file layout
and the rules, is {doc}`../developer/parity`.
