# Port research code

How to bring a dynamic algorithm from a research repository into dynG so that the port gives
**the same results as the original, byte for byte**, and is not slower. This is the procedure of
PLAN Section 6.3, as it was followed for `sssp` (MOSP-OpenMP and MOSP-CUDA) and `cycle_count`
(CycleEnumeration-GPU); the {doc}`../history/index` pages show the result for each original.
{doc}`add_an_algorithm` is the same work from the other side, for an algorithm without an
original.

The rules that make a port trustworthy:

- The original is **read-only**: it is never built in place, never modified and its git state is
  never changed. Every build is a `git archive` copy in the scratch directory `$DYNG_SCRATCH`
  (default `~/Projects/dyng-work`).
- The port matches **one pinned commit** (a full SHA in `parity/references.toml`), which should be
  the original's *fixed* code: when the paper's snapshot has bugs, fix them in the original first
  (a branch there, with its own `CHANGES.md`), then pin that commit. The algorithm page records
  the difference in "Paper vs fixed code".
- Parity is **exact**: byte equality of the output files (or bit equality of the counts), no
  tolerance (ADR 0013). A difference is a bug in the port until the two originals (where there
  are two) agree with each other and the port does not.

## 1. Pin the original and build the references

Add a `[[reference]]` to `parity/references.toml`: the clone's directory name, the upstream URL,
the pinned `commit`, the paper's `baseline_tag` and `baseline_commit`, the build command, the
compiler, the environment of every run, the original's own test command and the inputs the archive
does not contain. Then:

```bash
git clone https://github.com/<owner>/<Original>.git ~/Projects/<Original>   # once; read-only
parity/build_reference.sh <Original>          # unpatched and patched copies, built
parity/build_reference.sh --test <Original>   # ... and the original's own tests in the copy
```

`build_reference.sh` builds **two** copies: an *unpatched* one for performance baselines and a
*patched* one (additive export patches only, in `parity/export_patches/<Original>/`) that writes
the goldens. Export patches add programs next to the unchanged sources (a golden exporter, a trace
dump); they never edit an original file, because instrumentation changes the timing.

## 2. Export the goldens and cross-check the originals

Write a corpus generator in `parity/export_goldens.py` (the small, fast cases the original's
tests use, the regression cases of its `CHANGES.md`, ties, empty and delete-all batches, the
boundaries of packed words, a few paper-scale runs), then:

```bash
parity/export_goldens.py <algo>               # goldens in $DYNG_SCRATCH/goldens/<algo>
```

The goldens stay outside the repository (hundreds of MB); `parity/goldens.toml` records one
SHA-256 per case, so `parity/compare.py` detects a changed corpus. Small fixtures that the unit
tests need go to `cpp/tests/data/` (keep them under 1 MB in total).

If two originals compute the same thing (MOSP-CUDA and MOSP-OpenMP), compare them on the corpus
**once, before the port** (the M1a cross-check): a port can only be blamed for a difference the
originals do not have.

## 3. Map the timed regions

Before writing the port, write `parity/timed_regions/<algo>.toml`: for each timer of the original
(its report lines or stage CSV), the dynG profiler stages whose times add up to the same region.
The performance gates of PLAN 8.6 ("the paper-timed region within 1.05x") are defined by this map;
`parity/perf_ab.py` loads it and refuses to run when a region is missing. Name the stages after the
template's hooks (`<algo>.identify_affected`, `<algo>.loop`, ...), so the map reads the same for
every algorithm.

## 4. Scaffold

```bash
python3 scripts/new_algorithm.py <algo> --family fixed_point --backends seq,omp \
    --title "..." --computes "..."
```

The scaffold is green on the first build (a recompute placeholder behind the conformance kit).
Fill the manifest (`cpp/src/algorithms/<algo>/manifest.toml`): `origin` (repository, commit and
the ported paths), `paper`, `cite` (entries in `docs/references.bib`), `tier = "custom_engine"`
when a backend keeps the original's hand-fused kernel, and `maturity = "experimental"` for the
first merge. `python3 scripts/regen.py` updates the tables, the registry and CODEOWNERS.

## 5. Port straight

Copy the original's engine into `cpp/src/algorithms/<algo>/` with **mechanical changes only**:

| In the original | In dynG |
|---|---|
| names in its own style | the naming table of ADR 0004 (snake_case, `dyng::` / `dyng::detail::`) |
| `exit()`, return codes, `cerr` | exceptions (`invalid_argument_error`, `io_error`, ...; the message says what to do) |
| global state, `using namespace std` | members of the problem or the workspace |
| raw `cudaMalloc`, the default stream | `buffer` from the resources' memory resource, `res.stream()` |
| scratch arrays allocated per call | the workspace leased from the resources (ADR 0015), sized once |
| timers | profiler stages (the names of step 3) |
| `setenv("CUDA_MODULE_LOADING", ...)` | `resources::warm_up()` |

Every ported file starts with the SPDX lines and the provenance header (`ci/provenance_check.py`
fails a file that names an original's symbol without it):

```cpp
// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
// Derived from <Original>@<commit>:<path> (<the functions taken>)
```

On the CPU, the code goes into the hooks of the problem (Tier A: `prepare`, `commit`,
`identify_affected`, `seed`, `loop`, `finalize`, or `count` for an aggregate delta). On CUDA a
hand-fused kernel goes **verbatim** behind `enact_fused` (Tier B), where the framework still owns
Step 0, the commit, the statistics and the error checks ({doc}`../developer/framework`). A
sequential backend is required; when the original's sequential code is not a clean reference
(MOSP's `sequentialSOSPUpdate` prints and reads files), adapt it into the hooks and check it
against an independent oracle (`testing::dijkstra`, the subset-DP cycle oracles).

## 6. Prove parity

Add the algorithm to the compat driver (`tools/compat`, a clone of the original's CLI with its
flags) and replay the goldens on every backend:

```bash
cmake --preset parity && cmake --build --preset parity
parity/compare.py                             # sssp; `parity/compare.py cycle_count` for cycle_count
ci/check.sh --parity                          # the same inside the local gate
```

Then make the conformance kit green on every backend (`ctest --preset dev -L conformance`): it
checks the contract (C1-C12: equality with `compute()` after every batch, stale results, empty
batches, composition with other results, determinism, allocation budgets). Record two
**mutations** (a deliberate bug in the port, such as counting one cycle length twice) that the
suite must catch.

## 7. Record the performance baseline

Time the port against the **unpatched** original with `parity/perf_ab.py` (A/B/A/B, medians of
at least 5 runs, at least 20 with CUDA events for regions under 10 ms; on CUDA with the clocks
locked, ADR 0018; the script takes the machine's exclusive lock `$DYNG_SCRATCH/perf.lock` itself),
and record the kernels' register counts and occupancy (`perf_ab.py kernels`):

```bash
parity/perf_ab.py run --exe build/parity/tools/compat/dyng-compat-mosp --backend openmp \
    --graph roadNet-CA --lock-clocks none
```

The verdicts and the machine state go into the milestone's certificate in `parity/results/`.

## 8. Refactor onto the shared pieces, one commit at a time

Move the straight port onto the framework and the shared containers, frontiers and operators
**one commit at a time**, re-running steps 6 and 7 after every commit. A refactor that breaks
parity is reverted; one that costs more than the gate's tolerance is reverted or kept only behind
`engine::operators`, and the algorithm page says why.

## 9. Finish

- The algorithm page (the nine sections of PLAN 9.5, with "Differences from the paper",
  "Paper vs fixed code" and "Mapping from the original code"), and a history page for a new
  original.
- An example in `examples/cpp` (and `examples/python`), tested like the others.
- The Python binding (`python/bindings/`, the typed module in `python/dyng/`, the stubs:
  `python scripts/regen.py --stubs`) and the CLI command (`dyng <algo> compute|update`).
- The BibTeX entry, the `CITATION.cff` reference, the CHANGELOG entry and the parity certificate.
- The maturity: `experimental` until the algorithm has its binding, its example and its
  benchmark record; then `stable` (PLAN 6.3 step 10).
