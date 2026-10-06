# Robustness checks: fuzzers, mutation checks, long property runs

Three checks look for bugs that the unit tests, the conformance kit and the golden parity do not
reach by construction (PLAN Sections 8.1 and 8.4; ADR 0032). None of them is a required check of
a pull request: each explores inputs at random or takes hours, so a failure is a new bug to fix,
not a property of the pull request that met it.

| Check | What | Where | When |
|---|---|---|---|
| Reader fuzzers | libFuzzer on every file reader, with ASan and UBSan | `cpp/fuzz`, `ci/fuzz.sh`, `.github/workflows/fuzz.yml` | pull requests that touch a reader (60 s per target), weekly (10 minutes per target), on demand |
| Corpus replay | the seed corpus and every fixed finding, once | CTest `fuzz.replay.<target>` (label `fuzz`), `cpp/tests/io/fuzz_regression_test.cpp` | every test build, so every pull request |
| Golden mutation checks | each recorded bug, put back, must make its golden suite fail | `parity/mutate.py` | before each minor release, and when a mutation point changes (local: needs the golden corpus and a GPU) |
| Long property runs | the Hypothesis properties with many more, non-derandomized examples | profile `full` of `python/tests/conftest.py`, `.github/workflows/property.yml` | weekly, on demand |

## Reader fuzzers

Every reader of `dyng::io` has a libFuzzer target in `cpp/fuzz` (its README lists them): Matrix
Market, edge lists (and their Matrix Market path), the MOSP CSR triplet, MOSP batches
(`insert.txt` / `delete.txt`), `.dgt` batch files and the result files (distances, parents). The
binary batch files (`.dgb`) and the hypergraph files (`.hg`) come with M8 (0.3.0); each reader
added later gets a target in the same pull request.

A target splits its input into reader options (the first line: byte i selects option i) and the
file contents (lines `@@` separate the files of a reader that opens several), writes the files and
calls the reader. It requires that

- a rejected input raises only an exception the reader documents (`io_error`,
  `invalid_argument_error`, `out_of_memory_error`); any other exception, a sanitizer report, a
  crash or a hang is a finding;
- an accepted input gives a consistent result (ids in range, sizes that agree, the documented
  order);
- where a writer exists, writing the result and reading it back gives the same data;
- no single allocation exceeds 2 GB and no input takes 10 s (`-malloc_limit_mb`, `-timeout`): a
  file of a few kilobytes that makes a reader reserve gigabytes is a finding too.

The command line's text batches are parsed in Python, where libFuzzer does not reach:
`python/tests/test_reader_robustness.py` feeds them, and `dyng.io`'s readers, arbitrary text with
Hypothesis under the same rules (documented exceptions only, round trips).

### Running them

```bash
ci/fuzz.sh                         # every target for 60 s
ci/fuzz.sh --time 600 csr_triplet  # one target for 10 minutes
DYNG_FUZZ_JOBS=4 ci/fuzz.sh        # four fuzzing processes per target (libFuzzer -fork)
```

The script configures the preset `fuzz` (Clang only: `CXX=clang++-18` by default; libdyng is
compiled with `-fsanitize=fuzzer-no-link,address,undefined`, `-O1`, libstdc++ assertions, and only
the modules the readers need), builds the targets, replays the seed corpus and the reproducers
(`ctest --preset fuzz`), then fuzzes each target on a copy of its seed corpus in
`build/fuzz-runs/corpus/<target>`. On the shared development machine, run it under the shared
perf lock and niced: `flock -s "$DYNG_SCRATCH/perf.lock" nice -n 10 ci/fuzz.sh`.

`fuzz.yml` runs the same script on GitHub's runners with Clang 18 (`clang-18`,
`libclang-rt-18-dev`): on pull requests that change a reader, the parser utilities, the fuzzers or
the workflow (60 s per target), weekly on `main` (10 minutes per target) and on demand (any
duration). A finding fails the job and uploads its reproducer (artifact `fuzz-findings`).

### A finding

1. Reproduce it: `build/fuzz/cpp/fuzz/fuzz_<target> <reproducer>` (one run), and minimize it if
   it is large: `fuzz_<target> -minimize_crash=1 -runs=10000 <reproducer>`.
2. Fix the reader. A reader must report malformed input as `io_error` with the path, line and
   column, and must not size an allocation by a count in the file that the rest of the file
   cannot hold.
3. Add the reproducer to `cpp/fuzz/regressions/<target>/` (every test build replays it) and a test
   to `cpp/tests/io/fuzz_regression_test.cpp` that states what the reader must do (the error,
   and, for an allocation finding, the bound: that executable replaces `operator new` and records
   the largest request).

### Findings so far

| Date | Target | Finding | Fix |
|---|---|---|---|
| 2026-10-06 | `fuzz_matrix_market` | a 74-byte file whose size line announced 3,000,000,000,003 entries made the reader reserve 3 GB (min(entries, 2^26) entries, twice for a symmetric matrix) before it found the file empty | the reservation is bounded by the entries the file can hold (4 bytes each) |
| 2026-10-06 | `fuzz_csr_triplet` (its memory use: 280 MB resident at 500 inputs/s led to the review) | a RowPtr announcing 10^9 edges made the strict ColInd reader reserve 4 GB; with m edges in RowPtr and ColInd, a Values file of one line of K weights made the strict Values reader allocate m * K weights (1.6 GB for m = K = 20,000) | both reservations are bounded by what the file can hold (an index takes 2 bytes, a weight line one line break) |
| 2026-10-06 | `python/tests/test_reader_robustness.py` (Hypothesis) | the command line's text batches (`dyng cycle_count update --batch`, parsed in Python) raised `OverflowError` for an id beyond 64 bits and `UnicodeDecodeError` for a file that is not UTF-8 | both are `dyng.FileFormatError` with the path and line |

Each fix has its reproducer and test (above). After the fixes every target ran for 5 and then 15
minutes on the development machine without a finding (in the 15-minute run between 0.47 million
inputs, `fuzz_csr_triplet`, and 5.6 million, `fuzz_result_io`).

## Golden mutation checks

`parity/mutate.py` puts each recorded bug back into a `git archive` copy of the current commit,
builds the compatibility driver of its golden suite (`parity-cuda` preset) and replays the suite
on the mutation's backend: the control (no mutation) must pass on every suite and backend the
mutations use, and every mutation must make its suite fail.

| Suite (golden corpus) | Mutations (each on the sequential, OpenMP and CUDA backend) |
|---|---|
| `sssp` (MOSP-OpenMP@c352151, `dyng-compat-mosp`) | a skipped subtree invalidation: only the roots of a deleted tree edge are invalidated (PLAN 8.4, the count-to-infinity revert) |
| `cycle_count` (CycleEnumeration-GPU@0a976ad, `dyng-compat-cycle-enum`; the CUDA backend against the set `cycle_count_cuda`) | double counting 5-cycles; a weakened ownership rule (PLAN 8.4, the two bugs the original recorded; host searches and device kernels) |
| `mosp` (the MOSP outputs of the `sssp` corpus: the combined graph's tree and distances, the path costs) | a combined-graph edge weighted by the first tree that holds it only; the path costs of every objective read from the first objective's weights (shared by the backends) |

```bash
parity/mutate.py list
parity/mutate.py run --json mutation.json             # everything (about 30 minutes: one build each)
parity/mutate.py run --suites mosp --json mosp.json   # one suite
parity/mutate.py run --mutations cycle_count_weak_ownership
```

The run of 2026-10-06 (commit `a99e968`, `parity/results/M6b-mutation.json`): the control passed on
the sssp and cycle_count corpora with the sequential, OpenMP (4 threads) and CUDA backends, and all
eleven mutations were detected: sssp 365 failing cases on each backend, mosp 398 (combined-edge
weights) and 397 (path costs) of the 495 cases, cycle_count 4 (double-counted 5-cycles) and 14
(weakened ownership) of its cases on the host and on CUDA.

It needs the golden corpus in `$DYNG_SCRATCH/goldens` (`parity/export_goldens.py`) and a GPU for
the CUDA mutations (`$DYNG_TEST_GPU`, default 1), so it runs on the development machine, before
each minor release (docs/developer/release.md, step 3, records it with `parity/certify.py check`),
and whenever a mutation point changes: `parity/tests` checks that every textual mutation still
matches its code exactly once and that every `defines` mutation names a hook that exists. The
cycle_count mutations are also CTests (`ctest -L mutation`), which run them against the unit and
randomized suites on every build.

## Long property runs

The pytest suite's Hypothesis properties (`python/tests/test_properties.py`, `test_mosp.py`)
compare sssp, cycle_count, mosp, the multi-result update and the tutorial algorithms
(`dynamic_bfs`, `triangle_delta`) with pure-Python oracles on random graphs and chains of batches.
Pull requests run the derandomized profile `ci` (200 examples per property, the same inputs every
time). The profile `full` explores: 10,000 examples per property by default
(`DYNG_HYPOTHESIS_EXAMPLES`), a new random seed every run, and the failing example printed as a
`@reproduce_failure` blob. With 2,500 examples the seven properties took 82 s on the development
machine (2026-10-06, all passed).

```bash
DYNG_HYPOTHESIS_PROFILE=full python -m pytest python/tests/test_properties.py python/tests/test_mosp.py
DYNG_HYPOTHESIS_PROFILE=full ci/python.sh       # the whole suite, as property.yml runs it
```

`property.yml` runs `ci/python.sh` with the profile `full` weekly and on demand (with a chosen
number of examples). A failure: replay it with the printed blob, fix the bug, and keep the input
as an `@example` of the property.
