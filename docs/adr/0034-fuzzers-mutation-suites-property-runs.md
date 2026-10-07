# ADR 0034: Reader fuzzers, the golden mutation suites and the long property runs

- **Status:** Accepted under delegation (2026-10-06; GOVERNANCE.md, "Delegation of technical
  ADRs"). It carries out the M6b acceptance criteria for fuzzers, mutation checks and the full
  Hypothesis profile (PLAN 8.1, 8.4, 11.3 M6) and records where it differs from the plan's
  sketches. It changes no rule the author approved: no gate or measurement protocol, no parity
  rule (the golden comparisons are unchanged; the mutation checks only add evidence that they
  fail when they should), no licence, no name; the public API is unchanged.
- **Date:** 2026-10-06
- **Deciders:** the AI assistant, on the author's behalf (M6b, step docs-pages-fuzz)
- **Number:** written as 0032 and renumbered 0034 in the M6b review: the parallel milestone M6a
  (branch `m6a-cuda-wheels`) uses 0030, 0031 and 0032, and M6b's tutorial-algorithms ADR is 0033.

## Context

PLAN 4.2 sketches `fuzz/` at the repository root with `fuzz_matrix_market.cpp`,
`fuzz_edge_list.cpp`, `fuzz_hg.cpp` and `fuzz_batch_text.cpp`; PLAN 5.7 says "every reader has a
libFuzzer target"; PLAN 7.3 lists the option `DYNG_BUILD_FUZZERS` (Clang only); PLAN 8.1 runs
fuzzing nightly. PLAN 8.4 lists the mutations of `parity/mutate.py` for sssp, CycleEnum, triads
and label propagation, none for mosp. The M6b criteria ask for targets for every text reader and
the binary readers, a CMake option or preset, a hosted job with Clang 18 that runs each target for
a bounded time on a committed seed corpus, every crash fixed with a regression test; `mutate.py`
extended to the cycle_count and mosp goldens; and a separate pytest profile `full` with a manual
or weekly workflow.

The readers take paths, not buffers; several open more than one file (the CSR triplet three, a
MOSP batch two); most have options that change the parse (weights, id mapping, symmetry,
duplicates, threads) and several id and weight types.

## Decision

1. **Where.** `cpp/fuzz/` (next to `cpp/tests`, as the M6b task names it), not the root `fuzz/` of
   the PLAN 4.2 sketch: the fuzzers are C++ test code of the library and build with its CMake
   project. Six targets cover every reader of 0.2: `fuzz_matrix_market`, `fuzz_edge_list` (also
   its Matrix Market path), `fuzz_csr_triplet`, `fuzz_legacy_batch` (MOSP `insert.txt` /
   `delete.txt`), `fuzz_batch_text` (`.dgt`) and `fuzz_result_io` (distances, parents). **There
   is no binary reader yet:** the `.dgb` batches moved to M8 (M7 retrospective, deviation 3) and
   ship with 0.3.0, as do the `.hg` files (`fuzz_hg`); each gets its target in the pull request
   that adds it.
2. **Input layout.** The first line of an input selects the reader options (byte i chooses option
   i modulo its choices; a digit counts as its value, so seeds are readable), the rest is the file
   content, and lines `@@` separate the files of a multi-file reader. The target writes the files
   into a private temporary directory and calls the reader. The layout lets libFuzzer mutate the
   options and the content together without the clang-only `FuzzedDataProvider`, so the same
   source builds with any compiler.
3. **What a target checks.** A rejected input may raise only the exceptions the reader documents
   (`io_error`, `invalid_argument_error`, `out_of_memory_error`); an accepted input must be
   consistent (ids in range, sizes, the documented order), and where a writer exists, write then
   read must give the same data; only the first read may reject the input, so any exception in
   the round trip (a copy the reader rejects) is a finding, and CTest `fuzz.selftest.<target>`
   (a build that corrupts every written file) shows that the check can fail. The corpus replay
   `fuzz.corpus.<target>` runs with `-malloc_limit_mb=256`. ci/fuzz.sh runs libFuzzer with
   `-malloc_limit_mb=2048` and
   `-timeout=10`: **a small file that makes a reader allocate gigabytes is a finding**, even
   though the reader would report `out_of_memory_error` or succeed in reserving it.
4. **Two builds of every target.** With `DYNG_BUILD_FUZZERS=ON` (preset `fuzz`: Clang, libdyng
   instrumented with `fuzzer-no-link,address,undefined` through `DYNG_SANITIZE`, Debug with
   `-O1` and libstdc++ assertions, static, only the sssp algorithm module, which every build
   has) the libFuzzer executables and CTest `fuzz.corpus.<target>`; and in **every test build**
   `dyng_fuzz_replay_<target>` (the same source with a small `main`) and CTest
   `fuzz.replay.<target>`, which replay the seed corpus and the reproducers of fixed findings.
   So each pull request replays every finding with GCC and Clang, Debug and Release, and the
   sanitizer presets.
5. **CI.** `fuzz.yml`: pull requests that touch the readers, the parser utilities, the fuzzers or
   the workflow (60 s per target, the criteria's bound), weekly on `main` (10 minutes per target)
   and on demand; Clang 18 from Ubuntu (`clang-18`, `libclang-rt-18-dev`); the logic is in
   `ci/fuzz.sh`, so local and hosted runs are the same. **Not a required check** (the
   orchestrator decides on required checks): fuzzing is randomized, and a finding is a new bug,
   not a property of the pull request that met it. A finding is fixed with a reproducer in
   `cpp/fuzz/regressions/<target>/` and a test in `cpp/tests/io/fuzz_regression_test.cpp`; that
   executable replaces `operator new` and records the largest request, so allocation findings are
   regression-tested with the regular compilers too.
6. **The golden mutation suites.** Each mutation of `parity/mutate.py` names its suite: `sssp`
   and `mosp` replay the golden corpus of MOSP-OpenMP@c352151 through `dyng-compat-mosp` (its
   cases hold the original's MOSP outputs: the combined graph's tree and distances and the path
   costs), `cycle_count` the corpus of CycleEnumeration-GPU@0a976ad through
   `dyng-compat-cycle-enum` (the CUDA configuration against the set `cycle_count_cuda`). A
   mutation is a textual replacement that must match once, or the `#if
   defined(DYNG_MUTATION_...)` hooks already in the sources (the cycle_count ones of the CTests),
   passed to the host and the CUDA compiler. Every suite has a mutation on each backend:
   - cycle_count (PLAN 8.4): double counting 5-cycles and the weakened ownership rule, in the host
     searches (sequential, OpenMP) and in the device kernels (CUDA), where the original recorded
     them;
   - mosp (PLAN 8.4 names none; chosen to put back the two parts of the combined-graph step that
     MOSP-OpenMP's fix M-d made exact): a combined edge weighted by the first tree that holds it
     only (sequential, OpenMP, CUDA), and the path costs of every objective read from the first
     objective's weights (the shared host code; sequential).
   The control replays every (suite, backend) pair the chosen mutations use.
7. **The long property runs.** The Hypothesis profile `full` (python/tests/conftest.py): 10,000
   examples per property by default (`DYNG_HYPOTHESIS_EXAMPLES`), **not derandomized** (the
   pull-request profiles are, so a pull request never fails on an input another one did not
   see; the long run must explore), no example database, the `@reproduce_failure` blob printed.
   `property.yml` runs `ci/python.sh` with it weekly and on demand (not a required check). The
   properties gained the tutorial algorithms over chains of batches.

## Consequences

- The first runs found three allocation findings in the C++ readers, and the Hypothesis test of
  the Python-side readers (`python/tests/test_reader_robustness.py`) two exceptions of the command
  line's text batches (CHANGELOG, "Fixed"; docs/developer/robustness.md, "Findings so far"), now
  fixed with reproducers and tests; after the fixes every target ran for 5 and then 15 minutes
  without a finding.
- Nothing enforces a target for a new reader except review: cpp/fuzz/README.md,
  docs/developer/robustness.md and docs/api/file_formats.md say that every reader has one.
- `parity/mutate.py run` takes about 30 minutes for all eleven mutations (one `parity-cuda` build
  each, 2026-10-06: all detected, the control passed; `parity/results/M6b-mutation.json`) and
  needs the golden corpus and a GPU, so it stays a local check before minor releases (release.md
  step 3), as PLAN 8.4 says.
- The weekly jobs (`fuzz.yml`, `property.yml`, `docs.yml`'s link check) report on `main` only;
  someone has to look at their results (GitHub notifies the workflow's last editor on failure).
