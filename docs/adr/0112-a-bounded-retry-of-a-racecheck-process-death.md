# ADR 0112: A bounded retry of a racecheck run whose process died

- **Status:** Accepted under delegation (2026-11-20; R040, step stability-and-carryovers;
  GOVERNANCE.md, "Delegation of technical ADRs"), technical scope only. It carries out R031's open
  item 6 for 0.4.0 as R040's criterion 2 names it ("a bounded retry of that outcome only, still
  failing on any reported hazard"). It changes no rule the author approved: no gate, no parity rule,
  no sanitizer's verdict on a hazard or an error; a release certificate still needs every step of
  `ci/gpu_local.sh` passed in one run (`parity/certify.py`'s `GPU_LOCAL_STEPS`), and the retried
  runs are listed in the run's summary.
- **Date:** 2026-11-20
- **Deciders:** the AI assistant, on the author's behalf
- **Builds on:** ADR 0062 (the steps a certificate needs), R030's deviation 7 and R031's deviation 7
  and open item 6 (`docs/developer/retrospectives/R030.md`, `R031.md`), M2b's open item 8.

## Context

`ci/gpu_local.sh` runs compute-sanitizer's racecheck over CUDA test executables in eight of its
steps. Three times in the project's history one such run ended with

```text
========= Error: process didn't terminate successfully
=========     The application may have hit an error when dereferencing Unified Memory from the host. ...
========= Target application returned an error
========= RACECHECK SUMMARY: 0 hazards displayed (0 errors, 0 warnings)
```

and no failed test: M2b (`dyng_cycle_count_cuda_tests`, during the fixture tests), R030
(`dyng_triad_count_conformance_cuda_tests`, C1 / C11 / C12) and R031 (`dyng_cbst_scenario_tests`).
In R030's and R031's logs the test process printed nothing at all; that does not date the death,
because the test's standard output goes to a pipe and is block-buffered (a process that ends by a
signal loses its buffer: R040's runs below print nothing either until they exit). The message is
the one compute-sanitizer gives when the target process ends abnormally (by a signal) rather than
by returning. Every one of them passed on every re-run (12 in M2b, 4 in R030, 3 in R031), and each
time the whole `ci/gpu_local.sh` (4.5 to 8 hours) was run again for the certificate.

## Investigation (R040)

The three racecheck commands were run again in a loop on GPU 1 from the `parity-cuda` build of
this branch, as `ci/gpu_local.sh` runs them, each through a small wrapper that records the test
process's own exit status (128 + n for a signal n) next to the sanitizer's, with a monitor of
GPU 1 (processes, utilization, memory, clocks), the host's load and its free memory every 10 s,
first on an otherwise idle GPU 1 and then with a second CUDA process on GPU 1 beside the sanitizer
(the label_propagation CUDA tests in a loop). The records are in
`$DYNG_SCRATCH/runs/r040-stability/racecheck/` (`summary.txt`, one log per run, the monitors).

The results are in the section "Results of the investigation" at the end of this ADR.

## Decision

1. **Every compute-sanitizer call of `ci/gpu_local.sh` goes through `sanitize`**, which is the
   script's `heavy` (the shared lock, niced, the GPU1-QUIET calendar) for every tool.
2. **For racecheck only, one outcome is run once more**: the sanitizer's exit status is not 0, its
   output has "Error: process didn't terminate successfully", every RACECHECK SUMMARY says "0
   hazards displayed (0 errors, 0 warnings)", there is no other `========= Error` or `=========
   Warning` line, and no `[  FAILED  ]` line of GoogleTest. Then the same command runs once more,
   and its result is the run's result.
3. **Bounded**: at most one retry per command. A second death fails; a hazard, an error, a warning
   or a failed test in either run fails at once; memcheck, synccheck and initcheck are never
   retried (no death of theirs was ever seen, and a death there is not this outcome).
4. **Recorded**: the script prints the retry with the command into the log, and the summary
   (`gpu_local_summary.md`) says below its table "Sanitizer runs retried after a process death (ADR
   0112): none." or the count and the commands. The table's rows are unchanged, so
   `parity/certify.py` reads the steps as before; a release record with a retry is visible to the
   reviewer and the retrospective reports it.
5. **Not covered**: the racecheck of triad_count's make-test replays runs inside `compare.py
   triad_count` through a wrapper of its own; a process death there fails that step as before (it
   was never seen there).
6. **Tests**: `ci/tests/test_gpu_local_sanitizer_retry.py` cuts the two functions out of the script
   and runs them with a fake compute-sanitizer: a pass runs once; a death with nothing reported
   runs once more and passes on a passing second run; a second death fails; a hazard, a warning, a
   failed test or a test failure without a death fail at once, as does a hazard after the retry;
   the other tools are never retried; no sanitizer call of the script bypasses `sanitize`.

## Consequences

- A transient death at a sanitizer's start no longer costs a whole second `ci/gpu_local.sh` run of
  several hours; under the branch note of R040 the protocol of R031 (the step run again in full)
  remains the rule for any other failure.
- The retry can hide a death that a defect of dynG causes only rarely and only under the
  sanitizer, if it happens once in two runs and never shows a hazard, an error or a failed test.
  The deaths seen so far were at the process's start or during tests that pass under every other
  tool (memcheck, synccheck, initcheck run the same executables) and in plain runs; the summary
  lists every retry, so a recurring one is seen and investigated.

## Alternatives considered

- **Record it and keep running the whole script again** (R031's protocol): correct but costly; the
  certificate of 0.4 has more racecheck runs than 0.3's (label_propagation, slack_csr, hyper_sssp).
- **Retry every failing sanitizer run**: would retry real hazards and failed tests; never.
- **Retry the whole step**: a step holds several executables and tools; one command is the unit
  that died.
- **A longer timeout or other sanitizer flags**: no timeout was involved, and no flag is known to
  change a death at start-up.

## Results of the investigation

(R040 adds them when the loops end.)
