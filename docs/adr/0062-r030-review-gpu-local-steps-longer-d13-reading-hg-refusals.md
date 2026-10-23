# ADR 0062: The R030 review: gpu_local's steps by release series, the longer D13 reading, and the `.hg` reader's refusals

- **Status:** Accepted under delegation (2026-10-22; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It tightens the release certificate's check of a record (a
  rule of `parity/certify.py`, not a gate or a tolerance), carries out ADR 0024's rule 3 for
  ADR 0061's reading as written, corrects a misstatement of ADR 0061, and makes the `.hg` reader
  refuse three files of other formats before `.hg` becomes a stable-tier format at 0.3.0. It
  changes no rule the author approved: the gates, their protocol (ADRs 0018, 0021, 0024, the
  one-layout rule of 2026-10-08), the parity rules and the frozen signatures are untouched.
  What the reading leaves above the bar is listed for the author (section 2).
- **Date:** 2026-10-22
- **Deciders:** the AI assistant, on the author's behalf (R030, the review fixes)
- **Amends:** ADR 0061 (its reading; the restructured guard's numbers). **Builds on:** ADR 0024
  (rules 1-3), ADR 0045 (the `.hg` files), ADR 0056 (the 0.3 certificate), R020's
  `REQUIRED_CHECKS` (`10dc59d`).

## Context

The independent review of the branch at `75d6940` (the certificate of 0.3.0rc1 written)
confirmed, among others:

1. `REQUIRED_CHECKS` required only memcheck, synccheck and racecheck of a gpu_local record. The
   0.3 steps of `ci/gpu_local.sh` (the CBST store's, the hypergraph container's and
   triad_count's sanitizers and replays, count_local_patterns) can be skipped with
   `DYNG_GPU_SKIP`, and some skip a part of themselves when their inputs are missing while still
   recording "passed", so a 0.3 record without them would have certified.
2. ADR 0061's dynG-against-dynG reading left two sub-regions above ADR 0024's 1.02 bar
   (`hypergraph.apply.insert` on sequential, 1.013-1.043; `hypergraph.build`, 1.019-1.085)
   after the 17-layout reading, without the longer reading rule 3 requires (170 rounds over 85
   layouts on OpenMP), and kept the insert step's cost without listing it for the author; and
   its sentence on the restructured guard misstated the guard's own numbers.
3. The reading tool wrote "exclusive perf lock" into its records without checking it and named
   its sides by build paths since deleted.
4. A Matrix Market file passed to the `.hg` reader was read as a hypergraph without an error.

## Decision

### 1. gpu_local's steps by release series

`parity/certify.py` `GPU_LOCAL_STEPS` lists the steps of `ci/gpu_local.sh` (by their names in
its summary) by the release series that added them: from 0.1 the build, the GPU and CPU tests,
the CUDA golden corpora, memcheck, synccheck (sssp and cycle_count) and racecheck; from 0.2
mosp's synccheck, the plugin wheel and clang-tidy; from 0.3 the CBST store's sanitizers, its
two device replays, `bin/main` and the host-model lockstep, the hypergraph container's
sanitizers and replays, triad_count's sanitizers and count_local_patterns. A certificate of a
series needs every step of its series and the earlier ones **passed**; "skipped" and the new
result **"partial"** (`ci/gpu_local.sh` records the triad_count step so when its coauth batch or
its replays were skipped for missing inputs) are not failures of the script, but do not count.
A test checks that every listed name is one the script records, so a renamed step fails the
certificate instead of passing it silently. The 0.1 and 0.2 lists are what those certificates
recorded as their required steps; the 0.3.0rc1 record of `75d6940` had every step passed.

### 2. The longer D13 reading (ADR 0024 rule 3)

The longer reading was taken on the release's **final library** (`c9de41d`: the D13 fix and the
`.hg` reader's refusals of section 3, side B) against `e6f87ed` (side A, as in ADR 0061), both
`dyng-compat-escher` of the `parity` preset from `git archive` copies built anew, with
`VERSION` 0.3.0.dev0 on both sides (so that only the library differs), by the fixed tool of
section 4 (the exclusive lock recorded as "held exclusively by this process", each side's
commit checked against its sources). The apply mode over **85 layouts, 170 rounds** per side
(rule 3) on coauth-DBLP and syn1M, sequential and openmp with 8 threads; the update modes again
over 17 layouts, 51 rounds (type 1 on sequential and on coauth-DBLP, h_motif on syn1M with 8
threads), because side B is a later library than ADR 0061's. 2026-10-22 23:29 to 2026-10-23
04:32 UTC; the contamination monitor flagged 3 rounds, which were repeated. Record:
`parity/results/R030-d13-longer-reading.json` (`$DYNG_SCRATCH/runs/r030-fix/d13`); read with
`parity/ab_modes.py --bar 1.02`. Medians in ms, A -> B, and the ratio B / A:

| Case / backend / mode (rounds) | `hypergraph.apply` | `triad_count.update` | `.apply.insert` | `hypergraph.build` |
|---|---|---|---|---|
| coauth sequential, apply (170) | 4,703.5 -> 4,444.2 **0.945** | | 109.3 -> 111.8 **1.023** | 6,669.3 -> 6,991.0 **1.048** |
| coauth openmp 8, apply (170) | 2,871.1 -> 2,779.8 **0.968** | | 84.4 -> 85.6 1.014 | 1,870.9 -> 1,889.5 1.010 |
| syn1M sequential, apply (170) | 2,640.9 -> 2,524.8 **0.956** | | 95.9 -> 101.7 **1.060** | 2,996.0 -> 3,136.1 **1.047** |
| syn1M openmp 8, apply (170) | 1,862.2 -> 1,800.6 **0.967** | | 73.9 -> 75.1 1.016 | 929.2 -> 941.7 1.013 |
| coauth sequential, update type 1 (51) | 4,697.7 -> 4,425.5 0.942 | 4,708.3 -> 4,436.0 **0.942** | **1.023** | **1.047** |
| coauth openmp 8, update type 1 (51) | 2,840.8 -> 2,748.0 0.967 | 2,847.9 -> 2,755.6 **0.968** | 1.016 | **1.021** |
| syn1M sequential, update type 1 (51) | 2,620.1 -> 2,505.8 0.956 | 2,632.0 -> 2,517.7 **0.957** | **1.059** | **1.046** |
| syn1M openmp 8, update h_motif (51) | 1,867.2 -> 1,807.3 0.968 | 5,030.7 -> 4,884.0 **0.971** | 1.019 | 1.012 |

**The regions the criterion names are within the bar**, faster on the final library: every
`hypergraph.apply`, `apply_ms`, `triad_count.update` and `update_ms` region reads 0.942-0.971.
On sequential they are bimodal with side A in the slow mode and side B in the fast one in every
round (170 / 170 against 1 / 170): code placement, not the code (side A is the same sources as
ADR 0061's, built anew in another directory, and its apply reads 3 % slower than then,
4,703.5 against 4,552.1 ms on coauth sequential).

**Still above the bar after the longer reading**, both on sequential only (on 8 threads every
sub-region reads 1.010-1.019 over 170 rounds; the coauth update's 1.021 over 51 rounds is the
same region read 1.010 over 170):

- `hypergraph.apply.insert`: 1.023 (coauth, +2.5 ms over three batches) and 1.060 (syn1M,
  +5.8 ms), 0.1-0.2 % of the apply. Located in ADR 0061: the per-group `try` and check of
  `v2h_append`'s guard (1.022 with the JCC-erratum mitigation flags on both sides, so not only
  placement). A fix is possible (on the serial path the loop could run outside any OpenMP region,
  where an exception needs no guard), but it is a library change whose effect on this CPU is
  again dominated by code placement, after the certificate's library was fixed.
- `hypergraph.build` (the construction before the batches; its line-graph row build is the
  changed region): 1.046-1.048. Located in ADR 0061 as code placement (1.007 with the mitigation
  flags on both sides).

Neither is a region that the criterion or a gate names (the host backends have no gate, D3; on
cuda the construction is inside triad_count's end-to-end gate, which the certificate reads on
this library). ADR 0024's rule 3 makes such a region "a regression to be located and fixed";
both are located and **not fixed**, which is a deviation from the measurement protocol, which is
not delegated: it is listed for the author's acknowledgement (GOVERNANCE.md, open decision D14;
docs/developer/retrospectives/R030.md, step 4; the release pull request), with the serial-path fix of the insert step as
the remedy if the author asks for it (a later library change, with its own reading).

**The restructured guard (ADR 0061, corrected).** ADR 0061 said that the restructured guard
(`parallel_chunks`, one `try` per thread) "did not help in the release build (+2 to +7 %)". Its
own diagnostics read otherwise: in the release build it is faster than the per-row form on
sequential (6,076.7 against 6,513.0 ms in the one session with both, 2.3 % above base) and
+8.1 % over base on 8 threads (+4.7 % padded; the per-row form was not in those sessions); with
the mitigation flags on both sides the per-row form is within 0.3 % of base on both backends and
the restructured guard 1.8 % (sequential) and 2.6 % (8 threads) above it. The choice of the
per-row form stands on the mitigation-flag reading, and the record's sentence is corrected
(`parity/results/R030-d13-reading.json`, `diagnostics.reading`).

### 3. The `.hg` reader refuses three files of other formats

`io::read_hypergraph` refuses, at the first line, a `.dgb` batch file (the magic `DYNGBAT1`), a
`.dgt` batch file (a first line `%dgt` followed by a separator or the end of the line) and a
Matrix Market file (a first line starting `%%MatrixMarket`, any case), with an `io_error` that
names the format and its reader. These lines were comments under the comment rule (a first
character `%`), so the `.hg` format's version 1 changes before it becomes stable at 0.3.0: a
deliberate difference from ESCHER-GPU's loader, which skips them. No `.hg` file of either
original, of the datasets registry or of the golden sets starts with them, so no replay or
reading input changes; the format page says so. The messages of the common mistakes name their
remedy in C++, Python and the command line, quoted tokens are cut after 32 bytes, and
`throw_io_error` writes control bytes as `\xHH`. This is a library change after the first
certificate, so the certificate of 0.3.0rc1 is measured again on it (R030 step 4).

### 4. The reading tool

`parity/experiments/hypergraph_host_ab.py` takes the exclusive lock itself
(`perf_ab.perf_lock`) and records what it found, records each side with its commit, the
executable's SHA-256 and its build settings, refuses a side whose source tree is not its
commit's, and puts layouts beyond a file name's 255 bytes into directory components (the
first try of the longer reading stopped at round 32 on that).

## Consequences

- A gpu_local record of 0.3 or later with a 0.3 step skipped or partial fails `certify.py
  write`; on a machine without the goldens or the coauth-DBLP file the script still runs, and
  says which part it left out.
- A step added to `ci/gpu_local.sh` for a release is added to `GPU_LOCAL_STEPS` with it.
- `.hg` version 1 is the rules of 0.3.0, the three refusals included (PLAN 5.9).
