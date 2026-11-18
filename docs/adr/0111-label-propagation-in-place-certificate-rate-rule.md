# ADR 0111: The rate rule of label_propagation's in_place certificate

- **Status:** Accepted (2026-11-18). The rule is the author's, **decided by the author on
  2026-11-18** (GOVERNANCE.md, approvals log, the row "The in_place certificate's rate rule"; the
  STOP of the one further in_place reading of the M10b review fixes). **This ADR's wording is
  pending the author's approval before M10b's pull request** (GOVERNANCE.md, Open decisions,
  M10 (AD)). Its implementation (the frozen table, the comparator, the certificate check, the
  tests) is accepted under delegation (GOVERNANCE.md, "Delegation of technical ADRs"), technical
  scope only. M10b, the review fixes (PLAN 8.3, 8.5; criteria 2 and 12). It refines ADR 0078 (the
  0.4 certificate rules) for the `label_propagation_inplace` set and ADR 0065 (the in_place
  calibration) for how a certificate reads the frozen envelope; it changes no threshold, no
  envelope, no calibration and no other set.
- **Date:** 2026-11-18
- **Deciders:** S M Shovan (the rule); the AI assistant, on the author's behalf (its
  implementation)
- **Number:** M10's blocks (0063-0079, 0100-0109) are used up; 0110-0119 is the release line's
  block (ADR 0063), and this rule is one the release (R040) applies too; 0110 is left for the
  release line's first ADR, and 0120 is taken by the unmerged ADR 0120 fix branch. So 0111.

## Context

`schedule::in_place` is not deterministic on CUDA, in LabelPropagation-CUDA@a276a3a (O) as in the
port: its half-sweeps read labels that other threads of the same launch write, so the iterations
of a batch depend on the thread schedule. AD5 (decided by the author on 2026-10-28 and
2026-10-29) therefore judges it against a calibration frozen before the port (ADR 0065,
`parity/results/M10a-inplace-calibration.json`, `parity/goldens.toml`): over N = 10 reference
runs of the unpatched O per case of I (5k5, 50k, 500k, 5m), the labels within T_mean, T_max and
T_flips (k = 3 times the largest pairwise value, with floors), the components exact, and the
per-batch iterations and their sum inside the envelope [0.9 x the minimum, 1.1 x the maximum] of
the reference runs (at least 2 beyond them). Until now the certificate's replay
(`compare.py label_propagation inplace`) required **every one of 3 port runs per case** to lie
inside all of it.

That rule failed again and again on runs whose labels, components, iteration sums and traces were
all inside, on one batch's iteration count in the schedule's tail: M10a step 4 (5k5 batch 4, 182
against 170.5), the M10a review fixes (50k batch 9, 88 against 78.1) and after the merge of main,
M10b step 1 (two runs), M10b step 4 (50k batch 8, 122 against 104.5), and three readings of the
M10b review fixes' final code (50k batch 9: 86, 81 and 82 against 78.1). The author decided each
as a recorded miss, not a pass (the approvals log's in_place tail-event rows of 2026-10-30,
11-04, 11-11, 11-12, 11-14 and 11-18). The 30-run evidence of 2026-11-14 at 50k
(`parity/results/M10b-review/inplace-rate-50k.json`) shows why: **the unpatched original itself
is outside the frozen envelope in 2 of 30 runs** (batches 9 and 4), the port in 2 of 30 (batch 9
both). With an original rate near 2/30 at 50k, three runs fail the every-run rule with probability
about 1 - (28/30)^3 = 0.19 at 50k alone, so the original would fail the certificate as often as
the port does. The every-run rule cannot tell a port that behaves like the original from one that
does not.

## Decision

### 1. The rule (the author's, 2026-11-18)

For each case of the `label_propagation_inplace` set (the four cases of the frozen thresholds):

1. **30 fresh port runs and 30 fresh runs of the unpatched original**
   (LabelPropagation-CUDA@a276a3a, `parity/build_reference.sh --variant unpatched`), on the same
   GPU in one session.
2. A run is **outside the frozen envelope** when one of its per-batch iteration counts or their
   sum lies outside the frozen `iterations_lo` / `iterations_hi` / `iterations_sum_lo` /
   `iterations_sum_hi` of the case.
3. **The case fails if the port's count of runs outside the envelope is significantly higher than
   the original's**: a one-sided Fisher exact test (the alternative "the port's rate is higher")
   at 5 %, i.e. the case fails when p < 0.05.
4. **The labels, flips, iteration sums and traces are still checked on every port run as now**:
   each port run's labels within T_mean, T_max and T_flips of every reference run, its components
   exact, its iteration sum inside the frozen sum bounds; the deterministic stages (components,
   G', the initial labels, the initial frontier) against the trace golden, as before. A port run
   that fails one of these fails the case whatever the rate.
5. **Unchanged**: the envelope, the thresholds, the calibration (AD5, ADR 0065), k, the floors,
   the trace golden, every other golden set, the gates. The readings that missed under the
   every-run rule stay recorded as misses; nothing is replaced or deleted.

R040 uses the same rule at the release.

### 2. What the test detects, stated plainly

The p-value is the exact hypergeometric tail P(X >= a) of the 2 x 2 table (a of the port's 30 runs
outside, b of the original's 30), computed with integers (`fractions.Fraction`), no SciPy. Some
values with 30 runs each:

| port outside | original outside | p | verdict |
|---|---|---|---|
| 2 | 2 | 0.694 | passes |
| 3 | 0 | 0.119 | passes |
| 4 | 0 | 0.0562 | passes |
| 5 | 0 | 0.0261 | fails |
| 6 | 1 | 0.0514 | passes |
| 30 | 2 | 4.2e-15 | fails (AD9's plain Jacobi mutation: every run outside) |

So the rule tolerates the original's own tail and a few runs more; it fails a port whose tail is
clearly heavier (from 5 of 30 when the original has none) and any systematic change of the
schedule's iteration counts. Everything a schedule's tail does not explain (the labels, the
components, the sums, the deterministic stages) stays judged run by run, with no allowance.

### 3. How it is carried out (under delegation)

- **The frozen rule**: `parity/goldens.toml`, `[sets.label_propagation_inplace.rate_rule]`
  `runs = 30`, `alpha = 0.05`, `test = "fisher_exact_one_sided"`, equal to
  `label_propagation_goldens.RATE_RULE` (a test keeps them equal).
- **The comparator**: `compare.py label_propagation inplace --rate --exe COMPAT` runs, per case,
  the port and the original alternately (port run k, then original run k) with the same argv and
  environment, so both see the same GPU (the caller's `CUDA_VISIBLE_DEVICES`) and the same
  session; it refuses `--runs` other than the rule's, a goldens.toml rule other than this one, and
  a recording twin as the original. Its record keeps every port run (labels, components,
  iterations, the run-by-run verdict `checked_every_run`), every original run's iterations, the
  table, the p-value and the verdict. Without `--rate` the comparator keeps the every-run check of
  `--runs` port runs (default 3).
- **The certificate**: `compare.py label_propagation certificate --kind inplace` carries the rule
  in the applied tolerance (`rate_rule`, per case) and the tables (`rate`: each side's runs and
  runs outside, the p-value, whether every port run passed its run-by-run checks, the trace
  mismatches). `certify.py` requires the applied rule to be the frozen one and **recomputes the
  test from the table** (`rate_rule_problems`): a case the replay passed fails the certificate if
  either side has not exactly 30 runs, the port's count is significantly higher, the recorded
  p-value is not the table's, a port run failed its run-by-run checks, the trace mismatched, or the
  table is missing (so a record of the every-run rule no longer certifies the set).
- **`ci/gpu_local.sh`** keeps its every-run cloud (5k5 and 50k, 3 runs each, stricter than the
  certificate): it is a local check, not the certificate. Whether it moves to the rate rule is
  left to the author with this ADR's wording.
- **Tests** (`parity/tests/test_label_propagation_goldens.py`, `test_label_propagation_port.py`,
  `test_release_cert.py`): the exact p-values; the frozen table; the comparator on synthetic runs
  (the original's own rate passes; AD9's plain Jacobi in_place mutation, every port run outside,
  fails; a run's labels beyond T_max or its sum outside the envelope fails the case although the
  rate is the original's; other run counts and rules refused); the certificate (a passed dry run;
  each faulty table above fails; a gated part removed still fails the dry run).
- **AD9's mutation on the real code**: the plain Jacobi in_place mutant
  (`DYNG_MUTATION_LP_INPLACE_PLAIN_JACOBI`, the parity-cuda build's mutant library linked into
  `dyng-compat-dynlp`) read under the rule; the record is in `parity/results/M10b.md` section 5.8.

## Consequences

- The certificate of the in_place set judges the port against the original's own behaviour in
  the same session, not against a fixed number of runs that the original itself fails about one
  time in five.
- A certificate reading takes 60 runs per case instead of 3 (about 15 minutes for the four cases
  on an RTX A5000).
- The rule has limited power for small rate increases (section 2): a port with a slightly heavier
  tail can pass. The run-by-run checks bound everything else.
- The every-run readings that missed (M10a and M10b) stay recorded misses.

## Alternatives considered

- **More runs under the every-run rule**: the original itself fails it more often the more runs
  are taken.
- **A wider envelope or a recalibration**: AD5 freezes the calibration; widening it is the author's
  and was not asked.
- **A binomial test against a fixed rate**: the original's rate is not frozen and depends on the
  GPU and the session; measuring it in the same session is what makes the comparison fair.
- **A chi-square test**: the counts are small; Fisher's exact test needs no approximation.
