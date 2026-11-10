# ADR 0100: The hook-round count of the damped trace parity, against the original's own values

- **Status:** Accepted (2026-11-05). The rule is the author's, **decided by the author on
  2026-11-05** (GOVERNANCE.md, approvals log, the row "The hook-round count of the damped trace
  parity"; the STOP of the M10a review fixes after the merge of main); the evidence record, the
  comparator and the certificate check that carry it out are accepted under delegation
  (GOVERNANCE.md, "Delegation of technical ADRs"), technical scope only. M10a, the review fixes
  (PLAN 8.3, 8.5; the M10a acceptance criteria 6, 12 and 19). It refines ADR 0075 (the fused
  engine's damped byte parity) and ADR 0078 (the 0.4 certificate rules) for one field and changes
  nothing else of them: no golden, no other field, no criterion.
- **Date:** 2026-11-05
- **Deciders:** S M Shovan (the rule); the AI assistant, on the author's behalf (its
  implementation)

## Context

The fused engine's damped replay (`compare.py label_propagation damped`, ADR 0075) compares every
field the trace golden records, the batch inputs aside (criterion 12: "every trace field is
equal"). One of them is `connected_components.hook_rounds`, the number of Shiloach-Vishkin rounds
of the connected components of G' (the golden's "hook-round delta" of criterion 6, recorded from
the original's `cc.hookRounds` through `prof::counter`, ADR 0065).

The damped replay on the merged library (`d327950`, 2026-11-04) was identical in 9 of 10 cases
of P+; at 500k every byte, every label bit, the results.csv columns and every other trace field
(the component ids among them) were equal, but batch 2's hook-round count was 3 against the
golden's 4. In 70 fresh runs of each side at 500k on GPU 1
(`parity/results/M10a-merged/hook-rounds-500k.json`), **LabelPropagation-CUDA's own recorder
twin (its unchanged objects) takes 3 rounds instead of 4 at batch 2 in 2 runs**, the port in 3;
every other batch's count equals the golden's in every run. The hook kernel, verbatim from the
original (its SASS is the original's, `parity/results/M10a-merged/sass-port.json`), reads parents
that other threads of the same launch lower with `atomicMin`, so the number of rounds until a hook
changes nothing depends on the thread schedule. The count is therefore not a deterministic field
of the original, and the exact check of it was not exact (a STOP; PLAN's rule "an exact check is
not exact" routes it to the author).

## Decision

### 1. The rule (the author's)

`connected_components.hook_rounds` is read as a **non-deterministic counter of the original**: the
damped trace parity compares it against the set of values the original's own runs produce for
that case and batch. The labels, the per-batch label bits, the results.csv columns, the component
ids and every other trace field stay compared exactly, and the goldens are unchanged. The existing
reading is judged again under the rule, not re-rolled.

### 2. The evidence (`parity/results/M10a-hook-rounds.json`)

The comparator's input is a committed record of the original's value sets, per case of P+ and
per batch: the golden's value (one run of the twin, exported twice) and every value of the twin
in the fresh runs of a committed source record. It holds one case today, 500k from
`hook-rounds-500k.json` (70 twin runs: `{3, 4}` at batch 2, the golden's value alone at the
other nine). A case or batch the record does not list accepts **the golden's value only**, so the
field stays exact wherever the original's own variation was not recorded; a new value of the
original (a later reading that finds one) is added only through a new committed twin record,
never from the port's runs. The record names each source with its SHA-256, and
`label_propagation_port.original_values_problems()` checks that every set is exactly what its
source's golden list and twin runs give (the port's runs in the same source are not evidence).

### 3. The comparator (`parity/label_propagation_port.py`)

`ORIGINAL_VALUE_FIELDS` names the one field. `compare_traces(..., original=, accepted=)`: a field
of that list that differs from the golden is not a mismatch when it is a scalar i64 on both sides
and **both** the port's and the golden's values (matched by their SHA-256, so a digest trace is
judged the same) are in the batch's set; it is appended to `accepted` instead. Without `original`
(every other replay: the stage trace, the container, in_place's deterministic stages) the field is
exact as before; the deterministic replay's field list (`TRACE_FIELDS`) never held it. `damped`
records, per case, `original_values_accepted` and, once, `original_value_sets` (the fields, the
evidence record and its SHA-256). `damped-rejudge --record R --json OUT` judges an existing damped
record again under the rule without a run: it moves only recorded `hook_rounds` mismatches whose
two values (by the recorded SHA-256 prefixes) are in the set, and passes a case only if nothing
else was recorded against it (the labels, the label bits and results.csv equal, a mismatch list
shorter than the 20 a record keeps, its golden of the digest mode the replay runs the case in).

### 4. The certificate (`parity/label_propagation_cert.py`, `parity/certify.py`)

A damped certificate record carries the rule it applied as its `tolerance`
(`original_value_sets`: the fields, the evidence record, its SHA-256 and every accepted
difference with its case, batch and both values). `certify.original_value_set_problems`, called
for `label_propagation_damped` only, fails the certificate unless the committed evidence passes its
own check, the record names that evidence by its SHA-256 and those fields alone, the tolerance holds
nothing else, and every accepted difference has both values in the evidence's set of its case and
batch. A damped record without a tolerance compared every field exactly and passes as before.

### 5. The existing reading, judged again

`compare.py label_propagation damped-rejudge` over the merged library's record
(`parity/results/M10a-merged/damped.json`, 2026-11-04) gives
`parity/results/M10a-merged/damped-adr0100.json`: **10 of 10 cases identical**, the one accepted
difference 500k batch 2, `hook_rounds` 3 (golden 4; the original's set `{3, 4}`). The gpu_local
step's damped record on the merged library (`gpulocal-damped-cuda-fused.json`) was already exact
in all its 9 cases and stays so. The certificate record
`parity/results/M10a-gates/parity-label_propagation_damped-cuda-fused.json` was written again
from the judged record, and the 0.4 certificate's dry run of label_propagation passes.

## Consequences

- The damped byte parity no longer depends on the thread schedule happening to give the golden's
  round count; the bytes of the result and every deterministic field are still exact.
- A hook-round count outside the original's recorded values still fails the replay and the
  certificate, as does any difference in another field (tests in
  `parity/tests/test_label_propagation_port.py` and `parity/tests/test_release_cert.py`).
- The evidence can only widen through committed runs of the original's twin; widening it is a
  change of a record certify.py checks against its sources, visible in review.
- M10b's operators engine keeps the rule for its fused path; a refactored engine (AD6: within
  c·ε with identical counts, or non-default) is judged by the same evidence.

## Alternatives considered

- **Keep the field exact** (the conservative default while the STOP was open): a replay would pass
  only when the schedule happens to give the golden's count, about 96 % of runs at 500k batch 2.
  Not taken: the author decided the field is not deterministic in the original.
- **Drop the field from the comparison**: would accept any count, also one the original never
  produces (for example a port that does more or fewer hook rounds by a change of the kernel).
  Not taken: the author's rule compares it against the original's values.
- **Accept the port's own observed values**: the port's runs are what is being judged, not
  evidence; only the original's twin runs and the golden define the set.
