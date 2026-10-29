# ADR 0065: The DynLP recorder twin, the label_propagation golden kinds, the in_place calibration and O's --check margins

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M10a step 1 (PLAN 6.3 step 1, 6.4.7, 8.3; the M10a
  acceptance criteria 2, 5, 6 and 7). The parity rules are the author's: AD5 (the in_place
  calibration), AD6 (damped byte parity), AD7 (the harmonic oracle), AD10 (the budgets, no
  release asset in 0.4), **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log,
  the rows "The parity rules of 0.4 (sheet C1-C8)" and "Licensing, data provenance, budgets and
  scope of 0.4 (sheet G1-G3, H1-H3)"), and ADR 0013 (parity and goldens), ADR 0036 (the
  link-time recording twin). This ADR records how they are carried out on the reference side.
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  reference-side-baseline-calibration)

## Decision

### 1. The recorder twin (no source patch; ADR 0036's method)

`parity/export_patches/LabelPropagation-CUDA/build.sh` relinks the PATCHED copy's UNCHANGED
`obj/*.o` of `bin/dynLP` and `bin/propagation_test` (O's own link lines, from `make -n -B`) with
`parity/exporters/dynlp/{dynlp_trace.cpp, recorder_dynlp.cpp | recorder_dynlp_hub.cpp}` through
`-Xlinker --wrap=<mangled>` for the symbols of `wrap_dynlp.txt` / `wrap_hub.txt` (taken with `nm`
from the archive build; every one is called across objects, which is what `--wrap` redirects).
The originals are kept as `parity_export/orig/`; `diff -r` of a fresh archive against the patched
tree lists added files only (`bin/`, `obj/`, `parity_export/`, `.dyng-reference`). Every
`__wrap_` calls `__real_` with the same arguments; recording is on only when
`DYNG_DYNLP_TRACE_DIR` is set (a wrapper then synchronizes the stream it was given and copies what
it records: timing changes, the stream order does not).

- **Wrap points and records** (per batch b unless stated): `runDynLP` before (the config, the full
  CSR with value bits, the true labels, every `BatchData`) and after (the final and IrLP label
  bits, the `BatchResult` columns, the return value); `generateSparseGraph` (its arguments and
  return) and `generateBatches` (with `DYNG_DYNLP_GENERATOR_ONLY=1`: the CSR, the labels and every
  batch, then exit 0 before `runDynLP`); `launchApplyVertexChanges` (the batch marker, exactly one
  per batch, with the device batch and the status and gt_class after the call);
  `launchAppendEdges` after (`row_end`, the slot `col_ind` and value bits, `row_ptr` in b0, the
  sticky error flags); `launchBuildSparsifiedSubgraph` (tau and batchStamp, the G' count, G' as
  sorted (min, max) pairs); `launchConnectedComponents` (numComponents, the component id of each
  vertex of Ins \ Del in list order, the `cc.hookRounds` delta through `prof::counter`);
  `launchLabelInitialization` (W0 / W1 per component as uint64, the initial labels of Ins \ Del);
  `launchIterativePropagation` before (the initial frontier: affected and ACTIVE and gtClass < 0,
  sorted; the labels) and after (iterations, the label bits, the `prop.frontierVisits` and
  `prop.persistentIterations` deltas, `ws.nextTag`); `runIrLPSnapshot` (the per-snapshot label
  bits and iterations); propagation_test's `launchIterativePropagation` (the hub graph as passed,
  status, gt_class, affected, the labels before and after, the rule, iterations, counters). A
  stage that is not called in a batch has no record there (absent).
- **Format** (`dynlp_trace.hpp`; the reader `parity/dynlp_trace.py`, no numpy needed): binary
  little-endian; a header (`DYNGDLT1`, version 1, flags); records of a tag, the batch, a
  per-tag sequence and typed fields (i32, i64, u64, f32 as IEEE-754 bits, u8, f64 bits, text),
  each stored raw or, with `DYNG_DYNLP_TRACE_DIGEST=1` (the 5M cases), as its element count and
  the SHA-256 of its bytes; an `end` record counting the records (a trace without it is refused).
  A trace is never overwritten. The reader re-encodes a trace byte for byte (`encode`), which the
  canonical goldens use.
- **check_twins** (`parity/exporters/dynlp/check_twins.py`): on damped 5k5 and 50k, the twin with
  recording off and on prints the unpatched O's `dynlp_labels.txt` bytes and its `results.csv`
  columns iterations, frontier_visits, components and accuracy; with recording on, one marker per
  batch and the final label bits print O's file; propagation_test's damped lines and verdicts are
  O's, off and on.

### 2. The golden sets (`parity/label_propagation_goldens.py`)

Under `$DYNG_SCRATCH/goldens/<set>`, each listed in `parity/goldens.toml` with the SHA-256 of its
`MANIFEST.sha256` and per-case digests (AD10: no release asset in 0.4):

| Set | Corpus | Holds |
|---|---|---|
| `label_propagation_generator` | G | bytes cases: O's four `graph/csr*.txt` files (checked against the unpatched O's own files and against the text of the recorded arrays) and the generator trace (CSR value bits, true-label bits, every batch); digest cases 5m, 5m1, 50m (`--noGraphFiles`): the digest trace only |
| `label_propagation_damped` | P+ | the unpatched O's `dynlp_labels.txt`, the deterministic `results.csv` columns and tau, the twin's per-batch label bits (5m: per-batch SHA-256) |
| `label_propagation_trace` | P+ | the twin's canonical trace (every record above, the timings left out; 5m in the digest mode) and a per-batch summary |
| `label_propagation_hub` | hub | the inputs and both damped calls; the in-place call's outputs of 5 runs as a distribution under `inplace/`, outside the manifest |
| `label_propagation_irlp` | 5k5, 50k | IrLP's label bits and iterations per snapshot, O's `irlp_labels.txt` |
| `label_propagation_inplace` | I | item 3 below (a distribution, exported once) |

`label_propagation_check` holds O's `--check` logs (section 4), outside any manifest.
P+ = {5k5, 50k, 500k} damped at delta 1e-4, 5k5 and 50k at delta 1e-6, 5k5 and 50k at seeds 1 and
2, and 5m by digest; G = 5k5, 50k, 500k as bytes, 5m, 5m1, 50m by SHA-256, seeds 1-3 at 5k5 and
50k, avgDegree 3 and 7 at 50k.

- **Hashing** (M9a finding 4): a manifest line is the SHA-256 of a file's content (a `.zst` file
  decompressed); `PROVENANCE.json` (the GPU, the copy, the run times, the zstd version) is hashed
  by nothing; no run-varying value (time_ms, eval_ms, IrLP's time) is in a hashed file.
- **Export twice** (PLAN 8.3; ADR 0013; M9a finding 4): `export --twice` builds two FRESH patched
  copies (`build_reference.sh --variant patched --fresh --scratch $DYNG_SCRATCH/ref/
  m10a-export-<n>`), exports the first on GPU 1 under the shared lock and the second on GPU 0
  inside a `locklog.py run --reset-gpu 0 EXCL` window (no timing), requires identical
  `MANIFEST.sha256` files, installs the first and removes the copies. No export starts below
  20 GB free. Every comparison of an export can fail (the twin against the unpatched O of the
  same copy; the final bits printed against O's files; the generator text against the arrays and
  O's files; IrLP's iterations and last snapshot against O's `results.csv` and
  `irlp_labels.txt`; `--validate` leaving the DynLP labels unchanged; the hub's damped calls
  across runs).
- **verify** (`compare.py label_propagation --verify`): the manifest against `goldens.toml`,
  every file against the manifest, the case digests, and per set: one marker per batch, the
  host status of the batches (O's host pass, `final_status`) equal to the device status of the
  last batch in every raw trace, the last batch's bits printing O's label file, the generator
  text equal to the recorded arrays, the frozen in_place thresholds recomputed from the
  calibration record.

### 3. The in_place calibration (AD5)

- **Runs:** N = 10 unpatched `--update inplace` runs per I case (5k5, 50k, 500k, 5m) on GPU 1
  under the shared lock at default clocks, plus 3 per case on GPU 0 at the base lock inside an
  EXCL window; `--quiet --noValidate --noGraphFiles`; each run's `dynlp_labels.txt` (zstd) and
  `results.csv` under `label_propagation_inplace/runs/<case>/`, with its SHA-256, GPU, clocks and
  time. Also N = 3 runs at delta 1e-6 on 5k5 and 50k (`tight/`, for M10a step 2's class margin).
- **The vertex set:** ACTIVE and not ground truth after the last batch: O's host status of the
  deterministic batches (`final_status`, from a generator-only run of the twin; verify checks it
  against the device status of the raw traces). The labels are O's 6-decimal file; a class is
  "label >= 0.5" of that text (a label printed as 0.500000 is class 1), the rule the port's file
  is read by as well.
- **Per case:** for all 45 pairs of the GPU-1 runs the mean and max |dF| and the class flips;
  T_mean = 3 x the largest pairwise mean, T_max = 3 x the largest pairwise max (floors 1e-6),
  T_flips = ceil(3 x the largest pairwise count); per batch the iteration median, minimum and
  maximum, and the median of the sums. The components must be identical in every run (else
  STOP). Every GPU-0 run against every GPU-1 run must fall inside T_mean, T_max and T_flips, its
  per-batch iterations within +/- 10 % of the median (at least +/- 2) and its sum within +/- 10 %;
  otherwise it is flagged for the author in GOVERNANCE.md's sub-block `M10 (AD)`.
- **Frozen:** the thresholds, iteration medians and components per case are in
  `parity/goldens.toml` (`[sets.label_propagation_inplace.thresholds."<case>"]`), the record in
  `calibration.json` (its SHA-256 in `goldens.toml`) and committed as
  `parity/results/M10a-inplace-calibration.json`. Recalibration only by ADR with the author's
  approval; `recompute_thresholds` lets certify.py's frozen-tolerance check (M10a step 5) pass
  an approximate replay only with the tolerance recomputed from the committed record.

### 4. O's --check margins (AD7)

`label_propagation_goldens.py margins`: the unpatched O with `--check` (verbose, so every
batch's label line is printed) and `OMP_NUM_THREADS=8` on GPU 1, every output line timestamped as
it arrives (`stdbuf -oL`): per batch mean |F - F*| against 2000 x delta, the margin
(2000 delta / mean), the PCG iterations and residual, the skips (`SKIPPED`: PCG did not reach
1e-12) and the host time of the oracle (from the check line before the label line to it). C =
{5k5, 50k} at delta 1e-6, both rules, `--validate` at 5k5: exit 0, every margin >= 2, 0 skips
(otherwise STOP). L = {500k, 5m} at delta 1e-4, both rules, time-boxed to 2 h for all of L.
Records: `parity/results/M10a-check-margins.json`, the logs in `label_propagation_check`.

## Outcome (M10a step 1)

See `parity/results/M10a.md` (sections 2-7) for every number; in short: check_twins passed;
the components are identical in every calibration run; the two exports are identical; the SASS
is identical (ADR 0066); the margins on C hold on every batch but one (5k5 damped, delta 1e-6,
batch 1: 1.32, O's own figure; the STOP of step 1, decided by the author below).

## Amendment (2026-10-29): the author's decisions on AD5's iterations and AD7's margin

**Decided by the author on 2026-10-29** (S M Shovan; GOVERNANCE.md, approvals log, the rows
"The in_place iteration rule (AD5, sheet C1)" and "The AD7 margin on C (AD7, sheet C3; the STOP
of M10a step 1)"):

- **The iteration rule of section 3** is replaced: an in_place port run's iterations per batch,
  and their sum, must lie inside the envelope [0.9 x the minimum, 1.1 x the maximum] of the 10
  GPU-1 reference runs (per batch, and for the sum); per batch at least `iterations_floor` = 2
  beyond the minimum and the maximum (the floor is moot for the sums, all above 1000), never
  below 0. Where O is stable this equals +/- 10 % of the median; it widens only where O spreads
  itself. `label_propagation_goldens.py`'s `iteration_envelope` / `envelope_of` compute it from
  the committed record (`M10a-inplace-calibration.json`, unchanged), `inplace_toml` freezes it in
  `parity/goldens.toml` (`iterations_lo`, `iterations_hi`, `iterations_sum_lo`,
  `iterations_sum_hi`, and `iterations_rule = "envelope"` in the rule), `verify` checks the frozen
  envelope against the record, and `iterations_inside` is the check M10a step 4 reads the port
  by. It was frozen in its own commit before any port run was read by it. `envelope-check` puts
  every reference run against it: all 13 per case (10 on GPU 1, 3 on GPU 0) lie inside
  (`parity/results/M10a-inplace-envelope.json`), which settles the GPU-0 cross-check flag. The
  label thresholds and the record are unchanged; `iteration_window` (the median rule) stays only
  because the record's `gpu0_outside` and self-check were computed by it.
- **The margin of section 4:** AD7's bound 2000 delta stays on every batch, schedule, engine and
  backend; the go/no-go margin of 2 is waived for 5k5 damped, delta 1e-6, batch 1 only (O's own
  margin 1.32), because the damped schedule is deterministic, so run-to-run noise cannot erode
  its headroom. The certificate reports every batch's margin; every other batch of C keeps 2.

## Consequences

- The port (M10a steps 2-5) replays fixed records: `lp_host_model` against the traces, the
  container against `row_end`, slots and status, the fused engine against the label bits, the
  generator against G, the in_place cloud against the frozen thresholds.
- About 120 KB of committed test data may be used by M10 in all (AD10); this step commits no file
  under `cpp/tests/data`.

## Alternatives considered

- **A source patch with recording hooks.** Rejected (ADR 0036): the twin relinks O's unchanged
  objects, so the recorded code is O's machine code.
- **Hashing the raw traces.** Rejected: they carry O's timings (time_ms, eval_ms, IrLP's time);
  the canonical traces leave exactly those fields out.
- **Bit-level labels for the calibration.** O's unpatched runs (AD5: unpatched) write only the
  6-decimal text; the floor of 1e-6 is above its rounding (5e-7).
