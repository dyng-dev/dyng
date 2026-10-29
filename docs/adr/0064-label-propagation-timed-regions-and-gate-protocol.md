# ADR 0064: The label_propagation timed regions, gate protocol and harness (M10a)

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M10a step 1 (PLAN 6.3 step 3, 6.4.7, 8.6; the M10a
  acceptance criteria 1 and 4): the map of LabelPropagation-CUDA@a276a3a's timer onto the port,
  the harness keys it needs and the pre-port baseline's protocol. The gates, their rows and the
  measurement protocol are the author's: AD2, AD3 and AD4, **decided by the author on
  2026-10-28** (GOVERNANCE.md, approvals log, the row "The measurement protocol and gates of 0.4
  (sheet B1-B6)"), B5 (the memory reading, the same row) and the earlier binding decisions (the
  one-layout protocol of ADRs 0018 / 0021 against the originals; D1, the gated Nsight Systems
  peak). This ADR changes none of them: it writes down which timer of the original is compared
  with which part of the port, and how the reference side is measured.
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  reference-side-baseline-calibration)
- **Builds on:** ADRs 0018 and 0021 (clocks), 0024 (dynG against dynG), 0048 and 0056 (the
  triad_count map, records and audit), 0063 (the M10 plan of record).

## Context

LabelPropagation-CUDA@a276a3a (O) times each batch of `runDynLP` (src/dynLP.cu) with two CUDA
events on its compute stream: `batchStart` at :418, recorded **before** the host net-effect pass
(:435-479: the effective insertions Ins \ Del, the host status, the id checks), and `batchEnd` at
:642, after Step 3 (the propagation, which ends with the ground-truth enforcement kernel); the
elapsed time is `results.csv`'s `time_ms` (utils.cu:265-300). Inside the region: the batch's
transfers on a second stream from page-locked memory (:493-520, overlapped with Step 1a through
two events), the deletions and status changes (b2), the edge append (b3), and, only when
`numInserted > 0`, sparsification, connected components and label initialization (b4-b6; the
`--check` copies at :584-598 and the `--verbose` count copy at :592-598 are inside the region),
then the propagation (b7). Outside: the evaluation (:661-716, `eval_ms`), `--validate`'s IrLP,
the setup before the loop (:290-410: every allocation sized for the largest batch and
`cudaHostRegister` of every batch vector, :217) and the generation (main.cu). PLAN 6.4.7 gates
"the sum of per-batch `time_ms`" at <= 1.05x and says "the damped-rule baseline for the stable
gate is measured before the port"; PLAN 6.3 step 3 requires the map before the port.

## Decision

### 1. The map (`parity/timed_regions/label_propagation.toml`, section `[reference.dynlp_cuda]`)

- **`update`**, per batch from b0 (`first_batch = 0`: b0, the 1 % ground truth and the edges
  among it, is an ordinary update, counted, timed and consuming a propagation tag), rows
  `update[b0]` ... `update[b10]` (a case reads the rows of its own batches), reported
  (`gate = "none"`); `report_sum = true` with `sum_gate = "compute"`: the row `update[sum]` is
  the gated row of AD2 (<= 1.05x; <= 1.10x with at least 20 runs where O's median is below
  10 ms). O: the sum of `csv:time_ms` over b0..b(B-1). Port: the sum of the CUDA-event time
  bracketing each public `label_propagation::update()` call on the resources' stream in
  `dyng-compat-dynlp` (O's own mechanism), written as `time_ms` in O's `results.csv` format, with
  the host wall time from the call to its return beside it (`host_ms`, reported).
- **`eval`**: `csv:eval_ms`, summed, reported.
- **`end_to_end`**: the process wall time (bench/run.sh's `wall_s`), gate `end_to_end` (1.10x).
- **`device_memory`**: the Nsight Systems peak of live device allocations of the whole process
  (D1), gate `memory` (1.05x).
- **Stage rows** (`stage.*`, reported, from a separate diagnostic reading only: O with
  `--profile`, whose stage boundaries synchronize; the port with `--sync-stages`): O's
  `b1.host.batchIds`, `b2.deletions+status`, `b3.appendEdges`, `b4.sparsify`,
  `b5.connectedComponents`, `b6.labelInit`, `b7.propagation` against the sketch's stages
  (`label_propagation.normalize`, `.commit`, `.identify_affected.sparsify`, `.components`,
  `.seed`, `.loop` + `.finalize`). The port's stage names are provisional until its first commit
  (M10a step 4 may rename them, with the map, in one commit).
- **Scope rules** (AD2, the same on both sides): generation outside the region; the batch arrays
  page-locked and the workspaces reserved for the largest batch before the timer (O in its setup;
  the port with `pinned_host` arrays, `compute()` on the empty capacity graph and
  `result::reserve`); timed rounds with `--quiet --noValidate` and without `--check`,
  `--profile` or `--sync-stages`; the evaluation after the timer, every batch; one process per
  case; tau published; `CUDA_MODULE_LOADING=EAGER` and no `OMP_*` on both sides. `end_to_end`
  with bench/run.sh's argv, the graph files written into a fresh per-run scratch directory on
  the same filesystem for both sides and deleted after the round; `--noGraphFiles` only at 50M,
  on both sides.
- **Decided by the author on 2026-10-28** (AD2-AD4, recorded here for the map's readers): damped
  is gated, in_place reported; 21 rounds at 50k and 500k, 11 at 5m and 5m1; the clock per case
  and schedule from a recorded 3-round boost probe, else base; default clocks recorded only;
  50M x 11 recorded in M10 (3 rounds at base) and gated once in the 0.4.0 certificate (5 rounds,
  base) unless 5 rounds would take more than 2 h or more than 30 % of the rounds are rejected;
  PLAN 6.4.7's figures (3ecb438, unlocked clocks, a loaded host) are `paper_reference` only.

### 2. The harness keys (`parity/perf_ab.py`, `parity/bench_suite.py`)

M10's half of the joint key specification with M11 (orchestrator's reconciliation of
2026-10-27; M11 adds a list-valued `original` and the `e2e:` keys in its own functions):

- `first_batch` (0 or 1, default 1; only on a per-batch region): the index of the first row;
  `update[b0]` rows are accepted only with `first_batch = 0` (`perf_ab.first_batch`).
- `sum_gate` (`"compute"` or `"none"`, default `"none"`; it needs `report_sum = true`): gates the
  existing report_sum row `<name>[sum]` (no new `gate_sum` key; `perf_ab.sum_gate`,
  `region_rows`, `row_gate`); `bench_suite.region_map` gives that row its gate.
- O's report keys (`perf_ab.dynlp_report_keys`): `csv:time_ms`, `csv:eval_ms`,
  `stages:<b1..b7>`, `process wall time`, `nsys:peak_live`; the gate kind `memory` is accepted
  only on the `nsys:peak_live` row of this map (`perf_ab.memory_gate`).
- Unknown keys, report keys and gates are still refused. `ci/tests/test_lp_regions.py` checks
  every rule, and that every earlier map reads the same rows and gates as at B0 (b2b1524; a
  SHA-256 of the canonical reading) and every suite under `benchmarks/paper` validates and reads
  its 45 committed 0.3.0 readings to the same regions.

### 3. The pre-port baseline (`perf_ab.py label_propagation baseline`)

`parity/label_propagation_perf.py` (reached as `perf_ab.py label_propagation`): the UNPATCHED
copy's `bin/dynLP` alone, GPU 0, inside a `locklog.py run --reset-gpu 0 EXCL` window per case and
schedule (R1; at most 2 h); per round one process with bench/run.sh's argv and its output in a
fresh scratch directory (deleted after the round; no round below 20 GB free); the contamination
monitor (rejected rounds repeated, at most 2 x rounds) and triad_perf's audit; per case and
schedule the 3-round boost probe and its decision (boost only if every busy sample of every round
is at the lock and no round was contaminated, else base), the reading at that lock (21 / 11
rounds), and 11 rounds at default clocks. A per-round guard (damped: the per-batch iterations,
components, frontier_visits, accuracy and the label file's SHA-256 equal the first round's;
in_place: the components; tau always) makes every comparison able to fail; a failure stops the
reading. Per batch: `time_ms`, iterations, components, frontier_visits, `eval_ms`; per round the
wall time, the host peak RSS (wait4) and the output's disk use; per reading the medians of the
map's rows and the short-region rule (PLAN 8.6: a damped `update[sum]` median under 10 ms is
read at 1.10x with at least 20 runs). The nsys peak of each case and schedule:
`perf_ab.py label_propagation memory --side original` on GPU 1 inside a `locklog.py run
GPU1-QUIET` window, nvidia-smi before, every 5 s during and after (a reading with a foreign
process is not counted). These records are never a gate reference: the gates re-measure O in
the same session as the port (AD3).

### 4. The port side

`run` (A/B/A/B of O and `dyng-compat-dynlp`), `kernels` (both sides under ncu at base) and
`engines` (fused against operators, AD6) have their command lines and record shapes
(`label_propagation_perf.RECORD`) fixed now and tested; they refuse to run until the port exists
(M10a steps 4-5).

## Consequences

- The map is committed before any file under `cpp/src/algorithms/label_propagation/`,
  `cpp/include/dyng/label_propagation.hpp`, `cpp/src/graph/` or `cpp/include/dyng/graph/`
  changes (the branch's log shows the order).
- The baseline's numbers and the per-case short-region rule are in
  `parity/results/M10-baseline-<date>.json` and `parity/results/M10a.md`.

## Alternatives considered

- **Gate each batch** (as triad_count's `update[b1..b3]`). Not taken: AD2 gates the sum, which
  is PLAN 6.4.7's region; several batches are short (b0 under 1 ms), whose per-batch reading
  would need 1.10x and 20 runs each without being a paper figure.
- **A new key `gate_sum`.** Not taken: the joint specification gates the existing report_sum row
  with `sum_gate`, so a map has one place for the sum.
- **Comparing with PLAN 6.4.7's published figures.** Not taken (AD3): they were taken at 3ecb438
  on unlocked clocks and a loaded host.
