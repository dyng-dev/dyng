# ADR 0078: label_propagation's gate readings, the suite ics26_dynlp and the 0.4 certificate

- **Status:** Accepted under delegation (2026-10-30; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. M10a step 5 (PLAN 8.3, 8.5, 8.6; the M10a acceptance criteria 10,
  18 and 19). The gates, their protocol and AD4's release gate are the author's, **decided by the
  author on 2026-10-28** (GOVERNANCE.md, approvals log: AD2-AD4 and B5 in the row "The measurement
  protocol and gates of 0.4 (sheet B1-B6)"; AD5-AD7 in "The parity rules of 0.4 (sheet C1-C8)",
  refined by the rows of 2026-10-29 and 2026-10-30); ADR 0024 for the dynG-against-dynG reading
  and ADRs 0018 / 0021 for the one-layout protocol are unchanged. Nothing here changes a bar, a
  round count or the protocol.
- **Date:** 2026-10-30
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  kit-mutations-gates-certificate)

## Decision

### 1. The suite `benchmarks/paper/ics26_dynlp.yaml`

In the format of `ipdps26_escher.yaml` (title, schema 1, suite, algorithm, paper, harness,
datasets, batches, backends, metrics, baselines, tolerance, readings, paper_reference):

- **harness:** tool `perf_ab.label_propagation`, inputs `$DYNG_SCRATCH`, executable
  `tools/compat/dyng-compat-dynlp`, presets `{openmp: parity, cuda: parity-cuda}`, region map
  `parity/timed_regions/label_propagation.toml`, goldens `label_propagation_damped`.
- **datasets** are LabelPropagation-CUDA's configurations 50k, 500k, 5m, 5m1 and 50m, generated
  in-process by both sides, each pinned by its generator parameters and the golden set
  `label_propagation_generator`'s digest (its CSR, labels and batches); **batches** the same
  configurations (`legacy.dynlp_vertex_arrival`, seed 42).
- **metrics:** `update[sum]` (compute), `end_to_end`, `device_memory` (memory), and reported rows;
  **tolerance** {compute 1.05, compute_short 1.10, short_region_ms 10, end_to_end 1.10, memory
  1.05}; **backends:** cuda on GPU 0, `CUDA_MODULE_LOADING=EAGER`, no `OMP_*`; **baselines:**
  LabelPropagation-CUDA `a276a3a4`, unpatched.
- **readings**, each with a **schedule** key (damped | in_place), one window per case:
  `damped-50k`, `damped-500k` (21 rounds), `damped-5m`, `damped-5m1` (11 rounds), gated, at the
  baseline's recorded lock (boost for every case); `memory-damped` (gated, GPU 1, D1 / B5);
  `in_place-*` (the same cases and rounds, reported); `damped-default-clocks` (11 rounds at 50k,
  500k, 5m; ADR 0018, recorded); `kernels-500k` (ncu at base, reported); `damped-50m` (3 rounds at
  base) and `in_place-50m` (one run), recorded (AD4); and **`release-damped-50m`**, the release-only
  50m `update[sum]` gate of AD4 (5 rounds at base, `--noGraphFiles` on both sides, read once for
  the 0.4.0 certificate at R040) with `release_gate: {state, reason}`.
- **paper_reference:** O's results/README.md figures (3ecb438, in place, default clocks, medians
  of 3, a loaded host: flagged) and the step-1 baseline; never a gate reference.

### 2. `parity/bench_suite.py` (additive)

`TOOLS`, `REGION_MAPS`, `MAP_KEYS` and `EXECUTABLES` know label_propagation. `validate` gains
`validate_label_propagation`: the configurations and their digests, every reading's schedule and
cases, a gated reading damped (AD2), with AD3's rounds and at the recorded lock, the memory
reading on GPU 1, the release-only reading a gated damped run of 50m at base with 5 rounds and a
`release_gate` whose state is `gated` or `recorded_only` and whose reason is not empty; another
algorithm's suite may not use `schedule`, `release_only` or `release_gate`; a recorded
label_propagation reading may have fewer than 5 rounds (AD4's 3 and 1). `plan` writes one
`perf_ab.py label_propagation run|memory --side both|kernels` per case and leaves a release-only
reading out unless `--release-only`; `summarize` reads the records with `lp_read_record` (the
command, the schedule, the case, the rounds, the lock, the guard, a parity-cuda port; rows
`<case>/<schedule>`). Every earlier suite validates and reads the same numbers (the step-1
snapshot tests and the certificate tests are unchanged).

### 3. `parity/certify.py` for 0.4

- `REQUIRED_SUITES['ics26_dynlp']` since (0, 4); `EQUIVALENCE_TARGETS` and `EQUIVALENCE_FILES`
  gain `dyng-compat-dynlp` (one entry per line, below M10's comment); no (0, 4) `REQUIRED_CHECKS`
  entry; M10's sub-list of the one `GPU_LOCAL_STEPS[(0, 4)]` key holds the label_propagation step
  of `ci/gpu_local.sh`.
- `LABEL_PROPAGATION_REPLAYS_SINCE = (0, 4)` (after `TRIAD_MODES`, the `TRIAD_REPLAYS_SINCE`
  precedent) and `LABEL_PROPAGATION_REPLAYS`: generator (sequential), trace (sequential, cuda),
  damped and hub (`cuda-fused`, a parity-cuda build only: the record's build is checked), irlp
  and inplace (cuda), oracle (sequential and cuda; the new set `label_propagation_oracle` of
  `parity/goldens.toml` with its coverage table: C on both, L on cuda). The records are
  `parity/label_propagation_cert.py`'s (`compare.py label_propagation certificate`), converted
  from the replays' `--json`, which carry the port's commit and build (`stamp_port`). A 0.3 or
  older certificate lists the sets without a verdict.
- **The frozen-tolerance check** `frozen_tolerance_problems`, called only for the calibrated
  kinds: an inplace replay passes only if its tolerance per case (T_mean, T_max, T_flips, the
  iteration envelope) is the one recomputed from `parity/results/M10a-inplace-calibration.json`
  (`label_propagation_goldens.recompute_thresholds`, `envelope_of`), whose SHA-256 goldens.toml
  freezes and whose thresholds goldens.toml holds; an oracle replay only with the bound factor
  2000, k = 3 and the m of `parity/results/M10a-class-margin.json` (frozen by its SHA-256).
- **`missing_gates`** reads label_propagation per gate case: `update[sum]` and `end_to_end` of
  every gated damped run and `device_memory` of the gated memory reading. AD4's 50m row is
  required in a write of the 0.4 series or later that is not a dry run exactly when the reading's
  state is `gated`; when it is `recorded_only` the certificate shows it as recorded only, with its
  reason; a dry run lists it as outside the dry run (pending R040); a version before 0.4 never
  requires it (nor the suite). The certificate validates every suite file it reads, so a suite
  with a changed tolerance, preset or schedule, or a `recorded_only` without its reason, fails it.

### 4. AD7's class rule at delta 1e-4

AD7 gives the class rule (no flip where \|F* - 0.5\| > 3m) with C's conditions, m taken from the
original's own delta 1e-6 runs (m = 2.9217e-3). At delta 1e-4 the original's own 50k labels (the
damped golden's label bits) flip beyond 3m in batches 2-7, so the rule cannot be a parity rule
there. `compare.py label_propagation oracle-port --cases L|weekly` and the certificate's oracle
kind therefore gate, at delta 1e-4, the bound mean \|F - F*\| <= 2000 delta and the PCG
convergence (0 skips), and report the flips beyond 3m; at delta 1e-6 (C) all three are gated.
First taken as the most conservative option of a new question in an author-reserved area, it was
**decided by the author on 2026-10-30** (GOVERNANCE.md, approvals log, the row "AD7's class rule
at delta 1e-4").

### 5. The readings

On the final M10a library (`72eb020`; `parity/results/M10a.md` section 12.7, the compacted
records `parity/results/M10a-gates-final.json`), one exclusive window per case and schedule on
GPU 0 at the baseline's recorded lock (boost, base at 50m), the memory in GPU-1 quiet windows:

| Case (damped) | update[sum] (1.05) | end_to_end (1.10) | device memory (1.05) |
|---|---|---|---|
| 50k (21 rounds) | 1.006 | 1.097 | 1.0005 |
| 500k (21 rounds) | 1.005 | 1.032 | 1.0005 |
| 5m (11 rounds) | 0.988 | 1.006 | 1.0005 |
| 5m1 (11 rounds) | 1.014 | 1.029 | 1.0018 |

All 12 gated rows pass; the guard held in every round, the audit flagged no round of a gated
reading, and the rejected rounds (1, 7, 0, 0) were repeated. Reported: in_place `update[sum]`
1.008 / 1.005 / 1.008 / 1.037 with the iteration medians equal within 1 %; the default clocks
1.017 / 1.002 / 0.998 (50k / 500k / 5m); ncu at base: every ported kernel with the original's
registers and occupancy; 50m at base: damped `update[sum]` 1.014 and `end_to_end` 1.065 over 3
rounds (10 min 50 s, 0 rejected), in_place 1.011 / 1.078 over one. **AD4's feasibility
statement:** 5 rounds take about 18 min (under 2 h) with no rejected round (under 30 %), so
`release-damped-50m` is `gated` (its reason in the suite states this basis). The first
reading at 5m1, on `0c815a1`, had missed the bar (1.080); its cause and fix are ADR 0079, and
no bar, round count or protocol was changed.

**Amendments of the M10a review.** (1) A label_propagation set passes only when the passed cases
of each required backend cover the set's cases (its coverage table, else its case table, the
in_place thresholds' cases or the hub's one case): a replay of a subset certifies nothing of the
rest. (2) certify.py re-checks every label_propagation record of the summary with
`bench_suite.check_lp_record` (the rounds, one sample per round and the medians those of the
samples, the guard, the clock lock; the memory sides counted, nvidia-smi before, during and after,
no foreign process, the side's own process seen during the reading), the
`recheck_triad_readings` precedent. (3) The oracle's frozen tolerance is read from what the
replay applied (the bound / delta of every batch, the class limit / m, the class rule exactly at
delta 1e-6), never from constants. (4) The memory sampler polls every 0.25 s. The readings on the
fixed library: `parity/results/M10a.md` section 13.4 (12 / 12 gated rows within; 50m damped 1.011
/ 1.060 over 3 rounds, the release gate stays `gated`).

## Consequences

- R040 reads `release-damped-50m` with `bench_suite.py run --release-only`; it may switch the
  state to `recorded_only` under the same AD4 condition, with the reason recorded.
- M11's suite and replays join the same tables below their own comments.
- On the merged tree, a scoped dry run passes only over records measured on the merged library's
  code. M11a's records predate the merge, so `hyper_sssp`'s dry run on M10a's tree fails until
  they are measured again. The author ruled on 2026-11-11 (GOVERNANCE.md's approvals log) that
  this belongs to the integrator, the second of M10b and M11b to freeze, who runs both scoped dry
  runs on the final tree (`parity/results/M10a.md` section 14.9).
