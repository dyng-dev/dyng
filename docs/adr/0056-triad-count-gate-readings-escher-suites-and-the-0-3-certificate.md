# ADR 0056: The triad_count gate readings, the ESCHER suites and the 0.3 certificate

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out M9a step 5 (the M9a acceptance criteria 12, 13
  and 14; PLAN 6.4.6, 8.5, 8.6, 10.3). The measurement protocol is the author's and is used as it
  is: PLAN 8.6 (the tolerances, the A/B protocol), ADRs 0018 and 0021 (the clock lock), the
  one-layout decision of 2026-10-08, the memory decision of 2026-10-07 (store accounting) and the
  D1 reading of 2026-10-15 (the gated Nsight Systems peak). **Pending the author** (taken on the
  plan's defaults until the 0.3.0 release, decisions of 2026-10-15): the suites and their split
  (D8), the gate rows, cases and rounds (D7), the coverage table the certificate requires (D6).
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step gates-suites-certificate)
- **Builds on:** ADR 0018 and ADR 0021 (clocks), ADR 0024 (dynG against dynG), ADR 0048 (the
  triad_count timed regions and goldens), ADR 0050 (the replays), ADR 0051 (the driver and
  `perf_ab.py triad_count`), ADR 0053 (the CUDA port).

## Context

M9a's last step reads the gates of PLAN 6.4.6 and 8.6 for `triad_count` against
ESCHER-GPU@abcf9b2 (A) and makes them part of the release certificate from 0.3 on: two paper
suites (PLAN 8.5 names `ipdps26_escher.yaml` and `tkde26_escher_plus.yaml`), their records, and
`parity/certify.py`'s rules for the 0.3 series. `perf_ab.py triad_count` (ADR 0051) wrote records
for the go/no-go pre-reads and the early reading of step 3, not records a suite summary or a
certificate could read: no gate per row, no input digests, no per-record clock control the
summary compares with the suite, no store-accounting rows.

## Decision

### 1. The records of `perf_ab.py triad_count` (schema 2)

1. **run**: results keyed `<case>/<pattern>` (`--patterns`; the gated readings are h_motif's,
   `main`); every row of the region map the port records, with its gate kind and, for a gated
   region, its gate (`perf_ab.gate_limit`: 1.05, 1.10 below 10 ms, 1.10 end to end) and verdict;
   the reference (A's unpatched copy, built by its Makefile), the port's commit and build (the
   parity-cuda preset is checked), the clock control (`protocol.clocks.control`), the rounds, the
   per-round guard (A's printed counts, and the port's, equal the golden case's in every round,
   accepted or repeated; a failure stops the reading and leaves no record), the rounds the
   monitor rejected and repeated, and the input digest taken by the measuring process:
   coauth-DBLP's file, or a synthetic input's rows as the golden set stores them (both sides
   generate the input in-process from A's parameters and the seed; generators::legacy reproduces
   A's rows byte for byte, M9a step 1, and the guard requires the golden's full count).
2. **memory**: the gate `device_memory` (D1) as `original` / `port` peaks of live allocations;
   the gate `store_memory` as the store accounting per point, each point marked whether the gate
   reads it (section 3); the Nsight Systems details beside them (the NVTX ranges of each peak,
   the peaks per range, live / pool / slack / held bytes at the end of every batch, the
   footprint at the live peak and between batches, flagged above 1.05, not gated); the triad
   bytes live during `hypergraph.apply`, measured as the difference between the peaks inside
   `hypergraph.apply` of the counting port and of the same process with `--no-count` (a third
   nsys run); foreign processes on the GPU checked before and after each case (the reading stops
   if one is there).
3. **kernels** (recorded, not gated; ADR 0018 rule 1): the update's subtract and add kernels of
   both sides (A's `countWedges` and the pattern's counting kernel; the port's transplanted
   kernels) under `ncu --clock-control base --cache-control none`, selected with
   `--kernel-name-base function --kernel-name <regex> --launch-skip <the full count's launches>
   --launch-count <the batches' launches>` (two launches per phase, one for type 1), A/B
   alternation, medians per batch and phase of the kernel time and DRAM bytes, registers and
   launch geometry.

### 2. The two suites (decision D8 pending the author)

ESCHER-GPU's repository does not say which of its readings belong to the IPDPS paper (ESCHER) and
which to the TKDE extension (ESCHER+): its `results/README.md` has one table per measure and its
`README.md` names "ESCHER / ESCHER+" once, for the best-fit reuse insert. The split is therefore
ours:

- `ipdps26_escher.yaml`: the timing gates of ESCHER's paper-timed region (update per batch,
  full count, end to end) on coauth_b50k_s1-s3 (11 rounds) and syn1m / syn2m b50k s1-s3 (21
  rounds), all at the base clock lock (ADR 0021; coauth-DBLP as in M8b, the synthetic inputs
  because the recorded 3-round boost probe of M9a step 2 did not hold), with the default-clock
  reading (seed 1, 11 rounds) and the 100K / 200K coauth batches (base, 5 rounds) recorded;
- `tkde26_escher_plus.yaml`: the memory gates (`store_memory`, `device_memory`) on
  coauth_b50k_s1, syn1m_b50k_s1 and syn2m_b50k_s1, with StatHyper's types 1-3 on coauth_b50k_s1
  (timing, base, 5 rounds) and the memory at the 100K / 200K coauth batches recorded. The
  "batch sizes" reading of this suite is a memory reading, so that no reading runs twice.

Both pin the datasets: coauth-DBLP by its file's SHA-256; syn1M and syn2M by A's generator
parameters and the SHA-256 of their rows per seed (the golden set's stored `.hg`, M9a step 1).
`parity/bench_suite.py` validates them against the golden set (every case a group of
`triad_count_escher_gpu`, every digest the golden set's, coauth-DBLP's gated readings at base)
and refuses a record whose cases or patterns are not the reading's, with fewer rounds than the
reading's runs, a missing batch row, a guard that did not pass in every round, a kept
contaminated round, a flagged contamination audit (a gated reading), a foreign GPU process during
a memory case, or a store-accounting point the gate reads missing. A suite gates timing only
where it has a gated run reading on the backend, so `tkde26_escher_plus` lists the memory metrics
only.

### 3. The memory gates

- **store_memory** (the author's decision of 2026-10-07, carried into M9): dynG's
  `store_stats().device_bytes` against A's `DynamicHypergraph::deviceBytes()` at the same
  points, <= 1.05: on coauth-DBLP after the build and after each of the three batches (A's values
  from the golden set's recording of the same batches; the two end points must equal A's own
  `[memory]` print of the same run, or the reading stops); on syn1M and syn2M at A's two print
  points (after the build, after the last batch), the values the golden recording and A's print
  agree on, the middle points recorded but not gated. Rows `store_memory[build]`,
  `store_memory[b<k>]`, in bytes.
- **device_memory** (D1): the peak of live device allocations (`cudaMalloc` +
  `cudaMallocAsync`, kind Device; A's Thrust temporaries included) of both whole timed processes
  (no `--check`) under `nsys profile -t cuda,nvtx --cuda-memory-usage=true`, GPU 1, the shared
  lock, no foreign process; port / A <= 1.05.

### 4. The contamination audit

The monitor (perf_ab.MachineMonitor) repeats a round with foreign CPU load or a foreign GPU
process. R020 showed rounds that pass it while the machine is in a storm's tail (load average
27-30 from tasks blocked on mounts). Every triad round records the 1-minute load average and
`procs_blocked` at the start and end of each side; a gated reading is flagged when a round's
load average exceeds 16 (the readings of M9a step 2 saw 2-9 with A's parallel host build) or
`procs_blocked` exceeds 4, or a foreign compute process was on the GPU, and the summary refuses
it: the reading is taken again (`bench_suite.py run --skip-existing` after the record is moved
aside). No round is ever kept contaminated (`--keep-contaminated` does not exist for
triad_count).

### 5. The 0.3 certificate (`parity/certify.py`)

1. `REQUIRED_SUITES` gains `ipdps26_escher` and `tkde26_escher_plus` from (0, 3); a 0.1 or 0.2
   certificate does not need them.
2. `record_region` and `recheck_gates` read the `store_memory` rows point by point (bytes) from
   the record next to the summary, like every other gated row.
3. `missing_gates`: a metric named as a row (`update[b1]`) is read by that row only, so a summary
   without `update[b2]` misses that gate even though `update[b1]` folds to the same region.
4. The triad goldens are required from (0, 3), not in the unconditional `REQUIRED_REPLAYS`: for
   every backend of the set's coverage table (`goldens.toml`, D6) passed replays with the
   algorithm counter (an oracle replay is not evidence of the algorithm), by key and by content,
   covering every case the table gives the backend (all four patterns). `compare.py triad_count
   --certificate <results>` writes them (config `<backend>[/t<N>]:triad/<mode>`); a planted
   defect or a `--no-coverage` run never makes one.
5. `write --dry-run <algorithm> --results <outside benchmarks/results/>`: one algorithm's suites
   and golden sets judged as in a release, every other required part listed as outside the dry
   run; never a release's certificate. M9a's dry 0.3 run is of `triad_count`.
6. The golden-mutation record needs the triad mutations because `parity/mutate.py list` holds
   them (unchanged rule).

## Outcome (M9a step 5)

Every gate is within its bar (`parity/results/M9a.md` section 5): the timing gates 45 / 45 rows,
ratios 0.858-1.028 (update per batch 0.858-1.028, full count 0.999-1.002, end to end
0.941-0.987; coauth-DBLP 11 rounds, syn1M / syn2M 21 rounds, all at the base lock), no reading
flagged by the audit; the memory gates 11 / 11 rows (store accounting 1.0000 at every gated
point, the Nsight Systems peak 1.0010 / 1.0015 / 1.0011); the footprint between batches 1.63-1.69
of A's held bytes (the pool's reserve), flagged and reported. The final replays are exact on the
coverage table (cuda 300, sequential 120, openmp 142 at 1 and 8 threads, identical), and the dry
0.3 certificate of `triad_count` passes `recheck_gates`; its first run found that the
`device_memory` rows did not recompute (a ratio from bytes next to MiB rounded to 0.1), fixed in
the summary and the certificate before the records were summarized again.

## Consequences

- A 0.3 certificate fails without both ESCHER suites, without a store-accounting point, without
  a per-batch update row, or without the triad replays of a backend or a mode.
- The memory readings need GPU 1 free of other processes; the release's step 2
  (`docs/developer/release.md`) says so.
- The records of M9a's readings are committed compacted under `parity/results/M9a-gates-*.json`;
  the dry certificate under `parity/results/M9a-certificate-dryrun.json`.

## Alternatives considered

- **One suite for both papers.** Rejected: PLAN 8.5 names two suites, and the decision D8 keeps
  them; their split is recorded above as ours.
- **Gating the store accounting at every point on the synthetic inputs too.** The author's
  decision of 2026-10-07 reads coauth-DBLP after the build and after each batch; on syn1M and
  syn2M the acceptance names A's two print points. The middle points are recorded (they are
  equal on every reading so far).
- **The triad bytes during the apply from the allocation trace alone** (the pool's allocations
  outside the hypergraph's ranges): the stream-ordered pool hides which component an allocation
  serves; two processes that differ only in counting give the number without attributing
  allocations.
