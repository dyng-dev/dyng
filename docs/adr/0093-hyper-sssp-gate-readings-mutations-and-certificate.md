# ADR 0093: hyper_sssp's gate readings, its mutation suite, the 0.3 no-regression readings and its part of the 0.4 certificate

- **Status:** Accepted. The protocol and the bars are the author's: HD3 (the gated rows), HD4 (the
  protocol of the readings), HD6 (memory), HD7 (the fallback decision in a gate round), HD16 (the
  dynG-against-dynG bar and the D14 path) and HD18, **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log, "The measurement protocol and gates of 0.4 (sheet B1-B6)", "The
  parity rules of 0.4 (sheet C1-C8)" and "Licensing, data provenance, budgets and scope of 0.4"),
  on PLAN 8.4, 8.6 and the decisions of 2026-10-07, -08, -15 (D1) and -23 (the D14 precedent).
  The technical parts (the harness commands, the suite file, the record checks, the certificate
  plumbing, the mutation suite's form) are accepted under delegation (GOVERNANCE.md, "Delegation
  of technical ADRs"). Part of M11a step 5 (criteria 16-22).
- **Date:** 2026-10-30
- **Deciders:** S M Shovan (HD3, HD4, HD6, HD7, HD16, HD18, 2026-10-28); the AI assistant, on the
  author's behalf, for the technical parts (M11a, step mutations-noregression-gates-certificate)
- **Builds on:** ADR 0018 / 0021 (clocks), ADR 0024 (dynG against dynG), ADR 0048 (per-batch
  rows), ADR 0056 (the audit, the memory rows, the certificate), ADR 0061 section 4 (GoogleTest
  rows of a golden-mutation record), ADR 0062 (`GPU_LOCAL_STEPS`), ADR 0080 (the map and the
  protocol), ADRs 0081-0092 (the port).

## Context

M11a steps 1-4 built the reference side, the `slack_csr` store and the as-is `hyper_sssp` against
MOSP_ESCHER@4b86159 (B), each exact by id on B's golden set. Step 5 finishes correctness (the
mutations of PLAN 8.4, the compute-sanitizer steps), freezes M11a's code, shows that the 0.3 paths
did not move (the R030 placement lesson: code placement can move a region without a source change)
and takes the binding readings: every gated row of HD3 on the 16 paper configurations, the memory
gates of HD6, and the certificate plumbing that makes a 0.4 release depend on them.

## Decision

### 1. The mutation suite `hyper_sssp` (PLAN 8.4; criterion 16)

`parity/mutate.py` gains the suite `hyper_sssp`: hyper_sssp's GoogleTest suites (the hand cases,
the result replays of the committed subset by id, its kit) and the container's under `slack_csr`
(the store, line-graph and overflow tests, the id replays). A suite may list its executables per
configuration (`config_targets`): a mutation of configuration `cuda` runs the CUDA executables
(the cuda backend only), `sequential` the host ones, so each of B's sites is detected in its CUDA
code and in its host twin on its own. Twelve mutations: #1 and #2 (the CUDA engine and the
sequential twin), #3 (the CUDA store and the host store), #4 (the allocator's H2V unfill, the
host model the CUDA store shares), #6 (`hypergraph_h2v_not_deduplicated` against this suite), #7
and #8 (CUDA and host); #5 (`cbst_graph_own_ids`) and #4's CBST counterpart
(`cbst_unfill_drops_last`) are re-runs of M8a's mutations in the cbst suite. Mutations that
change the code the same way (the twins) share one copy and one build. Every GPU job of a run
first asks the shared calendar (`locklog.py quiet-ok`, with `DYNG_LOCKLOG_MILESTONE`), as does
every heavy command of `ci/gpu_local.sh`. The CTests of the hypergraph mutants of M11a's sites
carry the label `hyper_sssp`.

### 2. The 0.3 no-regression readings (criterion 20; HD16)

- **SASS identity** (`parity/sass_identity.py`): the escher store's objects, `escher_store.cu`,
  the sorted-merge line graph and triad_count's kernels of the final library against the merge
  base, kernel by kernel (triad_sass.py's normalized instruction stream and resources). The hash
  an anonymous namespace's mangled name carries depends on the source tree's path and is removed.
- **Store bytes** (`parity/experiments/escher_store_bytes.py`): the escher store's
  `store_stats().device_bytes` after the build and after every batch (the `[memory]` lines the
  0.3 store gates read) and the batch lines of both builds, byte for byte.
- **ADR 0024 readings** (`parity/experiments/hypergraph_host_ab.py`, 17 layouts x 51 rounds, read
  with `parity/ab_modes.py --bar 1.02`): `hypergraph.build`, `.apply`, `.apply.insert` and
  `.apply.delete` on escher storage on sequential and openmp-8 (coauth-DBLP's ESCHER variant and
  syn1m); `hypergraph.apply` and `triad_count.update[b1..b3]` on cuda (coauth and syn1m, base
  clocks). The tool gains a cuda backend (the GPU, the clock lock for the whole reading, EAGER,
  `--sync-stages` on both sides) and reads `hypergraph.apply.delete`. A region above the bar is
  located and listed for the author (the D14 path), never a silent pass.

### 3. The gate readings (criteria 18 and 19; HD3, HD4, HD6, HD7)

- **The suite** `benchmarks/paper/ia3_hsosp.yaml` (validated by `parity/bench_suite.py`):
  coauth-DBLP and coauth-MAG-Geology with the golden set's digests; the batches replayed from the
  golden set (HD5); B unpatched; PLAN 8.6's tolerances; a recorded 3-round boost probe per
  dataset; the gated runs of the 16 paper configurations at their probe's clocks, or at base
  where the reading cannot hold boost (21 alternating rounds). **The clock rule, one for both
  datasets** (M11a review): a dataset whose probe holds is read at boost unless its 21-round
  reading loses the boost lock in a large share of its attempts (coauth-DBLP: 36 of 56 rounds of
  dblp_hyperedge_25000; Geology: 23 of 44 attempts of geology_hyperedge_200000), and is then read
  at base for all its configurations; the first reading kept Geology at boost, which read its
  gated rows from the boost rounds that survived the power cap, so both datasets are now read at
  base; the default-clock reading of P50 (11 rounds, recorded); the gated memory reading of
  HD6's three configurations and the recorded 200K ones; the ncu kernel table (recorded);
  `paper_reference` = B's README tables (07bbceb, CUDA 12.9, default clocks), which no gate reads.
- **One window per configuration group**: `bench_suite.py plan` gives one timed command per
  configuration (its own record), so a reading never needs a window of more than 2 h; the
  readings ran in exclusive-lock windows of the shared calendar per dataset
  (`tools/locklog.py run --reset-gpu 0 EXCL ...`), the memory reading in a GPU1-QUIET window.
- **`hsosp_perf.py memory`**: both whole processes plain (the guards; the graph-scope accounting)
  and under `nsys profile -t cuda,nvtx --cuda-memory-usage=true` (the port with `--nvtx`). The
  gate `device_memory` is the peak of live device allocations, port / B <= 1.05. The gate
  `graph_memory[build|b1..b3]` compares B's graph_mb scope with the port's slack_csr store, mirror
  and two results in bytes: B prints the scope only after each batch and in MB to 6 digits, so B's
  bytes are derived from the golden's `DeviceH2H::deviceBytes()` (recorded at the exits of
  `buildDeviceH2H` and `applyBatch`) plus its two states (`HsospState::deviceBytes()` of
  `caps.maxHyperedges` nodes), and the derivation must print exactly as B's own graph_mb after
  every batch of the same run. nvidia-smi is sampled before, every 5 s during and after the
  reading; a foreign process anywhere makes it not count (exit 3). The port's bench prints
  `[memory] load|batch <k> graph_scope_bytes <n>` (stderr, untimed) and takes `--nvtx`.
- **`hsosp_perf.py kernels`**: both processes under `ncu --clock-control base --cache-control
  none`, every launch of the engine's kernels (the pairs of `SASS_KERNELS`), the load's launches
  skipped by position: everything before the first update's `pack_state` (B's first kernel of
  `hsospUpdate`, hsospDevice.cu:1094); per kernel and per batch.
- **`hsosp_perf.py sass [--store]`** (no GPU): the transplanted kernels against B's unpatched
  objects, paired by name: the engine's (engine.cu.o against hsospDevice.cu.o) and, with
  `--store` (added by the M11a review, which found the store missing from criterion 18's record),
  the slack_csr store's fourteen (slack_csr_store.cu.o, slack_csr_h2h.cu.o and
  incidence_delta.cu.o against hsospDevice.cu.o and hsospDelta.cu.o); each record lists the
  kernels no label pairs on either side.
- **The round guards fail closed** (M11a review): a full reading's port line must carry the
  result group (digests, fallback, mismatch_static) of every batch; a GPU process that has gone
  is the memory reading's own only if it was seen alive as its descendant; the audit's quiet
  maximum is read from step 1's pre-read record and recorded with its source.
- **Record checks** (`bench_suite.hsosp_read_record`, rechecked by `certify.py`): the
  configurations, the protocol fields (the GPU, the exclusive window, EAGER, one layout,
  `--sync-stages`, the unpatched original, the clocks), the probe's clock decision for a gated
  reading, the medians recomputed from the samples, the round guards, the audit, HD7's scope
  (a configuration whose decisions differ in more than a third of its rounds is a problem listed
  for the author), every gated row of every batch, and both memory gates in bytes.

### 4. The certificate (criterion 21)

- `REQUIRED_SUITES['ia3_hsosp'] = {hyper_sssp, (0, 4)}` at M11's anchor.
- The set `hyper_sssp_mosp_escher` also needs the slack_csr store check on cuda (config
  `cuda:hyper_sssp/store`, every case: `compare.py hyper_sssp --part delta --backend cuda
  --certificate`); the result replays write `<backend>:hyper_sssp/<mode>`.
- `HYPERGRAPH_ID_REPLAYS` takes a per-entry and a per-feature `since`: MOSP_ESCHER's sessions
  replayed whole on `slack_csr` storage (feature `whole_sessions`, config
  `<backend>:hypergraph_ids/slack_csr`: `hypergraph_ids.py --storage slack_csr --certificate`) of
  `cbst_trace_mosp_escher_{test, dblp50k_hyperedge}` and the set
  `cbst_trace_mosp_escher_dblp50k_vertex` from (0, 4); the 0.3 requirements are unchanged.
- `EQUIVALENCE_TARGETS` and `EQUIVALENCE_FILES` gain `dyng-compat-hsosp` and `dyng-compat-escher`.
  **This widens the equivalence check of every version's certificate**: a metadata-only
  difference between a measured commit and a release now needs these two drivers byte-identical
  too, whatever the version the certificate is written for (`equivalence` builds and compares the
  lists as they are in the tree that writes it).
- `recheck_gates` reads `graph_memory` rows in bytes from the record's `graph_accounting` and
  rechecks hyper_sssp's readings against their records (every gated row of every batch and both
  memory gates). No (0, 4) `REQUIRED_CHECKS` entry.

## Outcome (M11a step 5)

The records are `parity/results/M11a.md` section 11 and the JSON records it names; M11a's code
freeze is `c78e4ef` (no library, driver or build change after it).

- **Mutations:** the control passes and 14 of 14 mutations are detected, each by failed tests of
  its own configuration's executables (`M11a-mutate-hyper_sssp.json`).
- **The 0.3 paths:** the 14 CUDA objects of the escher store, `escher_store.cu`, sorted-merge and
  triad_count are SASS-identical to the merge base; the escher store's device bytes and batch
  lines are identical on three cases. ADR 0024's readings put some regions of the escher-storage
  `hypergraph.apply` above the 1.02 bar: `.apply.delete` on every backend of coauth-DBLP (up to
  1.033 on openmp-8 and 1.035 on cuda), and on cuda `.apply` (1.021-1.029), `.apply.delete` and
  `.apply.insert` (1.025-1.033) of syn1M, confirmed by the 85 x 170 reading. **They were located as
  code placement**: the stages' host functions on cuda have the same instructions at other
  alignments, the device code is identical, and both sides rebuilt with R020's flag set FB read
  every region within the bar (apply 0.991-1.012, delete 1.001-1.009, insert 0.737-1.001). The one
  changed function in them, the host allocator's `erase` (the dead-key bit set of ADR 0089),
  costs 0.7-0.9 % of the host delete stage, within the bar. triad_count's PLAN 8.6 gates against
  ESCHER-GPU@abcf9b2 hold on the final library (update 0.857-1.023, full_count 1.002-1.003,
  end_to_end 0.971-0.996). As HD16 and the D14 precedent direct, the regions are listed for the
  author's acknowledgement (GOVERNANCE.md, Open decisions, M11 (HD)), not fixed in code.
- **Gates against MOSP_ESCHER:** **271 of 271 gated rows within their bars** (Geology at boost,
  coauth-DBLP at base after its 21-round reading could not hold boost): maintain 0.402-0.795,
  sosp_update 0.991-1.066 (1.066 under the 1.10 bar where B's median is 1.23 ms), static
  0.980-1.037, dynamic_comparable 0.637-0.855, dynamic 0.146-0.444, load_plus_dynamic
  0.672-0.846; no fallback decision differed from the golden's in any kept round (HD7: no
  "scope differs"); no audit flag. Memory: the nsys peak 0.477-0.557 of B's, the graph-scope
  accounting 1.0000 after the build and every batch.
- **Certificate:** the dry 0.4 certify restricted to hyper_sssp passes and fails with one gate row
  or one replay removed; M10's scoped dry run waits for M10's entries on the branch.

## Consequences

- A 0.4 certificate cannot pass without hyper_sssp's 16 x 16 timed rows, its memory rows, its
  replays in every parity mode and the store check.
- The suite is read per configuration; a retake of one configuration replaces one record.
- M11b reads the shared sssp engine against the same suite (HD13: engine::automatic picks it only
  if byte-identical and within every gate).

## Alternatives considered

- **B's graph_mb as printed** (MB to 6 digits, no point after the build): rejected for the bytes
  derived from the golden, cross-checked against every print.
- **One record per dataset**: a 2-h window could not hold a dataset's eight configurations with
  repeats and the M10 track's windows between them; one record per configuration keeps each
  window short.
- **An ncu `--launch-skip` count per process**: the load's launch count is schedule-dependent (B's
  own varies between runs); splitting at the first update's kernel is exact per run.
