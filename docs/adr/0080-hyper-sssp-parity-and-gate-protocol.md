# ADR 0080: The hyper_sssp parity and gate protocol (the timed regions, the parity modes, the gate readings, the original's cross-checks)

- **Status:** Accepted. The protocol is the author's: the decisions HD2-HD7 and HD16 of the 0.4.0
  decision sheet, **decided by the author on 2026-10-28** (the author accepted every default of
  the reconciled sheet; GOVERNANCE.md, approvals log, the rows of 2026-10-28 "The measurement
  protocol and gates of 0.4 (sheet B1-B6)" and "The parity rules of 0.4 (sheet C1-C8)"), on top
  of PLAN 8.6 (the tolerances, the A/B protocol), the decisions of 2026-10-07 (store accounting;
  O16 `erase_first`, H-SOSP parity runs on `insert_first`), 2026-10-08 (the one-layout protocol
  of ADRs 0018 / 0021 against the originals, built exactly as their Makefiles build them) and
  2026-10-15 (D1, the gated Nsight Systems peak). The technical parts (the file layout of the
  map, the loader functions, the stage names, the scope wording) are accepted under delegation
  (GOVERNANCE.md, "Delegation of technical ADRs"). It carries out M11a step 1 (PLAN 6.3 steps 1-3,
  6.4.8, 8.3, 8.6; the M11a acceptance criteria 1 and 4) and fixes the reference side that M11a
  steps 3-5 measure against.
- **Date:** 2026-10-28
- **Deciders:** S M Shovan (the protocol, HD2-HD7, HD16, 2026-10-28); the AI assistant, on the
  author's behalf, for the technical parts (M11a, step regions-exporter-goldens-oracle)
- **Builds on:** ADR 0013 (parity and goldens), ADRs 0018 and 0021 (clocks), ADR 0024 (dynG
  against dynG), ADR 0036 (the recording twins), ADRs 0041-0045 (the hypergraph container),
  ADR 0048 (triad_count's protocol, whose form this follows), ADR 0056 (the contamination audit).

## Context

`hyper_sssp` is the port of MOSP_ESCHER@4b86159's (B's) H-SOSP pipeline: single-source shortest
paths over the line graph of a dynamic hypergraph (stepping into hyperedge h costs its weight),
maintained batch by batch. B's driver, `bin/hsospBench --hg`, reads a hypergraph, adds a virtual
source and target hyperedge, generates three consecutive paper batches (hyperedge or vertex
kind) in process, and times each batch with host `steady_clock` timers separated by device
synchronizations: `t_escher_ms` (its three CBSTs), `t_delta_ms` (the host incidence update and
the GPU line-graph delta), `t_csr_apply_ms` (the device CSR apply), `t_sosp_update_ms` (the
budgeted update), `t_static_ms` (a recompute into a second state) and their sum
`t_dynamic_total_ms`; one `[e2e]` line reports the load, the summed dynamic and static times and
the wall time. PLAN 6.3 asks, before any port code, for the timed-region map (step 3), the goldens
from the pinned original (step 1) and the original checked against its own oracle (step 2).

The 0.3 paths gave B's CBSTs a port (the ESCHER store, M8). The H-SOSP store is a different design
(`DeviceIncidence` and `DeviceH2H`: slack rows, a host incidence model, an order-free delta), and
50-71 % of B's dynamic total is CBST maintenance that nothing on the SSSP path reads and that the
`slack_csr` store does not have (HD3's rationale). Gating only the total would let that
improvement mask a regression (PLAN 8.6).

## Decision

### 1. The timed-region map (`parity/timed_regions/hyper_sssp.toml`)

1. **Gated per batch** (`per_batch = true`, rows `<name>[b1]` .. `<name>[b3]`, `report_sum =
   true`; HD3):
   - `maintain`: `csv:t_delta_ms + csv:t_csr_apply_ms` against the whole `hypergraph.apply` under
     `slack_csr` (Step 0, ids, incidence, translate, line-graph apply, any rebuild). dynG's scope
     also holds the id assignment that B times in `t_escher_ms` (its h2v insert): the row is
     conservative for dynG.
   - `sosp_update`: `csv:t_sosp_update_ms` against `hyper_sssp`'s stages after the commit
     (`identify_affected`, `seed`, `loop`, `recompute`, `finalize`), the budget fallback
     included.
   - `static`: `csv:t_static_ms` against a recompute into a second, pre-allocated result on the
     maintained line graph (`hyper_sssp.compute`, the driver's `static_ms`): no rebuild, copy or
     allocation inside the region.
   - `dynamic_comparable`: `csv:t_delta_ms + csv:t_csr_apply_ms + csv:t_sosp_update_ms` against
     the whole `hyper_sssp::update` call (`hyper_sssp.update`, the driver's `update_ms`).
   - `dynamic`: `csv:t_dynamic_total_ms` (PLAN 6.4.8's figure) against the whole update; it is
     **flagged** in the improvement table, because B's total includes CBST maintenance that
     nothing on the SSSP path reads.
2. **Gated end to end:** `load_plus_dynamic` = `e2e:load + e2e:dynamic` against the driver's load
   from the start of main (`resources::warm_up`, the read, the build including the line graph, the
   upload, the initial compute) plus the summed updates. Gate `end_to_end` (1.10).
3. **The bars:** a `compute` row is at most 1.05x where B's median is at least 10 ms, otherwise at
   most 1.10x with at least 21 rounds (PLAN 8.6; `perf_ab.gate_limit`). `end_to_end` at 1.10x.
4. **Reported** (gate `none`): `escher` (`t_escher_ms` against the id-allocation sub-stage
   `hypergraph.apply.ids`), `delta`, `csr_apply` (HD3: 1.5-28 ms on host timers and scopes that
   are not one to one), the sub-stages of the update, `load`, `static_sum`, `wall`, the process
   wall time, and an improvement table kept apart from the gate table (PLAN 8.6).
5. **The stage names** the port must emit (a later rename amends the map in the same commit):
   `hypergraph.apply.{normalize, ids, incidence, translate, line_graph}` inside
   `hypergraph.apply`, and `hyper_sssp.{update, identify_affected, seed, loop, recompute,
   finalize, compute}` (with `hyper_sssp.normalize` and `hyper_sssp.commit` from the framework's
   hook names). `hyper_sssp.recompute` is emitted on every update (0 ms without a fallback), so
   that `sosp_update` is defined on every batch. The line-graph delta (`translate`) runs inside
   the commit, in `hypergraph.apply.translate`; the sketch's separate `hyper_sssp.translate`
   stage is not used (the deviation is the line-graph ADR's, M11a step 2).
6. **Scope rules**, the same on both sides:
   - dynG reads its exported batch before its timer starts (HD5: B's exported batches are
     replayed; it is a host batch, so Step 0 is inside the region); B generates its batch
     in-process, outside its timers;
   - a device synchronization before each timer starts (B ends every stage with
     `cudaDeviceSynchronize`; the port runs with `--sync-stages` and synchronizes before each
     driver timer);
   - `--verify none` on B and no oracle on dynG in a timed round; the update == static check is
     untimed on both sides (B's `compareDistances` runs after its static timer);
   - the port runs with `--sync-stages`;
   - capacities are planned as B plans them (`hsospBench.cu` lines 211-218 and 496-511): planned
     = batch x batches for hyperedge batches (0 for vertex batches) + 65536; `maxHyperedges` =
     rows (the virtual rows included) + planned + 1024; `headroomFactor` 1.3;
     `extraPayloadInts` = planned x (maxcard + 272) + 8 Mi; the line-graph and mirror headroom
     1.6;
   - the one asymmetry (conservative for dynG) is the id assignment inside dynG's maintain.
7. **The loaders: M11's half of the joint key specification** (reconciled with M10 on
   2026-10-27; M10 adds `first_batch` and `sum_gate` in functions of its own, and the second
   merger keeps both `REGION_KEYS` additions). `parity/perf_ab.py`, each in a function of its own:
   - **a list-valued `original`, the summed csv keys** (`check_summed_original`): `original` and
     `original_report` are lists; every `original_report` entry is a timer the script can read;
     an entry is never repeated (a repeat would count a timer twice); the values of a row are the
     sum of its entries (`hsosp_original_value`);
   - `t_dynamic_total_ms` joins `HSOSP_CSV_TIMES`;
   - **the `e2e:` keys** `e2e:load`, `e2e:dynamic`, `e2e:static` and `e2e:wall` (`hsosp_e2e_keys`,
     `check_e2e_keys`): the fields of B's `[e2e]` line, read once per process, so they are refused
     in a per-batch region; `parse_hsosp_e2e` and `parse_hsosp_csv` read B's outputs;
   - **the gated rows** (`check_hsosp_gates`): `load_hsosp_regions()` refuses a map whose gated
     regions are not exactly HD3's (`maintain`, `sosp_update`, `static`, `dynamic_comparable`,
     `dynamic` per batch, `load_plus_dynamic` end to end): an unknown or a missing gated row name
     is refused, as an unknown timer is.
   The earlier maps (`sssp.toml`, `mosp.toml`, `hypergraph.toml`, `triad_count.toml`) load and
   read as before (tests in `parity/tests/test_hsosp_regions.py`).

### 2. The parity modes (HD2; PLAN 6.4.8, 8.3)

- **Id map:** dynG id = B id - 1. B's virtual source row {s} (B id 1) and target row {t} (B id
  m + 2) are loaded by the harness as ordinary hyperedges at B's positions with container weight
  1 (the M8b precedent); B's ops naming them (B skips them via `isVirtual`) are filtered by the
  harness and counted as `filtered`.
- **by_id:** `slack_csr` storage and line graph, `id_reuse::insert_first`,
  `capacity_policy::mosp_escher(extra of the case)`, B's CSR / mirror headroom and node capacity
  (internal harness knobs), `hyper_sssp`'s INTERNAL virtual-endpoint mode (B's source row seeded
  at distance 0; the source and target rows weigh 0 inside `hyper_sssp` only; no public name),
  and B's `UpdateConfig` per call. Distances and parents equal B's by id.
- **by_id_public:** the same container with the public source-vertex API (source = B's s).
  Distances equal B's on every real hyperedge; a parent may differ only where B's parent is its
  source row (dynG: -1) or its virtual target row, each such case counted and checked against
  dynG's oracle.
- **by_key:** a container loaded as a user loads it (`read_hypergraph(..., hsosp())`, no virtual
  rows, the default `erase_first`, `slack_csr`), the public API, B's batches resolved by content:
  distances equal by (sorted vertex set, weight); parents form dynG's canonical tree.
- **Canonical encodings** (golden and port alike): distances int64 little-endian by id over
  0..max(bound_B, bound_dynG) - 1, dead or unreachable = INT64_MAX / 4 (B's
  `HostHypergraph::INF`); parents int32, 0-based by id, -1 = none; pairs sorted (min, max) id
  pairs, the deleted list then the inserted list. A digest is the SHA-256 of an encoding together
  with its count (the bound), so a port with a larger id bound hashes B's bound and shows its
  extra ids dead.
- **Coverage** (HD2): cuda on T and all 16 P configurations; sequential on T plus
  `dblp_hyperedge_50000`, `dblp_vertex_50000` and `geology_hyperedge_50000`; openmp at the
  container level only (HD14: `hyper_sssp` has no OpenMP backend in 0.4). The table per case is
  in `parity/goldens.toml` (ADR 0081).
- **The fallback decision** of a batch is not a parity target (HD7; PLAN 8.3 does not list it):
  it is tabulated per batch against B's golden, and on the gate rounds it is a per-round guard
  (section 3).

### 3. The gate readings (HD3, HD4, HD6, HD7)

- **Baseline:** the UNPATCHED copy of B at 4b86159 (`parity/build_reference.sh`, `make -j{jobs}
  all NVCC=/usr/local/cuda-13.1/bin/nvcc CUDA_ARCH=sm_86`, B's `-O3 -lineinfo`), re-measured in
  the same session as the port; the README's 07bbceb / CUDA 12.9 tables are `paper_reference`
  only.
- **Cases:** all 16 configurations `<dblp|geology>_<hyperedge|vertex>_<25000|50000|100000|200000>`
  at B's defaults (seed 20260725, `--maxcard 25`, `--del 50`, `--batches 3`, maxIterations 4096,
  workBudget 1.0), `--verify none` passed explicitly (B's default is `--verify first`).
- **Rounds:** 21 alternating rounds per configuration (A/B/A/B, medians), GPU 0 under the
  exclusive perf lock, one layout (fixed-length names; ADRs 0018 / 0021), EAGER on both sides,
  `resources::warm_up`, no `OMP_*` (B runs its host work on 56 unpinned threads).
- **Clocks:** a recorded 3-round boost probe per dataset; base where the probe does not hold
  boost (ADR 0021; ADR 0048 section 1.4's recorded probe). `parity/hsosp_perf.py probe` records
  it (this step's quiet pre-read).
- **Short rows:** host `steady_clock` timers on both sides (B has no CUDA events), with an `ncu`
  base kernel table of both sides recorded beside them.
- **Contamination:** the monitor and the ADR 0056 audit with the thresholds 16 (1-minute load
  average) and 4 (`procs_blocked`), unless B's own quiet maximum exceeds 8; then 2x that maximum,
  or an audit of `foreign_runnable` (HD4). The quiet pre-read records B's maxima and the choice
  (`parity/results/M11a.md`).
- **Per-round guards:** B's `h2h_pairs`, `seeds` and `overflow_rebuilds` equal the golden's and
  `mismatch_static` is 0; the port's per-batch distance, parent and pair digests equal the
  golden's; both sides' fallback decisions equal the golden's. A round that differs is repeated
  (at most doubling the rounds); a configuration whose decisions differ in more than a third of
  its rounds is recorded as "scope differs" and listed for the author (HD7).
- **Recorded, not gated:** a default-clock reading (P50, 11 rounds), the `ncu` table, the SASS
  instruction counts of the transplanted kernels against B's.
- **Memory** (HD6; D1): gated, the Nsight Systems peak of live device allocations of both whole
  processes (load, build, initial SSSP, three batches with their static baselines, `--verify
  none`), port / B at most 1.05, on `dblp_hyperedge_50000`, `dblp_vertex_50000` and
  `geology_hyperedge_50000`, read on GPU 1 under the shared lock in a GPU-1 quiet window of the
  shared calendar (`tools/locklog.py run GPU1-QUIET`); and the graph-scope accounting (B's
  `graph_mb` = DeviceH2H + mirror + two states, against dynG's `slack_csr` store + mirror + the
  two results, from internal counters) at most 1.05 after the build and after each batch. The
  driver mirrors B's capacities and keeps a second result, as B keeps state B. Recorded: the
  200K configurations, the footprint, B's `escher_mb` (improvement table) and `dev_mem_mb`.
- A gated row above its bar after one-class-per-commit fixes is STOP S5, never a waiver; no gate,
  tolerance or protocol is relaxed and no row is waived.

### 4. B against its own oracle (PLAN 6.3 step 2; M11a criterion 4)

- A fresh `make test` of the unpatched copy, every case PASS.
- Every P configuration with `--verify all` (B's oracle: Dijkstra on the line graph rebuilt from
  the incidence lists, and the canonical-parent check, on every batch): `mismatch_static` 0 and
  the oracle OK on every batch; `h2h_pairs`, `seeds` and `overflow_rebuilds` equal the export's
  (`--verify none`) exactly; the fallback decision per batch compared under the repeat rule of
  section 3.
- STOP S1 if `h2h_pairs`, `seeds` or `overflow_rebuilds` disagree, if the fallback still differs
  after the repeats, or if `mismatch_static` or the oracle fails: a finding against the original
  is reported, never worked around.

## Consequences

- The port (M11a steps 2-5) is measured against a fixed map: it is committed before any file
  under `cpp/src/hypergraph/slack_csr/`, `cpp/src/hypergraph/line_graph/incidence_delta*`,
  `cpp/src/algorithms/hyper_sssp/` or `cpp/include/dyng/hyper_sssp.hpp` exists (the git log
  shows the order).
- `hypergraph.toml`'s `[reference.mosp_escher]` stub (M8a) keeps its reported `escher` region and
  points at this map.
- The improvement table is where the port's lack of CBST maintenance shows; the gate table has
  rows (`maintain`, `dynamic_comparable`) where it cannot hide a regression.

## Alternatives considered

- **Gate `delta` and `csr_apply` separately** (the sheet's alternative to HD3). Not chosen by the
  author: the scopes are not one to one (B's Step 0 checks sit in its host incidence update; a CSR
  rebuild lands in `csr_apply` on one side and inside the line-graph apply on the other).
- **Gate only the dynamic total.** Not chosen: B's CBST maintenance would mask any regression of
  the SSSP path.
- **CUDA events for the short rows.** B has none; adding them would patch the timed copy, which
  PLAN 8.3 forbids. Host timers on both sides with an `ncu` table beside them instead.
