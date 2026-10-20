# ADR 0051: The generators' ESCHER host model, the port driver `dyng-compat-escher run` and the triad A/B harness

- **Status:** Accepted under delegation (2026-10-15; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It carries out the driver and harness parts of M9a step 2 (PLAN
  6.4.6, 8.6; the driver and harness parts of the M9a acceptance criteria 12 and 13). No approved
  rule changes: the gates, their tolerances and protocol (PLAN 8.6; ADRs 0018, 0021; the author's
  decisions of 2026-10-07, 2026-10-08 and D1 of 2026-10-15) are used as they are.
- **Date:** 2026-10-15
- **Deciders:** the AI assistant, on the author's behalf (M9a, step
  oracle-harness-driver-prereads)
- **Builds on:** ADR 0024 (one heap layout), ADR 0045 (the legacy generators), ADR 0048 (the region
  map and its scope rules).

## Context

The gates compare A's whole timed process with the port's (D1: the Nsight Systems peak of live
device allocations of the whole process, port / A <= 1.05). M8b measured the container alone:
both peaks lay in the H2H merge (A 3,018.9 MiB, dynG 3,019.1 MiB on coauth_b50k_s1), leaving
about 150 MiB for everything `triad_count` adds. M8b's generator
`escher_batch_generator::next(res, hypergraph)` reads the live state through `hypergraph::h2v()`,
which builds a **public decoded device view** (about 50 MB on coauth-DBLP) that is cached per
version and stays live through the next `apply()`'s merge: a driver drawing its batches that way
would add that view to the gated peak, and A has no such view (its generator reads its host rows,
`DynamicHypergraph::h2v_`, which its `update` keeps in step).

## Decision

### 1. `generators::legacy::escher_host_model` (tracked API, `dyng/generators/legacy.hpp`)

- `escher_host_model<vertex_t>`: the live rows by id on the host, built from the list the
  hypergraph was built from (`from_hyperedges`: hyperedge i = id i) and kept in step by
  `apply(batch, hyper_apply_summary)`: the deleted ids' rows are cleared, then each inserted
  hyperedge, its members sorted and deduplicated as the container stores them, is set at its id
  from `inserted_ids` (A's `DynamicHypergraph::update` host model; the rows never shrink, as A's
  do not). It validates its input (host memory, offsets, members in range, non-empty insertions,
  one non-negative id per insertion) before it changes anything (strong guarantee).
- `escher_batch_generator::next<weight_t = unweighted>(const escher_host_model<vertex_t>&)`: the
  same draws as `next(res, hypergraph)` from the same engine state. Both overloads share one
  implementation of A's `BatchGenerator::next` (the live ids in increasing order, the partial
  Fisher-Yates of the deletions, sampled or random insertions; random insertions draw vertices up
  to max(graph.max_vertex, num_vertices - 1)); the existing `next(res, hypergraph)` is unchanged
  in what it draws (the M8b tests, `dyng_hypergraph_ids_tests --legacy-generator`, still pass).
- Tests: in step with a hypergraph over 6 batches, under both `id_reuse` orders, both insert modes
  and three batch sizes (ids reused, the bound going down, everything deleted), `next(model)` draws
  exactly `next(res, h)`'s batches, the engines end in the same state, and the model's rows equal
  `h2v()` by id; the validation and the strong guarantee.
- Proof against A: `parity/triad_replay.py draws` runs the driver below with `--no-count
  --dump-batches` on make-test (15 groups), syn1m_b50k_s1-s3, syn2m_b50k_s1 and coauth_b50k_s1-s3:
  every batch file (split one record per file as the exporter does) has the SHA-256 of the
  golden's, and the batch and Hypergraph lines are A's (`parity/results/M9a-draws-escher_gpu.json`).

### 2. The port driver `dyng-compat-escher run` (`tools/compat/escher/escher_run.cpp`)

- A's argv verbatim (`N maxCard minCard V [capacity] [alignment]` or `--input F [capacity]
  [alignment]`, `--batches`, `--batch-size`, `--deletions`, `--insertions`, `--seed`,
  `--insert-mode`, `--check`, `--timing FILE` in A's JSON format), plus `--pattern
  {h_motif,type1,type2,type3}`, `--backend`, `--threads`, `--sync-stages`, `--nvtx`, `--no-count`
  and `--dump-batches FILE` (never in a timed run). An alignment other than A's default 4 and
  `--reference` are refused (the oracle replays are `compare.py triad_count`).
- A's order (`src/driver.cu` `run()`): the clock at the top of `main()`; `resources::warm_up()`;
  the load (`io::read_hypergraph(F, escher_gpu())` or `generators::legacy::escher_hypergraph`
  from one `std::mt19937_64(seed)`; `load_ms`); A's `Hypergraph:` line; the hypergraph under
  {`erase_first`, `escher_gpu(capacity)`, `sorted_merge`} (the line graph and row views built at
  construction) and the host model (the list is released after it, as A moves its rows into its
  model); the full count (step 3); per batch the draw from the model (`generate_ms`, the stage
  `driver.batch.generate`), then the update (`update_ms`, step 3) or with `--no-count`
  `hypergraph::apply()` alone (`apply_ms`), each timer between two device synchronizations; the
  model follows the summary; A's `Batch b (...)` line; `--check` runs `check_integrity()` outside
  the timers; `total_ms` after the last batch. The `--timing` file holds the driver timers (the
  `port_report` names of `parity/timed_regions/triad_count.toml`) and every profiler stage per
  batch; stderr has `[memory]` lines with `store_stats().device_bytes` after the build and every
  batch.
- **No public view in the gated process**: the driver never calls `h2v()`, `v2h()` or `h2h()`.
- In step 2 only `--no-count` runs; without it the driver exits 2 ("counting arrives with
  triad_count").

### 3. The A/B harness `parity/perf_ab.py triad_count run|memory` (`parity/triad_perf.py`)

- `run`: A's unpatched `build/main` (or `build/type<N>`) against the driver (`--backend cuda
  --sync-stages`), the exclusive perf lock, GPU 0, `perf_ab.ClockLock` (`--lock-clocks
  base|boost|none`; `ncu --clock-control reset` at the end), `perf_ab.MachineMonitor` (a
  contaminated round is repeated; it gives up after 2 x rounds rejections),
  `CUDA_MODULE_LOADING=EAGER` and no `OMP_*` on both sides, one heap layout (one `--timing` file
  name for every round of both sides), A/B alternation, medians; per round the load average and
  `procs_blocked` before and after each side; a guard per round (A's counts equal the golden's;
  the port's counts once it counts, its batch lines with `--no-count`); the rows of the region map
  the port records; one Nsight Systems API trace per side for the host synchronizations inside
  each batch's update range. With `--no-count`, the end-to-end pre-read of step 2 (section 4).
- `memory`: both whole processes (no `--check`) under `nsys profile -t cuda,nvtx
  --cuda-memory-usage=true` on GPU 1 with no foreign process on it: the peak of live device
  allocations (memKind Device: `cudaMalloc` and `cudaMallocAsync`, A's Thrust temporaries
  included), the NVTX ranges of each peak, the peak per range, live / pool / slack / held at the end
  of each batch's range, and the footprint (allocations outside the pool plus the pool) at its
  peak and at the live peak (`hypergraph_perf.memory_by_range`, extended additively); beside it the
  store accounting per point against A's `deviceBytes()` (the golden set's recording) and A's two
  print points.

### 4. The go/no-go pre-reads of step 2 (not gates)

(a) End to end: base lock, GPU 0, 5 A/B rounds on coauth_b50k_s1 and syn1m_b50k_s1, A's
`A_noncount = total - init.full_count - sum(update.subtract + update.add)` against the driver's
`--no-count` total; target port <= A_noncount + 0.04 x A_total. (b) Memory: the nsys peaks of A's
process and of the `--no-count` driver on coauth_b50k_s1, syn1m_b50k_s1 and syn2m_b50k_s1;
target port <= 1.03 x A; the triad budget for step 3 = 1.05 x A_peak - port_peak - 1% of A_peak.
(c) The boost probe (3 rounds at boost on syn1m_b50k_s1 and syn2m_b50k_s1) decides the syn gate
lock; coauth stays at base (ADR 0021). The thresholds are internal margin targets, never
tolerances.

## Outcome (M9a step 2)

The driver draws A's batch files byte for byte on all 22 proof groups. (a) End to end, met:
coauth_b50k_s1 port 2,204.9 ms against A_noncount 2,720.7 ms (target 3,097.4), syn1m_b50k_s1
1,539.1 against 1,548.9 (target 1,648.2). (b) Memory, met: port / A = 1.0001 (coauth), 1.0002
(syn1m), 1.0000 (syn2m), both peaks in the H2H merge; the triad budgets for step 3 are 120.6,
45.1 and 88.4 MiB. (c) The boost lock does not hold on syn1m / syn2m (A's process in every
round), so their gates run at base. `parity/results/M9a.md` section 2.4.

## Consequences

- The gated port process holds only what A's holds (the stores, H2H and row views, and from step
  3 the counter's buffers); the 50 MB public view that M8b's generator would have kept is gone.
- `escher_host_model` is a tracked addition to `generators/legacy.hpp` (the API baseline lists
  it); the Python binding is M9b's.
- The host model costs host memory and time inside the end-to-end region, as A's host rows do.
  It is not inside the port's gated `update` (the driver applies it after the update timer),
  whereas A's host rows are maintained inside A's `update.maintain`: the asymmetry is recorded in
  ADR 0048 (scope rules) and the region map, the driver times the host model as
  `host_model_ms`, and the harness reports `update_with_host_model[b<N>]` beside the gate (M9a
  review). The host model reserves its row vector with geometric headroom (an exact reservation
  moved every row whenever the id bound grew: 28-41 ms of host work after the coauth batches'
  update timer in an Nsight Systems trace).

## Alternatives considered

- **Dropping the view after each draw** (a `release_views()` on the frozen container): would need
  a frozen-header change (`hypergraph/*`), which is not allowed.
- **Drawing from `to_hyperedge_list()`**: a full download and copy per batch, no view kept, but
  ids are compacted (the generator needs ids), and the download would add host syncs and time.
- **Prestaged batches** (read from `.dgb`): against the scope rules (ADR 0048: generation happens
  outside the region but in the process, as A's).
