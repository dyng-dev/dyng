# Retrospective: M1b (CUDA `sssp`, CUDA core, performance harness and gates)

Status: **in progress.** Each implementation step appends its section; the close-out adds the
milestone summary, the acceptance record, the lessons and the re-estimate.

## Step 1: the OpenMP gates and the M1a carry-over (cpu-gates, 2026-09-27)

Goal: close the M1a carry-over (retrospective M1a, "Open items carried forward" 1 and 2; ADR 0013
"Open M1b blocker") so that the OpenMP backend meets the PLAN 8.6 gates against
MOSP-OpenMP@c352151 on the four graphs of PLAN 6.4.2, before the CUDA gate is read.

### Done

- **Workspace sharing (ADR 0015).** `detail::workspace_pool` (`cpp/src/framework/workspace.hpp`)
  owned by every `resources` handle and shared by its copies. `compute()`, `from_arrays()`,
  `clone()` and each result's part of `update()` lease the pooled `sssp_workspace`, size it once
  and return it: the K objectives of `update_each()` share one workspace, as `mospUpdate()`
  shares its `SospWorkspace`; a steady-state update allocates no scratch memory; concurrent calls
  on copies of a handle lease distinct workspaces; a run that throws discards its workspace (a
  failed run may leave `in_far` flags set). The per-objective change lists moved into the
  workspace. Public additions: `resources::release_workspaces()`, `resources::workspace_bytes()`;
  `set_memory_resource()` releases the idle workspaces. The M1a pre-touch
  (`sssp.workspace.pretouch`) and the "first touch counted" reading of the A/B are gone: both
  sides now pay objective 0's first touch inside its region.
- **Graph construction without a transposition.** The in-edges are built on first use and cached
  per graph state (thread-safe); `run_update()` builds them inside the commit for the updated
  graph only. `from_csr()` checks the CSR in one parallel pass and takes a CSR that already has
  the requested form as it is; a new rvalue overload moves the arrays. The M1a load-time
  transposition (213 ms on roadNet-CA) is gone.
- **Parallel where the original is, identical results.** On the OpenMP backend: the CSR assembly
  of `apply` (degrees, offsets, block-wise copies), the import and the `validate_inputs` checks of
  `from_arrays()` (a chain walk like the engine's invalidation; any problem re-runs the
  sequential checks, so the message is unchanged). The transposition's scratch arrays are no
  longer zero-filled by one thread.
- **Reading.** `read_file()` reads a file with one allocation and one `fread` (the 64 KB-chunk
  growth copied large files several times); the distance and tree files are parsed in one pass
  with `std::from_chars` (falling back to the strict reader on anything irregular); the Values
  file is parsed straight into the objective-major columns (no conversion pass).
- **Harness.** `dyng-compat-mosp` builds the K results one after the other on its own handle and
  splits the input phase in its report; `timed_regions/sssp.toml` compares every objective as
  measured and gates `apply` and `end_to_end` at 1.10x; `perf_ab.py` reads the `end_to_end` gate,
  knows the local-batch radius of each graph (bench/prepare.sh: 110, 160, 110, 200) and writes
  schema 3. Benchmark inputs for roadNet-PA, rgg_n_2_20_s0 (`rgg`) and road_usa (`road_usa_g`)
  prepared with the unpatched original's `mospPrep` (2.4 GB under `$DYNG_SCRATCH/bench/mosp`).
- **Tests** (227 on `dev`, all green; also `cpu-only`, `asan`, `tsan`): the pool (reuse,
  distinct overlapping leases, discard on failure, 8 threads x 2000 leases), sssp on the pool
  (one workspace for K objectives, constant pool bytes over repeated updates, two graphs and two
  handles, failed update, concurrent computes, parallel validation messages), lazy in-edges,
  `from_csr` equal to `from_edges` for every property combination (copy and move, both backends),
  OpenMP apply assembly equal to sequential on 80K-vertex graphs, fast and strict readers agree.
  Every intermediate commit of the step builds with `-Werror` and passes its tests.
- **Parity:** the golden replay (495 cases x sequential / OpenMP 1, 4, 16) is byte-identical
  after the workspace change, after the I/O and apply changes (`ci/check.sh --parity`) and on the
  committed code in both the `parity` and the `dev` preset (JSON records); the A/B checks byte-identical outputs on all 12
  graph x batch combinations and equal `invalidated` counters in every round.

### Measured (details and every number: `parity/results/M1b.md`)

OpenMP A/B at `79400d2` against the unpatched MOSP-OpenMP@c352151, 28 threads pinned, medians of
21 alternating rounds, all three batches (50K safe, 50K unsafe, 10K local); ratio dynG / original:

| Graph | SOSP region per objective (gate 1.05x) | apply (gate 1.10x) | end to end (gate 1.10x) |
|---|---|---|---|
| roadNet-PA | 0.78-0.95x | 0.78-0.91x | 0.80-0.83x |
| roadNet-CA | 0.81-0.99x | 0.84-0.88x | 0.76x |
| rgg_n_2_20_s0 | 0.82-1.01x | 0.77-0.80x | 0.83-0.85x |
| road_usa | 0.84-0.96x | 0.79x | 0.77-0.81x |

**Every gate is met** (36 per-objective, 12 apply and 12 end-to-end readings). In M1a, on
roadNet-CA alone, objective 0 was 1.07-1.44x with its moved first-touch cost counted and end to
end 1.30-1.41x. Golden replay: 495/495 byte-identical on sequential and OpenMP 1/4/16 threads in
the `parity` and `dev` presets.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 4.7.2 / 5.1: workspaces live inside `result` objects | the `resources` handle owns a workspace pool; runs lease (ADR 0015) | MOSP shares one workspace across the objectives; per-result workspaces cost K times the memory and made the per-objective gate unmeetable without moving cost out of the region (M1a) |
| PLAN 4.7.1 `resources` members | + `release_workspaces()`, `workspace_bytes()` | the cache needs a release and a memory report (PLAN 8.6 memory gate) |
| PLAN 5.2 `graph::from_csr(res, csr_view, props)` | + `from_csr(res, csr&&, props)` | taking the arrays over avoids a copy of the whole graph (about 1 GB on road_usa) |
| `store_transposed`: "keep the in-edges" | the in-edges are built on first use and cached per graph state; `dyng::update()` builds them inside the commit | a graph built only to receive a batch never needs its own transposition (MOSP builds the reverse graph once, of the updated graph) |
| PLAN 8.6 gates: compute regions and end to end | also the `apply` region (the original's `apply batch` + `prepare`) gated at 1.10x | the draft map of M1a announced it; it keeps the non-kernel work honest |
| — | parallel CSR assembly in `apply`, parallel `from_arrays` checks, one-pass readers | the end-to-end gap was in the surroundings, not the kernel (see Lessons); outputs and messages unchanged |

### Lessons

1. **The end-to-end gap was I/O plumbing.** The kernels were at parity since M1a; the 1.3-1.4x
   end-to-end ratio came from a load-time transposition nobody used, file reads that copied every
   file several times, a two-pass token parser, a sequential weight-layout conversion and
   sequential validation. Profile the whole run stage by stage (the compat tool's split report)
   before touching an engine.
2. **Scratch memory belongs to the executor, not to the value.** Once the workspace moved to
   `resources`, the "first touch counted" bookkeeping, the moved-cost caveat of ADR 0013 and the
   K-fold memory all disappeared together.
3. **The original's own variance on this machine is large** (roadNet-PA end to end 306-418 ms
   between rounds; road_usa's `read_graph` 2.1-3.9 s): only medians of 21 alternating rounds are
   readable; 5-7-round probes during development gave misleading single ratios.

### Open items

- The host `graph::apply()` still builds a new CSR per batch (the straight port of
  `applyChangeBatch`, exempt from I9 until the resident apply), and the OpenMP engine's
  per-thread gather lists allocate per round (the straight port of `ListGather`).
- The `OMP_WAIT_POLICY=active` A/B of M1a was not re-recorded (the gates are read with the
  default policy, as in M1a).
- Next steps of M1b: CUDA core, resident device graph, the fused kernel, the CUDA gates, the
  `edge_t` benchmark (ADR 0009), `ci/gpu_local.sh`, `cuda-build.yml`.
