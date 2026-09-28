# Retrospective: M1b (CUDA `sssp`, CUDA core, performance harness and gates)

Status: **complete (2026-09-28) with one open gate decision; reviewed and fixed.** Steps 1-4 each
appended their section; Step 5 (finish) added the documentation, the Doxygen synchronization rule,
the final verification from a fresh clone of every preset, the milestone summary, the acceptance
record, the consolidated deviations, the lessons and the re-estimate. An independent review then
found 19 problems; all are fixed and the gates were re-measured on the fixed code (section
"Review and fix step" at the end). One acceptance reading is still not met as written and waits
for the author (criterion 3a, CUDA road_usa local 10K batch, 1.054-1.060x as measured, ADR 0018).

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

## Step 2: CUDA core (cuda-core, 2026-09-27)

Goal: the CUDA foundation the fused sssp kernel and the resident device graph build on (PLAN
4.6, 4.7.1-4.7.3, 7.1-7.4, 7.8, 8.8): the CUDA build, `resources::cuda()`, streams, the
CCCL-shaped memory resources, device buffers, errors, device error flags, `warm_up()`, device
workspaces, the GPU gate and the compile-only CI.

### Done

- **Build.** `DYNG_ENABLE_CUDA` (ON when a CUDA compiler is found; the presets pin it),
  `cmake/cuda_architectures.cmake` (`native`, `release` per toolkit, or an explicit list; CUDA >=
  12.4 enforced; set before `enable_language(CUDA)` and any target), nvcc using the same host
  compiler, shared cudart, `cuda_std_17`, `--extended-lambda --expt-relaxed-constexpr`,
  `-lineinfo` in RelWithDebInfo and with `DYNG_CUDA_LINEINFO`, `DYNG_CUDA_DEBUG_SYNC` (Debug).
  `.cu` files enter a module through `CUDA_SOURCES`; host files that call the runtime are
  ordinary sources guarded by `#if DYNG_HAS_CUDA`, so the CPU builds and clang-tidy see them.
  Host code keeps `-Wall -Wextra -Wpedantic -Werror`; nvcc runs with `--Werror=all-warnings` and
  `-Xcompiler=-Wall,-Wextra,-Werror`. Public headers stay host-only (the self-containment test
  still compiles them with g++ in the CUDA builds).
- **Presets:** `dev-cuda` (native), `release-cuda` (release list), `parity-cuda` (sm_86,
  `-O3 -lineinfo -fmad=true`, MOSP-CUDA's flags), `sanitize-cuda`, `ci-cuda12`, `ci-cuda13`;
  the CPU presets pin `DYNG_ENABLE_CUDA=OFF`.
- **Core (ADR 0016).** `resources::cuda(device, stream)` (device validated, properties recorded
  once incl. cooperative launch and SM count, a test hook to force the cooperative flag off),
  `scoped_device` on every enqueuing call, `synchronize()`, `warm_up()` through a kernel registry
  (`DYNG_REGISTER_KERNEL`); `backend_available(cuda)` checks for a visible device.
  `cuda_async_memory_resource` (own `cudaMemPool_t`, unlimited release threshold, 256-byte
  alignment, `used_bytes()` / `reserved_bytes()`), `default_device_memory_resource(device)`
  (per device, never destroyed), `pinned_host_memory_resource`; copies between every pair of
  spaces through `cudaMemcpyAsync` on the handle's stream and device; `DYNG_CUDA_TRY`,
  `DYNG_CUDA_TRY_NO_THROW`, `DYNG_CHECK_KERNEL(stream)`; `device_error_flags` (+ `.cuh` raise);
  `fill_async` kernels (7 explicit instantiations); `scratch_buffer<T>` for device workspaces;
  CCCL 3.x adapters in both directions (ADR 0003).
- **Tests** (`dyng_cuda_tests`, label `gpu`, 27 tests): resources and device record, invalid
  devices, user streams, current-device preservation across calls and copies (two GPUs), memory
  resource rules, warm-up of every registered kernel, pools (stream-ordered reuse without new
  reservations, OOM recovery), pinned memory, device buffers (round trips, resize, move, pinned
  staging, kernel writes), every fill instantiation, error macros and their messages, launch
  failures, the device error word and its exception mapping, the CCCL adapters, device
  workspaces (reuse, zero allocations in steady state counted through a counting resource,
  growth, release by `release_workspaces()`, `set_memory_resource()` and the last handle). The
  CPU tests adapt to CUDA builds (the default backend becomes `cuda`).
- **Gates.** `ci/gpu_local.sh` (dev-cuda build, `ctest -L gpu` on GPU 1, `ctest -L cpu` of the
  CUDA build, compute-sanitizer memcheck with leak check, clang-tidy with the CUDA build's
  compile database, a Markdown summary, optional PR comment). `ci/build_cuda.sh` and
  `.github/workflows/cuda-build.yml` (compile-only in `nvidia/cuda` 13.1.1, 13.3.1 and 12.9.2
  devel containers; per-kernel registers / stack / shared memory per architecture and the
  library size in the job summary).

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 7.4: `dev` is the native-architecture CUDA build | `dev` stays CPU-only (pinned); CUDA builds are `dev-cuda`, `release-cuda`, `parity-cuda`, `sanitize-cuda`, `ci-cuda12/13` (ADR 0016 item 9) | the CPU gate (`ci/check.sh`, `cpu.yml`) must not depend on an installed toolkit, and the Step 1 OpenMP gates stay on the unchanged `parity` build |
| PLAN 4.7.3: `DYNG_CHECK_KERNEL()` | `DYNG_CHECK_KERNEL(stream)` | the debug synchronization must be per stream; the stream-less form would need `cudaDeviceSynchronize`, which PLAN 4.6 rule 7 forbids |
| PLAN 4.2: `util/cuda_check.cuh`, `core/memory_cuda.cu` | `util/cuda_check.hpp`, `core/memory_cuda.cpp`, `core/cuda_runtime.cpp` | they need only the runtime API, so by the naming table (`.cuh` / `.cu` = needs nvcc) they are host files; the host compiler checks them with `-Wpedantic` |
| PLAN 4.7.2: built-in resources | + `default_device_memory_resource(device)`, `default_pinned_host_memory_resource()`, `cuda_async_memory_resource::used_bytes()` / `reserved_bytes()`, equality operators on the built-in resources | a default resource must outlive every buffer (process-wide, never destroyed); the memory report and the I9 tests need the pool counters; the CCCL concept requires equality |
| PLAN 4.7.1: "host staging is owned by resources" | the handle carries a staging resource (pinned for CUDA); staging buffers live in workspaces | pinned allocation is slow and synchronous: buffers must be allocated once, which the workspace pool already guarantees |
| PLAN 8.8 `cuda-build.yml`: latest 12.x and 13.x | + CUDA 13.1.1, the development toolkit | the only toolkit that is also built and tested on the development machine |

### Lessons

1. **Test the CUDA 12 path locally even without a CUDA 12 machine.** A conda toolkit (CUDA 12.9
   in an existing environment, used read-only with `CUDAHOSTCXX=/usr/bin/g++`) showed that the
   CCCL 2.x of CUDA 12 lacks the 3.x memory-resource concept; the adapters are now guarded, and
   `ci-cuda12` builds with `-Werror` and its GPU tests pass on the CUDA 12.9 runtime.
2. **compute-sanitizer counts failed API calls as errors.** Tests that make calls fail on purpose
   are grouped in the suite `CudaApiErrors`, which the gate runs without API-error reporting;
   every other test must be free of API errors too.
3. **CUDA 13 reports an oversized block as `cudaErrorInvalidValue`** (CUDA 12:
   `cudaErrorInvalidConfiguration`); tests should not pin launch-error codes.

### Open items

- The profiler's `device_ms` from CUDA events (PLAN 4.7.5) is not wired yet; the CUDA gate step
  needs it (or `perf_ab.py`'s own event timing) for regions under 10 ms.
- `engine::automatic`'s use of the recorded cooperative-launch flag, the co-resident block count
  (per kernel, from the occupancy API) and the fused kernel are the next step; so are the
  resident device graph, the CUDA A/B, the `edge_t` benchmark (ADR 0009) and the sssp page.
- `cuda-build.yml` has not run on GitHub yet (the repository is not pushed); CUDA 13.3.1 was not
  built locally.

## Step 3: the CUDA sssp engine (sssp-cuda, 2026-09-27)

Goal: port MOSP-CUDA@e220ee2's SOSP engine verbatim behind the fused engine of the CUDA backend,
with the resident device graph (the batch still applied on the host), compute / from_arrays /
device results, engine selection, `generators::legacy` for MOSP's change generator, and the CUDA
side of the harness (`dyng-compat-mosp`, `compare.py`, `perf_ab.py`) (PLAN 4.5.4, 5.1, 6.3,
6.4.1, 6.4.2; ADR 0017).

### Done

- **Fused engine** (`cpp/src/algorithms/sssp/fused.cuh`, `cuda.cu`): `sospPersistentKernel` and
  the host side of `sospUpdateGpu()` / `sospFromScratchGpu()` with mechanical changes only
  (names, templates on the index types, dyng streams, scratch buffers from the handle's pool,
  exceptions). The int32 instantiation compiles to the original's 59 registers and 616 bytes of
  parameters (`cuobjdump --dump-resource-usage` of `bin/mosp` and of the parity-cuda
  `libdyng.so`), so the occupancy-derived grid is the same (256 x 256 threads). Two additions
  outside the search: the deterministic `affected` counter in the unpack pass (224 more SASS
  instructions, all in the unpack pass; without it the SASS is byte-identical to the original's;
  the pass writes only changed pairs, so it moves no more bytes than the original's) and a
  host-side parent-cycle check on the control block.
- **Engine selection** before anything is changed: `automatic` / `fused` need the recorded
  cooperative-launch flag, else `not_supported_error` naming the host backends; `operators`
  throws (0.2). Tested with the forced flag (`resources_access::force_cooperative_launch`).
- **Resident device graph** (`cpp/src/graph/device_graph.{hpp,cu}`, from `uploadDeviceGraph`):
  a graph built with CUDA resources keeps its host CSR and a device copy per state (out- and
  in-edges, objective-major columns), uploaded on first use and inside the commit
  (`graph.upload`); placement by backend class with `clone(res)` to move graphs and results.
- **sssp on CUDA:** compute, update (the per-objective change lists from the commit's
  per-(insertion, objective) byte classification, uploaded in `sssp.changes`), `from_arrays`
  (host or device arrays, checked on the host, uploaded), `clone` across host and device, device
  result arrays; stats as in PLAN 5.1 (`engine_used = fused`).
- **Profiler:** `profiler_options::cuda_events` fills `device_ms` from CUDA events.
- **generators::legacy::mosp_changes()** (MOSP-OpenMP's generator, the same code as MOSP-CUDA's):
  15 committed fixture cases made by both originals' `mospPrep changes` are reproduced byte for
  byte for every index type; `dyng-compat-mosp changes` reproduces all twelve benchmark batches of
  the four gate graphs (seed 777) byte for byte.
- **Harness:** `dyng-compat-mosp --backend cuda` (warm-up first, CUDA-event rows in the timing
  CSV, download before writing), `compare.py --configs cuda`, CTest
  `parity.sssp.mosp_cuda_e220ee2`, `perf_ab.py run --backend cuda`, the completed
  `[reference.mosp_cuda]` map in `parity/timed_regions/sssp.toml`, and a golden-corpus step in
  `ci/gpu_local.sh`.
- **Tests:** `dyng_sssp_cuda_tests` (label `gpu`): the shared suites `sssp_test.cpp`,
  `sssp_random_test.cpp` and `sssp_fixture_test.cpp` compiled with `DYNG_TEST_CUDA=1` (the hand
  cases and the 34 MOSP fixtures byte-exact on cuda; randomized cross-backend equality cuda =
  openmp = sequential against Dijkstra and `check_sssp_tree(require_canonical)`, for all three
  index types, canonical and perturbed input trees), plus `sssp_cuda_test.cpp`: the forced
  no-cooperative-launch path, `engine::operators`, placement and clones, the device copy per
  state, device and host inputs of `from_arrays`, an overflowing imported tree (poisoned result),
  the packing boundary n = 2^17 - 1 (pull, push and compute), the 320 x 320 large-weight fallback
  (weights 2 * 10^9 with ties, and random up to 2^31 - 1), steady-state allocations (an update
  allocates exactly what the upload of the new graph state allocates), profiler device times,
  kernel registration. `dyng_generators_tests` (label `cpu`).

### Measured

- Golden corpus on cuda (`compare.py --configs cuda`, parity-cuda preset): **495 / 495
  byte-identical**, every group, `invalidated` equal.
- A first CUDA A/B (roadNet-PA, 5 alternating runs, GPU 0, not the gate record): SOSP region per
  objective 0.98-1.00x of MOSP-CUDA, apply 0.55-0.60x, end to end 0.82-0.84x; outputs
  byte-identical, `invalidated` equal in every run. The gate record (four graphs, >= 20 runs) is
  the next step's.
- road_usa (5 runs): 50K safe 1.00-1.01x per objective, apply 0.71x, end to end 0.73x; local 10K
  1.05-1.09x per objective. That gap is the GPU boost clock (1.69 vs 1.90 GHz during the kernels,
  nsys GPU metrics; equal kernel times under Nsight Compute's locked clocks), after the unpack's
  `affected` count was made free (`parity/results/M1b.md` 6.4).

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 5.1: `compute` "@async on CUDA" | `compute` synchronizes once (ADR 0017 item 7) | the workspace's stamp generation, which the next run must continue from, is known only after the kernel (as in `sospFromScratchGpu`) |
| PLAN 4.2 file names (`graph/transpose.cu`, `algorithms/sssp/kernels.cuh`, `instantiate.cu`) | `graph/device_graph.{hpp,cu}`; `algorithms/sssp/fused.cuh` + `cuda.cu` (instantiations in `cuda.cu`) | the device graph is a single port of `uploadDeviceGraph` (upload + device transposition); the fused engine has no separate kernels besides the persistent one |
| PLAN 4.7.5 stage names `sssp.identify_affected` ... `sssp.finalize` | on cuda one stage, `sssp.enact_fused`, replaces them | Tier B replaces those hooks by one launch; `sssp.changes` (upload of the change lists) is a sub-stage of the update |
| PLAN 8.6: regions < 10 ms "timed with CUDA events or nsys kernel sums" | the CUDA gate compares host times of the same scope (the original's timer is host time up to its synchronization); dynG's CUDA-event time is recorded next to it | the original has no device timer; comparing dynG's device time with the original's host time would favour dynG |
| PLAN 4.6 rule 5 (placement) | enforced by backend class: sequential and openmp share host graphs; a CUDA graph keeps a host CSR as well | the host apply is the straight port (device apply later); the CPU backends have always shared the host storage |
| — | the kernel counts `affected` and the host checks for a parent cycle; `stats::packed_parents` may differ between cuda and the host engines right at the packing limit | `affected` is part of `update_stats`; a cycle would otherwise give a silent wrong tree; the two originals choose the packing slightly differently and each port stays byte-equal to its original |
| PLAN 5.8 `generators::legacy` over graphs | `mosp_changes(csr_view, options, report*)` | MOSP's generator reads a CSR; a view needs no resources and works for any placement (`g.to_csr(res).view()`) |
| — | `dyng-compat-mosp changes` (the `mospPrep changes` flags) | a drop-in check of the generator on the benchmark batches |

### Lessons

1. **A template over the index types costs nothing when the int32 instantiation is the
   original's code:** 59 registers on both sides. Additions belong outside the hot loops (the
   `affected` count lives in the unpack pass only).
2. **Report lines are an interface.** The harness regex expects `(invalidated N,` first in the
   per-objective line; the first CUDA replay "failed" 495 cases on a reordered line while every
   file was equal. New fields go at the end.
3. **Compile shared suites twice instead of parameterizing across labels.** One source per suite,
   a `DYNG_TEST_CUDA` build of it in the `gpu` executable, keeps the `cpu` label free of GPU work
   and runs every hand case on cuda. Give the second build's cases a CTest prefix: CTest merged
   the same names from two executables and mixed their labels.
4. **Measure the clock before blaming the code.** With byte-identical SASS the port was 7 % slower
   on one workload; Nsight Compute (locked clocks) and nsys GPU metrics showed a boost-clock
   difference caused by what ran on the GPU before the timed region. Host-timed GPU regions of a
   few tens of ms need the clock state recorded next to them.
5. **An added counter can be free:** comparing before writing (write only changed pairs) moves no
   more bytes than the original's unconditional write, where "read, then write everything" cost
   2 %.

### Open items

- The CUDA performance gates (four graphs, three batches, >= 20 runs, `parity/results/M1b.md`),
  the `edge_t` benchmark with ADR 0009, and the `cuda-build.yml` check on GitHub are the next
  steps. The GPU boost-clock effect (`parity/results/M1b.md` 6.4) makes host-timed per-objective
  regions depend on the GPU lead-in; the gate protocol must control for it (locked clocks need
  root; otherwise record the GPC clock during the kernels, e.g. nsys `--gpu-metrics-devices`, and
  compare at equal clocks or with an equal lead-in), and must not move work to hide it. If the
  gate needs device-side times of the original, nsys kernel sums of `sospPersistentKernel` are
  the only source (not wired into `perf_ab.py`).
- A CUDA graph keeps a host copy of its CSR (memory) and uploads each new state; the device
  apply (PLAN 6.4.1) replaces both later.
- `compute()` on cuda is synchronous (ADR 0017 item 7).

## Step 4: parity and performance of the CUDA backend (cuda-parity-perf, 2026-09-27/28)

Goal: prove M1b parity and performance (PLAN 6.4.2, 8.3, 8.5, 8.6; acceptance criteria 2-4 and
7): byte parity of every backend with both originals on the whole corpus, the CUDA and OpenMP
gates on the four graphs, the fused kernel's resources, and the `edge_t` benchmark with ADR 0009.
The step was interrupted once by a usage limit; the second session verified the uncommitted work
of the first, kept it, and finished the step.

### Done

- **The corpus from MOSP-CUDA's own tools** (`711ad76`, `2b39fac`): `export_goldens.py
  --reference MOSP-CUDA --compare-to` re-exports all 495 cases with MOSP-CUDA@e220ee2's tools (its
  `bin/main` test cases, `mospTest`, `stressTest` / `parallelStressTest`, `mospPrep`, `bin/mosp
  --validate`, `sospUpdateGpu` through the export tool) and requires the committed corpus file for
  file: identical. `compare.py` configurations take `/int32` or `/int64`.
- **Byte parity on the final code** (`0f0fba9`): 495 / 495 on sequential, OpenMP 1/4/16/28,
  sequential and OpenMP with int64 offsets, CUDA and CUDA with int64 offsets; the paper-scale
  outputs of both A/Bs byte-identical; `invalidated` equal in every timed round.
- **Harness** (`0ecc062`, `49943ac`): `perf_ab.py kernels` (both programs under Nsight Compute,
  clocks locked to base: per-objective kernel time, DRAM bytes, registers, grid, occupancy
  limits) and `perf_ab.py edge-type` (int32 against int64 offsets); `dyng-compat-mosp
  --edge-type`.
- **The `edge_t` default** (`988a113`, ADR 0009): int32, with `detail::checked_edge_count` in
  every construction path (`capacity_error` naming the int64 instantiation); tested at the
  boundary.
- **The GPU clock state** (ADR 0018, proposed): the as-measured and the controlled-clock readings
  are recorded side by side; nothing is moved to change the clock state.
- **OpenMP engine** (`9439733`, `0f0fba9`): the per-thread lists live in the workspace on their
  own cache lines (`util/thread_list.hpp`); the near-far rounds pass three barriers instead of six
  (`list_gather::gather_pair()`, `nowait` loops), which fixed the road_usa local-batch reading
  (section "Measured") and makes every local batch 0.63-0.89x of the original.
- **Records**: `parity/results/M1b.md` sections 7-12 (the certificate), the JSON records next to
  it, ADR 0009's table on the final code, the sssp page's performance section.
- **Tests**: `list_gather` (`gather_pair()` equals two `gather()` calls, in thread order, after an
  `omp for nowait`), `thread_list` (alignment, no shared lines, capacity kept),
  `checked_edge_count` at the int32 boundary, the harness smoke tests for `kernels` and
  `edge-type`. `ci/check.sh` and `ci/gpu_local.sh` (build, `ctest -L gpu` and `-L cpu` of the CUDA
  build, the CUDA golden replay, compute-sanitizer memcheck, clang-tidy) green on the final code.

### Measured (every number: `parity/results/M1b.md` sections 7-12)

All on port `0f0fba9` (clean), parity presets, exclusive perf lock, GPU 0, 21 alternating
rounds per batch (11 for `edge_t`), medians:

- **Byte parity:** 495 / 495 in 9 configurations (sequential, OpenMP 1/4/16/28, sequential and
  OpenMP 4 with int64 offsets, CUDA, CUDA with int64 offsets); 24 paper-scale output checks
  byte-identical; `invalidated` equal in every timed round.
- **CUDA, as measured:** 34 / 36 per-objective readings within the gate (0.98-1.01x); road_usa's
  local 10K batch exceeds on objectives 0 and 1 (1.065x, 1.063x; objective 2 1.037x): **FAIL**
  under the strict reading. apply 0.55-0.74x, end to end 0.83-0.92x.
- **CUDA, locked clocks (Nsight Compute):** 36 / 36 kernels within 0.992-1.014x (road_usa local:
  0.992-0.995x), DRAM bytes 0.99-1.03x, 59 registers and a 256 x 256 grid on both sides. nsys GPU
  metrics: at the same clock (1.694 GHz) both kernels take 22.5 ms; the original's usually run at
  1.89 GHz after its 470 ms upload stage, dynG's at 1.69 GHz.
- **OpenMP:** 36 / 36 within the gate (0.63-0.99x; local batches 0.63-0.89x), apply 0.78-1.02x,
  end to end 0.75-0.91x. Before the barrier change the straight port read 1.12x on road_usa's
  local batch (objectives 1 and 2) in two campaigns, 0.96x in step 1's.
- **`edge_t`:** int64 costs 3-4.5 % on the CUDA SOSP region of the 50K batches (1.5-2 % on
  road_usa), -0.6 to +4.4 % on OpenMP; default int32 (ADR 0009).

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 6.4.2 / 8.6: the CUDA per-objective gate as host-time medians | recorded as measured **and** at locked clocks (Nsight Compute `--clock-control base`), never merged; the as-measured miss on road_usa's local batch stays a FAIL until the author decides (ADR 0018, open decision) | the RTX A5000's DVFS state during the timed kernels depends on the GPU work that precedes them in a one-shot process, not on the kernel; locking the clocks for whole runs needs root |
| MOSP-OpenMP's `ListGather` used as is (ported straight) | `list_gather::gather_pair()` and `nowait` loops: three barriers per near-far round instead of six | the straight port missed the local-batch gate on road_usa in two campaigns (barrier latency); outputs unchanged; reported separately from the straight port (PLAN 8.6) |
| MOSP's per-region `std::vector` locals | per-thread `thread_list`s kept in the workspace | no per-round allocations, no shared cache lines; same lists and order |
| PLAN 8.3 `parity/certify.py` writing `benchmarks/results/<version>/parity.json` | the certificate is `parity/results/M1b.md` plus the JSON records of `compare.py` / `perf_ab.py` | `certify.py` is a release tool (0.1); the records hold every field it would collect |
| PLAN 8.6: >= 5 runs | 21 alternating rounds per batch (11 for the `edge_t` benchmark) | every per-objective region is gated, and the regions under 10 ms need >= 20 |

### Lessons

1. **A 5 % gate on a barrier-bound loop measures the machine as much as the code.** The same
   binary read 0.96x in one campaign and 1.12x in the next; the original has the same two modes.
   Fixing the cause (six barriers per round where three suffice) removed the mode instead of
   averaging over it. Instrumenting the loop per thread found the cause in one run after hours of
   excluding the usual suspects (layout, huge pages, power, false sharing).
2. **Record the controlled reading next to the gate reading, never instead of it.** The locked-clock
   kernel table shows that the CUDA kernel is the original's (0.99-1.01x everywhere); the
   as-measured table shows what a user of a one-shot process sees. Both are true, and only the
   author can decide which one the gate means (ADR 0018).
3. **Re-exporting the corpus with the second original is cheap insurance.** "Byte-identical to
   the goldens" now means both originals' tools produce the same 11,799 files.
4. **Keep the tree clean during a measurement campaign.** The harness stamps every record with
   `HEAD` and a dirty flag at write time; editing a tracked file while it ran would have marked
   the records dirty. Docs and name fixes were written after the last record.
5. **Interrupted sessions need a written state.** The first session left uncommitted work and
   JSON records from several commits; the second had to re-derive which record belonged to which
   code. The final campaign re-measured everything on one clean commit.

### Open items

- **The CUDA per-objective gate on road_usa's local 10K batch** (objectives 0 and 1, 1.065x and
  1.063x as measured) needs the author's decision (ADR 0018, "Open decision"): (A) one A/B with
  the GPU clocks locked by root (`nvidia-smi -lgc/-lmc` on GPU 0, then reset), (B) accept the
  proposed verdict rule (different P-states recorded, locked-clock reading within the gate, end
  to end within the gate: all three hold), or (C) keep it as a known M1b gate miss. Until then
  acceptance criterion 3a is not met as written. Nothing was moved in dynG to change the clock
  state.
- The PLAN 8.3 `parity/certify.py` / `benchmarks/results/<version>/parity.json` tooling is not
  written; `parity/results/M1b.md` and its JSON records carry the same content for M1b.
- The OpenMP near-far loop could drop to two barriers per round (one gather barrier, the region
  end) with a lock-free prefix sum; not needed for the gate, left for the operators engine of 0.2.
- `perf_ab.py`'s load-average guard counts kernel threads in uninterruptible sleep (a CIFS mount
  on the lab machine pushed it to 244 without CPU load); a run-queue sample would be a better
  contamination monitor (PLAN 8.5).
- As in step 3: a CUDA graph keeps a host copy of its CSR; `compute()` on cuda synchronizes.

## Step 5: finish (2026-09-28)

Goal: close M1b (the step "finish" of the milestone): the sssp page, README, CHANGELOG, Doxygen
clean with `@sync` / `@async` on CUDA-capable functions, REUSE and provenance, both local gates
green from a clean tree, and this close-out. The step was interrupted once by a usage limit; the
second session found the tree clean at `0fef14f` (nothing of the first session was left
uncommitted) and did the whole step.

### Done

- **Doxygen synchronization rule** (`3f683fb`, `e330e9e`). The public functions that take a
  `stream_ref` or a `memory_resource_ref`, and the members that do stream-ordered work with their
  own stream (`buffer`'s destructor, `resize()` and move assignment; `resources::
  release_workspaces()`), now carry `@sync` or `@async` with a sentence saying what is ordered on
  which stream: the memory resources' `allocate` / `deallocate`, the `buffer` constructors,
  `resources::cuda()`, `set_memory_resource()`, `release_workspaces()`. `ci/doxygen_coverage.py`
  enforces it (before, only functions taking `resources` were checked); the rule was checked to
  fail on four removed tags.
- **Example on the GPU** (`10805b4`): `sssp_update ... cuda` runs the update with the fused engine
  and writes the same file; CTest `example.sssp_update.cuda` (label `gpu`) compares it with MOSP's
  output and is skipped (77) without a device; unknown backends are rejected.
- **sssp page** (`73f7f05`): the CUDA engine table (`automatic` / `fused` / `operators` with and
  without cooperative launch), the determinism level, a CUDA API snippet (compiled against the
  headers), absolute times next to the ratios, the gate thresholds of the table corrected
  (1.10x under 10 ms), and **"Paper vs fixed code"** (PLAN 6.3 step 10): the correctness fixes
  between the papers' code (`baseline-2026-09`) and the pinned originals, and the per-objective
  times of the papers' code, of the fixed code (its own record and the M1b A/B) and of dynG.
- **README and CHANGELOG** (`fbb2508`): the CUDA build (toolkit, presets, architectures), the local
  GPU gate and its knobs (`DYNG_TEST_GPU`, the perf lock, `DYNG_GPU_SKIP`), the example on cuda,
  the parity check with MOSP-CUDA, the status table.
- **Found by building every preset** (the first full run, on `fbb2508`):
  - `SsspWorkspace.TheObjectivesShareOneWorkspaceAndASteadyStateAllocatesNone/openmp` failed in
    the `relwithdebinfo` preset (workspace 157,584 -> 169,872 bytes after the warm-up; about 1 in
    180 runs in `dev` too). Since Step 4 the OpenMP per-thread lists live in the workspace, and a
    list grows when its thread takes a larger share of a round than before, which the dynamic
    schedule decides. `sssp_workspace::thread_list_bytes()` now separates them; the test requires
    the rest of the workspace to stay exactly constant, the lists never to shrink, and the whole
    workspace to stay constant on the sequential backend (`5fbf149`; 320 repetitions under
    parallel load without a failure). The sssp page, `thread_list.hpp`, the engine header and
    ADR 0015 state the exception to I9 instead of "allocates nothing" (`eff3684`, `0086ee7`).
  - GCC 12 at `-O2` warned (`-Wstringop-overflow`, not an error in `relwithdebinfo`) on the
    sequential engine's colour reset; it now fills the whole state array (`4609c94`).
- **Verification** of every preset from a fresh clone of the final code (section "Final
  verification").

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 9.1: `@sync` / `@async` "on CUDA-capable functions" | the check defines CUDA-capable mechanically (a `resources`, `stream_ref` or `memory_resource_ref` parameter, plus a named list of stream-ordered members) | the XML has no notion of "CUDA-capable"; the list is short and lives next to the rule |
| PLAN 8.5 / 9.5: "paper vs fixed code" against the papers' numbers | against the papers' **code** (`baseline-2026-09`), from the originals' own `results/README.md`, plus the M1b A/B | neither original re-derived the papers' run times (MOSP-CUDA `CHANGES.md`, "Not done"); the same-machine comparison is the reproducible one |
| PLAN 9.6: README quickstarts extracted and run by `ci/docs.sh` | not yet; the CUDA snippet of the sssp page was compiled by hand against the headers | the extraction step belongs to the docs tooling of M5 |
| invariant I9 (no allocation in a steady-state update) | holds except for the OpenMP per-thread lists, which grow geometrically to the largest share a thread has taken (ADR 0015 update) | bounding them up front costs a round's size per thread; MOSP allocates them in every region |

## Milestone summary

M1b was carried out in five implementation steps between 2026-09-27 15:40 and 2026-09-28 04:20
(CDT): the OpenMP gates and the M1a carry-over (Step 1), the CUDA core (Step 2), the CUDA sssp
engine (Step 3), parity and performance (Step 4), and this finish (Step 5); two steps were
interrupted by a usage limit and resumed. 46 commits on `main` since the M1a close-out
(`c54eb6a`); nothing was pushed by a step. M2a (`m2-cycle`) and M4 (`m4-infra`) ran in parallel on
their own branches and worktrees and are not part of this record.

| Area | Size after M1b (lines, tracked) | After M1a |
|---|---:|---:|
| Library: public headers and sources (`cpp/include`, `cpp/src`; 1,540 lines of `.cu` / `.cuh`) | 16,000 | 9,900 |
| Tests (`cpp/tests`, without the fixture data) | 7,000 | 4,000 |
| Parity harness, compat driver, CI, scripts, CMake, examples (without the result records) | 7,300 | 3,800 |
| Documentation (Markdown, ADRs, algorithm and API pages) | 4,700 | 1,900 |

The tracked tree is 2.5 MB, of which 720 KB are the JSON records of `parity/results/` and 209 KB
the committed test inputs (limit 1 MB).

What M1b delivered, in one list:

- **CUDA foundation** (ADRs 0003, 0016): the CUDA build and presets, `resources::cuda()`,
  `stream_ref`, the CCCL-shaped `memory_resource_ref` with `cuda_async_memory_resource` as the
  default, pinned host memory, device `buffer`, device error flags, `warm_up()`, device
  workspaces in the handle's pool, profiler device times from CUDA events.
- **CUDA sssp** (ADR 0017): the resident device graph (uploaded per state, in-edges built on the
  device), MOSP-CUDA@e220ee2's persistent cooperative kernel ported verbatim behind the fused
  engine (the same 59 registers, 616 parameter bytes and 256 x 256 grid), engine selection with
  the forced no-cooperative-launch path, device results, `from_arrays()` / `clone()` across
  spaces.
- **OpenMP at the gates** (ADR 0015): workspace sharing through the pool, lazy in-edges,
  parallel checks and assembly, one-pass readers, per-thread lists in the workspace, three
  barriers per near-far round.
- **`generators::legacy::mosp_changes()`**, bit-exact with MOSP's `mospPrep changes`.
- **Harness:** CUDA in `dyng-compat-mosp`, `compare.py`, `perf_ab.py` (`run`, `kernels`,
  `edge-type`), the corpus re-exported by MOSP-CUDA's own tools, `ci/gpu_local.sh`,
  `ci/build_cuda.sh` and `cuda-build.yml`.
- **Decisions:** ADR 0009 (int32 offsets, checked), ADR 0015 (workspace sharing), ADR 0018
  (the GPU clock state in the CUDA gate; open).

### Acceptance record

| # | Criterion | Evidence | Status |
|---|---|---|---|
| 1 | A fresh clone configures and builds every preset (host `-Werror`) and passes all tests: CPU labels via `ci/check.sh`, GPU labels via `ci/gpu_local.sh` | section "Final verification": all 13 configure presets from one fresh clone of `0086ee7`, every test label green | met |
| 2 | CUDA byte-identical to MOSP-CUDA e220ee2 on the 495-case corpus plus the M1b cases; CUDA = OpenMP = sequential on the corpus and in randomized differential tests against `testing::dijkstra` + `check_sssp_tree(require_canonical)` | 495/495 in 9 configurations incl. cuda and cuda with int64 offsets (`parity/results/M1b.md` 7.3); the M1b cases are corpus groups (packing n = 2^17 - 1, the 320 x 320 large-weight grid, 100 `stressTest` + 100 `parallelStressTest` seeds) and CUDA unit tests; the corpus re-exported by MOSP-CUDA's own tools is file-identical (11,799 files); the randomized suites run cuda next to the host backends (`dyng_sssp_cuda_tests`); replayed again on cuda from the fresh clones of Step 5 | met |
| 3a | CUDA per-objective SOSP region <= 1.05x (<= 1.10x under 10 ms) of MOSP-CUDA | 34/36 readings within the gate (0.98-1.01x); road_usa local 10K objectives 0 and 1 read 1.065x and 1.063x as measured (objective 2 1.037x); at locked clocks all 36 kernels read 0.992-1.014x (ADR 0018). **Re-measured after the review** on `674fc3b` with the contamination monitor: 33/36 within the gate, road_usa local 10K 1.060 / 1.060 / 1.054x, clean rounds, 0.989-0.991x at locked clocks | **not met as written**; the author's decision is open (ADR 0018: A, B or C) |
| 3b | CUDA end to end <= 1.10x | 0.83-0.92x (apply 0.55-0.74x) | met |
| 3c | OpenMP per-objective region and end to end within the same gates against MOSP-OpenMP c352151 (28 threads pinned); the M1a carry-over closed | 36/36 per objective (0.63-0.99x), end to end 0.75-0.91x, apply 0.78-1.02x; workspace sharing (ADR 0015), lazy in-edges in `from_csr`, parallel tree validation | met |
| 4 | Fused kernel registers and occupancy no worse than the original's | `cuobjdump --dump-resource-usage`: 59 registers, 616 parameter bytes on both sides; 4 blocks of 256 threads per SM, the same 256 x 256 grid (`parity/results/M1b.md` section 11) | met |
| 5 | `engine::automatic` = fused, `not_supported_error` with a clear message without cooperative launch (forced flag tested); streams and memory resources per PLAN 4.7; no allocation inside `update` after warm-up for a stable workload | engine selection before any change, message naming the host backends (`sssp_cuda_test.cpp`, forced flag); `resource_ref`-shaped `memory_resource_ref`, `cuda_async_memory_resource` default (ADRs 0003, 0016); the algorithm phase of an update (workspace, change lists, fused engine) allocates nothing in steady state (counting-resource test); the per-batch upload of the new graph state allocates its arrays from the stream-ordered pool, because the host apply builds a new CSR per batch as MOSP does | met, with the graph upload exempt until the device apply (as the host apply's I9 exemption of M1a) |
| 6 | `generators::legacy` reproduces MOSP's change generator bit-exactly | 15 committed fixtures from both originals' `mospPrep changes`, all index types; the twelve benchmark batches of the gate graphs (seed 777) byte for byte | met |
| 7 | `edge_t` benchmark done, ADR 0009 fixes the default; an ADR on workspace sharing | int64 offsets cost 3-4.5 % on the CUDA 50K batches, -0.6 to +4.4 % on OpenMP: int32 with checked construction (ADR 0009, accepted); ADR 0015 | met |
| 8 | `cuda-build.yml` (compile-only) exists and builds locally in an equivalent way; docs updated; this retrospective | `ci/build_cuda.sh ci-cuda13` (CUDA 13.1) and `ci-cuda12` (CUDA 12.9) from the fresh clone (below); the hosted run waits for the push; the sssp page (CUDA backend, engines, performance table, paper vs fixed code), README, CHANGELOG, this file | met (the hosted run waits for the push) |

### Final verification (Step 5)

All from one fresh `git clone` of `0086ee7` (the code as committed; this retrospective was
added after it and re-checked with the lint steps), on the development machine (GCC 12.2, CUDA
13.1, RTX A5000; GPU tests on GPU 1; every heavy step under the shared perf lock). Logs:
`$DYNG_SCRATCH/runs/m1b-finish/final/`.

| Command (preset) | Result |
|---|---|
| `ci/check.sh --parity` (`cpu-only`, `dev`, `parity`) | all steps passed: clang-format; `cpu-only` 229/229 and `dev` 241/241 (`ctest -L cpu`, `-Werror`); clang-tidy naming; REUSE; provenance (86 files); harness smoke tests; Doxygen + convention check (105 compounds, incl. the new `@sync` / `@async` rule); pre-commit; `parity` preset golden replay (sequential and OpenMP) passed |
| `ci/gpu_local.sh` (`dev-cuda`, `-Werror`) | build; `ctest -L gpu` 83/83 on GPU 1 (incl. `example.sssp_update.cuda`); `ctest -L cpu` 241/241 in the CUDA build; the golden corpus on cuda 495/495 byte-identical; compute-sanitizer memcheck with leak check 0 errors (4 passes); clang-tidy on the CUDA branches |
| `release`, `relwithdebinfo` | build without warnings; `ctest -L cpu` 229/229 each |
| `asan` (`-Werror`) | `ctest -L cpu` 229/229 (skips by design: impossible allocations under sanitizers) |
| `tsan` (`-Werror`, OpenMP off) | `ctest -L cpu` 200/200 |
| `release-cuda` (sm_75-sm_120 + PTX) | build without warnings; `ctest -L gpu` 77/77 |
| `parity-cuda` (MOSP-CUDA's flags, sm_86) | build; `ctest -L gpu` 84/84; `compare.py --configs cuda`: 495/495 byte-identical (manifest `668145c6...`) |
| `sanitize-cuda` (`-Werror`, launches checked) | `ctest -L gpu` 77/77 |
| `ci/build_cuda.sh ci-cuda13` (CUDA 13.1) | compile-only release build, `-Werror`: passed; `sssp_persistent_kernel<int, int, int>` 59 registers on sm_86 |
| `ci/build_cuda.sh ci-cuda12` (CUDA 12.9, conda toolkit read-only) | compile-only release build, `-Werror`: passed |

The first full run (on `fbb2508`) failed in `relwithdebinfo` on the flaky steady-state test and
showed the `-Wstringop-overflow` warning; both are fixed (Step 5, "Found by building every
preset"), and the run above is the rerun on the fixed code.

### Deviations, consolidated

Every deviation is in the table of the step that made it; the ones that matter beyond M1b:

1. **Scratch memory belongs to `resources`** (ADR 0015): a workspace pool per handle, leased by
   runs, instead of workspaces inside results (PLAN 4.7.2, 5.1); `release_workspaces()`,
   `workspace_bytes()`.
2. **Graph construction:** in-edges built on first use; `from_csr(res, csr&&, props)`; the
   `apply` region gated at 1.10x in addition to PLAN 8.6's regions.
3. **Presets:** `dev` stays CPU-only; the CUDA builds are separate presets (`dev-cuda`,
   `release-cuda`, `parity-cuda`, `sanitize-cuda`, `ci-cuda12/13`) (ADR 0016).
4. **CUDA placement:** a graph belongs to its backend class; a CUDA graph keeps a host CSR and a
   device copy per state; `compute()` on cuda synchronizes once (ADR 0017).
5. **Stage names on cuda:** one `sssp.enact_fused` stage replaces the four hooks (Tier B).
6. **The CUDA gate:** host times of the same scope on both sides (the original has no device
   timer), recorded as measured and at locked clocks, never merged (ADR 0018, open).
7. **OpenMP engine beyond the straight port:** three barriers per round (`gather_pair()`,
   `nowait`) and per-thread lists in the workspace; the straight port's records kept separately.
8. **Additions to the ported kernel:** the deterministic `affected` count in the unpack pass and
   a host-side parent-cycle check; `stats::packed_parents` may differ between cuda and the host
   engines right at the packing limit.
9. **Records instead of tools:** `parity/results/M1b.md` with JSON records instead of
   `certify.py` / `benchmarks/results/<version>/parity.json`; 21 alternating rounds instead of 5.
10. **Documentation checks:** `@sync` / `@async` enforced on a mechanical definition of
    "CUDA-capable"; "paper vs fixed code" against the papers' code rather than the papers'
    numbers.
11. **Invariant I9 exceptions:** the OpenMP per-thread lists (schedule-dependent growth, ADR 0015)
    and, on CUDA, the per-batch upload of the new graph state (until the device apply).

### Lessons

1. **Machine time for performance gates is the scarce resource.** The final campaign (4 graphs
   x 3 batches x 21 rounds for both backends, the locked-clock kernels and the `edge_t`
   benchmark) held the exclusive lock for about 2 h 40 min; every gate iteration before it cost
   30-60 min. Parallel milestones (M2, M4) share that lock: schedule GPU campaigns, do not
   assume them.
2. **A gate is only as good as its control of the machine.** Two causes of false readings were
   the machine, not the code: DVFS performance states on the GPU (ADR 0018) and barrier-latency
   modes on the CPU (Step 4). Record the machine state (clocks, P-state, run queue) next to every
   timing, and decide in advance, in an ADR, what the gate compares when the state differs.
3. **Straight ports first, measured changes after, both kept on record.** Every change beyond
   the straight port (workspace sharing, parallel assembly, three-barrier rounds) came with a
   measurement and left byte parity intact; the straight port's numbers stay in the certificate.
4. **Byte parity scales when the originals are cross-checked first.** Both originals' tools
   produce the same 11,799 files; every CUDA mismatch during the port was dynG's by construction,
   and the corpus replays in under a minute per configuration.
5. **Mechanical checks beat review for conventions.** Extending the `@sync` / `@async` rule
   from `resources` parameters to streams and memory resources found twelve public functions
   without the tag, all written in Steps 2-3 and unnoticed since.
6. **Build every preset from a fresh clone before closing a milestone.** `ci/check.sh` and
   `ci/gpu_local.sh` cover four presets; the other nine found a schedule-dependent test (a 1-in-180
   flake that the gates had never hit) and an optimizer-only warning.
7. **Interrupted sessions need a clean tree or a written state.** Step 4's first session left
   uncommitted work and mixed records; Step 5's interruption left nothing, because every piece
   was committed as soon as it was checked.

## Re-estimate of the roadmap

**Measured:** the M1a retrospective re-estimated M1b at 4-6 days of focused work and 1.5-2 weeks
of calendar time (plan: 2 weeks). M1b took five sessions over about 13 hours of wall-clock time;
the final measurement campaign alone held the exclusive lock for about 2 h 40 min, and the
earlier A/B runs of Steps 1, 3 and 4 added to that. The independent review and its fix
step (M1a's found 25 defects) are still to come. M2a (the CPU half of M2) and M4 were completed
in parallel on their own branches in the same period.

**What this changes.** The coding of a port with a good original is faster than the M1a
re-estimate assumed; the remaining time is dominated by (i) measurement campaigns that need the
exclusive lock of one shared machine, (ii) reviews and their fix steps, (iii) the author's
decisions and account actions, and (iv) merges of parallel branches. The re-estimate therefore
lowers the focused effort and keeps most of the calendar time.

| Milestone | Plan (working weeks) | M1a re-estimate (focused) | Now: focused AI effort | Calendar incl. author gates | Main risk |
|---|---|---|---|---|---|
| M1b CUDA `sssp` + harness | 2 | 4-6 days | done (about 13 h), plus a review-and-fix step of 1-2 days | ADR 0018 decision; review | the open gate reading; review findings |
| M2 `cycle_count` | 2-3 | 4-6 days | M2a done; M2b (CUDA) 3-5 days after the M1b merge | 1-1.5 weeks | two kernel families, device batch apply, GPU campaigns competing for the lock |
| M3 framework + conformance kit + 0.1 API freeze | 2-3 | 5-8 days | 4-7 days | 2-3 weeks | re-running parity and both GPU/CPU gates per refactor commit (lock time); the author's API sign-off |
| M4 GitHub repository and infrastructure | 1-2 | 1-3 days | done on `m4-infra` (merge pending) | author settings actions | first hosted runs of the CUDA compile job |
| M5 Python CPU wheel, CLI, docs | 2-3 | 5-8 days | 4-7 days | 2-3 weeks | nanobind + scikit-build-core, Sphinx `-W`, TestPyPI release candidate |
| **0.1.0** | **12-17** (from M0) | **4-6 weeks** | **about 3-4 weeks** from now | **about 6-8 weeks** | merges of three branches; reviews |
| 0.1.x (M6) | 4-6 | 1.5-2 weeks | 1-2 weeks | 3-4 weeks | GPU runner decision (O11); wheel sizes |
| 0.2.0 (M7-M9) | 8-12 | 3-4 weeks | 2.5-3.5 weeks | 6-8 weeks | the operators engine must stay within 1.05x of the fused one (a long GPU campaign per refactor); the CBST merge |
| 0.3.0 (M10-M11) | 8-12 | 3-4 weeks | 2.5-3.5 weeks | 6-8 weeks | DynLP float tolerances; conda-forge review |
| **Up to 0.3.0** | **about 32-47** | **12-16 weeks** | **about 9-13 weeks** | **about 20-27 weeks** | |

**Recommended order now:** the author decides ADR 0018 (option A needs one sudo command pair and
one 30-minute A/B on GPU 0); an independent review of M1b; merge M1b, then M2a and M4 into `main`
(M2b needs M1b's CUDA core); M2b; then M3. Measurement campaigns of parallel milestones should be
scheduled one after the other on the exclusive lock.

## Open items carried forward

For **the author**:

1. **ADR 0018** (acceptance criterion 3a): (A) one A/B with root-locked clocks on GPU 0
   (`nvidia-smi -lgc/-lmc`, then reset), (B) accept the proposed verdict rule (its three
   conditions hold for road_usa's local batch), or (C) keep the reading as a known M1b gate miss.
2. Review M1b; ADRs 0015-0017 are proposed and are accepted with the 0.1 API freeze (M3).

For **M1b's review-and-fix step / M2b / M3**:

3. A CUDA graph keeps a host copy of its CSR and uploads each new state (allocating from the
   pool); the device apply (PLAN 6.4.1) removes both. `compute()` on cuda synchronizes (ADR 0017
   item 7).
4. `perf_ab.py`'s load-average guard counts kernel threads in uninterruptible sleep (CIFS); a
   run-queue sample would be a better contamination monitor (PLAN 8.5).
5. The OpenMP near-far round could drop to two barriers with a lock-free prefix sum (not needed
   for the gate; the operators engine of 0.2 is the place).
6. `parity/certify.py` and `benchmarks/results/<version>/parity.json` (PLAN 8.3) are release
   tooling for 0.1.
7. The README quickstart extraction (`ci/docs.sh`, PLAN 9.6) comes with the docs tooling of M5.

For **M4 / the first push**:

8. `cuda-build.yml` has not run on GitHub yet (the CUDA 13.3.1 container was never built
   locally; 13.1 and 12.9 were).

## Review and fix step (M1b review, 2026-09-28)

An independent review of M1b (at `6ab4f25`) found 19 problems, each confirmed by a second
reviewer. All 19 are fixed in this step, in small commits `043fcae`..`674fc3b` (plus the records
and this section), each checked by the tests it touches; nothing was deferred. The step was
interrupted once by a usage limit before any change was made; the resumed session found the tree
clean at `6ab4f25` and did the whole step.

### Findings and fixes

| # | Finding (severity) | Fix | Commit | Checked by |
|---|---|---|---|---|
| 1 | The fused kernel's `invalidated` counter races with the insertion-head appends (high; inherited from MOSP-CUDA@e220ee2) | per-thread counts summed per warp before the barrier (`count_into`); ADR 0017 amendment | `d5c52bc` | the CUDA sssp suite under `compute-sanitizer --tool synccheck` 49/49 (four failed before), a new `synccheck` step in `ci/gpu_local.sh` (`a0ee1a9`); 59 registers kept; golden corpus 495/495 on cuda and cuda/int64 |
| 2 | `grow()` freed the old device arrays on their own stream while the copy reading them ran on another (medium) | `buffer::set_stream()`; the old arrays move to the updating stream first | `2a9d0a2` | `SsspCuda.VertexGrowthThroughAHandleOnAnotherStream` (two explicit streams) |
| 3 | `@sync` of `sssp::update()` / `compute()` omitted the graph upload's synchronization (low) | `@sync` text, sssp page, ADR 0017 item 7 amendment | `28df43c` | Doxygen check |
| 4 | No contamination monitor or per-round GPU clock record in the gate records (medium) | `perf_ab.py`: per round and side the foreign CPU load, run queue, GPU P-state / SM and memory clocks / utilization every 50 ms, foreign compute processes on the GPU; contaminated rounds repeated; lock status recorded | `2a8950e` | harness tests (a spinning process is detected); the re-measured gate records below |
| 5 | The CUDA `apply` region left `sssp.import` out of dynG's side (low) | added to `port_all_results` (as in the OpenMP map) | `043fcae` | harness test; re-measured records |
| 6 | Default-stream handles are per-thread streams, so `synchronize()` and the workspace pool's reuse were unsound across threads (high) | workspace fences (`detail::cuda_stream_fence`): a CUDA lease records an event, the next lease on another stream waits for it, `release_workspaces()` / the pool's destructor wait before freeing; docs of `resources`, `stream_ref`, `buffer`; ADR 0016 item 10, ADR 0015 update | `e89a43c` | `CudaWorkspace.ALeaseOnAnotherThreadsDefaultStreamWaitsForThePreviousLease` (fails without the wait: the second thread read the old zeros) and `...ReleasingFromAnotherThreadWaitsForTheLastLease` |
| 7 | `copy_policy` never read; device batches rejected on CUDA (high) | `core/staging.hpp`: inputs not in host memory are copied once under the policy in graph builds, `graph::apply`, `dyng::update` / `update_each` / `sssp::update` and `from_arrays`; ADR 0016 item 11 | `182bfc6` | `SsspCuda.BatchArraysInDeviceMemoryAreCopiedOnce`, `...TheCopyPolicyGovernsImplicitCopies`, `...GraphBuildsTakeDeviceArrays`, `...FromArraysTakesHostOrDeviceArrays` |
| 8 | `sssp::update` crashed when only `insert_src` was in device memory (medium) | staging (7) plus `expect_host_batch()` on all eight arrays before any host read | `182bfc6` | each batch array in device memory in turn, on cuda and the host backends |
| 9 | `copy()` and `update_each()` did not deduce from mutable views (medium) | `copy()` deduces from `dst` only; an `update_each()` overload for `array_view<result_t*>`; callers simplified | `d6f7184` | `Copy.ToVectorAndCopyOnHost`; the callers compile in the natural form |
| 10 | `graph::to_backend()` missing and unrecorded; `to_space()` limited to the handle's space (medium) | `graph::to_backend(res)` (= `clone(res)` in this release), named by the placement errors; `to_space()` picks the resource by space; ADR 0017 item 3 amendment | `d6f7184` | `CudaBuffer.ToSpaceReachesEverySpaceOfTheHandle`, `SsspCuda.GraphsAndResultsStayWithTheirBackendUntilCloned` |
| 11 | `num_threads()` said 1 for CUDA while the host work used the global OpenMP default (low) | `resources::cuda(device, stream, host_threads = 0)` snapshots the count; ADR 0017 item 6 amendment | `8980145` | `CudaResources.HostThreadsAreFixedAtCreation` |
| 12 | Doxygen gaps: `@sync` on the synchronous allocation members, `@throws cuda_error` on the copies and buffers (low) | tags added; `ci/doxygen_coverage.py` covers `allocate_sync` / `deallocate_sync` of every resource and `stream_ref::synchronize()` | `d6f7184`, `7e95016` | the check fails on a removed tag (tried) and passes |
| 13 | `cuda-build.yml`'s "latest CUDA 13" was 13.3.1; 13.4.1 was out and unverified (medium) | the matrix builds 13.4.1 | `41b339e` | `ci/build_cuda.sh ci-cuda13` with conda-forge CUDA 13.4.92 and GCC 13.4: `-Werror` build passed |
| 14 | The local equivalent never used the containers' GCC 13; GCC 14 failed with `-Werror` (medium) | the test's per-insertion vector (a GCC 13/14 `-Wfree-nonheap-object` false positive) replaced | `9aacfbb` | `cpu-only` with GCC 13.4 and GCC 14.4: build with `-Werror` and 229/229 tests each; `ci-cuda13` (CUDA 13.4) and `ci-cuda12` (CUDA 12.9) with GCC 13.4 as host compiler |
| 15 | `native` architectures silently fell back to sm_75 without a visible GPU (low) | `dyng_check_native_cuda_architectures()`: release list and a warning | `319ebeb` | configure with `CUDA_VISIBLE_DEVICES=""` (sm_75..sm_120) and on the machine (86-real) |
| 16 | `cuda-build.yml` did not cache ccache (low) | ccache installed, launchers set, cache per toolkit; container paths for the caches | `41b339e` | YAML check (the hosted run waits for the push) |
| 17 | The `compat_mosp` cuda tests failed instead of skipping without a device (low) | exit code 77 mapped in `run_and_compare.cmake`, `SKIP_RETURN_CODE 77`; `gpu_local.sh` requires a visible test GPU | `7f96e1b`, `a0ee1a9` | `CUDA_VISIBLE_DEVICES=7 ctest -L gpu`: all skipped; GPU 1: passed |
| 18 | clang-tidy and all of `ci/check.sh` ran outside the perf lock (low) | `heavy()` in `check.sh`; tidy through it in both scripts | `720af36`, `a0ee1a9` | both gates run under the lock |
| 19 | `DYNG_WITH_NVTX` missing; `profiler_options::nvtx` silently ignored (low) | the option (CUDA::nvtx3, build interface only) and ranges per profiler stage | `cdd14d5` | `nsys --trace=nvtx` shows the stage ranges; `Profiler.NvtxRangesDoNotChangeTheRecord` |

### Verification of the final code (`674fc3b`)

All on the development machine (GCC 12.2, CUDA 13.1, RTX A5000; GPU tests on GPU 1; every
heavy step under the shared perf lock, the timing under the exclusive one). Logs:
`$DYNG_SCRATCH/runs/m1b-review/`.

| Command (preset, code) | Result |
|---|---|
| `ci/check.sh --parity` (`cpu-only`, `dev`, `parity`; `dea0896`, code equal to `674fc3b`) | all steps passed: clang-format, `cpu-only` 230/230 and `dev` 242/242 (`-Werror`), clang-tidy naming, REUSE, provenance (88 files), harness (17 tests), Doxygen + convention check (105 compounds, with the extended `@sync` rule), pre-commit, golden replay on the `parity` preset |
| `ci/gpu_local.sh` (`dev-cuda`, `674fc3b`) | build; `ctest -L gpu` 91/91; `ctest -L cpu` 242/242; golden corpus on cuda 495/495; memcheck 0 errors; **synccheck of the CUDA sssp suite: all tests pass** (new step); clang-tidy |
| golden replay records (`674fc3b`) | 495/495 in 9 configurations (`M1b-review-sssp-*.json`) |
| `cpu-only` with conda-forge GCC 13.4 and GCC 14.4 (`9aacfbb`) | `-Werror` build, `ctest -L cpu` 229/229 each |
| `ci/build_cuda.sh ci-cuda13` with CUDA 13.4.92 + GCC 13.4, `ci-cuda12` with CUDA 12.9 + GCC 13.4 (`9aacfbb`) | compile-only release builds with `-Werror` passed; the int32 fused kernel 59 registers on sm_86 |
| fresh clone of `674fc3b`: `release`, `relwithdebinfo`, `asan`, `tsan` (`ctest -L cpu`), `release-cuda`, `sanitize-cuda` (`ctest -L gpu`) | each builds without a warning; `release` / `relwithdebinfo` / `asan` 230/230, `tsan` 201/201, `release-cuda` and `sanitize-cuda` 85/85 |
| performance campaign (`674fc3b`, exclusive lock, 06:26-08:02 including waits for other sessions' shared locks) | below |

### Performance, re-measured with the contamination monitor

`parity/results/M1b.md` section 13 has every number; medians of 21 alternating rounds per
batch, parity presets, GPU 0, with the contamination monitor (one round rejected in the whole
campaign, none of the CUDA ones; foreign load of accepted rounds at most 1.46 cores; no foreign
GPU process).

| Reading | Result |
|---|---|
| CUDA per-objective SOSP region, as measured | 33 / 36 within the gate (0.979-1.010x); **road_usa local 10K 1.060 / 1.060 / 1.054x: still a FAIL as written** (step 4: 1.065 / 1.063 / 1.037x) |
| CUDA fused kernels at locked clocks (road_usa) | 0.989-0.991x (local 10K), 1.009-1.011x (50K); 59 registers, grid 256 x 256 on both sides |
| CUDA apply / end to end | 0.66-0.96x / 0.80-0.90x (apply now includes `sssp.import`) |
| OpenMP per-objective / apply / end to end | 36 / 36 (0.63-0.99x) / 0.79-1.03x / 0.77-0.90x |
| `invalidated` and trees in the A/Bs | equal in every timed round, byte-identical outputs |

The monitor settles what the road_usa miss is not: it is not contamination. It does not settle
ADR 0018's condition (a): sampled every 50 ms over each process, both sides show the same clock
states, and the 65 ms of kernels per round cannot be separated from the surrounding work. ADR 0018
now says so; the author's decision stays open, and option A (locked clocks for the whole A/B) is
the one that would measure the gate as written.

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| MOSP-CUDA's kernel ported verbatim (PLAN 6.3 step 5) | `invalidated` counted per thread instead of read from the list counter (ADR 0017 item 1) | the original races on that read; the public contract calls the counter deterministic and the parity checks compare it exactly; trees and SASS of the search unchanged |
| PLAN 4.7.4: a copy of `resources` "refers to the same stream" | true for explicit streams; the default `cudaStreamPerThread` is each thread's own stream, documented; the library orders its own pooled memory across streams with events (ADR 0016 item 10) | the per-thread stream has no handle that another thread could use; pinning handles to a thread would forbid the concurrent read-only calls of 4.7.4 |
| PLAN 4.7.1: `resources::cuda(device, stream)` | + `host_threads` (ADR 0017 item 6) | the host-side work of a CUDA call runs on OpenMP threads, whose count belongs to the handle (PLAN 4.6 rule 6, 4.7.4) |
| PLAN 4.7.1: "if [an input's space] does not match the backend, the library copies once" | the match is against where the input is read: the host, for every backend in this release; host inputs of CUDA calls are not implicit copies (ADR 0016 item 11) | the batch is applied, graphs are built and trees are checked on the host (the straight ports); calling host arrays "misplaced" would make the fastest path warn |
| PLAN 5.2: `graph::to_backend(res)` next to `clone(res)` | both, identical in this release | a CUDA graph keeps its authoritative CSR on the host until the device apply |
| PLAN 8.5 item 2: "the contamination monitor" | foreign CPU load from `/proc/stat` minus the harness's own CPU time, run queue without the timed program's threads, NVML samples via `nvidia-smi -lms 50`, foreign GPU processes every second | the originals' `gpumon.sh` rule plus the CPU side the load average could not show; `pynvml` is not installed |

### Lessons

1. **A verbatim port inherits the original's races.** The golden corpus could not show the
   `invalidated` race because the original's plain runs never expose it; a sanitizer's scheduling
   did. Run the ported kernels under `synccheck` / `racecheck` once, and keep the one that finds
   nothing false in the gate.
2. **"Per-thread default stream" is a different stream per thread.** Every stream-ordered free
   and every "same stream" assumption has to name the thread; the library's own reuse needs
   events, and the documentation has to say what `synchronize()` really waits for.
3. **Documented options must do something.** `copy_policy` and `profiler_options::nvtx` were
   stored and documented but read nowhere; a grep for the readers of every public setting is a
   cheap review step.
4. **Test the CI's host compiler, not the development machine's.** GCC 13/14 disagreed with GCC
   12 on one test; conda-forge compilers (read-only environments in `$DYNG_SCRATCH/tools`) make
   the container toolchains checkable locally.
5. **Record the machine next to every timing.** The per-round monitor turns ADR 0018's rule (a)
   from a claim backed by three profiled runs into a property of the gated samples.

### Open items after the review

- **ADR 0018** (acceptance criterion 3a) is still the author's decision; the monitored
  re-measurement shows the miss is not contamination and that the gated samples cannot establish
  rule 3's condition (a), which leaves option A (locked clocks for the whole A/B, a root action)
  as the way to read the gate as written.
- `cuda-build.yml` (now CUDA 13.4.1, 13.1.1 and 12.9.2, with ccache) runs on GitHub only after the
  push; its three toolkits were built locally with the containers' host compiler generation
  (GCC 13) from conda-forge environments in `$DYNG_SCRATCH/tools` (13.1 with the system GCC 12).
- The implicit copies of device inputs are host staging copies until the device apply (PLAN
  6.4.1) makes device batches the matching space (ADR 0016 item 11).
- The monitor's GPU samples cover whole processes; clock readings inside the timed kernels would
  need profiler metrics on every gated round (ADR 0018 update).
- The records of this step add about 840 KB of JSON to `parity/results/` (the per-round monitor
  data); the tracked tree is about 3.4 MB. A compact record format (or moving the per-round data
  out of the repository) is worth deciding before 0.1.

## Acceptance fix step (M1b acceptance, 2026-09-28)

The independent acceptance verifier (at `a304d90`) passed criteria 1, 2 and 4-8 and failed 3(a):
its own default-clock A/B read road_usa's local 10K batch at 1.061 / 1.059x on objectives 0 and 1
(gate 1.05x), as the committed records did, and PLAN 8.6 has no clock-state exception. It named
option A of ADR 0018 (locked clocks for the whole A/B) as the way forward, which was believed to
need root. The step was interrupted once by a usage limit before any change was made; the resumed
session found the tree clean at `a304d90`.

### Done

| Item | Commit |
|---|---|
| Locked clocks without root: Nsight Compute holds a clock lock for every process on the GPU while the process it profiles lives; `perf_ab.py run --backend cuda --lock-clocks` with `boost` (the default), `base` or `none` starts the idle helper `parity/clock_lock/clock_holder.cu` under `ncu --clock-control`, the monitor requires every busy GPU sample of both sides to be at the locked clocks and allows the helper's context, `ncu --clock-control reset` runs at the end; harness tests for the clock check, the allowed process and the no-op mode | `c25fe6d` |
| `DYNG_WITH_NVTX` defaults to ON only when `nvtx3/nvToolsExt.h` exists (the verifier's CUDA 12.8 conda toolkit has `CUDA::nvtx3` but no headers); `cuda-build.yml` names the `LD_LIBRARY_PATH` a conda-forge toolkit needs locally (the verifier's minor findings) | `92d9fe4` |
| The CUDA gate at locked clocks on all four graphs, plus road_usa at base clocks and a fresh default-clock reading of road_usa's local batch | `3e6de97` |
| ADR 0018 update, `parity/results/M1b.md` section 14, the sssp page, `parity/README.md`, CHANGELOG, this section | this commit and `3ecb8fd` |

### Measured (`parity/results/M1b.md` section 14)

| Reading | Result |
|---|---|
| CUDA per-objective SOSP region, clocks locked (boost: SM 1695 / memory 7601 MHz), 4 graphs x 3 batches, 21 rounds | **36 / 36 within the gate**, 0.977-1.028x; road_usa local 10K **0.993 / 0.994 / 0.995x**; unimodal per-round times |
| CUDA apply / end to end, locked | 0.67-0.87x / 0.76-0.90x |
| road_usa at base clocks (SM 1170 MHz) | 0.993-1.002x per objective |
| road_usa local 10K at default clocks (ungated) | 1.063 / 1.060 / 1.062x, bimodal as before; the original's kernels reached P0 more often (118 of 480 busy samples against 68 of 525) |
| Outputs / `invalidated` | byte-identical, equal in every round |
| `ci/check.sh` (`c25fe6d`) | all checks passed (harness 20 tests) |
| `ci/build_cuda.sh ci-cuda12` with the conda-forge CUDA 12.8 toolkit and GCC 12.2 (`92d9fe4`; the verifier could not reproduce ci-cuda12 because of the NVTX headers) | release architecture list (sm_75..sm_120), `-Werror`, 0 warnings, rc 0; NVTX detected as absent; the int32 fused kernel 59 registers on sm_86 |

The OpenMP gate and the other criteria were not re-measured: no library or kernel code changed
in this step (the changes are the harness, the NVTX detection of the build and documentation).

### Deviations from the plan (pragmatic choices, same intent)

| Plan | What was done | Why |
|---|---|---|
| PLAN 8.6: the CUDA per-objective gate, protocol without a clock rule | the gate is read with the GPU clocks locked for the whole A/B (ADR 0018 rule 4, `--lock-clocks boost`); the default-clock reading stays recorded next to it, ungated | the default-clock reading measures the GPU's DVFS response to the GPU work each program does before its kernels (the original's upload stage vs dynG's earlier tree uploads), not the code; locked clocks are the standard control, the plan's gate threshold is unchanged, and neither program is altered or profiled |
| ADR 0018 option A: `sudo nvidia-smi -lgc/-lmc` | the same lock through Nsight Compute's clock control, no root | root is not available; the lock is verified in every sample |

### Lessons

1. **Look for an unprivileged path before calling a step blocked on root.** The profiler's clock
   control, already used by `perf_ab.py kernels`, was the tool; what was missing was the
   observation that its lock applies to the whole GPU while the profiled process lives.
2. **A killed profiler session leaves the clocks locked**, for everyone on that GPU, until `ncu
   --clock-control reset`. The harness resets unconditionally and unwinds on SIGTERM / SIGHUP.
3. **Never walk `/` on this machine.** A `find /` issued to locate a header descended into the
   automounted `/dfs` network share and mounted thousands of CIFS sub-shares; the desktop daemons
   of every logged-in user then loaded the CPUs (load average above 50) for about half an hour.
   Search the toolkit directories by name instead.

### Open items

- ADR 0018: the author may still decide that the default-clock reading should be gated too
  (today it is recorded and ungated); ADRs 0015-0018 are accepted with the 0.1 API freeze.
- The records of this step add about 720 KB of JSON (per-round monitor data); the compact record
  format of the review's open items is still to be decided.
- `ci/perf_gate.sh` (PLAN 8.6) should pass `--lock-clocks boost` for the CUDA gates (it is the
  default of `perf_ab.py run`).
