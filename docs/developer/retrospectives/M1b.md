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
