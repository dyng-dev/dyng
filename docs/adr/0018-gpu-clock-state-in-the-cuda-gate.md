# ADR 0018: The GPU clock state in the CUDA performance gate

- **Status:** Accepted (2026-09-28, by the author: option B). The CUDA gate is read with the clocks
  locked for the whole A/B (`perf_ab.py run --backend cuda --lock-clocks boost`, both sides, checked in
  every busy sample); default-clock readings are recorded and published, not gated
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

PLAN 6.4.2 and 8.6 gate the CUDA `sssp` port per objective: dynG's `sssp::update` region against
MOSP-CUDA@e220ee2's `obj<k>/sosp_update_gpu` timer, <= 1.05x for regions of 10 ms or more,
medians of alternating runs of the two programs (one process per batch, as the original's bench
runs it). `perf_ab.py run --backend cuda` measures exactly that, as host times of the same scope on
both sides (the original has no device timer; dynG's CUDA-event time is recorded next to it and is
within 0.01 ms of its host time).

The RTX A5000 changes its performance state by itself (DVFS). Measured with nvidia-smi and with
Nsight Systems GPU metrics on GPU 0 (`parity/results/M1b.md` section 8.2):

- idle: P8 (SM 210 MHz, memory 405 MHz);
- a process with a CUDA context, GPU mostly idle: **P2**, SM clock up to 1695 MHz, **memory
  7601 MHz**;
- after roughly half a second of sustained GPU load: **P0**, SM about 1920 MHz, **memory
  8001 MHz**; it falls back to P2 within about a second once the load stops.

Both programs spend seconds on the host (reading the text inputs, applying the batch) with the GPU
idle, so what runs on the GPU in the half second before objective 0 decides the state of the
timed kernels:

- MOSP-CUDA runs its whole "upload" stage right before objective 0 (road_usa: about 470 ms of GPU
  work: the graph, the reverse CSR built on the device, the K trees, the workspace allocations);
  its kernels run in P0.
- dynG uploads the K trees earlier (in `result::from_arrays`, before the host apply) and its graph
  upload inside the commit takes about 130 ms; its kernels start in P2. In three nsys-profiled
  runs of each program the original's kernels ran at 1.84-1.90 GHz after one at 1.69 GHz, dynG's
  at 1.69-1.83 GHz; at the same clock (1.694 GHz) both took 22.5 ms.

On the gate graphs this matters only where the kernels are short and the graph is large: road_usa's
10K local batch (three kernels of about 21 ms). There the as-measured reading of the final gate
record is **1.065 / 1.063 / 1.037x** for objectives 0 / 1 / 2
(`M1b-final-perf-cuda-road_usa_g.json`, port `0f0fba9`; 1.06x in every earlier campaign), while
the other 33 per-objective readings of the four graphs are 0.98-1.01x. With the clocks held equal
the kernels are equal: under Nsight Compute with the clocks locked to base (`perf_ab.py kernels`,
21 alternating runs) road_usa local 10K reads 0.992-0.995x per objective and all 36
per-objective kernels of the suite are within 0.99-1.01x of the original's, with 59 registers,
the same 256 x 256 cooperative grid and 0.99-1.03x the DRAM bytes (the `affected` count). The code is the original's; the difference is the clock state that
the work *before* the timed region leaves behind.

Locking the clocks for whole runs (`nvidia-smi -lgc/-lmc`, or the PowerMizer "prefer maximum
performance" mode) needs root or a change of the shared machine's GPU settings, which this step did
not make. Moving dynG's tree uploads next to the kernels, or adding GPU busy work before them,
would raise the clock for the benchmark only; it would not change the kernel, would make dynG's
update slower end to end, and would tune the library to the idle pattern of a one-batch process.

## Decision

1. **Two readings of every per-objective CUDA region are recorded**, never merged:
   - *as measured* (PLAN 8.6, the gate as written): host times of the region, default clocks,
     `perf_ab.py run --backend cuda`, >= 20 alternating runs;
   - *at controlled clocks*: the fused kernels of both programs under Nsight Compute with the
     clocks locked to base (`perf_ab.py kernels`: `ncu --clock-control base --cache-control none`,
     kernel time, DRAM bytes, registers, grid, occupancy limits), >= 20 alternating runs. It
     needs no root.
2. **Nothing is moved to change the clock state.** dynG's stages stay where its API puts them;
   the harness adds no GPU work to either side.
3. **Proposed verdict rule** (not in PLAN 8.6; in force only once the author accepts it): a
   per-objective CUDA region that exceeds its gate as measured passes if (a) the two sides'
   kernels ran in different performance states (recorded: NVML / nsys GPU metrics), (b) the
   controlled-clock reading of the same region is within the gate, and (c) end to end is within
   its gate. Otherwise it fails. Until then the as-measured miss is reported as a **FAIL** in
   `parity/results/M1b.md`.
4. **Preferred protocol when root is available:** lock the application clocks on GPU 0 for the
   whole A/B (`sudo nvidia-smi -i 0 -lgc <f>,<f> -lmc <f>,<f>`, reset with `-rgc -rmc`) and read
   `perf_ab.py run --backend cuda` as measured; that makes rule 3 unnecessary.

## Decision taken (2026-09-28)

The author chose **option B**, in the form the M1b acceptance fix made possible without root: the
CUDA per-objective gate of PLAN 8.6 is read **with the GPU clocks locked for the whole A/B**
(`perf_ab.py run --backend cuda --lock-clocks boost`: an idle helper holds the lock through
`ncu --clock-control`, applied equally to both programs, and every busy GPU sample of both sides must be
at the locked clocks or the round is repeated). This is standard practice for comparing GPU kernels.
The **default-clock** reading is still measured, recorded and published next to it (for example
road_usa's local 10K batch reads about 1.06x at default clocks because of the P-state the preceding
work leaves behind), but it is **not gated**. Options A (root-locked clocks) and C (a strict
default-clock gate) were not taken. The same rule applies to every later CUDA gate (`cycle_count`,
`mosp`, `triad_count`, `label_propagation`, `hyper_sssp`).

## Consequences

- The CUDA gate record carries a clock-controlled kernel table next to the host-time table;
  `perf_ab.py kernels` stays part of the harness (`ci/perf_gate.sh` later).
- Short GPU regions measured in one-shot processes are documented as clock-sensitive (the sssp
  page's performance notes); users who time a single update after an idle period see the same
  effect with any code.

## Update (M1b review, 2026-09-28): the gated samples with a contamination monitor

The review found that the gate records carried no per-round machine state, so rule 3(a) rested on
three nsys-profiled runs outside the gated campaign. `perf_ab.py` now records, per round and side,
the foreign CPU load, the run queue, and the GPU's P-state, SM and memory clocks and utilization
every 50 ms (plus foreign compute processes on the GPU), and repeats contaminated rounds. The
CUDA gate was re-measured on the fixed code (`parity/results/M1b.md` section 13,
`M1b-review-perf-cuda-road_usa_g.json`):

- road_usa's local 10K batch still exceeds as measured: **1.060 / 1.060 / 1.054x**; the rounds
  were clean (foreign load at most 0.17 cores, no foreign GPU process, none rejected);
- both sides show the same two per-round modes (about 21.3 and 22.6 ms for objective 0) in
  different proportions (original 16 of 21 rounds in the fast mode, dynG 3 of 21);
- at locked clocks the kernels read 0.989-0.991x (`M1b-review-kernels-cuda-road_usa_g.json`);
- the 50 ms, whole-process clock record shows the **same** states on both sides (the same maximum
  SM clock, 1905 MHz, in every round, and P0 samples on both); the roughly 65 ms of kernels per
  round are one or two samples and cannot be separated from the work around them.

Consequence for the decision: the gated samples do not establish condition (a) of rule 3, so rule 3
as written would also leave the reading a FAIL. Establishing (a) on gated samples would need the
clock sampled inside the kernel windows (for example Nsight Systems GPU metrics on every gated
round, or timestamps of the timed region from both programs), which the unpatched original cannot
provide without a profiler. Option A (locked application clocks for the whole A/B, a root action)
is therefore the only option left that measures the gate as written with the clock state
controlled; options B and C are unchanged otherwise. The decision stays with the author.

## Update (M1b acceptance fix, 2026-09-28): locked clocks without root

The acceptance verifier re-measured the default-clock reading independently (road_usa local 10K
1.061 / 1.059 / 1.014x) and failed criterion 3(a), naming option A (locked clocks for the whole
A/B) as the way forward. Option A turned out not to need root:

- Nsight Compute locks the GPU's clocks while it profiles a process (`--clock-control
  base|boost`), and the driver keeps them locked **for every process on the GPU** until the
  profiled process ends (profiling is open to non-admin users on this machine,
  `RmProfilingAdminOnly: 0`). Checked on GPU 1: a spin kernel of another process took 36.1 ms
  under a `base` lock instead of 24.9 ms, and 24.9 ms under a `boost` lock, where sustained load
  at default clocks drifts to 24.5 ms.
- `perf_ab.py run --backend cuda --lock-clocks boost` (the default now) starts
  `ncu --clock-control boost parity/clock_lock/clock_holder.cu` on the timed GPU: the helper runs
  one kernel, then idles until the harness closes its standard input. **Neither timed program is
  profiled or changed.** The monitor accepts the helper's idle context and requires every busy GPU
  sample of both sides, in every accepted round, to be at the locked clocks (otherwise the round
  is repeated). At the end the helper exits normally (ncu restores the clocks) and `ncu
  --clock-control reset` runs in any case, because a killed ncu session leaves the clocks locked
  (seen once on GPU 1 during the trials and reset the same way).
- `boost` locks the RTX A5000 at SM 1695 MHz, memory 7601 MHz: the P2 state in which CUDA work
  runs until DVFS promotes a long busy period to P0. `base` locks SM 1170 MHz, memory 7601 MHz.

Result (`parity/results/M1b.md` section 14; all four graphs, three batches, 21 alternating
rounds, `boost`): every per-objective CUDA region is within its gate, 0.977-1.028x (road_usa
local 10K **0.993 / 0.994 / 0.995x**, original 22.7 ms); apply 0.67-0.87x; end to end
0.76-0.90x; outputs byte-identical and `invalidated` equal in every round. The same A/B at `base`
clocks on road_usa and a fresh default-clock reading of road_usa's local batch are recorded next
to it; the default-clock reading still shows the 1.06x of the P-state difference and is kept as a
separate, ungated row.

What this changes:

- Decision 4 is the gate protocol for the CUDA per-objective regions from now on, and the rule 3
  waiver is withdrawn (not needed). Decision 1 still holds: the default-clock reading and the
  Nsight Compute kernel reading are recorded next to the gate, never merged into it.
- The open decision reduces to one question for the author: whether the default-clock reading
  (which measures the GPU's DVFS response to the work each program does before its kernels, not
  the port) should also be gated. Until the author says so, the locked-clock reading is the
  gate, as rule 4 proposed.
