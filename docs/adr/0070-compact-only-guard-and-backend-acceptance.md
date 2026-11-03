# ADR 0070: The compact-only guard of the 0.1-0.3 algorithms, the backend-acceptance sentence, and the ADR 0024 reading of M10a step 3

- **Status:** Accepted under delegation (2026-10-29; GOVERNANCE.md, "Delegation of technical
  ADRs") for its technical choices. It carries out AD11 (sheet E5: one central check refuses
  slotted and slack graphs in the existing graph algorithms and in `dyng::update()`'s
  composition; `triad_count` takes hypergraphs only and is unaffected) and records AD19's backend
  sentence (sheet H1: no OpenMP backend of label_propagation in 0.4; OpenMP resources raise
  `not_supported_error` naming sequential and cuda), both **decided by the author on 2026-10-28**
  (GOVERNANCE.md, approvals log, the rows "API semantics of 0.4 (sheet E2-E10)" and "Licensing,
  data provenance, budgets and scope of 0.4 (sheet G1-G3, H1-H3)"), and the bar of AD11 / HD16
  (sheet B6: ADR 0024 at 1.02 with the D14 path; the row "The measurement protocol and gates of
  0.4 (sheet B1-B6)"). M10a step 3 (PLAN 4.6, 5.1; the M10a acceptance criteria 9 and 10).
- **Date:** 2026-10-29
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  host-container-sequential-driver)

## Context

`row_layout::slotted` (ADR 0069) changes what a graph's storage means: `view().out` describes the
slots (entries up to a row end, the unused slots -1, entries to deleted vertices kept). Every
algorithm of 0.1-0.3 (`sssp`, `mosp`, `cycle_count`, and the tutorials `dynamic_bfs` and
`triangle_delta`), the device copy of the CUDA backend (the upload, `graph.upload`) and the device
merge of `batch_semantics::set()` read the out-edges as an exact CSR; given slotted rows they
would read the wrong graph without an error. PLAN 5.1's sentence "every algorithm accepts every
preset" concerns the batch presets of compact rows (ADR 0010); AD11 makes it "every preset its
layout supports".

## Decision

### 1. One central check

`detail::expect_compact_layout(const graph_properties&, std::string_view consumer)`
(`cpp/src/graph/graph_impl.hpp`, defined in `graph.cpp`) raises `not_supported_error` for any
layout but `row_layout::compact`, naming the consumer, the graph's layout, and the fix (copy the
edges into a compact graph: `graph::from_csr(res, g.to_csr(res).view(), props)` with
`props.layout = row_layout::compact`). It is the only implementation of the refusal; it is called:

- by `graph_access::expect_placement()`, the placement check every graph algorithm already makes
  at its entries (so the scaffold's algorithms of `scripts/new_algorithm.py` inherit the guard);
  label_propagation calls the new `graph_access::expect_placement_any_layout()` instead;
- first, before the other requirement checks, in the requirement functions of `sssp`, `mosp`,
  `cycle_count` and `triangle_delta` (their `expect_graph` / `expect_requirements`, which run
  before the placement check and would otherwise answer with a misleading
  `invalid_argument_error`, e.g. "needs the in-edges"), and as the first statement of the
  update entries of `sssp`, `mosp` and `cycle_count` (which check the result's version before the
  placement), so that `compute`, `update`, `from_arrays` and `dyng::update()` over any of them
  refuse a slotted graph with the named error before anything changes;
- by the upload of the compact device copy (`graph_impl::device_edges`, `graph.upload`) and by the
  device merge of the set semantics (unreachable for slotted rows, whose semantics is
  `net_effect()`; the check documents and enforces it).

Nothing else in those algorithms changes: one call per entry, and `@throws not_supported_error`
documentation in `sssp.hpp`, `mosp.hpp`, `cycle_count.hpp`, `dynamic_bfs.hpp`,
`triangle_delta.hpp` and `update.hpp` (documentation only; `triad_count.hpp` and `hypergraph/**`
are not touched: triad_count takes hypergraphs). The tests are `cpp/tests/graph/
compact_guard_test.cpp` (every entry, the composition, and the graph unchanged) and
`slotted_test.cpp` (the device copy).

### 2. The backend sentence (AD19)

**An algorithm accepts the backends its manifest lists; any other raises not_supported_error
naming them.** This is the rule label_propagation follows in 0.4 (its manifest lists sequential,
and cuda from M10a step 4; OpenMP resources raise `not_supported_error` naming `sequential` and
`cuda` and the fix `dyng.Resources("sequential")`), and M11's `hyper_sssp` cites it. It is not an
amendment of ADR 0010 (whose line 54 concerns the batch presets, not the backends).

### 3. What the guarded algorithms compile (for step 5's re-reading)

The readings of section 4 time code compiled from these sources; step 5 re-reads every region
whose compiled code changes after this step (any change of these files, or of the code placement
of libdyng, which every change of the library can move):

- `sssp`: `cpp/src/algorithms/sssp/{sssp.cpp, sequential.cpp, openmp.cpp, cuda.cu, operators.cu}`
  with `{problem.hpp, fused.cuh, kernels.cuh, operators.cuh}`;
- `mosp`: `cpp/src/algorithms/mosp/{mosp.cpp, sequential.cpp, openmp.cpp, cuda.cu}`, `problem.hpp`,
  and `sssp`'s engines;
- `cycle_count`: `cpp/src/algorithms/cycle_count/*.{cpp,cu}` and its headers;
- shared, read by all three: `cpp/src/graph/` (the container: `graph.cpp`, `apply_host.cpp`,
  `apply_set_host.cpp`, `apply_set_device.cu`, `device_graph.cu`, `graph_impl.hpp`,
  `normalized_batch.hpp`, `device_graph.hpp`, `slotted_rows.hpp`), `cpp/src/framework/`
  (`composition.cpp` and its headers), `cpp/src/operators/`, `cpp/src/core/`, `cpp/src/util/`,
  and the public headers `cpp/include/dyng/{core,graph}/*` and `update.hpp`.

Step 3 changed, of these: `graph.cpp`, `apply_host.cpp`, `apply_set_host.cpp`, `device_graph.cu`
(float instantiations), `apply_slotted_host.cpp` (new), `graph_impl.hpp`, `normalized_batch.hpp`,
`apply_common.hpp`, `composition.cpp` (the Step 0 of `net_effect` and the float instantiation), the
graph headers (appended fields and members), and one guard call in `sssp.cpp`, `mosp.cpp`,
`cycle_count.cpp` and `triangle_delta.cpp`.

### 4. The ADR 0024 reading (B0 = `b2b1524` against the final library of this step)

Each algorithm and backend in its own exclusive window of at most 2 h, logged through
`dyng-work/tools/locklog.py`; bar 1.02 on every gated region; a sub-region above the bar from code
placement only takes the D14 path (GOVERNANCE.md, open-decision sub-block `M10 (AD)`); a whole
gated region above 1.02 is a STOP unless it comes from code placement only.

**Reading** (2026-10-29, 36 windows; `parity/results/M10a.md` section 10.5, records
`parity/results/M10a-ab-b0-*.json`): `b2b1524` against `2fa0f27`, both `parity-cuda`, with the
step's command forms (17 heap layouts, 51 rounds, 34 for road_usa and cycle_count; 85 layouts /
170 rounds for the OpenMP local10k updates) and rule 3's longer readings (170 rounds on OpenMP,
102 on CUDA). Every gated region of mosp (CUDA and OpenMP), of sssp CUDA and of cycle_count's
kernels and updates is within 1.02 (`compute` 0.946-1.010, `sosp_update` 0.942-1.017 by ADR 0024's
mode rule, sssp CUDA `apply` 0.952-1.020, every `end_to_end` of sssp and mosp 0.980-1.008). Above
the bar by 0.1 %: sssp OpenMP roadNet-PA safe50k `apply` 1.021 (170 rounds) and cycle_count CUDA
Twitch `update_end_to_end` 1.021 / `static_end_to_end` 1.020 (102 rounds). Both are located
stage by stage (`graph.apply` and `graph.transpose`; `graph.build`, +26 ms) to functions whose
machine code is identical in both libraries at other alignments, and the same source pair reads
0.79-0.99 when built without CUDA and 0.98-1.00 with the JCC-erratum mitigation flags on both
sides (`M10a-ab-b0-diagnosis.json`): code placement only. They take the D14 path (listed for the
author in `M10 (AD)`); nothing was changed to move them (no alignment flag, the decision of
2026-10-08 and docs/roadmap.md).

**Reading on the final M10a library** (M10a step 5, 2026-11-02 / 03, 29 windows;
`parity/results/M10a.md` section 12.9, records `parity/results/M10a-ab-final-*.json`):
`b2b1524` against `72eb020` (the sources of steps 4 and 5 included: `graph_impl.hpp`,
`graph.cpp`, `apply_slotted_host.cpp`, `slotted_rows.hpp`), the same command forms and rule-3
readings. mosp (CUDA and OpenMP) and cycle_count CUDA are within 1.02 everywhere (cycle_count's
Twitch excursions now 1.002 / 1.001). **Six whole gated regions of sssp are above the bar:**
OpenMP `apply` roadNet-PA 1.025 / 1.026 and roadNet-CA 1.026 / 1.035 (safe50k / unsafe50k, 170
rounds), CUDA `apply` roadNet-PA 1.026 / 1.033 and `end_to_end` roadNet-PA unsafe50k 1.021 (102
rounds). Located (`M10a-ab-final-diagnosis.json`): the `apply` regions' functions are
instruction for instruction identical at other alignments and read 0.957-0.987 with the
JCC-erratum mitigation on both sides (OpenMP; on CUDA `apply` 1.002-1.006, `end_to_end`
1.009-1.014): code placement; half of the CUDA `end_to_end` excursion is the CUDA context's
creation, about 5 ms longer because
`CUDA_MODULE_LOADING=EAGER` also loads label_propagation's kernels: a cost of the larger library,
not placement. This is a STOP of M10a step 5 for the author (GOVERNANCE.md, Open decisions,
`M10 (AD)`); nothing was changed to move them.

## Consequences

- A slotted graph cannot be passed to any algorithm but label_propagation by mistake; the error
  names the fix.
- New algorithms get the guard through `expect_placement()` and opt in to slotted rows explicitly.

## Alternatives considered

- **A per-problem trait read by the framework** (e.g. `accepts_slotted_rows`). Rejected for 0.4:
  the framework headers are included by triad_count, whose object files must not change (AD11),
  and the participant interface is in the frozen `update.hpp`.
- **Teaching the 0.1-0.3 algorithms to read slotted rows.** Out of scope for 0.4 (AD11).
