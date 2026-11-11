# ADR 0063: The M10 plan of record (label_propagation, 0.4)

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs") for its technical sequencing. The structure it records is the author's: AD1 and the
  sheet's A2 and A3, **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, the
  row "The 0.4.0 plan: the decision sheet of M10 and M11, and the structure (sheet A1-A3)"). The
  ownership of the carried items and the ADR blocks are the orchestrator's delegated choices of
  2026-10-27 (the reconciliation of the M10 and M11 plans).
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  reference-side-baseline-calibration)

## Context

M10 ports `label_propagation` (DynLP, ICS 2026) from LabelPropagation-CUDA@a276a3a (O) for
0.4.0 (PLAN 6.4.7, 11.4; Appendix F moved M10 from 0.3 to 0.4). M11 (`slack_csr` and
`hyper_sssp`) is ported in the same release, in parallel. The author accepted every default of
the 0.4 decision sheet on 2026-10-28 (AD1-AD21 for M10); the rows of that date in GOVERNANCE.md's
approvals log record each decision by its AD id, and this milestone's ADRs cite them.

## Decision

### 1. The split and the parity order (AD1)

M10 is two pull requests, both in 0.4.0:

- **M10a** (branch `m10a-label-propagation`): parity and the gates against O. Its five steps, in
  this order, each building on the evidence of the one before:
  1. `reference-side-baseline-calibration`: O's side before any dynG code: the reference copies
     and the link-time recorder twin (ADR 0065), the timed-region map and protocol (ADR 0064),
     the O-only harness and the pre-port baseline of both schedules, the frozen in_place
     calibration (AD5), the golden sets exported twice, O's own `--check` margins (AD7) and the
     build-flags go/no-go (ADR 0066). Nothing under `cpp/src/algorithms/label_propagation/`,
     `cpp/include/dyng/label_propagation.hpp`, `cpp/src/graph/` or `cpp/include/dyng/graph/` is
     written in this step; the region map, the baseline records and the calibration record with
     its thresholds come first in the branch's log.
  2. `oracles-generator-writers`: the host-only `dyng::testing` oracles (`harmonic_solution`,
     `union_find`, `lp_host_model`) checked against the goldens, and the byte-identical legacy
     generator (AD8).
  3. `host-container-sequential-driver`: the float slotted graphs with vertex batches under
     `net_effect` on the host, the central compact-only guard (AD11), the sequential backend and
     `dyng-compat-dynlp`.
  4. `cuda-device-apply-fused-engine-parity`: the device apply, the verbatim CUDA fused engine
     and IrLP, damped byte parity (AD6), the calibrated in_place cloud (AD5) and the single-stream
     port of O's transfer/compute overlap (AD21).
  5. `kit-mutations-gates-certificate`: the conformance kit, the mutations (AD9), the
     sanitizers, the PLAN 8.6 gates (AD2-AD4) and the 0.4 certificate plumbing.
- **M10b** (a new branch from origin/main after the author merges M10a): the public surfaces and
  the freeze: the operators engine within c x eps (AD6), `row_layout::slack` and the streaming
  mode, the library vertex-arrival generator, the formats (AD13), Python, the command line, the
  kNN recipe (AD18), the docs, the API review and the freeze.

Parity with O is the top bar: a divergence is accepted only as an explicit, documented
comparator rule with an ADR (C8), never as a tolerated mismatch.

### 2. The base, the tracks and the ADR blocks (sheet A2, A3)

- **Base:** `m10a-label-propagation` was created from origin/main at `b2b1524`, the merge of
  P040 (the 0.4.0.dev0 post-release pull request with the empty anchor blocks in the shared
  files), after 0.3.0's release (main `20bc2d0`, the merge of PR #18 "Release 0.3.0rc1", + R031,
  0.3.0 final, + P040). Every later merge of origin/main into the branch is a merge commit (never
  a rebase), made at the latest before step 5's final gate readings, and is recorded in
  `docs/developer/retrospectives/M10a.md`.
- **Two parallel tracks:** M10 in `/home/sskg8/Projects/dyng-work/wt-m10`, M11 in
  `.../wt-m11` (branch `m11a-hyper-sssp`, expected to merge first). They share only the shared
  files, the formats, one calendar of the exclusive perf lock in windows of at most 2 h and the
  GPU-1 quiet windows, both taken and logged only through `dyng-work/tools/locklog.py` (R1).
- **ADR numbers:** M10 uses 0063-0079, then 0100-0109; M11 uses 0080-0099; 0110-0119 are for the
  release line (P040, any 0.3.x fix, R040; P040 takes 0110 if it needs one). The rows of
  `docs/adr/README.md` stay in numeric order: the second track to merge inserts its rows between
  their neighbours.
- **The shared files** (P040's anchors; the orchestrator's reconciliation rule 2 for the fixed
  positions where P040 has no slot): M10's entries go only at M10's anchors (`# M10
  label_propagation`), never a second `GPU_LOCAL_STEPS[(0, 4)]` key, one entry per line in
  `EQUIVALENCE_TARGETS` / `EQUIVALENCE_FILES`, the CHANGELOG entries only in `### M10:
  \`label_propagation\` (0.4)` (scripts/new_algorithm.py runs with `--no-changelog`), the open
  decisions only in GOVERNANCE.md's sub-block `M10 (AD)`; M11's entries and anchors are never
  edited; only label_propagation's `[[planned]]` block of `cpp/src/algorithms/planned.toml` is
  deleted; a generated file is never merged by hand (`scripts/regen.py`, the stub generator and
  `ci/api_snapshot.py` reproduce it after every merge of main).

### 3. The carried items: all owned by M11

PLAN 4.7.4 and the author's instruction of 2026-10-23 ("each with its own gate reading") carry
three items into 0.4; the orchestrator gave all three to M11 (2026-10-27):

1. `operators::openmp_exec::for_each`: M11a step 2, commit 1;
2. `escher::flatten_rows`: M11a step 2, commit 2;
3. the D13 class in sssp's OpenMP regions (including `list_gather`'s `omp single`): M11b step 1;

each with its own reading there. M10 changes none of them: the branch's own commits never touch
`cpp/src/operators/execution.hpp`, `cpp/src/hypergraph/escher/flatten.*`,
`cpp/src/algorithms/sssp/openmp.cpp`, `cpp/src/util/thread_list.hpp` or
`cpp/src/util/list_gather.hpp` (`git log --no-merges b2b1524..HEAD --not origin/main -- <those
paths>` is empty; what arrives through a merge of origin/main does not count). If one of them
would need an edit for M10, M10 stops and reports it.

### 4. Joint items with M11 (delegated, 2026-10-27)

- `perf_ab` region keys: M10a step 1 adds `first_batch` and `sum_gate` (ADR 0064), M11a step 1 a
  list-valued `original` and the `e2e:` keys, each in its own function; the second merger keeps
  both.
- `perf_ab.py mosp --baseline-exe/--baseline-label/--layouts`: M10a step 3 (M11b step 1 reuses
  it; if an equivalent copy reaches main first, the second merger keeps one).
- The AD20 sentence of `core/stats.hpp` (the last paragraph of the `update_stats` class comment)
  and the AD19 backend-acceptance sentence: M10a step 3 (M11a step 4 writes the identical stats
  sentence if M10a has not merged, then one copy is kept).
- The `.dgb` plumbing: M11a step 2 defines `known_graph` / `known_hypergraph` and flag bit 3;
  M10b step 2 adds bit 4 `vertex_records` and `batch_dtype::float32` (AD13).

### 5. What stays with the author

Only (GOVERNANCE.md, Open decisions, sub-block `M10 (AD)`): AD15 / F1, the stability
confirmation at the 0.4.0 release; acknowledgements the D14 path routes to the author (a
dynG-against-dynG excursion above 1.02 from code placement only); and the protocol-routed flags
the steps define: a GPU-0 calibration cross-check outside the in_place thresholds, a public name
beyond AD12's list, a new question in an author-reserved area taken on its most conservative
option, and what a STOP raises. No decision of the 2026-10-28 sheet is listed there as pending.

## Consequences

- Each step's commits follow the order above; the retrospective records the base, every merge
  of main and the machine time of each part.
- M10b starts only after the author merges M10a, from origin/main.

## Alternatives considered

- **One pull request for M10** (the sheet's alternative to A1): not taken by the author.
- **Fully sequential tracks** (M11a, M11b, M10a, M10b): not taken by the author (about 3-4 weeks
  later).
