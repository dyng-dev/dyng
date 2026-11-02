# ADR 0091: hyper_sssp's as-is port and its internal virtual-endpoint mode

- **Status:** Accepted. The parity rule and the engines are the author's: HD2 (by_id through an
  internal virtual-endpoint mode with no public name; by_id_public; by_key; the coverage per
  backend), HD13 (the as-is kernels are `engine::fused`) and HD14 (sequential and cuda only),
  **decided by the author on 2026-10-28** (GOVERNANCE.md, approvals log, "The parity rules of 0.4
  (sheet C1-C8)" and "Licensing, data provenance, budgets and scope of 0.4"). The technical parts
  (the transplant, its rename map and hunks, the mechanical classes, the host port, the framework
  bend, the replay harness and its checks) are accepted under delegation (GOVERNANCE.md,
  "Delegation of technical ADRs"). It carries out M11a step 4 (PLAN 4.5, 4.8, 6.3 steps 5-7, 8.3;
  the M11a acceptance criteria 13, 14, 15 (the algorithm), 16 (sites #1 and #2) and 17 (the
  algorithm's parts)).
- **Date:** 2026-10-29
- **Deciders:** S M Shovan (HD2, HD13, HD14, 2026-10-28); the AI assistant, on the author's
  behalf, for the technical parts (M11a, step hyper-sssp-sequential-asis-cuda)
- **Builds on:** ADR 0038, 0042 and 0089 (the transplant method), ADR 0080 (the parity modes and
  the region map), ADR 0081 (the dumps and the golden set), ADR 0082 (the oracle), ADR 0086 and
  0089 (the slack_csr store and its delta), ADR 0090 (the API), ADR 0092 (the budget).

## Context

MOSP_ESCHER@4b86159 (B) runs H-SOSP in two engines over its slack CSR line graph: the CUDA engine
(`hsosp/src/hsospDevice.cu`: `hsospRecompute`, a pull / collect loop one warp per candidate, and
`hsospUpdate`, the packed `(distance << 32 | parent)` words with `atomicMin`, the invalidation of
the deleted tree edges' subtrees by pointer jumping, the pulls of the invalidated nodes, the
relaxations of the inserted pairs and push rounds under a work budget) and its host emulation
(`hypergraph/src/HostHypergraph.cpp:365-507`, `emulateSospRecompute` / `emulateSospUpdate`, the
oracle of its own tests). Both run from a virtual source row {s} (B id 1) and carry a virtual
target row {t} (B id m + 2), both of weight 0. dynG's public API takes a source vertex and has
no virtual rows (ADR 0090).

## Decision

### 1. The framework first: a fake, and one bend

Before any port code, a fixed-point fake on slack_csr storage (`test_hyper_hops`,
`cpp/tests/framework/fake_hyper_fixed_point.hpp`: hop counts and canonical parents from a source
hyperedge as Tier A hooks, one loop call per push round) runs through `update_enactor` on every
backend, with hyperedge batches, incidence edits, two results of different sources in one
`dyng::update()` and the three `on_limit` actions (six tests). It showed one bend: H-SOSP's budget
counts relaxations, which the iteration cap cannot express. The framework gains
`over_budget(ctx)` (ADR 0092 section 2); nothing else changed.

### 2. The sequential backend: B's host emulation as it is (Tier A)

`sequential.cpp` ports `emulateSospRecompute` and `emulateSospUpdate` with 0-based ids: the
roots and the children-list invalidation (`identify_affected`), the pulls of the invalidated
hyperedges and the inserted pairs' relaxations (`seed`), B's priority queue of (distance, parent)
keys with the lowest-id tie rule drained in one `loop` call, the recompute's pull rounds
(`recompute`, `compute()`). Its scratch is pooled per hypergraph (`hyper_sssp_workspace`), so an
update after the first allocates nothing (conformance check C8).

### 3. The cuda backend: B's engine transplanted (Tier B, `engine::fused`)

`parity/escher_port/transplant.py --map parity/escher_port/rename_map_hyper_sssp.toml` generates
`state.cuh`, `kernels.cuh` and `engine.cu` from B's `hsosp.cuh` and `hsospDevice.cu` (whole-token
renames to snake_case, the namespace `dyng::detail::hsosp`, the provenance header,
clang-format) and applies `manual_hunks_hyper_sssp.patch`: the device graph as the engine reads
it (the slack_csr store's rows, weights and last delta: B's `DeviceH2H` / `DeviceDelta` as views;
`zeroDegreeKernel` dropped, since the store's CSR apply zeroes the dead degrees, ADR 0089), the
zero-weight ids of the virtual mode at every weight read (`node_weight()`), the public mode's base
offers (section 4), a 64-bit `max_iterations`, the fallback reason in B's order of its tests, and
`budget_action::error`. `--check --rev 4537dd1` regenerates the files identically. Then one
mechanical class per commit, the cuda replay of make test exact by id after each:

| Commit | Class |
|---|---|
| `4537dd1` | c1: the transplant, `cuda.cu` (dynG's driver: reserve with growth, recompute, update, download / upload, the source flags), engine::fused on cuda |
| `96cdade` | c2: errors (`DYNG_CUDA_TRY`; the sticky device error word) |
| `54e4bcf` | c3: streams from the resources (the state's stream; asynchronous copies) |
| `e954584` | c4: memory (`dyng::buffer` owners instead of `cudaMalloc`, the raw pointers as aliases) |
| `f90b1d8` | c5: the stages (identify_affected, seed, loop, recompute on every update, finalize), the host synchronizations through one helper that counts them (I9), the host lists in the pooled workspace |

The kernels' machine code against B's (`parity/hsosp_perf.py sass`, `M11a-hsosp-sass.json`):
registers, stack and shared memory equal on every shared kernel but `pull_invalidated` (36
registers against 34: the source flags); the instruction streams equal on 10 of 14, the other
four (both pulls, `relax_inserted`, `push`) longer by 8-16 instructions, the zero-weight compare
and the base offer.

### 4. The two modes (HD2)

- **Public** (a source vertex s): every live hyperedge holding s is offered its own weight with
  parent -1 in its pulls (B's source row's offer, which wins every tie: -1 sorts first), a
  hyperedge that lost s while its parent was -1 is a root of the invalidation, one that gained s
  is offered its weight; the flags are kept per batch from the applied record.
- **Virtual-endpoint** (`detail::hyper_sssp_virtual`, `virtual_endpoints.hpp`; no public name):
  the source is a hyperedge at distance 0, never relaxed, and up to two ids weigh 0 inside
  hyper_sssp (B's {s} and {t} rows, held with weight 1 by the container: dynG id = B id - 1), plus
  B's recompute block size and a recompute into an existing result (the bench's static row). The
  update, the budget and the engines are the public algorithm's.

The test `ThePublicModeEqualsTheVirtualModeOnHPlusTheSourceRow` shows the two equal on H with a
source row added, on both backends.

### 5. The replay of the results (`--part results`)

`cpp/tests/parity/hsosp/hsosp_results.{hpp,cpp}` (`dyng_hsosp_replay --algorithm`,
`parity/compare.py hyper_sssp --part results`) replays every session of the golden set whole:
the container loaded and every batch converted as the by-id container replay does it (ADR 0089),
each batch applied by the update of the results B updates after it, each recompute B recorded a
recompute into the result's own arrays, each with B's UpdateConfig of that call. by_id compares
distances and parents with B's exactly (the canonical encodings of ADR 0080; digests outside
P50); by_id_public compares distances on the real hyperedges, counting the parents whose B
parent is a virtual row; by_key loads the rows as a user would and resolves B's ids by content:
each B id of a batch's insertions is mapped to the dynG id of the same position in the batch,
the counts must be equal, and the mapped dynG row must hold the sorted vertex set and the weight
of B's row in the dump (a mapping that pairs two different hyperedges is a mismatch). In both
public modes every call is also checked by the oracle itself (`dyng::testing::
check_hyper_sssp_tree` from the source vertex on the container's public `h2v()` and `weights()`):
dynG's distances and parents against `hyper_sssp_reference`, which shares no code with the store
or the engine. (Amended by the M11a review: the first version re-derived the canonical rule
inline from the slack_csr store's host model and mapped by_key's ids by position without checking
their content.) A replay also fails on an `apply_batch` without its `begin_batch` (a lost or
reordered record; the session stops), on a call of B's it could not follow (`not_checked`), and,
in digest mode, on a dynG distance or parent past B's bound that is not INF / -1 (the canonical
encodings span max(bound_B, bound_dynG) ids); and on cuda by_id, a make-test session whose
fallback decisions differ from B's in a repeat as well (HD7: those sessions lie far from the
budget; elsewhere the decision is tabulated, not a parity target). `--check-compute` compares every update with a fresh
`compute()`, `--repeat` and `--same-as` compare the digests of runs and backends. The comparator
catches a planted wrong distance or parent, a reordered batch and a dropped batch (its tests).
`dyng-compat-hsosp replay --algorithm` and `bench` (the update and static rows of hsospBench, the
44-column CSV, the `[e2e]` line) run the same code.

### 6. Results

All 0 mismatches (`parity/results/M11a.md` section 10): by_id on cuda over the whole set (25
cases, 685 updates, 211 recomputes, 212 million ids; 75 fallbacks against B's 75, none differing)
and on make test with update == compute; by_id_public on cuda over T and P50 and on sequential
over T; by_key on both over T and dblp_vertex_50000; by_id on sequential over T (update ==
compute) and the three P50 configurations of HD2, with digests equal to cuda's; ten cuda runs of
dblp_vertex_50000 with identical digests.

## Consequences

- The as-is engine is the cuda default (HD13); M11b's shared engine is checked against it.
- The virtual mode is a test and compat surface only: it stays in `detail`.
- `transplant.py --check` is part of the step's checks; a change of the transplanted files goes
  through the map, the hunks or a later class commit.
