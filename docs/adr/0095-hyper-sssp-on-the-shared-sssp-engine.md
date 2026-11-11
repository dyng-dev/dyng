# ADR 0095: hyper_sssp on the shared sssp engine (O12) and the engine the gates chose

- **Status:** Accepted under delegation (2026-11-10; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The rule that chooses the default engine is HD13, **decided by the
  author on 2026-10-28** (GOVERNANCE.md, approvals log, "Licensing, data provenance, budgets and
  scope of 0.4 (sheet G1-G3, H1-H3)", sheet H2): the shared sssp engine becomes `hyper_sssp`'s
  default only if it is byte-identical to the ported engine and within every gate; otherwise the
  as-is engine stays the default, the shared one stays selectable as `engine::operators`, and the
  reason and the numbers go on the algorithm page. The fallback decisions are tabulated, not a
  parity target (HD7, decided by the author on 2026-10-28); the dynG-against-dynG bar is ADR
  0024's 1.02 (HD16, decided by the author on 2026-10-28). It changes no rule the author
  approved: sssp's and mosp's headers, machine code and results, the gates and their protocol
  (ADRs 0018, 0021, 0024, 0080), the parity rules (HD2) and the budget semantics (HD8, ADR 0092)
  are untouched.
- **Date:** 2026-11-05 (the engine), 2026-11-10 (the readings and the decision)
- **Deciders:** the AI assistant, on the author's behalf (M11b, step shared-engine-node-weight)
- **Builds on:** ADR 0026 (sssp's CUDA operators engine), ADR 0029 (the packed words), ADR 0024
  (dynG against dynG), ADRs 0091-0093 (hyper_sssp's as-is port, its budget, its gates), PLAN
  4.5.3 (operators and the rule of two), 6.3 step 8, 6.4.8 and 12.2 O12.

## Context

PLAN 6.4.8 ports MOSP_ESCHER's H-SOSP kernels as they are first (M11a: `engine::fused`, ADR 0091)
and then replaces them "by the shared sssp engine with a node-weight functor and the undirected
(both directions) view, gated by parity and performance"; O12's default (accepted with the sheet
on 2026-10-28) is to move after the as-is port proves parity, keeping the budget as a policy.
sssp and mosp are stable from 0.1 / 0.2: their headers never change, and the step's goal is that
their machine code does not either.

## Decision

1. **Compile-time policies on sssp's operators engine** (`cpp/src/algorithms/sssp/operators.cuh`).
   The pull and the push read their graph through `graph_access<graph_t>`; sssp's `device_csr` is
   the default, written as today's code: compact rows `row_ptr[v] .. row_ptr[v + 1]`, the edge
   weight `weights[e]`, no base offers, no work counter, neighbour words read through L2
   (`__ldcg`), one thread per row. The policies a second graph type may set: slack rows
   (`util/slack_rows.cuh`, written once in util/ for every reader of slack rows on the device:
   M10b's slotted layout can reuse it), node weights (`w[col]` pushing u -> col, `w[v]` pulling
   into v), one symmetric view (the caller passes one view as `in` and `out`), per-node base
   offers (the pull also offers `(w(v), -1)`), a work counter in MOSP_ESCHER's units with the
   rows it counts (`counts_row`), plain neighbour loads (`peek`), and one warp per row
   (`pull_pass_warp`, `push_pass_warp`). The seeds (a single source, a source node at 0, base
   nodes) are the host loop's. sssp's kernels keep their names and parameters and call the
   generic bodies, which inline to the same instructions.
2. **Identity, proven and kept** (`parity/sssp_sass.py`): every kernel of sssp's and mosp's CUDA
   objects (the fused kernels, the 37 operators kernels, mosp's 11; every
   `DYNG_FOR_EACH_GRAPH_TYPE` instantiation) has the same instruction stream (only names
   normalized), resources and non-debug device sections as at the step's base (`3d331fa`, the
   library of `8e2f3df`), and every sssp and mosp object, host and CUDA, the same host
   instructions and relocations; only the line tables and the PTX text differ (they follow the
   source's line numbers). The normalization catches a planted one-instruction change (`plant`:
   one comparison of the pull, only the three pull kernels differ, with the same instruction
   count). sssp's fused engine, `sequential.cpp`, `openmp.cpp` and `cuda.cu` are untouched.
3. **hyper_sssp's `engine::operators`** (`cpp/src/algorithms/hyper_sssp/operators.{cuh,cu}`):
   sssp's kernels over `line_graph_view` (the slack_csr store's rows, node weights with the
   virtual mode's two zero-weight ids, the public mode's source flags, the work counter) and the
   host loop of sssp's operators engine (Tier A: identify_affected, seed, loop, finalize; the
   framework's `over_budget` then `recompute`). hyper_sssp's own kernels turn the store's pair
   delta and ids into `sssp_changes` (a deleted pair (a, b) -> a->b and b->a; a new, recreated or
   dead id v -> (parent[v] -> v), a root whatever its parent, the virtual source row never; a
   node of the public mode that lost the source -> (-1 -> v), a root exactly when its parent is
   -1) and relax the inserted pairs and the gained nodes after the pull (MOSP_ESCHER's
   relaxInsertedKernel on sssp's words; `seed_pairs`; pulling both endpoints as sssp's insertion
   heads is the other variant). The device arrays are the result's (as-is) device state; the
   as-is engine's work arrays serve as sssp's scratch (flag, ancestor, the candidate and frontier
   lists, the near and far lists, in_far), all zero or overwritten between runs of either engine;
   the stamps and their generation are shared (both only compare a stamp with a newer
   generation). The shared engine's own scratch is a control block per result and the update's
   lists in the pooled workspace (C8: a steady-state update allocates nothing).
4. **The budget, MOSP_ESCHER's triggers where they apply** (HD8): before each push round,
   `iterations >= max_iterations`, then relaxations `> work_budget x max(1, live entries)`,
   counted in MOSP_ESCHER's units (the row lengths of the invalidated nodes' pulls and of the
   pushed reachable nodes; not the inserted pairs' seeds, which MOSP_ESCHER does not count):
   with them the shared engine's fallback decisions equal B's on all 685 updates of T and P. The
   32-bit packed width of the as-is engine does not apply: sssp's packing holds n x the largest
   weight next to the parent bits; where it cannot (distance-only words), the update falls back
   with `distance_width` and compute() takes the ported recompute. An input distance that does
   not fit (pack_tree's overflow) is `distance_width`; a pointer jumping that does not end in
   ceil(log2 n) + 1 rounds is `invalidation_depth`. `budget_action::error` throws for the first
   two only, after the commit, as the as-is engine.
5. **The static solve on the shared engine** (`shared_compute`): compute(), the update's fallback
   and the bench's static recompute run sssp's near-far loop from the seeds with MOSP's
   defaultDelta over the line graph's entries (32 x the average entry weight / the average
   degree; the entry u -> v costs w(v)); an update keeps an unbounded width (every improved node
   joins the next round, MOSP_ESCHER's rounds), which reads faster on these line graphs. The
   ported recompute (hsospRecompute) stays for distance-only words.
6. **The internal knobs** (`hyper_sssp_operators_knobs`: `warp_rows`, `delta`, `seed_pairs`,
   `shared_compute`, `compute_delta`; `hyper_sssp_virtual::set_operators_knobs`, the tools'
   `--operators-knobs`) are not public. Their defaults are the variant the informal GPU-1 timings
   of the step chose (warp rows, unbounded update width, pair seeds, the shared compute with the
   default delta): the thread mapping read 2-3x slower on the DBLP line graph's rows, the
   insertion heads' pulls 3-4x slower in the seed, L2 neighbour loads and a capped warp grid
   about 15 % slower in the loop. Every variant was replayed by id on T and P (0 mismatches).
7. **The rule of two.** `invalidate_subtree` (the pointer jumping) and `packed_min` are now used by
   sssp and hyper_sssp; they stay in `cpp/src/algorithms/sssp/` instead of moving to
   `cpp/src/operators/`: the move renames every kernel (the namespace and the packing type are
   part of each kernel's mangled name), which changes sssp's and mosp's host objects and so would
   need the ADR 0024 and gate readings of criterion 2 for a rename. Recorded as a rename-only
   follow-up after 0.4.0 (docs/roadmap.md).
8. **The decision (HD13): the as-is engine stays the default.** The rule's first half holds:
   the shared engine is byte-identical to the as-is engine and to MOSP_ESCHER (by id on every
   batch of T and P, by_id_public on T + P50, by_key on T + dblp_vertex_50000; update == compute;
   C4 and C0-C12 on both engines; `parity/results/M11b.md` section 2.4) and within the memory
   gate (the nsys peak of live allocations 0.493-0.561 of B's on the three gated
   configurations, the graph scope 1.0000; section 2.7). Its second half does not: the full
   reading against unpatched MOSP_ESCHER@4b86159 (16 configurations, 21 alternating rounds,
   base clocks, M11a's protocol; section 2.6) has 4 of its gated rows above their bar, all
   `sosp_update` of coauth-DBLP's hyperedge batches where B's median is above 10 ms (bar 1.05):
   dblp_hyperedge_100000 b2 **1.051** and dblp_hyperedge_200000 b1-b3 **1.056, 1.061,
   1.059** (the as-is engine reads 1.001-1.024 on the same rows). Every other gated row is
   within: the shared static solve reads **0.690-0.740** of B's recompute (the as-is engine's
   0.981-1.036), Geology's `sosp_update` 0.781-0.994, `load_plus_dynamic` 0.641-0.810. So
   `engine::automatic` keeps `engine::fused` on cuda, `engine::operators` stays selectable
   with the same results, and the reason and the numbers are on the algorithm page (section 5).
   The cost is located (the stage timers of the two gate readings on dblp_hyperedge_200000):
   `hyper_sssp.identify_affected` +0.3-0.7 ms (sssp's pointer jumping runs ceil(log2 n) + 1
   bounded rounds where MOSP_ESCHER's stops when nothing changes), `hyper_sssp.seed` +0.3-0.9
   ms (the pull of the invalidated nodes and the inserted pairs' seeds as two launches over
   sssp's lists), the push loop +0.1-0.2 ms, on a 22-23 ms update. ADR 0024's reading of the
   shared engine against the as-is engine (17 layouts, 51 rounds, 102 where a row was above 1.02;
   section 2.8; reported, not a gate) says the same: `sosp_update` 1.056-1.065 on
   dblp_hyperedge_200000 (102 rounds), 0.981-1.022 on the other coauth-DBLP configurations,
   0.806-0.992 on Geology; `dynamic_comparable` 0.948-1.010.
   **The static solve.** HD13 keeps the ported recompute for static "if the shared compute
   reads slower"; it reads faster (0.69-0.74 of the ported recompute in ADR 0024's reading, from
   the near-far loop with MOSP's defaultDelta instead of MOSP_ESCHER's unbounded rounds). Since
   the rule chooses an engine whole, `engine::fused` keeps its ported recompute (compute(), the
   static baseline and the update's fallback) and `engine::operators` keeps the shared static
   solve; an as-is update with the shared static solve would be a third engine, which HD13 does
   not define (recorded as a follow-up after 0.4.0 in docs/roadmap.md, with the rename of
   decision 7 and the update's two located costs).
   **Not a STOP.** A gate miss of the shared engine is not one (criterion 3); sssp's and mosp's
   machine code is identical (decision 2), so no gate against MOSP-CUDA@e220ee2 or
   MOSP-OpenMP@c352151 can have moved (S7 does not arise). The ADR 0024 rows of the shared engine
   above 1.02 are not D14-path items: `engine::operators` is new code, not a change of stable
   code, and the default engine is unchanged.

## Consequences

- `engine::automatic` on cuda stays `engine::fused` (decision 8); `<dyng/hyper_sssp.hpp>`
  says so, and the shared engine is the documented alternative.
- `engine::operators` on cuda runs the shared engine (it threw `not_supported_error` in M11a);
  stats::engine_used reports it; the hand cases run every case on both engines, the kit runs
  C0-C12 a second time with `engine::operators` as its engine (the test-only tag
  `hyper_sssp_shared_engine`), C4 compares the two, and the committed subset's replay runs on
  both. The mutations #1 and #2 have twins in the shared engine (`DYNG_MUTATION_HSOSP_OPS_*`),
  and sssp's "skipped subtree invalidation" in the shared pointer jumping fails hyper_sssp's suite.
- The bench records the static recompute's nested stages as `static/<stage>` (the shared static
  solve opens the update's hook stages); the region map says so.
- `parity/results/M11b.md` section 2 holds the SASS record, the parity matrix, the readings, the
  gate table, the fallback table and the decision; the algorithm page holds the reason and the
  numbers.
