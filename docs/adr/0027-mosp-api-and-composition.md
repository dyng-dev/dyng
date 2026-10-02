# ADR 0027: The mosp API (0.2) and mosp as a composition of K sssp problems

- **Status:** Accepted under delegation (2026-09-30; GOVERNANCE.md, "Delegation of technical
  ADRs"). It freezes the reviewed sketch `docs/design/sketches/mosp.md` (M3, ADR 0023) with the
  changes below, and records three additive changes to frozen 0.1 headers and one behaviour
  relaxation of `sssp::update()`. It does not change a rule the author approved: the PLAN 8.6
  gates and their protocol (ADRs 0018, 0021), the parity rules and the sssp gate's measurement
  (the compat driver's default mode) are untouched; no 0.1 signature changes.
- **Date:** 2026-09-30
- **Deciders:** the AI assistant, on the author's behalf (M7, step mosp-core)

## Context

PLAN 6.4.4 ports MOSP's multi-objective update: K SOSP updates on one graph (MOSP-CUDA@e220ee2
`mospUpdate.cu`, MOSP-OpenMP@c352151 `mospUpdate.cpp`), the combined graph of the K trees and its
SOSP tree (`combinedGraphGpu.cu`, `combinedGraphCpu.cpp`), and the path costs (`mospPathCosts`).
The M3 sketch fixed the public shape and left three questions open (the path costs on the device,
`preferences` as a vector, the combined graph's storage). PLAN 4.5.2 makes mosp "plain C++
composition": K `sssp` problems over objective views plus a static `sssp` over the combined graph.

## Decision

1. **Public API** (`cpp/include/dyng/mosp.hpp`) as sketched: `options{preferences, delta,
   cuda_engine, compute_path_costs, validate_inputs, num_objectives}`, `stats : update_stats
   {batch, objectives, combined_edges, preference_scale}`, `result<vertex_t, distance_t>` with
   `distances(k)`, `parents(k)`, `combined_distances()`, `combined_parents()`, `path_costs()`,
   `preference_scale()`, `num_objectives()`, `from_arrays()` over K views, `compute(res, g,
   source, opt)`, `update(res, g, batch, r)`, `update_traits` for `dyng::update()`, and the
   constants `max_objectives = 64`, `max_preference_scale = 2^20`. Changes from the sketch:
   - `options::num_objectives` (appended): the first K weight columns, 0 = all (MOSP's
     `MospOptions::numberOfObjectives` and the `mosp -k` flag). Fixed at compute().
   - `options::delta` is the width of the K updates only; the combined solve uses the automatic
     width of the combined graph, as both originals do (`combinedGraphSosp*(..., delta = 0)`).
   - `path_costs()` covers the K objectives (MOSP's `mospPathCosts` covers every weight column of
     the graph; `dyng-compat-mosp -k K` writes the other columns from
     `testing::mosp_path_costs_reference()` to keep the original's file) and is host memory on
     every backend in 0.2 (sketch question 1: the host first, as the originals; the device later).
     It throws while `compute_path_costs` is false.
   - `stats::affected` counts the vertices whose combined distance or MOSP parent changed (the
     result keeps the previous MOSP tree: a second pair of combined arrays, swapped after each
     solve; the originals have no such counter).
   - `preferences` stays a `std::vector<int32_t>` (sketch question 2): the Python binding converts
     a sequence; no ABI concern for a C++17 source library.
2. **Composition.** `detail::mosp_problem` is the participant of one mosp result in
   `run_update()`: it owns K `problem_participant<sssp_problem>` (each with its own update
   enactor, the `sssp.*` stages nested in `mosp.objective`), forwards Step 0 and the normalized
   batch, and runs the finalize step after them. So one commit serves the K objectives, as in
   `mospUpdate()`, the K updates share the handle's one sssp workspace (ADR 0015), and
   `dyng::update(res, g, b, paths, other)` composes mosp with any result (conformance C10). Its
   budget (invariant I9) is the sum of the K sssp budgets plus the finalize step's
   synchronizations (cuda: the combined graph's size, the combined solve's, the end of the step).
3. **The combined solve through the framework.** `sssp_solve_view()` (internal, sssp) runs
   sssp's static enactor on a graph view that is not a container (the combined CSR, in pooled
   workspaces: sketch question 3, no allocation per update) into caller-owned arrays, with the
   backend's engine (sospFromScratch*), the packing bound `L (K + 1)` as in the originals. The
   sequential engine's distance-only parent recovery reads in-edges, so the sequential combine
   also builds the combined graph's in-edge CSR when the packing does not fit (the OpenMP and
   CUDA engines recover over out-edges).
4. **Additions to frozen 0.1 headers** (additive, reviewed here): `io::write_path_costs()` in
   `<dyng/io/result_io.hpp>` (MOSP's `mospCosts.txt` format, the `writeCosts()` of the `mosp`
   driver; the compat driver and the CLI need it); `#include <dyng/mosp.hpp>` in the umbrella
   `<dyng/dyng.hpp>`; the new `<dyng/testing/mosp_oracle.hpp>` (`combined_graph_reference()`,
   `mosp_path_costs_reference()`, PLAN 5.8). `<dyng/mosp.hpp>` joins the frozen list of
   `ci/api_snapshot.py` (the sketch said "frozen in M7").
5. **sssp::update() accepts a batch without insertions whatever its `num_weights`** (as
   `graph::apply()` always did). Before, an empty or deletion-only batch built with the default
   one weight column was rejected on a multi-column graph; mosp's objectives (and the kit's C10
   with a K = 3 partner) send such batches. A behaviour relaxation, no signature change.
6. **Conformance kit** (test code): `test_traits::num_weights` (optional, default 1) gives the
   kit's weighted graphs K columns (column 0 random, the others a fixed function of the edge and
   column 0, so a re-weighting raises some columns and lowers others); C7's weight-count check
   uses K + 1 columns; C10 builds its graph with the larger column count of the pair.
7. **Compat driver.** `dyng-compat-mosp --mosp [--pref p1,..,pK] [--no-path-costs]` is the clone
   of the whole `mosp` driver (combinedGraph/ outputs, `comb` report line, `RESULT compute_ms=`).
   Without `--mosp` the tool keeps its SOSP-only behaviour, so the sssp gates' timed regions and
   their measurement (`parity/timed_regions/sssp.toml`, `perf_ab.py`) are unchanged.
   `parity/compare.py` replays the corpus with `--mosp` and compares `combined/` as well.

## Consequences

- Byte parity of every MOSP output file with both originals on the 495-case corpus (default
  preferences) and on the committed preference cases (`cpp/tests/data/mosp_combined`, generated
  by `parity/fixtures/mosp/make_mosp_fixtures.sh` after MOSP-OpenMP and MOSP-CUDA agreed).
- Memory per result: K trees, two pairs of combined arrays, the host path costs; the combined
  CSR (up to K n edges) is pooled scratch.
- The "(a) compute" region of the gates maps to the `mosp.objective` samples (sssp's engine stages)
  plus `mosp.combine`, `mosp.combined_sssp` and `mosp.finalize` (the `affected` count is dynG's
  work inside the region); `mosp.path_costs` belongs to "(b) end to end", as the original computes
  the costs while writing.

## Amendment (M7, step mosp-parity-perf, 2026-10-01)

Technical, no public signature and no gate rule changed (recorded in
`docs/developer/retrospectives/M7.md`, step 3):

- **Parallel path costs.** On the openmp and cuda backends the path costs run on the handle's
  host threads (`mosp_path_costs_openmp`, `cpp/src/algorithms/mosp/openmp.cpp`): the children
  lists by atomic counts, a prefix sum and atomic fills, then a level-synchronous breadth-first
  traversal of the MOSP tree inside one parallel region. The values are those of `mospPathCosts`
  (sums along each tree path); when a tree edge is missing, the sequential traversal runs again so
  that the reported vertex is the same. The sequential backend keeps the sequential port. Reason:
  the sequential pass took 1.5-2.9 s on road_usa, inside `update()`, where the originals compute
  the costs while writing `mospCosts.txt`, concurrently with the other output files.
- **The tree's download is timed with the path costs.** On cuda, `mosp.finalize` is the `affected`
  count only; the download of the MOSP tree for the host's path costs moved into
  `mosp.path_costs` (one more synchronization when `compute_path_costs` is true), so the
  "(a) compute" region (above) holds what the originals' Steps 2-3 hold.
- **`affected` without a second MOSP tree on the device** (the device-memory gate, PLAN 8.6:
  <= 1.05x). The second pair of combined arrays kept for the `affected` count (12 bytes per
  vertex) put mosp's peak device memory at 1.068-1.075x of MOSP-CUDA's on the three road graphs
  (`parity/results/M7.md`, section 8). On cuda the combined solve now overwrites the previous MOSP
  tree in place and counts the changed vertices in its unpack pass, as an update counts its
  `affected` (`sssp_solve_view(..., count_changes)`: the fused kernel's `count_changes`
  parameter, which an update always sets, and the operators engine's counted unpack); the count
  kernel and one synchronization of `mosp.finalize` are gone (an update with
  `compute_path_costs` synchronizes K + 3 times with the fused engine). The host backends keep
  the second pair (host memory has no gate; their engines write the arrays during the search).
  Same values: `affected` is the count of the same comparison.

## Alternatives considered

- **A framework "composite problem" type** (problem_base with child problems): no second user
  yet (rule of two); the participant interface already composes.
- **Path costs over every weight column** (exactly MOSP's): mixes columns that are not
  objectives into the result; the compat driver reproduces the file instead.
- **Rebuilding the combined graph incrementally** from the K trees' changes: not in the originals;
  the full rebuild is O(K n) and a small part of the region (PLAN 6.4.4 watch item is the combine's
  atomics, kept warp-aggregated as in MOSP-CUDA).
