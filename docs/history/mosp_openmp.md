# MOSP-OpenMP

MOSP-OpenMP is the CPU (OpenMP) implementation of DynaMOSP, the dynamic multi-objective shortest
path update of the IPDPS 2025 paper and its IEEE TPDS 2025 journal version (S M Shovan et al.;
`dyng.citation("sssp")`). It updates one single-objective shortest-path tree (SOSP) per
objective after a batch of edge changes, then combines the K trees into a multi-objective tree.
dynG's `sssp` is its SOSP update; the multi-objective combination becomes `mosp` in 0.2.

| | |
|---|---|
| Repository | <https://github.com/SMShovan/MOSP-OpenMP> |
| Pinned commit | `c35215135341d5b5d1553458afe4b2226edc38fb` (branch `fix/correctness-perf`) |
| Paper snapshot | tag `baseline-2026-09` = `7284f50` ("Code Refactored For Modularity and Readability.") |
| History | 59 commits, 2023-09 to 2026-09, all by S M Shovan (31 AI-assisted) |
| Ported in | M1a (sequential and OpenMP backends, the MOSP graph semantics, the I/O), M5 (the `mospPrep` subcommands as `dyng prep`), M7 (the multi-objective combination: {doc}`../algorithms/mosp`) |
| Parity | byte-identical on the 495-case golden corpus, both host backends ([M1a certificate](https://github.com/dyng-dev/dyng/blob/main/parity/results/M1a.md), re-run on every later milestone) |

## What was ported, from which files

| MOSP-OpenMP@c352151 | dynG |
|---|---|
| `src/sospUpdateCpu.cpp`, `headers/sospUpdateCpu.h` (`sospUpdateCpu`, `sospFromScratchCpu`, `SospWorkspace`, `HostChanges`) | `sssp::update` / `sssp::compute` on `resources::openmp()` (`cpp/src/algorithms/sssp/openmp.cpp`), the workspace of ADR 0015 |
| `src/sequentialSOSPUpdate.cpp` (adapted) | `sssp` on `resources::sequential()` (`sequential.cpp`) |
| `headers/listGather.h` (`ListGather`) | `detail::list_gather` (`cpp/src/util/list_gather.hpp`) |
| `src/csrGraph.cpp`, `headers/csrGraph.h` (`applyChangeBatch`, `transposeCsrGraph`, the text readers and writers, `runConcurrently`) | `graph::apply` under `graph_properties::mosp_compatible()`, the transposition, `io::read_csr_triplet`, `io::read_legacy_batch`, the distance and tree writers |
| `src/changeGenerator.cpp`, `headers/changeGenerator.h` (`generateChangeBatch`, the modes, `--local`, `--safe`) | `generators::legacy::mosp_changes()`, `dyng generate mosp_changes`, `dyng prep changes` |
| `src/Dijkstra.cpp` (`dijkstraCsrGraph`), `src/validation.cpp` (`checkSospTree`) | `testing::dijkstra`, `testing::check_sssp_tree` |
| `src/mospUpdate.cpp` (`canonicalizeTree`, the per-objective loop) | `sssp::result::from_arrays(..., canonicalize)`, `dyng::update_each()`; the whole update: `mosp::update()` (M7) |
| `src/combinedGraphCpu.cpp`, `headers/combinedGraphCpu.h`, `headers/parallelCombinedGraph.h` (`combinedGraphSospCpu`, `preferenceScale`, `combinedEdgeWeight`, `mospPathCosts`) | the combine step of `mosp` on the OpenMP and sequential backends (`cpp/src/algorithms/mosp/`), `mosp::result::path_costs()`, `io::write_path_costs()` (M7) |
| `src/mospPrep.cpp` (`mtx2csr`, `widen`, `cache`, `changes`, `init`, `expected`) | `dyng prep ...` of the command line ({doc}`../api/cli`), byte-identical outputs |
| `src/mosp.cpp` (the `mosp` driver, per objective) | `tools/compat` `dyng-compat-mosp` (parity runs) and `dyng sssp update` |

## What was fixed in the original before the port

The paper was measured on `baseline-2026-09`. The branch `fix/correctness-perf` (recorded in the
original's `CHANGES.md`, with measurements in its `results/README.md`) fixed it; dynG ports the
fixed code:

- **M-a, count to infinity.** After a deleted tree edge, Step 1 gave the vertex its best
  *current* in-neighbour, which could be its own descendant: distances of reachable vertices
  could end too small (the stock OpenMP stress test failed intermittently; n = 6, seeds
  621705 / 250813: d(1) = 60 instead of 90), and a batch that disconnects vertices ran n rounds.
  Fixed by subtree invalidation and a monotone update, without the iteration cap and the BFS
  repair.
- **M-c, ties.** Parents of tied distances depended on the OpenMP schedule; the lowest-id parent
  is now chosen everywhere, so the trees equal Dijkstra's and MOSP-CUDA's byte for byte.
- **MP5**, the work-efficient engine (invalidation, pull, near-far push) that dynG's OpenMP
  backend ports; **H-M1 / H-M2**, the in-memory pipeline; **M-d**, the preference vector of the
  combined graph (for `mosp`, 0.2); **M-e**, the seeded change generator.
- **Packed format boundary** (a correctness fix of the input-validation work). The packed
  (distance, parent) words were sized for (n - 1) * maxWeight, but a relaxation forms a distance
  plus one more edge; at the boundary that candidate lost its top bits and a wrong small distance
  won (a path of 65,537 vertices with weights 2^31 - 1). The packing now requires n * maxWeight
  to fit, the counterpart of MOSP-CUDA's candidate-overflow fix. dynG's host backends keep the
  rule (they pack only when one more edge fits; `sssp::stats::packed_parents` reports whether
  the packed form was used).
- **M-f**, allocation and list merging in the original propagation loop (`ebf3674`): the
  candidate flags are allocated once and the thread lists merged at prefix-sum offsets
  (`ListGather`, which dynG ports as `detail::list_gather`) instead of under `omp critical`.
- Smaller fixes: `generateGraph` without `data/`, `runDijkstraCSR` on a graph whose batch deleted
  every edge, input validation of the loaders (weights in [1, 2^31 - 1], range-checked indices,
  complete tree files, a self-describing binary cache).

On the connectivity-safe 50K batch the fixed code is 7-33x faster per objective than the paper's
code built with `-O3` (the paper's own build used no optimization; `-O3` alone cuts its compute
by 2.4-2.6x); the table is on the {doc}`../algorithms/sssp` page ("Paper vs fixed code").

## What differs in dynG

- **The sequential backend** follows the tie rule of `sospUpdateCpu` (a parent changes only when
  its vertex is re-evaluated) instead of `sequentialSOSPUpdate`'s re-scan of all in-neighbours, so
  that every backend returns the same tree from any input tree (ADR 0006). From a canonical tree
  (every `compute()` result) all three agree with the original.
- **Barriers.** dynG's OpenMP rounds pass three barriers instead of six (`list_gather::gather_pair`),
  with the same lists and outputs; the per-thread lists keep their capacity in the workspace.
- **Errors** are exceptions with messages (`invalid_argument_error`, `io_error` with path and line)
  instead of `cerr` and exit codes; the drivers' strict argument checks are kept.
- **Not ported (yet):** the file-path wrapper `parallelCombinedGraph` (the combination itself is
  `mosp`, M7); the file-path API of the library (kept only in the parity adapters); about 1,400 tracked stress-test files, `html/`, `data/`, `output/` and the
  `tests/testCase0-9` directories (the goldens are regenerated by `parity/export_goldens.py`).

The mapping of every name is section 9 of {doc}`../algorithms/sssp`.
