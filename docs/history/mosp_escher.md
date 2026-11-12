# MOSP_ESCHER

MOSP_ESCHER (the repository `escher-mosp`) joined the two sibling projects of the author's
doctoral work, the ESCHER CBST store and the MOSP CUDA update, into one build tree, and added
**H-SOSP** (IA3 2026; `dyng.citation("hyper_sssp")`, key `hsosp2026`): single-source shortest paths
on a weighted dynamic hypergraph, through its line graph, under hyperedge and incidence batches.
dynG's `hyper_sssp`, the `slack_csr` storage of the `hypergraph` container with its incidence
edits are ports of it, and the second CUDA engine of `sssp` (0.2) follows the structure of its
MOSP half.

| | |
|---|---|
| Repository | <https://github.com/SMShovan/MOSP_ESCHER> |
| Pinned commit | `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (2026-09-26) |
| Paper snapshot | tag `baseline-2026-09` = `80cee62` |
| History | 51 commits, 2026-04 to 2026-09, all by S M Shovan; 50 AI-assisted (H-SOSP and the 2026 fixes) |
| Ported in | M7 (0.2: the host loop of kernels of its MOSP half, `mosp/src/sospUpdateGpu.cu`, is the structure of sssp's `engine::operators`, with MOSP-CUDA@e220ee2's semantics; ADR 0026), M8a (its CBST store, merged with ESCHER-GPU's, the `mosp_escher()` preset), M11a (the reference side, the `slack_csr` storage and its incidence edits, `hyper_sssp` on sequential and the as-is CUDA engine), M11b (the shared sssp engine, Python, the command line, the freeze) |
| Parity | the store's operation traces against both originals ([M8a](https://github.com/dyng-dev/dyng/blob/main/parity/results/M8a.md)); the ids, rows and line-graph pairs of every make-test session and paper configuration replayed whole; distances and parents by id after every update and recompute of the 25-case golden set `hyper_sssp_mosp_escher`, on sequential and both CUDA engines ([M11a](https://github.com/dyng-dev/dyng/blob/main/parity/results/M11a.md), [M11b](https://github.com/dyng-dev/dyng/blob/main/parity/results/M11b.md)) |
| Gates | against MOSP_ESCHER re-measured in the same session (16 paper configurations, 21 rounds): the update stage 0.99-1.05x of its time, the static recompute 0.98-1.04x, the maintenance 0.61-1.03x, end to end 0.65-0.79x, the peak of live device allocations 0.48-0.56x, the graph scope 1.00x ({doc}`../algorithms/hyper_sssp`, section 5) |

The code was a CUDA research prototype: the ESCHER CBST core (`escher/`) as the authoritative
store of the hypergraph's h2v, v2h and h2h relations, a host model (`hypergraph/HostHypergraph`),
the H-SOSP device graph and update (`hsosp/`) and a benchmark driver (`hsospBench`). Its 2026
fixes (the branch `fix/correctness-perf`) gave it `make test`, an independent oracle
(`HypergraphOracle.cpp`: the line graph rebuilt from the incidence lists, Dijkstra, the canonical
tree), CBST content checks, a real-hypergraph mode with the paper's batch model, and stage timers
in a 44-column CSV, which dynG's parity harness builds on.

## What was ported, from which files

| MOSP_ESCHER@4b86159 | dynG |
|---|---|
| `hsosp/src/hsospDevice.cu` (the kernels, `hsospRecompute`, `hsospUpdate`), `hsosp/include/hsosp.cuh` (`HsospState`, `UpdateConfig`) | `hyper_sssp::compute` / `update` on cuda, `engine::fused`: the kernels transplanted as they are (`parity/escher_port/transplant.py`, ADR 0091); `engine::operators`: the same steps on sssp's CUDA operators (ADR 0095) |
| `hsosp/src/hsospDelta.cu` (`DeviceIncidence`, `DeviceH2H`, `deriveDelta`, `applyDeltaToDevice`) | the `slack_csr` storage on cuda: the incidence mirror, the slack CSR line graph and the order-free pair delta inside the commit (ADR 0089) |
| `hypergraph/src/HostHypergraph.cpp` (`applyBatch`, `emulateSospRecompute`, `emulateSospUpdate`) | the host `slack_csr` store and its incidence edits (ADRs 0085, 0086); `hyper_sssp`'s sequential backend |
| `hypergraph/src/HypergraphOracle.cpp` | `dyng::testing::hyper_sssp_reference`, `check_hyper_sssp_tree`, `line_graph_delta` (ADR 0082) |
| `escher/` (the CBST core, `DynamicHypergraph`) | the ESCHER store's `mosp_escher()` preset (M8a) and the id allocator of `slack_csr` storage (best fit, fills and unfills; ADR 0085) |
| `hsosp/bench/hsospBench.cu`, its loader and batch model | `dyng-compat-hsosp to-dgb`, `replay`, `bench` (the parity driver, `tools/compat/hsosp`); `io::hypergraph_read_options::hsosp()`; the batches replayed from the golden set (HD5) |
| `mosp/src/sospUpdateGpu.cu` (the multi-kernel host loop of its MOSP half) | the structure of `sssp`'s `engine::operators` (0.2, ADR 0026; the semantics are MOSP-CUDA@e220ee2's) |

Not ported: the ESCHER CBSTs of the H-SOSP path (nothing on the shortest-path path reads them;
`slack_csr` storage keeps none), the virtual source and target rows (internal to the parity
harness's virtual-endpoint mode), the synthetic H-SOSP generators (`HypergraphGen`; after 0.4,
HD5), the experiment scripts and figures, and the SLURM wrappers.

## What was fixed in the original before the port

Recorded in the original's `CHANGES.md` (E: the ESCHER core, M: the MOSP half, S: H-SOSP, P:
performance), on coauth-DBLP and coauth-MAG-Geology, against `baseline-2026-09`:

- **E1-E9** the ESCHER core's defects, ported from ESCHER-GPU's fixes (the 32-bit CBST rank that
  overflowed above 65,535 records, stale subtree counts after insert, row terminators, erased keys,
  metadata lost on rebuild, truncated best-fit inserts, a wrong tail occupancy after unfill) and
  two in its adapters; at the paper's scale the original crashed or corrupted the h2v / v2h / h2h
  contents.
- **M1, M2** the MOSP half was an unfixed copy of MOSP-CUDA with its count-to-infinity defect; the
  fixed update was ported.
- **S1** the H-SOSP update counted to infinity after deletions, so every realistic batch ran its
  512-iteration cap and then a full recompute; it is now exact and incremental (invalidation,
  pull, push with a packed atomicMin; a budget falls back to the recompute, timed). **S2** the
  benchmark's check compared two results of the same kernels; every batch is now checked against
  the independent oracle. **S3** weights must lie in [1, 2^28]. **S4** device parents read with the
  wrong base. **S5** fallback iterations, a rebuild outside its timer and other processes' memory
  in the reported numbers. Operations on the virtual hyperedges and an empty generator pool were
  fixed too.
- **P1-P5** unification (the line-graph delta) on the GPU from an incidence mirror, device-side
  grouping and a warp-per-row CSR apply, warp-cooperative pull kernels (the static recompute 4.1x
  faster, which also lowers the reported speedups), the ESCHER maintenance and the host costs: on
  DBLP the dynamic time per 50K hyperedge batch went from 13.4 s to 0.17 s.

The wording of the items the paper's figures depend on (S1, S3, S4, S5, P3) and the speedups that
do not reproduce (recompute / dynamic 0.08-1.28 on the fixed code, against 1.3-12.1 in the paper)
is "Paper vs fixed code" on the {doc}`../algorithms/hyper_sssp` page (HD19).

## What differs in dynG

- **A source vertex, not a virtual row.** The public API takes a source vertex; the hyperedges
  holding it start at their own weight. MOSP_ESCHER's virtual source and target rows exist only in
  the parity harness's internal mode, which reproduces its results by id.
- **No CBST on the shortest-path path.** `slack_csr` storage keeps the incidence and the slack line
  graph and no ESCHER CBST, so the paper's dynamic total, which includes their maintenance, is not
  comparable stage by stage; the gates compare the stages both sides have.
- **Two CUDA engines with the same bytes.** The as-is kernels (the default, by the gates) and the
  shared sssp engine (selectable; ADR 0095).
- **Every backend's results are canonical and equal**, the host engine included; the work budget is
  also checked on the host, and `budget_action::error` raises instead of falling back.
- **Vertex distances** are the minimum of D(h) over a vertex's hyperedges (HD11), computed on
  request.
- **Ids.** Hyperedge ids and parents are 0-based (MOSP_ESCHER's host ids are 1-based, its device
  nodes 0-based: S4 came from that mismatch).

The mapping of every name is section 9 of {doc}`../algorithms/hyper_sssp`.
