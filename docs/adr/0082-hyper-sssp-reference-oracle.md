# ADR 0082: The hyper_sssp oracle (testing::hyper_sssp_reference, check_hyper_sssp_tree, line_graph_delta) and its go/no-go on MOSP_ESCHER's dumps

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. The three public names are the author's (HD12, decided by the
  author on 2026-10-28: `testing::{hyper_sssp_reference, check_hyper_sssp_tree,
  line_graph_delta}`); no other public name is added (the results are standard-library types).
  It carries out M11a step 1 (PLAN 5.8, 6.4.8 "Oracles in CI"; the M11a acceptance criterion 5).
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M11a, step
  regions-exporter-goldens-oracle)
- **Builds on:** ADR 0042 (`testing::rebuild_line_graph`, the same original algorithm, M8b),
  ADR 0049 (the isolated triad oracle), ADR 0080 (the canonical encodings), ADR 0081 (the dumps).

## Context

PLAN 6.4.8 asks for CI oracles of `hyper_sssp`: a line-graph rebuild, a Dijkstra and the canonical
tree, independent of the port, and PLAN 5.8 for oracles that share no code with what they check.
MOSP_ESCHER@4b86159 (B) has such an oracle, `hypergraph/src/HypergraphOracle.cpp` lines 25-171:
`rebuildLineGraph` (pairs per vertex, sorted, deduplicated), `lineGraphDelta` (the merge of two
line graphs), `referenceDistances` (a binary-heap Dijkstra where stepping into h costs w(h)) and
`checkSosp` (distances, then the canonical parent: the lowest-id tight neighbour). Before any port
code, the oracle must be shown to reproduce B on B's own data (go/no-go; STOP S2 otherwise).

## Decision

1. **The oracle** (`<dyng/testing/hyper_sssp_reference.hpp>`, `cpp/src/testing/
   hyper_sssp_reference.cpp`, derived from HypergraphOracle.cpp:25-171, with its provenance
   header): host only, standard library only, its own OBJECT library
   `dyng_testing_hyper_sssp_reference` inside `dyng::testing`.
   - `hyper_sssp_reference(h2v_offsets, h2v_members, num_vertices, weights, source,
     source_is_vertex = false)` returns `{distances, parents}` by 0-based id: the source is a
     hyperedge at distance 0 (B's virtual source row) or a vertex s, where every live hyperedge
     containing s starts at its own weight. Any non-negative weights, 0 included (B's virtual rows
     weigh 0). Dead (empty-row) and unreachable hyperedges are INT64_MAX / 4 (B's
     `HostHypergraph::INF`) with parent -1.
   - The canonical parent is the lowest-id live tight neighbour, D(p) + w(h) == D(h), as B's; the
     source has none, and with a vertex source a hyperedge reached from the source itself
     (it holds s and D(h) == w(h)) has none (-1): the public-mode counterpart of B's virtual
     source row, which has the lowest id and so always wins (HD2's by_id_public difference).
   - `check_hyper_sssp_tree(..., distances, parents)` returns `{distance mismatches, parent
     errors}`; parents are checked against the canonical parents of the REFERENCE distances, so a
     parent error counts even where the distances differ (B stops at the distances). Ids past the
     end of the given arrays count as INF / -1 (ADR 0080: a longer result must show its extra ids
     dead).
   - `line_graph_delta(pre, post, num_vertices)` returns `{deleted, inserted}`, sorted (a, b)
     pairs with a < b, from the two rebuilt line graphs (the id counts may differ).
   - A repeated member of a row counts once; no pair (h, h) is emitted. Malformed input throws
     `std::invalid_argument`, a distance reaching INF `std::overflow_error`. No OpenMP region: the
     oracle is single-threaded (coauth-DBLP's 250,891,568-entry line graph rebuilds in about a
     minute), so nothing can be thrown inside a parallel region.
2. **Isolation:** `hyper_sssp_reference_isolation` (CTest, `check_model_isolation.cmake`: the
   preprocessor's dependency list names no project file but its header; its object references no
   dynG symbol), so nothing of `cpp/src/algorithms/hyper_sssp`, of the container
   (`cpp/src/hypergraph`) or of the rest of dynG is included or linked. Two planted violations
   prove that the check fails, each for its own reason (the test passes only on the check's
   message): `hyper_sssp_reference_isolation_planted_include` (a source including
   `cpp/src/hypergraph/row_views.hpp`) and `..._planted_link` (an object referencing an undefined
   `dyng::detail` symbol). The check lives with its precedents (cbst_model_isolation,
   line_graph_oracle_isolation, triad_reference_isolation) in `cpp/tests`, run by `ci/check.sh`'s
   ctest step, rather than as a separate script under `ci/` (recorded in the retrospective).
3. **Tests** (`dyng_hyper_sssp_reference_tests`): B's forced-disconnection chain (both before and
   after the bridge is deleted), the vertex source, the tie-break, zero weights, dead rows,
   repeated members, the separate parent count, longer and shorter results, the line-graph delta,
   a brute force of each function over random hypergraphs (Bellman-Ford by definition; the
   adjacency by definition), and the refusals.
4. **The go/no-go** (`cpp/tests/parity/hsosp/hsosp_dump.{hpp,cpp}`: a host-only reader of the
   dumps of ADR 0081 and `check_with_oracle`; the tool `dyng_hsosp_oracle`; driven by
   `parity/hsosp_goldens.py oracle`): every session's hypergraph is rebuilt from its records (the
   loaded rows, then each batch's post-batch rows), and on every state the oracle (source = B's
   source row, B's weights, its virtual rows at 0) must reproduce the distances and parents of
   every recorded `hsospUpdate` and `hsospRecompute` by id, and `line_graph_delta(pre, post)` the
   deleted and inserted pairs of every batch, on every batch of T and of `dblp_vertex_50000` and
   `geology_hyperedge_50000`. A failure is STOP S2. The committed subset's test
   (`dyng_hsosp_dump_tests`) runs the same check in hosted CI and shows that a planted wrong
   distance, parent or pair fails it.

## Outcome (M11a step 1)

**GO**: on every batch of T and of `dblp_vertex_50000` and `geology_hyperedge_50000`, the oracle
reproduces all 788 recorded updates and recomputes of B by id (distances and parents, 0
differences) and `line_graph_delta` the pairs of all 728 batches (0 differences); the planted
changes of the committed subset's test are all detected (`parity/results/M11a.md` section 7).

## Consequences

- M11a's later steps check the port against this oracle (by_id_public and by_key, and the CI
  oracles of PLAN 6.4.8) and against B's goldens by id; the go/no-go shows the two agree on B's
  data before any port code exists.
- The results are `std::pair`s of standard containers, so the oracle adds exactly the three names
  of HD12.

## Alternatives considered

- **Reusing `testing::rebuild_line_graph`** (M8b). Rejected for the isolation rule: one oracle
  object includes no other project header; the rebuild is transliterated again (32-bit neighbour
  ids internally, to keep coauth-DBLP's line graph at 1 GB).
- **A result struct** (`hyper_sssp_tree_check`). Rejected: a public name outside HD12's list
  would be a naming item for the author.
- **B's parent rule unchanged for a vertex source** (the lowest-id tight neighbour only). It has
  no source row to win the tie, so a hyperedge holding the source could get a neighbour as its
  parent although it is reached from the source; -1 is the documented counterpart of B's virtual
  row.
