# Testing oracles

Reference solvers and checkers used by the tests, the parity harness and downstream users (`dyng::testing`).

`triad_reference` (`dyng/testing/triad_reference.hpp`) counts the triads of a hypergraph by brute
force, the oracle of `triad_count`: the 30-bin h-motif counts and StatHyper's types 1-3 over the
triangles (and adjacent pairs) of the line graph it rebuilds, following ESCHER-GPU's CPU reference
with 64-bit counts. It derives the h-motif classes itself, as the orbits of the seven Venn regions
under the permutations of the three hyperedges, numbered in ESCHER's order by a documented list of
representatives; bins 20-25 (the open triads) are always 0. Like `rebuild_line_graph` and
`cbst_host_model` it shares no code with what it checks (it includes no other dynG header, and a
test checks its object file), so it throws `std::invalid_argument` / `std::bad_alloc`.

The oracles of `label_propagation` (0.4, ADR 0068) follow the same rule (standard library only;
an include-graph and link test checks their object files):

- `harmonic_solution` (`dyng/testing/harmonic_solution.hpp`): the converged harmonic labels F* of
  a graph with seed labels, by Jacobi-preconditioned conjugate gradients in double to a relative
  residual of 1e-12 (at most 100,000 iterations), on the components that contain a seed (the
  others are counted as excluded); its reductions sum fixed chunks in a fixed order, so the result
  does not depend on the thread count. `label_error_against()` gives the mean and maximum
  \|F - F*\| and the class flips, `class_agreement_with()` the flips beyond a margin.
- `union_find` (`dyng/testing/union_find.hpp`): disjoint sets whose root is the smallest member.
- `lp_host_model` (`dyng/testing/lp_host_model.hpp`): an exact host model of vertex-arrival
  batches on fixed slot rows (single-use ids, the batch's net effect, the original's flags) and of
  `label_propagation`'s deterministic stages: the affected set V_aff, the sparsified subgraph G',
  its components in the original's numbering, the 64-bit supernode weights and the initial
  labels. It replays LabelPropagation-CUDA's recorded traces field for field.

`hyper_sssp_reference`, `check_hyper_sssp_tree` and `line_graph_delta`
(`dyng/testing/hyper_sssp_reference.hpp`, M11a, ADR 0082) are the oracle of `hyper_sssp`,
following MOSP_ESCHER's test oracle: the line graph rebuilt from the hyperedges, a node-weighted
Dijkstra (stepping into a hyperedge costs its weight) from a source hyperedge at distance 0 or a
source vertex (every hyperedge holding it starts at its own weight), the canonical parents (the
lowest-id tight neighbour), a check that counts distance and parent differences apart, and the net
line-graph change between two states of a hypergraph. They share no code with `hyper_sssp` or the
container either (an include-graph and link test, with a planted violation that must fail it), and
their results are standard-library types.

```{doxygengroup} testing
:content-only:
:members:
```
