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

```{doxygengroup} testing
:content-only:
:members:
```
