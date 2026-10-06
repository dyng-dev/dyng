# Glossary

```{glossary}
aggregate delta
  A property family: a global count (cycles, triads), updated by one signed recount around the
  changed elements under an {term}`ownership rule`, with no iteration.

approval checkpoint
  A point (A1-A5, ADR 0014) where an irreversible or public action waits for the lead
  maintainer's explicit approval.

batch
  A set of structural changes applied at once: edge, vertex, hyperedge or incidence insertions
  and deletions, and weight changes.

batch semantics
  The rules for interpreting a batch (order, upsert, missing deletions, self-loops, vertex
  growth). They belong to the graph (ADR 0010).

compute / update
  The static solve of an algorithm (the baseline and the oracle) / the batch update of an
  existing result, which also applies the batch to the container.

enactor
  The driver that calls a {term}`problem`'s hooks in the fixed order of the update template.

family
  {term}`fixed point` or {term}`aggregate delta`.

fixed point
  A property family: a value per element, iterated until no value changes (or a tolerance is
  met).

frontier
  The set of elements whose property may change (the thesis's F).

fused engine (Tier B)
  A hand-fused kernel that replaces Steps 1b to finalize of the template, inside the same
  pipeline.

internal-stable
  Documented and change-controlled for in-tree contributors, but not covered by SemVer for
  outside users.

maturity
  `experimental`, `stable` or `deprecated`; decides the SemVer coverage of an algorithm.
  `tutorial` marks teaching material (`dynamic_bfs`, `triangle_delta`): complete and tested like
  any algorithm, covered like `experimental`, and not a research algorithm.

operators engine (Tier A)
  An algorithm composed from the framework's shared operators.

oracle kind
  What "correct" means for an algorithm: `compute` (a chain of updates equals the static solve)
  or `reference` (both are within a tolerance of a converged reference).

ownership rule
  "Count an item once, at its smallest changed member"; required by every signed recount.

parity
  Proof that a port gives the same results (and similar performance) as the pinned original.

parity certificate
  The committed record per release: commits, toolchain, golden hashes, the pass/fail matrix and
  the performance table.

problem
  An algorithm's instantiation of the update template: its data and hooks.

row order / multi-edges
  Graph properties: rows kept sorted or in append order; parallel edges forbidden or allowed.

snapshot (old view / new view)
  Phase-typed read views of G_t and G_{t+1}; aggregate algorithms subtract on the old view and
  add on the new one.

stale result
  A result whose graph version no longer matches the graph; using it throws
  `stale_result_error`.

walking skeleton
  The first milestone (M1a): the thinnest end-to-end slice (CPU `sssp`) proven against an
  original before general infrastructure is built.
```
