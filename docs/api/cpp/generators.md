# Generators

Seeded generators of graphs, hypergraphs and batches (`dyng::generators`).

- `generators/hypergraphs.hpp`: `uniform_hypergraph()` and `random_hyperedge_batch()` with the
  batch mixes of the conformance kit (insertions, deletions, delete-then-reuse within a batch,
  heavy deletion, delete-all, re-insert, a batch larger than the hypergraph, duplicate members,
  dead-id deletions, a skewed hyperedge of 1,100 members, ids at the index limits, the empty
  batch). Their distributions are the library's own (`std::mt19937_64` and dynG's uniform integer
  and shuffle algorithms), so a stream is bit-exact per (seed, major version) with every compiler
  and standard library; golden hashes in the tests pin it.
- `generators::legacy` reproduces the random streams of the original research tools bit for bit,
  so the inputs of published experiments and of the parity harness can be regenerated without the
  original code: MOSP's change generator (`mosp_changes`), CycleEnumeration-GPU's batches
  (`cycle_enum_batch`), and ESCHER-GPU's synthetic hypergraph and batch generator
  (`escher_hypergraph`, `escher_batch_generator`, which reads the live state of a dynG hypergraph,
  or of an `escher_host_model` the caller keeps in step with it, and shares one `std::mt19937_64`
  with it in ESCHER-GPU's draw order; the host model draws the same batches without reading the
  hypergraph, so a timed driver keeps no public device view alive across `apply()`). These streams follow
  libstdc++'s algorithms, which dynG reproduces itself; `standard_library_matches_legacy_streams()`
  checks at run time that the build's standard library draws the same (true with libstdc++ 11 and
  later, which built the originals).

```{doxygengroup} generators
:content-only:
:members:
```
