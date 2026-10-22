# experimental

**EXPERIMENTAL PUBLIC; may change in any minor release.** `dyng::experimental::count_local_patterns<op_t>`
counts a new hypergraph pattern, given as an Op, with `triad_count`'s wedge engine, and keeps the
counts exact under hyperedge batches (ADR {doc}`../../adr/0058-experimental-local-patterns`). The
tutorial is {doc}`../../tutorials/count_a_new_hypergraph_pattern`.

Two headers: `<dyng/experimental/local_patterns.hpp>` compiles with a host compiler (the shapes, the
rows an Op reads and the set helpers, the stats, `access_rows()`); `<dyng/experimental/local_patterns.cuh>`
needs nvcc (`count_local_patterns`, its result and the kernels compiled for an Op), the one kind of
public header that does (PLAN Section 4.1). The experimental tier is tracked by the API check, which
reports its changes without failing on them; the changes are listed in the changelog.

```{doxygengroup} experimental
:content-only:
:members:
```
