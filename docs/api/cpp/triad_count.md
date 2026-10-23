# triad_count

hypergraph h-motif triad counts. The algorithm page is {doc}`../../algorithms/triad_count`.

The header is frozen and **stable from 0.3.0** (maturity `stable`, the author's decision of
2026-10-22, GOVERNANCE.md; {doc}`../../adr/0060-triad-count-api-review-and-freeze-0-3`): a
removed or changed declaration needs the `api-change` label and a CHANGELOG entry. Its extension
point, a pattern of your own counted by the same engine, is the {doc}`experimental` API
(`count_local_patterns`), which stays experimental: it may change in any minor release.

```{doxygengroup} triad_count
:content-only:
:members:
```
