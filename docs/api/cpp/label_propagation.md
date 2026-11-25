# label_propagation

Binary harmonic label propagation under vertex batches (DynLP). The algorithm page is
{doc}`../../algorithms/label_propagation`; the containers it reads are the slot rows of
{doc}`graph` ({doc}`../../concepts/row_layouts`), its stream generators are in {doc}`generators`
and its oracles (`dyng::testing::harmonic_solution`, `union_find`, `lp_host_model`) in
{doc}`testing`.

The header is **frozen** since the end of M10b ({doc}`../../adr/0109-label-propagation-api-review-freeze-and-kernel-groups`)
and **stable from 0.4.0** (maturity `stable`; SemVer, PLAN 5.9; the author's decision of
2026-11-18 on AD15, GOVERNANCE.md): a removed or changed declaration needs the `api-change` label
and a CHANGELOG entry with a migration note, and fields, enumerators and functions are only
appended. The same holds for `<dyng/generators/vertex_arrival.hpp>` ({doc}`generators`). The
three oracle headers ({doc}`testing`) are frozen too, but **outside the stable tier**, as the
testing oracles of 0.1-0.3 are.

```{doxygengroup} label_propagation
:content-only:
:members:
```
