# label_propagation

Binary harmonic label propagation under vertex batches (DynLP). The algorithm page is
{doc}`../../algorithms/label_propagation`; the containers it reads are the slot rows of
{doc}`graph` ({doc}`../../concepts/row_layouts`), its stream generators are in {doc}`generators`
and its oracles (`dyng::testing::harmonic_solution`, `union_find`, `lp_host_model`) in
{doc}`testing`.

The header is **frozen** at the end of M10b ({doc}`../../adr/0109-label-propagation-api-review-freeze-and-kernel-groups`):
a removed or changed declaration needs the `api-change` label and a CHANGELOG entry, and fields,
enumerators and functions are only appended. Its maturity stays `experimental` until the author
confirms `stable` at the 0.4.0 release (AD15, GOVERNANCE.md, open decisions). The same holds for
`<dyng/generators/vertex_arrival.hpp>` and the three oracle headers.

```{doxygengroup} label_propagation
:content-only:
:members:
```
