# hyper_sssp

shortest hyperpaths. The algorithm page is {doc}`../../algorithms/hyper_sssp`; the tutorial is
{doc}`../../tutorials/shortest_hyperpaths`.

The header is frozen since the end of M11b ({doc}`../../adr/0096-hyper-sssp-api-review-and-freeze-0-4`;
maturity `stable` in its manifest): a removed or changed declaration needs the `api-change` label
and a CHANGELOG entry. It is **stable from 0.4.0** (SemVer, PLAN 5.9; the author's decision of
2026-11-18 on HD17, GOVERNANCE.md), with the `slack_csr` storage and the incidence edits of the
hypergraph it reads ({doc}`hypergraph`). Its oracle, `<dyng/testing/hyper_sssp_reference.hpp>`
({doc}`testing`), stays tracked, not frozen, outside the stable tier.

```{doxygengroup} hyper_sssp
:content-only:
:members:
```
