# How to cite

If you use dynG in academic work, please cite **the software** and **the paper behind each
algorithm you use**.

| If you use | Please also cite |
|---|---|
| `sssp`, `mosp` | DynaMOSP (IPDPS 2025) and its journal version (IEEE TPDS 2025) |
| `cycle_count` | TruCy / DynTruCy (IEEE Transactions on Computers, submitted) |
| `triad_count`, the ESCHER hypergraph storage | ESCHER (IPDPS 2026) and ESCHER+ (IEEE TKDE 2026) |
| `label_propagation` | DynLP (ACM ICS 2026) |
| `hyper_sssp` | H-SOSP (IA3 workshop at SC 2026) |

From C++, `dyng::citation("sssp")` returns the BibTeX entries of an algorithm and
`dyng::citation()` all of them. GitHub's "Cite this repository" button uses `CITATION.cff`. Once
releases are archived on Zenodo, each release gets a DOI.

## References

```{bibliography}
:all:
```

## BibTeX

The entries above come from `docs/references.bib`, the single source of citations:

```{literalinclude} references.bib
:language: bibtex
:start-at: "@software"
```

## CITATION.cff

```{literalinclude} ../CITATION.cff
:language: yaml
```
