# Generators

Seeded generators of graphs and batches (`dyng::generators`). `generators::legacy` reproduces
the random streams of the original research tools bit for bit (today MOSP's change generator,
`legacy::mosp_changes`), so the inputs of published experiments and of the parity harness can
be regenerated without the original code.

```{doxygengroup} generators
:content-only:
:members:
```
