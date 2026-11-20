# Input and output

Readers and writers of the formats in {doc}`../file_formats`: MOSP text CSR, MOSP batches, Matrix Market, edge lists, distances and trees; the batch files `.dgt` (graph and hypergraph batches) and `.dgb` (binary, streamed by `batch_reader`); hypergraphs (`.hg`, with the preprocessing of ESCHER-GPU and of MOSP_ESCHER's H-SOSP as the option sets `hypergraph_read_options::escher_gpu()` and `hsosp()`).

The io headers are tracked by the API baseline, not frozen as a whole; the formats of dynG's own
are a stable tier of their own ({doc}`../file_formats`). Inside them, `io::write_labels` /
`read_labels` and the `.dgb` codes of 0.4 (`batch_dtype::float32`,
`batch_file_flags::vertex_records`) are frozen by declaration since the end of M10b (ADR 0109)
and **stable from 0.4.0** (the author's decision of 2026-11-18 on AD15, GOVERNANCE.md).

```{doxygengroup} io
:content-only:
:members:
```
