# History

dynG grew out of the research codes of the author's doctoral work. Each algorithm of dynG is a
port of one of them, pinned at a commit and proved equal to it by the parity harness
({doc}`../developer/parity`). These pages record, for every original that is part of a release,
where it came from, which commit was ported, what was fixed in the original before the port (the
code the papers measured is not the code dynG matches), what dynG took from it and what differs.
The per-file record is the provenance header `// Derived from <repo>@<commit>:<path>` of every
ported file ({doc}`../developer/provenance`).

| Original | Algorithm in dynG | Pinned commit | Paper snapshot | Released in |
|---|---|---|---|---|
| {doc}`mosp_openmp` | `sssp` (sequential, OpenMP); later `mosp` | `c352151` | `baseline-2026-09` = `7284f50` | 0.1 |
| {doc}`mosp_cuda` | `sssp` (CUDA); later `mosp` | `e220ee2` | `baseline-2026-09` = `ac29545` | 0.1 |
| {doc}`cycle_enumeration_gpu` | `cycle_count` | `0a976ad` | `baseline-2026-09` = `da2067d` | 0.1 |
| ESCHER-GPU | the ESCHER hypergraph store, `triad_count` | `abcf9b2` | | 0.3 (planned) |
| MOSP_ESCHER | `hyper_sssp`, the sssp operators engine | `4b86159` | | 0.2 (the operators engine) / 0.4 (`hyper_sssp`, planned) |
| LabelPropagation-CUDA | `label_propagation` | `a276a3a` | | 0.4 (planned) |

Every original is public under `https://github.com/SMShovan/<repository>`. The originals carried
no license; the basis for publishing their code under Apache-2.0 is recorded in
{doc}`../developer/provenance` and `GOVERNANCE.md`.

```{toctree}
:maxdepth: 1

mosp_openmp
mosp_cuda
cycle_enumeration_gpu
```
