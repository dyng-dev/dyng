# How-to guides

Short recipes for one task each. They assume you have built dynG
({doc}`../getting_started/install`).

```{toctree}
:maxdepth: 1

choose_a_backend
profile_an_update
read_and_write_files
run_parity
build_the_docs
add_an_algorithm
port_research_code
```

Planned, written with the code they describe: streams and memory (with an RMM pool), reproduce a
paper, use from PyTorch or CuPy (the CUDA plugins of 0.2; on the CPU, `torch.from_dlpack` of a
result array works today, {doc}`../api/python/index`), add an operator, add a backend, add a file
format, benchmark and read the performance gates, and debug CUDA (sanitizers, NVTX). The
{doc}`add_an_algorithm` outline becomes a full guide later; the release process is
{doc}`../developer/release`.
