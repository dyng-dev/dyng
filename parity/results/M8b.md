# M8b parity record: the hypergraph container

Date: 2026-10-13 onwards. M8b puts the public `hypergraph<vertex_t, weight_t>` on the ESCHER CBST
store of M8a and proves its ids against ESCHER-GPU@abcf9b2 (A) and MOSP_ESCHER@4b86159 (B) at
the level of the container (PLAN 6.4.5, 8.3, 11.3; ADR 0041). Each step adds its section.

| Item | Value |
|---|---|
| Branch | `m8b-hypergraph`, created by the orchestrator from the head of the accepted M8a branch `m8a-escher-store` at `b45e685` (M8a's pull request #14 is open: `git merge-base --is-ancestor m8a-escher-store origin/main` fails, so the base is M8a's head, not `origin/main`; `origin/main` `30c3eef` (0.2.0 released, 0.3.0.dev0) is already merged into it at `889953a`). `origin/main` is merged again once the author merges M8a. |
| References | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A), MOSP_ESCHER `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (B, the committed tree only) |
| Inputs | M8a's CBST trace sets (`parity/goldens.toml`, `[sets.cbst_trace_*]`, files under `$DYNG_SCRATCH/goldens/cbst_trace`): their batch records (A: `escher_gpu.build` / `escher_gpu.update`; B: `mosp_escher.bulk_load` / `begin_batch` / `finish_batch`) are the dumps of the container-level replay |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86 |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1; GPU 1 for every untimed run |
