# M8a parity record: the ESCHER CBST store

Date: 2026-10-07 onwards. M8a ports the CBST store of ESCHER-GPU@abcf9b2 (A) and merges the
behaviours of MOSP_ESCHER@4b86159 (B) into it, proven by CBST trace replay against both
originals (PLAN 6.4.5). Each step adds its section; step 1 (refs-traces) writes sections 1-8.

| Item | Value |
|---|---|
| Branch | `m8a-escher-store`, from `main` at `8b9067ea437bdc35e08e7019f6adcef03fe670f7` (`8b9067e`), which contains M7 (`7d64c90`, pull request #8: `git merge-base --is-ancestor 7d64c90 origin/main`) and M6b (pull request #10: `origin/m6b-hardening` is an ancestor) |
| References | ESCHER-GPU `abcf9b2bd42f896ec3103a6835edf713ed5e416c` (A), MOSP_ESCHER `4b86159ec0b3bfe6e18c718ff4bef1494d7276a5` (B, the committed tree only) |
| Toolchain | GCC 12.2.0, CUDA 13.1 (`/usr/local/cuda-13.1/bin/nvcc`), sm_86 |
| Host | Xeon Gold 6258R (28 cores, 56 threads), 124 GB, 2x RTX A5000 (sm_86); Debian 12, Linux 6.1; GPU 1 for every run of this step (none is timed) |
