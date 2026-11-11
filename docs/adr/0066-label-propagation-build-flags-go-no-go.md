# ADR 0066: The build-flags go/no-go of label_propagation (dynG's FAST_MATH line against O's Makefile)

- **Status:** Accepted under delegation (2026-10-28; GOVERNANCE.md, "Delegation of technical
  ADRs"), technical scope only. It records the day-1 go/no-go of M10a step 1 (PLAN 6.4.7, 7.4,
  8.6; acceptance criterion 3) before any port code. It changes no flag: the parity-cuda preset
  (`CMakePresets.json`, `-O3 -lineinfo -fmad=true` for every CUDA TU, PLAN 7.4) and
  `dyng_add_algorithm(... FAST_MATH)` (`cmake/dyng_add_algorithm.cmake`: `--use_fast_math` on the
  target's CUDA sources only, PLAN 6.4.7 and 8.6) are as they were.
- **Date:** 2026-10-28
- **Deciders:** the AI assistant, on the author's behalf (M10a, step
  reference-side-baseline-calibration)

## Context

Damped byte parity of the fused engine (AD6, decided by the author on 2026-10-28) holds only if
dynG compiles O's kernels to the same machine code. PLAN 6.4.7 says the preset "pins
`--use_fast_math -O3` for this target and sets `-fmad` explicitly". The question for day 1: does
the nvcc line dynG generates for a FAST_MATH CUDA TU under parity-cuda produce the SASS O's
Makefile line produces, before a line of the port exists?

## Decision

1. **The two lines.** dynG's line was taken from `compile_commands.json` of a scratch configure
   (`cmake --preset parity-cuda`, tests off) of the branch's tree with one throw-away
   `dyng_add_algorithm(NAME zz_probe ... FAST_MATH)` target (deleted afterwards; never
   committed):

   ```
   /usr/local/cuda-13.1/bin/nvcc -forward-unknown-to-host-compiler -I<dyng includes>
     -isystem <cuda>/include -isystem <cuda>/include/cccl -O3 -lineinfo -fmad=true -std=c++17
     "--generate-code=arch=compute_86,code=[compute_86,sm_86]" -Xcompiler=-fPIC
     -Xcompiler=-Wall,-Wextra --extended-lambda --expt-relaxed-constexpr --use_fast_math -x cu -c
   ```

   O's line (its Makefile's defaults, `make -n -B obj/<tu>.o` in a fresh archive):

   ```
   /usr/local/cuda/bin/nvcc -std=c++17 -arch=sm_86 -Iheaders -lineinfo -Xcompiler -Wall
     -Xcompiler -fopenmp -O3 --use_fast_math -c src/<tu>.cu -o obj/<tu>.o
   ```

   `-fmad=true` is the preset's global flag and is implied by `--use_fast_math`; the target adds
   no per-target `-fmad` flag. Both nvcc are the same binary (`/usr/local/cuda` ->
   `cuda-13.1`).
2. **The comparison** (`parity/label_propagation_sass.py`): O's six pipeline TUs
   (`dynamicGraph`, `sparsifySubgraph`, `connectedComponents`, `labelInitialization`,
   `iterativePropagation`, `validation`) compiled from a fresh `git archive a276a3a` with both
   lines (dynG's with O's `-Iheaders` added and its source and object replaced), and every
   function of `cuobjdump -sass` compared with only the address comments removed (the strictest
   normalization), plus `cuobjdump -res-usage` (registers, stack, shared and local memory).
3. **Result (go):** all **38 functions** of the six objects are identical, instruction for
   instruction and in their resources: O's 26 kernel functions (the six append, status and
   ground-truth kernels of the dynamic graph, the three sparsification and mark kernels, the six
   Shiloach-Vishkin kernels, the supernode weights and label initialization, the seven
   propagation functions (the frontier initialization, the persistent small-frontier kernel, the
   block kernel, the group kernel for 8 and 32 lanes, the damped commit and the ground-truth
   enforcement) and IrLP's two) and the 12 CUB kernels they instantiate (the radix sort and the
   scan by key of the append, the scan of the components).
   Nothing had to be removed through target options. The negative control shows the comparison
   can fail: `iterativePropagation.cu` with dynG's line minus `--use_fast_math` differs in 4 of
   its 7 functions (the persistent small-frontier kernel, the block and the two group propagation
   kernels). Record: `parity/results/M10a-sass-go-no-go.json`.

## Consequences

- The port's CUDA TUs are built with the line above (FAST_MATH on `dyng_label_propagation` only);
  M10a step 4 checks it over `compile_commands.json` (`--use_fast_math` only on the LP TUs,
  `-fmad=true` present) and compares the hot-loop SASS of every ported kernel with O's.
- The other flags of dynG's line (`--extended-lambda`, `--expt-relaxed-constexpr`, `-fPIC`, the
  CCCL `-isystem`) do not change O's device code at CUDA 13.1; a toolkit change re-runs this
  comparison.

## Alternatives considered

- **A per-target `-fmad=true`.** Unnecessary: the preset passes it to every CUDA TU and
  `--use_fast_math` implies it.
- **Normalizing immediates and branch targets** (as `parity/triad_sass.py`'s digest does).
  Not needed: the strict comparison of the instruction text already holds, and it is the
  stronger proof.
