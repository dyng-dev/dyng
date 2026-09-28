# ADR 0003: Language and toolkit floors

- **Status:** Accepted (M1b)
- **Date:** 2026-09-27
- **Deciders:** S M Shovan (lead maintainer)

## Context

The CUDA backend arrives in M1b. Before the first `.cu` file lands, the project needs fixed
floors for the language, the CUDA toolkit, the GPU architectures, CMake and the CUDA C++ Core
Libraries (CCCL), and a fixed shape for the memory-resource interface, which appears in public
headers (`memory_resource_ref`, PLAN Section 4.7.2) and therefore cannot change after the 0.1 API
freeze. The development and performance machine has CUDA 13.1 (CCCL 3.1.4, checked in
`/usr/local/cuda-13.1/include/cccl/cuda/std/__cccl/version.h`) and two RTX A5000 (sm_86); the
wheels must also serve CUDA 12 users (PLAN Section 7.8).

## Decision

1. **C++17** for host and device code (`cxx_std_17`, `cuda_std_17`); tested also as C++20 in CI.
   Public headers compile with a plain host compiler and include no CUDA or CCCL header (the
   header self-containment test builds them with g++ in every preset).
2. **CUDA >= 12.4** for source builds (`cmake/cuda_architectures.cmake` refuses older toolkits);
   release wheels use the latest 12.x (>= 12.8, needed for sm_100 / sm_120) and the latest 13.x.
   The two latest CUDA majors are supported (PLAN Section 7.8).
3. **Architectures** (one place: `cmake/cuda_architectures.cmake`, set before any target):
   `native` for the development presets (`dev-cuda`, `sanitize-cuda`); the release list
   `75-real;80-real;86-real;89-real;90-real;100-real;120` for CUDA >= 12.8 and 13.x
   (`75-real;80-real;86-real;89-real;90` for 12.4-12.7) for `release-cuda` and the CI presets;
   `86` (sm_86 SASS + compute_86 PTX, MOSP-CUDA's `-arch=sm_86`) for `parity-cuda`. **sm_75 is
   the floor** on every toolkit (CUDA 13 dropped Volta and older; nobody has asked for sm_70 on
   CUDA 12). The last entry of each list embeds PTX for forward compatibility.
4. **CMake >= 3.30** (the oldest version CI tests with CUDA 13; the conda environment pins
   `>=3.30,<4.5`).
5. **CCCL**: the version shipped with the toolkit (CUDA 13 moved it to `include/cccl`; nvcc adds
   the path). No newer CCCL is pinned through CPM for now.
6. **Memory-resource shape** follows the CCCL 3.x concept `cuda::mr::resource` (checked against
   CCCL 3.1.4): `void* allocate(stream, bytes, alignment)`, `void deallocate(stream, ptr, bytes,
   alignment) noexcept`, `void* allocate_sync(bytes, alignment)`, `void deallocate_sync(ptr,
   bytes, alignment) noexcept`, equality comparison, plus dynG's `memory_space space() const`.
   Only the stream type differs (`dyng::stream_ref` vs `cuda::stream_ref`, both a `cudaStream_t`),
   so the adapters in both directions are one line per member
   (`cpp/src/util/cccl_memory_resource.cuh`; the test `CcclAdaptersRoundTrip` checks
   `cuda::mr::resource_with<..., cuda::mr::device_accessible>` and allocates through both).
   The built-in resources (`host_memory_resource`, `cuda_async_memory_resource`,
   `pinned_host_memory_resource`) are equality-comparable for the same reason.

## Consequences

- The CCCL 2.x of CUDA 12 still has the older shape (`allocate_async(bytes, alignment, stream)`):
  on CUDA 12 the CCCL adapters are compiled out (`DYNG_HAS_CCCL3_MEMORY_RESOURCE` = 0) and the
  test skips; dynG's own interface is the same on both majors.
- `cuda::mr::resource_ref` is still experimental in CCCL 3.1 (behind
  `LIBCUDACXX_ENABLE_EXPERIMENTAL_MEMORY_RESOURCE`); dynG does not use it. The RMM adapter
  (`DYNG_WITH_RMM`, later) is written against the RMM release it targets, whose resource-ref type
  has been moving to the CCCL 3.x names; this ADR is re-checked then.
- `cuda-build.yml` compiles the release list with CUDA 13.1.1 (the development toolkit), the
  latest 13.x and the latest 12.x; the 12.x build was also checked locally with the conda CUDA
  12.9 toolkit.
