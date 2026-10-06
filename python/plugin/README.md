# dynG CUDA plugin (dyng-cu12, dyng-cu13)

This distribution adds the **CUDA backends** to [dynG](https://github.com/dyng-dev/dyng), the
library that keeps the results of graph algorithms (shortest paths, multi-objective shortest
paths, cycle counts) up to date while the graph changes in batches of edge insertions and
deletions. It is a plugin of the `dyng` package and is installed through its extras:

```bash
pip install "dyng[cu13]"    # NVIDIA driver for CUDA 13 (580 or newer)
pip install "dyng[cu12]"    # NVIDIA driver for CUDA 12 (525 or newer)
```

`dyng-cu12` is built with the CUDA 12 toolkit, `dyng-cu13` with CUDA 13. Each contains one
extension module with every backend of dynG (sequential, OpenMP and CUDA), compiled for the GPU
architectures from Turing (sm_75) on, with the CUDA runtime linked in: no CUDA toolkit is needed
at run time, only the NVIDIA driver. A plugin always has the same version as the `dyng` it
belongs to, and pip installs that version of `dyng` with it.

```python
import dyng

dyng.show_config()                       # which native module is active, and why
res = dyng.Resources.cuda(device=0)
g = dyng.Graph.from_edges([0, 0, 1, 2], [1, 2, 2, 3], [4, 1, 1, 5], resources=res)
tree = dyng.sssp.compute(g, source=0)    # runs on the GPU
```

You never import this package yourself: `import dyng` finds it through the entry point group
`dyng.backends` and uses it when a CUDA driver and a device are present; otherwise dynG uses its
CPU module and says why in `dyng.show_config()`. `dyng.use_cpu_only()` (before the first use) or
`DYNG_CPU_ONLY=1` forces the CPU module.

Documentation: <https://github.com/dyng-dev/dyng> (installation, troubleshooting, the API).
License: Apache-2.0 for dynG's own code; the wheel also contains third-party code (nanobind,
robin-map, the GCC runtime, the CUDA runtime and the CUDA C++ Core Libraries) under the licences
listed in its licence files (`THIRD_PARTY_LICENSES.txt`, `THIRD_PARTY_LICENSES_CUDA.txt`,
`NVIDIA_CUDA_EULA.txt`).
