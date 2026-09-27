# dyng 0.0.1 (name reservation)

**dynG: dynamic graph and hypergraph updates on GPUs** is a C++/CUDA library with Python
bindings for dynamic (batch-update) graph and hypergraph algorithms: it keeps the results of
algorithms such as single-source shortest paths, cycle counts, hypergraph triad counts and
label propagation up to date while the structure changes, without recomputing from scratch.

dynG is **under development**. This 0.0.1 release reserves the name `dyng` on PyPI; it contains
only a placeholder module:

```python
import dyng
print(dyng.__version__)   # 0.0.1
print(dyng.status())      # a short note on the project status
```

The first library release (0.1.0) will provide the CPU package `dyng` and, later, the CUDA
plugins `dyng-cu12` / `dyng-cu13`. Follow the project at https://github.com/dyng-dev/dyng.

License: Apache-2.0. dynG = dynamic graphs; inspired by Gunrock; not affiliated.
