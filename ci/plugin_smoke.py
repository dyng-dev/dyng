#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The smoke test of an installed CUDA plugin wheel (PLAN 5.4, 7.7; ADRs 0030 to 0032).

Run with the Python of an environment where ``dyng`` and ``dyng-cu<N>`` are installed (from
their wheels), from a directory outside the source tree. Two expectations:

``--expect cuda`` (a machine with a driver and a visible GPU: ``ci/plugin_wheels.sh``)
    The plugin's module is the active native module (``dyng_cu<N>._core``), built with the static
    CUDA runtime; ``sssp`` (both engines), ``cycle_count`` and ``mosp`` run on
    ``Resources.cuda(0)`` and their arrays, read from device memory, equal the sequential
    backend's element by element.

``--expect fallback`` (no driver or no device: the GitHub-hosted runners of ``wheels.yml``)
    The first use of ``dyng`` issues exactly one ``dyng.BackendWarning`` and dynG runs on
    ``dyng._core``; the selection reports the plugin as ``unusable`` with a reason containing
    ``--reason`` (``"no CUDA driver"`` on a runner without the NVIDIA driver); the plugin's own
    extension module still imports without a driver (it needs neither ``libcuda`` nor
    ``libcudart``) and reports its build (plugin, toolkit, static runtime, architectures); and
    ``sssp`` / ``cycle_count`` / ``mosp`` run on the CPU backends.

Both check the installed versions (``dyng`` and the plugin equal ``--version`` when given), that
the module was built for exactly the release list of CUDA architectures of its toolkit (PLAN
7.7; ``ci/wheel_check.py``'s :data:`RELEASE_ARCHITECTURES`, tied to
``cmake/cuda_architectures.cmake`` by its self-test), and print ``dyng.show_config()``. Exit
status 0 on success; an AssertionError otherwise.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.metadata as md
import sys
import warnings


def _graphs(dyng, np, res):  # type: ignore[no-untyped-def]
    """sssp (both engines), cycle_count and mosp of a small random graph on ``res``."""
    rng = np.random.default_rng(7)
    n, m = 200, 1500
    src = rng.integers(0, n, m, dtype=np.int32)
    dst = rng.integers(0, n, m, dtype=np.int32)
    w = rng.integers(1, 20, m, dtype=np.int32)
    keep = src != dst
    src, dst, w = src[keep], dst[keep], w[keep]
    g = dyng.Graph.from_edges(src, dst, w, num_vertices=n, resources=res)
    t = dyng.sssp.compute(g, 0)
    t_ops = dyng.sssp.compute(g, 0, cuda_engine="operators")
    h = dyng.cycle_count.compute(
        dyng.Graph.from_edges(src, dst, num_vertices=n, resources=res), max_length=4
    )
    gm = dyng.Graph.from_edges(
        src,
        dst,
        np.stack([w, (w * 7) % 13 + 1], axis=1),
        num_vertices=n,
        properties="mosp_compatible",
        resources=res,
    )
    mo = dyng.mosp.compute(gm, 0)
    return t, t_ops, h, mo


def _same(np, a, b) -> None:  # type: ignore[no-untyped-def]
    t, t_ops, h, mo = a
    ts, ts_ops, hs, ms = b
    eq = np.array_equal
    assert eq(t.distances.to_numpy(), ts.distances.to_numpy()), "sssp distances"
    assert eq(t.parents.to_numpy(), ts.parents.to_numpy()), "sssp parents"
    assert eq(t_ops.distances.to_numpy(), ts_ops.distances.to_numpy()), "sssp (operators)"
    assert eq(h.counts.to_numpy(), hs.counts.to_numpy()) and h.total > 0, "cycle_count"
    for k in range(2):
        assert eq(mo.distances(k).to_numpy(), ms.distances(k).to_numpy()), f"mosp distances {k}"
        assert eq(mo.parents(k).to_numpy(), ms.parents(k).to_numpy()), f"mosp parents {k}"
    assert eq(mo.combined_parents.to_numpy(), ms.combined_parents.to_numpy()), "mosp combined"
    assert eq(mo.path_costs.to_numpy(), ms.path_costs.to_numpy()), "mosp path costs"


def _check_architectures(build: dict) -> None:  # type: ignore[type-arg]
    """The module's architectures are the release list of the toolkit that built it."""
    from wheel_check import release_architectures  # ci/, next to this script

    got = str(build["cuda_architectures"]).replace(",", ";")
    want = release_architectures(str(build["cuda_toolkit"]))
    assert got == want, (
        f"the module was built for the CUDA architectures {got}, the release list of CUDA "
        f"{build['cuda_toolkit']} is {want}"
    )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--plugin", required=True, choices=["cu12", "cu13"])
    p.add_argument("--expect", required=True, choices=["cuda", "fallback"])
    p.add_argument("--reason", default="", help="fallback: a text the plugin's reason contains")
    p.add_argument("--version", default="", help="the version dyng and the plugin must have")
    args = p.parse_args(argv)
    plugin, package = args.plugin, f"dyng_{args.plugin}"

    dists = {d: md.version(d) for d in ("dyng", f"dyng-{plugin}")}
    print(f"plugin_smoke: installed {dists}")
    assert dists["dyng"] == dists[f"dyng-{plugin}"], dists
    if args.version:
        assert dists["dyng"] == args.version, (dists, args.version)
    eps = {e.name: e.value for e in md.entry_points(group="dyng.backends")}
    assert eps.get(plugin) == package, eps

    import numpy as np

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        import dyng
        import dyng._backend as backend

        sel = dyng._backend.selection()  # the first use: the module is chosen here
    from dyng import BackendWarning

    dyng.show_config()
    reports = {r.name: r for r in sel.plugins}
    assert plugin in reports, sel
    report = reports[plugin]
    seq = dyng.Resources.sequential()

    if args.expect == "cuda":
        assert not [w for w in caught if issubclass(w.category, BackendWarning)], caught
        assert backend.active_module_name.startswith(package), backend.active_module_name
        assert report.state == "chosen", report
        build = dyng.config()["build"]
        assert build["plugin"] == plugin and build["cuda_runtime"] == "static", build
        _check_architectures(build)
        assert dyng.config()["backends"]["cuda"], dyng.config()["backends"]
        cuda = dyng.Resources.cuda(0)
        on_gpu = _graphs(dyng, np, cuda)
        t, _, _, mo = on_gpu
        assert t.distances.device == "cuda:0" and mo.distances(0).device == "cuda:0"
        assert t.distances.__cuda_array_interface__["shape"] == (t.distances.shape[0],)
        _same(np, on_gpu, _graphs(dyng, np, seq))
        print(
            f"plugin_smoke: sssp, cycle_count and mosp ran on cuda ({backend.active_module_name});"
            " their arrays equal the sequential backend's"
        )
        return 0

    # fallback: no driver / no device
    ours = [w for w in caught if issubclass(w.category, BackendWarning)]
    assert len(ours) == 1, [str(w.message) for w in caught]
    print(f"plugin_smoke: the warning:\n{ours[0].message}")
    assert backend.active_module_name == "dyng._core", backend.active_module_name
    assert report.state == "unusable", report
    assert args.reason in report.reason, (args.reason, report.reason)
    assert sel.plugin is None and dyng.config()["build"]["plugin"] == "", dyng.config()["build"]
    # The plugin's module imports without a driver (no libcuda / libcudart needed) and is the
    # module the wheel promises.
    mod = importlib.import_module(package)
    assert mod.available() is False and mod.CUDA_MAJOR == int(plugin[2:])
    native = mod.native
    build = native.build_config
    assert build["plugin"] == plugin, build
    assert build["cuda"] and build["cuda_runtime"] == "static", build
    assert build["cuda_toolkit"].startswith(plugin[2:] + "."), build
    _check_architectures(build)
    assert native.__version__ == dists["dyng"], (native.__version__, dists)
    print(
        f"plugin_smoke: {package}._core imports without a driver: CUDA {build['cuda_toolkit']}, "
        f"architectures {build['cuda_architectures']}, runtime {build['cuda_runtime']}"
    )
    try:
        dyng.Resources.cuda(0)
    except Exception as e:  # the CPU module has no CUDA backend; the message says why
        print(f"plugin_smoke: Resources.cuda() on the CPU module: {type(e).__name__}: {e}")
    else:
        raise AssertionError("Resources.cuda() worked on the CPU module")
    omp = dyng.Resources.openmp(2)
    _same(np, _graphs(dyng, np, omp), _graphs(dyng, np, seq))
    print("plugin_smoke: dyng._core is active; sssp, cycle_count and mosp run on the CPU backends")
    return 0


if __name__ == "__main__":
    sys.exit(main())
