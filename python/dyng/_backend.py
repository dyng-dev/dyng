# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Select the native module of this process (PLAN Section 5.4, "CPU core + CUDA plugins").

The ``dyng`` wheel always contains ``dyng._core`` (sequential and OpenMP backends). The CUDA
backends come as plugin wheels (``dyng-cu12``, ``dyng-cu13``; ADR 0030), each registering an entry
point in the group ``dyng.backends`` that names a module with ``available() -> bool`` and
``native`` (its extension module, a superset of ``_core``); the plugins of dynG also offer
``CUDA_MAJOR``, ``__version__`` and ``status()`` (the driver's CUDA version and why the plugin
cannot run). Exactly one native module is active per process.

The module is chosen **on first use**, not at ``import dyng``: the first access to an attribute
of :data:`native` (the first graph, resources, reader, ``dyng.__version__``, ...) chooses it, and
only the chosen module is imported, so a plugin process never loads ``_core`` (a second static
libdyng and a second bundled OpenMP runtime). The rules (ADR 0031):

1. ``DYNG_CPU_ONLY=1`` in the environment, or :func:`use_cpu_only` before the first use, chooses
   ``_core`` without looking at the plugins.
2. A plugin of another version than ``dyng`` is never loaded (its bindings would not match the
   typed layer): ``pip install "dyng[cu13]"`` installs the plugin of the same version.
3. A plugin is a candidate when ``available()`` says it can run here (a driver of its CUDA major
   or newer and a visible device). Of the candidates, the plugin of the driver's CUDA major wins,
   then the newest older major (a cu12 plugin also runs on a CUDA 13 driver); ties go by name.
4. A candidate whose module fails to import is skipped (the next one is tried).
5. If plugins are installed but none can be used, ``_core`` is chosen with a
   :class:`~dyng.BackendWarning` that says why for each plugin; with no plugin installed the CPU
   module is chosen silently (the CPU-only install behaves as in 0.1.0).

:func:`selection` reports the choice (``dyng.show_config()`` prints it).
"""

from __future__ import annotations

import importlib
import os
import threading
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from importlib import metadata
from types import ModuleType
from typing import Any

__all__ = [
    "native",
    "active_module_name",
    "plugin_errors",
    "on_activate",
    "use_cpu_only",
    "selection",
    "PluginReport",
    "Selection",
]

#: Errors met while loading plugins (name -> message), for dyng.show_config().
plugin_errors: dict[str, str] = {}

#: The name of the active native module, or None before the first use.
active_module_name: str | None = None

_lock = threading.RLock()
_active: ModuleType | None = None
_selection: Selection | None = None
_cpu_only = False
_hooks: list[Callable[[ModuleType], None]] = []
_PACKAGE_DIR = os.path.dirname(os.path.abspath(__file__))


@dataclass(frozen=True)
class PluginReport:
    """What the selection found out about one installed CUDA plugin.

    Attributes:
        name: The entry point's name (``"cu13"``).
        module: The plugin's import package (``"dyng_cu13"``).
        version: The plugin's version, or None if it does not say.
        cuda_major: The CUDA major its module was built with, or None.
        driver_version: The driver's CUDA version as the plugin saw it (``1000 * major + 10 *
            minor``), or None without a driver (or when the plugin was not asked).
        state: ``"chosen"``, ``"available"`` (usable, another one was chosen), ``"unusable"`` (no
            driver, too old a driver, no device), ``"version mismatch"``, ``"failed"`` (loading
            the plugin or its module raised) or ``"not considered"`` (the CPU module was forced).
        reason: Why it was chosen or not, in words.
    """

    name: str
    module: str
    version: str | None
    cuda_major: int | None
    driver_version: int | None
    state: str
    reason: str


@dataclass(frozen=True)
class Selection:
    """The choice of the native module (see the module description).

    Attributes:
        module_name: The active module, ``"dyng._core"`` or ``"dyng_cu13 (plugin cu13)"``.
        plugin: The chosen plugin's name, or None for the CPU module.
        reason: Why this module was chosen.
        plugins: Every installed plugin, in name order.
        dyng_version: The installed ``dyng`` distribution's version (None when ``dyng`` runs from
            a source tree without metadata; then the version check is skipped).
    """

    module_name: str
    plugin: str | None
    reason: str
    plugins: tuple[PluginReport, ...] = field(default=())
    dyng_version: str | None = None

    @property
    def warning(self) -> str | None:
        """The fallback warning of rule 5, or None."""
        if self.plugin is not None or not self.plugins:
            return None
        if all(p.state == "not considered" for p in self.plugins):
            return None
        lines = [
            "dyng: a CUDA plugin is installed but cannot be used, so dynG runs on the CPU module "
            "dyng._core (sequential and OpenMP backends; Resources.cuda() is not available):"
        ]
        lines += [f"  {p.module} {p.version or '?'}: {p.reason}" for p in self.plugins]
        lines.append(
            "  (dyng.show_config() shows the choice; set DYNG_CPU_ONLY=1 or call "
            "dyng.use_cpu_only() to choose the CPU module without this warning)"
        )
        return "\n".join(lines)


def _driver_text(version: int | None) -> str:
    if version is None:
        return "no driver"
    return f"CUDA {version // 1000}.{(version % 1000) // 10}"


def _entry_points() -> Iterable[Any]:
    """The entry points of the group ``dyng.backends`` (replaced by the tests)."""
    return metadata.entry_points(group="dyng.backends")


def _dyng_version() -> str | None:
    try:
        return metadata.version("dyng")
    except metadata.PackageNotFoundError:  # a source tree on sys.path, without metadata
        return None


def _core() -> ModuleType:
    return importlib.import_module("._core", __package__)


def _probe(ep: Any, dyng_version: str | None) -> tuple[Any, PluginReport]:
    """Load one plugin (its package, not its module) and ask it whether it can run."""
    name, value = str(ep.name), str(ep.value)
    try:
        plugin = ep.load()
    except Exception as e:  # a broken plugin must not break the CPU package
        plugin_errors[name] = f"{type(e).__name__}: {e}"
        return None, PluginReport(
            name, value, None, None, None, "failed", f"cannot be loaded ({type(e).__name__}: {e})"
        )
    version = getattr(plugin, "__version__", None)
    major = getattr(plugin, "CUDA_MAJOR", None)
    major = int(major) if isinstance(major, int) else None
    if dyng_version is not None and version is not None and str(version) != dyng_version:
        reason = (
            f"version {version} does not match dyng {dyng_version}; install the plugin of the "
            f'same version: pip install "dyng[{name}]=={dyng_version}"'
        )
        return None, PluginReport(name, value, version, major, None, "version mismatch", reason)
    try:
        status = plugin.status() if callable(getattr(plugin, "status", None)) else None
        usable = bool(plugin.available())
    except Exception as e:
        plugin_errors[name] = f"{type(e).__name__}: {e}"
        return None, PluginReport(
            name,
            value,
            version,
            major,
            None,
            "failed",
            f"available() raised {type(e).__name__}: {e}",
        )
    driver = getattr(status, "driver_version", None)
    driver = int(driver) if isinstance(driver, int) else None
    if not usable:
        reason = str(getattr(status, "reason", "") or "") or "the plugin says it is not available"
        return None, PluginReport(name, value, version, major, driver, "unusable", reason)
    return plugin, PluginReport(name, value, version, major, driver, "available", "")


def _rank(report: PluginReport, driver_major: int | None) -> tuple[int, int, str]:
    """Sort key of a candidate: the driver's major first, then newer majors, then by name."""
    major = report.cuda_major
    if major is None:
        return (2, 0, report.name)
    if driver_major is not None and major == driver_major:
        return (0, 0, report.name)
    return (1, -major, report.name)


def _select(
    entry_points: Iterable[Any],
    *,
    cpu_only: bool = False,
    dyng_version: str | None = None,
    load_core: Callable[[], ModuleType] = _core,
) -> tuple[ModuleType, Selection]:
    """Choose the native module (the rules of the module description); imports only it."""
    eps = sorted(entry_points, key=lambda e: str(e.name))
    if cpu_only:
        why = "the CPU module was asked for (DYNG_CPU_ONLY=1 or dyng.use_cpu_only())"
        skipped = tuple(
            PluginReport(str(e.name), str(e.value), None, None, None, "not considered", why)
            for e in eps
        )
        return load_core(), Selection("dyng._core", None, why, skipped, dyng_version)
    probed = [_probe(ep, dyng_version) for ep in eps]
    reports = {r.name: r for _, r in probed}
    drivers = [r.driver_version for r in reports.values() if r.driver_version is not None]
    driver_major = max(drivers) // 1000 if drivers else None
    candidates = sorted(
        ((plugin, r) for plugin, r in probed if plugin is not None),
        key=lambda item: _rank(item[1], driver_major),
    )
    chosen: tuple[ModuleType, PluginReport] | None = None
    for plugin, r in candidates:
        if chosen is not None:
            reports[r.name] = _replace(r, "available", f"usable, but {chosen[1].module} was chosen")
            continue
        try:
            module = plugin.native
            built = getattr(module, "__version__", None)
            if dyng_version is not None and built is not None and str(built) != dyng_version:
                raise ImportError(f"its module is version {built}, dyng is {dyng_version}")
            if not module.build_config["cuda"]:
                raise ImportError("its module was built without CUDA")
        except Exception as e:
            plugin_errors[r.name] = f"{type(e).__name__}: {e}"
            reports[r.name] = _replace(
                r, "failed", f"its module cannot be loaded ({type(e).__name__}: {e})"
            )
            continue
        if driver_major is not None and r.cuda_major == driver_major:
            why = f"it matches the driver's CUDA major ({_driver_text(r.driver_version)})"
        elif r.cuda_major is not None:
            why = (
                f"the driver supports {_driver_text(r.driver_version)}; no plugin of that major "
                f"is usable, and CUDA {r.cuda_major} runs on it"
            )
        else:
            why = "it is available"
        reports[r.name] = _replace(r, "chosen", why)
        chosen = (module, reports[r.name])
    ordered = tuple(reports[str(e.name)] for e in eps)
    if chosen is None:
        why = (
            "no CUDA plugin is installed"
            if not ordered
            else "no installed CUDA plugin can be used (see the plugins)"
        )
        return load_core(), Selection("dyng._core", None, why, ordered, dyng_version)
    module, report = chosen
    return module, Selection(
        f"{report.module} (plugin {report.name})",
        report.name,
        f"plugin {report.name}: {report.reason}",
        ordered,
        dyng_version,
    )


def _replace(report: PluginReport, state: str, reason: str) -> PluginReport:
    return PluginReport(
        report.name,
        report.module,
        report.version,
        report.cuda_major,
        report.driver_version,
        state,
        reason,
    )


def _cpu_only_requested() -> bool:
    return _cpu_only or os.environ.get("DYNG_CPU_ONLY", "").strip() not in ("", "0")


def active_module() -> ModuleType:
    """The active native module, chosen (and its hooks run) on the first call."""
    global _active, _selection, active_module_name
    module = _active
    if module is not None:
        return module
    warning: str | None = None
    with _lock:
        if _active is None:
            module, chosen = _select(
                _entry_points(), cpu_only=_cpu_only_requested(), dyng_version=_dyng_version()
            )
            for hook in _hooks:
                hook(module)
            active_module_name = chosen.module_name
            _selection = chosen
            _active = module
            warning = chosen.warning
        module = _active
    if warning is not None:
        from .errors import BackendWarning

        warnings.warn(warning, BackendWarning, skip_file_prefixes=(_PACKAGE_DIR,))
    return module


def selection() -> Selection:
    """The choice of the native module (made now if it was not made yet)."""
    active_module()
    assert _selection is not None
    return _selection


def on_activate(hook: Callable[[ModuleType], None]) -> None:
    """Run ``hook(module)`` when the native module is chosen (at once if it already is)."""
    with _lock:
        if _active is not None:
            hook(_active)
        else:
            _hooks.append(hook)


class _Native:
    """The active native module, resolved on first attribute access (see the module docs)."""

    __slots__ = ()

    def __getattr__(self, name: str) -> Any:
        return getattr(active_module(), name)

    def __repr__(self) -> str:
        return f"<dyng native module {active_module_name or '(not chosen yet)'}>"


#: The active native module (``dyng._core`` or a CUDA plugin's), chosen on first use.
native: Any = _Native()


def use_cpu_only() -> None:
    """Force the CPU module ``dyng._core`` (sequential and OpenMP) for this process.

    Call it before the first use of dynG (the first graph, resources, reader or
    ``dyng.__version__``; importing ``dyng`` does not count). Once ``_core`` is active, further
    calls do nothing. ``DYNG_CPU_ONLY=1`` in the environment does the same. Neither looks at the
    installed CUDA plugins, so neither warns about them.

    Raises:
        RuntimeError: if a CUDA plugin module was already chosen by an earlier use.
    """
    global _cpu_only
    with _lock:
        if _active is None:
            _cpu_only = True
            active_module()
            return
        if active_module_name != "dyng._core":
            raise RuntimeError(
                f"dyng.use_cpu_only(): the native module {active_module_name} was already "
                "chosen by an earlier use of dynG; call use_cpu_only() before the first graph, "
                "resources or reader, or set DYNG_CPU_ONLY=1 in the environment"
            )
