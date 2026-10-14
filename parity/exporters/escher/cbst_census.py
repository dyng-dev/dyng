#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The nm census of the CBST recording twins (ADR 0036; M8a acceptance criterion 2).

    cbst_census.py ESCHER-GPU|MOSP_ESCHER [--scratch DIR] [--json OUT]
    cbst_census.py --copy <patched copy> --name ESCHER-GPU|MOSP_ESCHER [--json OUT]

`-Xlinker --wrap=<symbol>` redirects only the references to <symbol> that cross object files, so
the recorder sees every call of a CBST entry point if and only if every object that references
an entry point it does not define has that reference wrapped. For every twin of the PATCHED copy
(parity_export/link/<name>.cmd: the original's own link command) the census runs `nm` on every
object and on every member of every archive on the command line, finds the object that defines
the CBST API (CBSTOperations' members and the *CBST free functions; not the kernels) and fails
when another object references an API symbol that is neither wrapped (the export patch's
wrap_symbols.txt) nor a read-only accessor (ACCESSORS). It does the same for the batch hooks
(wrap_hooks.txt) and checks that each twin defines __wrap_<symbol> for every wrapped symbol that
one of its objects references. Calls inside the defining object (CBSTOperations::fill ->
fillCBST, MOSP_ESCHER's compactCBST inside fillCBST and insertCBST, ESCHER-GPU's payload growth)
are not calls between objects: the recorder sees them in the state digest after the wrapped call.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
# Read-only accessors of the API: they change nothing, so they are not wrapped.
ACCESSORS = {
    "CBSTOperations::context() const",
    "CBSTOperations::deviceBytes() const",
    "CBSTOperations::findAndPrint(std::vector<int, std::allocator<int> > const&) const",
    "degreeBinLaunches()",
}
# Executables of the originals' builds and tests that are not twins: they must not link the CBST
# API (tests/local/local_tests exists once tests/run_tests.sh has built it).
NOT_TWINS = {"ESCHER-GPU": ["build/cbst_test"], "MOSP_ESCHER": ["tests/local/local_tests"]}
# The object that defines each original's batch hooks.
HOOK_OBJECT = {"ESCHER-GPU": "dynamic_hypergraph.o", "MOSP_ESCHER": "DynamicHypergraph.cpp.o"}


def demangle(names: list[str]) -> dict[str, str]:
    if not names:
        return {}
    out = subprocess.run(
        ["c++filt"], input="\n".join(names), capture_output=True, text=True, check=True
    ).stdout.splitlines()
    return dict(zip(names, out, strict=True))


def nm_objects(path: Path) -> dict[str, list[tuple[str, str]]]:
    """{object name: [(type, symbol)]} of an object file or of every member of an archive."""
    out = subprocess.run(
        ["nm", "-A", "-P", str(path)], capture_output=True, text=True, check=True
    ).stdout
    objects: dict[str, list[tuple[str, str]]] = {}
    for line in out.splitlines():
        # "<file>[<member>]: <symbol> <type> <value> <size>" or "<file>: <symbol> <type> ..."
        head, _, rest = line.partition(": ")
        parts = rest.split()
        if len(parts) < 2:
            continue
        sym, typ = parts[0], parts[1]
        objects.setdefault(head, []).append((typ, sym))
    if path.suffix == ".o":
        objects.setdefault(str(path), [])
    return objects


def read_list(path: Path) -> list[str]:
    syms = []
    for line in path.read_text().splitlines():
        line = line.split("#", 1)[0].strip()
        if line:
            syms.append(line)
    return syms


def is_api(demangled: str, mangled: str) -> bool:
    if "__device_stub" in mangled or "CBSTNode*" in demangled:
        return False  # kernels and their host stubs
    return (
        demangled.startswith("CBSTOperations::")
        or bool(re.match(r"^\w*CBST\w*\(", demangled))
        or demangled == "degreeBinLaunches()"
    )


def census(copy: Path, name: str) -> dict:
    export = REPO / "parity" / "export_patches" / name
    wraps = set(read_list(export / "wrap_symbols.txt"))
    hook_wraps = set(read_list(export / "wrap_hooks.txt"))
    report: dict = {"reference": name, "copy": str(copy), "twins": [], "problems": []}
    for cmd_file in sorted((copy / "parity_export" / "link").glob("*.cmd")):
        if cmd_file.name.endswith(".twin.cmd"):
            continue
        twin = cmd_file.name[: -len(".cmd")]
        tokens = shlex.split(cmd_file.read_text())
        inputs = [t for t in tokens if t.endswith((".o", ".a"))]
        objects: dict[str, list[tuple[str, str]]] = {}
        for t in inputs:
            objects.update(nm_objects(copy / t))
        defined: dict[str, str] = {}  # global symbol -> defining object
        for obj, syms in objects.items():
            for typ, sym in syms:
                if typ in ("T", "W"):
                    defined.setdefault(sym, obj)
        names = demangle(sorted(defined))
        # The API is what the object that defines CBSTOperations defines (helpers of other
        # objects with "CBST" in their names, e.g. ESCHER-GPU's prepareCBSTData in utils.o, are
        # host data preparation, not tree operations).
        owner = {defined[s] for s, d in names.items() if d.startswith("CBSTOperations::")}
        api = {s for s, d in names.items() if is_api(d, s) and defined[s] in owner}
        api_objects = sorted({defined[s] for s in api})
        hooks = {s for s in hook_wraps if s in defined}
        entry = {
            "twin": twin,
            "objects": len(objects),
            "api_defined_in": api_objects,
            "api_symbols": len(api),
            "references": [],
            "unwrapped": [],
        }
        referenced_wrapped = set()
        for obj, syms in sorted(objects.items()):
            for typ, sym in syms:
                if typ != "U":
                    continue
                if sym in api and defined[sym] != obj:
                    d = names[sym]
                    if sym in wraps:
                        kind = "wrapped"
                        referenced_wrapped.add(sym)
                    elif d in ACCESSORS:
                        kind = "accessor"
                    else:
                        kind = "UNWRAPPED"
                        entry["unwrapped"].append({"object": obj, "symbol": d})
                    entry["references"].append(
                        {
                            "object": Path(obj.split("[")[0]).name
                            if "[" not in obj
                            else obj.rsplit("/", 1)[-1],
                            "symbol": d,
                            "kind": kind,
                        }
                    )
                elif sym in hooks and defined[sym] != obj:
                    referenced_wrapped.add(sym)
                    entry["references"].append(
                        {
                            "object": obj.rsplit("/", 1)[-1],
                            "symbol": demangle([sym])[sym],
                            "kind": "hook",
                        }
                    )
        # hook-level functions that other objects call but the hook list does not wrap
        hook_obj = HOOK_OBJECT[name]
        hook_defs = {s for s, o in defined.items() if o.endswith(hook_obj)}
        twin_syms = {
            sym
            for _typ, sym in nm_objects(copy / "parity_export" / "bin" / twin).get(
                str(copy / "parity_export" / "bin" / twin), []
            )
        }
        missing = sorted(s for s in referenced_wrapped if f"__wrap_{s}" not in twin_syms)
        if hook_defs and not hooks:
            entry["unwrapped"].append({"object": hook_obj, "symbol": "(no batch hook linked)"})
        entry["missing_wrappers"] = missing
        for u in entry["unwrapped"]:
            report["problems"].append(f"{twin}: {u['object']} references {u['symbol']} unwrapped")
        for m in missing:
            report["problems"].append(f"{twin}: no __wrap_{m} in the twin")
        if len(api_objects) != 1:
            report["problems"].append(f"{twin}: the CBST API is defined in {api_objects}")
        report["twins"].append(entry)
    report["not_twins"] = []
    for rel in NOT_TWINS[name]:
        exe = copy / rel
        if not exe.is_file():
            report["not_twins"].append({"executable": rel, "present": False})
            continue
        syms = [sym for objs in nm_objects(exe).values() for _t, sym in objs]
        hits = sorted(
            d
            for m, d in demangle(syms).items()
            if (d.startswith("CBSTOperations::") or re.match(r"^\w*CBST\w*\(", d)) and is_api(d, m)
        )
        report["not_twins"].append({"executable": rel, "present": True, "cbst_symbols": hits})
        if hits:
            report["problems"].append(f"{rel} is not a twin but has CBST symbols: {hits[:3]}")
    if not report["twins"]:
        report["problems"].append("no twins (parity_export/link/*.cmd) in the copy")
    report["passed"] = not report["problems"]
    return report


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("name", nargs="?", choices=["ESCHER-GPU", "MOSP_ESCHER"])
    p.add_argument("--name", dest="name_opt", choices=["ESCHER-GPU", "MOSP_ESCHER"])
    p.add_argument("--copy", help="the patched copy (default: build_reference.sh --print-dir)")
    p.add_argument(
        "--scratch",
        default=os.environ.get("DYNG_SCRATCH", str(Path.home() / "Projects" / "dyng-work")),
    )
    p.add_argument("--json")
    args = p.parse_args(argv)
    name = args.name or args.name_opt
    if not name:
        p.error("name the reference")
    if args.copy:
        copy = Path(args.copy)
    else:
        out = subprocess.check_output(
            [
                str(REPO / "parity" / "build_reference.sh"),
                "--variant",
                "patched",
                "--scratch",
                args.scratch,
                "--print-dir",
                name,
            ],
            text=True,
        )
        copy = Path(out.strip().splitlines()[-1])
    report = census(copy, name)
    for t in report["twins"]:
        kinds: dict[str, int] = {}
        for r in t["references"]:
            kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
        print(
            f"{t['twin']}: {t['objects']} objects; API defined in "
            f"{', '.join(Path(o).name for o in t['api_defined_in'])}; references from other "
            f"objects: {kinds}"
        )
    for n in report["not_twins"]:
        print(
            f"{n['executable']}: not a twin; "
            + (
                "not built"
                if not n["present"]
                else f"CBST API symbols: {len(n['cbst_symbols'])} (it calls no CBST operation)"
            )
        )
    for pr in report["problems"]:
        print("PROBLEM: " + pr)
    print(f"{name}: census {'passed' if report['passed'] else 'FAILED'}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    sys.exit(main())
