#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The SASS and resource readings of triad_count's counting kernels (M9a step 3; PLAN 6.3 step 7,
8.6 "Kernel checks"; ADR 0053).

    parity/triad_sass.py report --object build/.../engine/counters.cu.o [--json OUT]
    parity/triad_sass.py report --escher [--json OUT]      # ESCHER-GPU's unpatched build objects
    parity/triad_sass.py compare A.json B.json             # exit 1 if a kernel differs

For each counting kernel (the wedge count, triangle_kernel of MotifOp and Type3Op, pair_kernel of
Type2Op, key_kernel of Choose3Op) the report holds `cuobjdump -res-usage`'s registers, stack,
shared and local memory, and from `cuobjdump -sass`: the instruction count, the opcode histogram,
the hot loops (every backward branch: the instructions from its target to the branch, by size)
and a digest of the normalized instruction stream (addresses, comments and the function's own name
removed), so two builds of a kernel compare equal exactly when their machine code does.

The report's "provenance" entry says what was read (M9b review): the objects (absolute path,
SHA-256), the build tree of each (its CMakeCache: the source tree, the build type and the CUDA
compiler with its version), the source tree's commit when the report is made (with "+dirty" when
its tracked files differ from it; so build first, in a clean clone, then report), the cuobjdump
version and the date. `compare` prints both provenances and compares the
kernels only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
ESCHER_BUILD = (
    Path(os.environ.get("DYNG_SCRATCH", Path.home() / "Projects" / "dyng-work"))
    / "ref"
    / "ESCHER-GPU@abcf9b2"
    / "unpatched"
    / "build"
)

# kernel label -> (regex of the mangled name of dynG's, regex of ESCHER-GPU's)
KERNELS = {
    "count_wedges": (r"count_wedges_kernel", r"countWedges"),
    "triangles<motif>": (r"triangle_kernel.*motif_op", r"triangleKernelI7MotifOp"),
    "triangles<type3>": (r"triangle_kernel.*type3_op", r"triangleKernelI7Type3Op"),
    "pairs<type2>": (r"pair_kernel.*type2_op", r"pairKernelI7Type2Op"),
    "keys<choose3>": (r"key_kernel.*choose3_op", r"keyKernelI9Choose3Op"),
}
# Where ESCHER-GPU's kernels are (one object each is enough: the TUs instantiate the same code).
ESCHER_OBJECTS = ["HMotifCount.o", "type1.o", "type2.o", "type3.o"]


def cuobjdump() -> str:
    for c in [shutil.which("cuobjdump"), "/usr/local/cuda-13.1/bin/cuobjdump"]:
        if c and Path(c).exists():
            return c
    raise SystemExit("triad_sass: cuobjdump not found")


def resource_usage(obj: Path) -> dict[str, dict]:
    out = subprocess.run(
        [cuobjdump(), "-res-usage", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    res: dict[str, dict] = {}
    name = None
    for line in out.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        if name and "REG:" in line:
            res[name] = {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", line)}
            name = None
    return res


def sass(obj: Path) -> dict[str, list[str]]:
    out = subprocess.run(
        [cuobjdump(), "-sass", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in out.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = re.match(r"\s*/\*([0-9a-f]{4,})\*/\s+(.*?)\s*;", line)
        if cur and m:
            funcs[cur].append((int(m.group(1), 16), m.group(2)))
    return funcs


def normalize(ins: str, name: str) -> str:
    ins = ins.replace(name, "<self>")
    ins = re.sub(r"0x[0-9a-f]+", "<addr>", ins)
    return re.sub(r"\s+", " ", ins).strip()


def analyse(name: str, code: list[tuple[int, str]]) -> dict:
    ops: dict[str, int] = {}
    for _, ins in code:
        op = re.sub(r"^@!?U?P\w+\s+", "", ins).split()[0]
        ops[op] = ops.get(op, 0) + 1
    loops = []
    for i, (addr, ins) in enumerate(code):
        m = re.search(r"\bBRA\b.*?(0x[0-9a-f]+)", ins)
        if m:
            target = int(m.group(1), 16)
            if target <= addr:
                first = next(j for j, (a, _) in enumerate(code) if a >= target)
                loops.append(i - first + 1)
    stream = "\n".join(normalize(ins, name) for _, ins in code)
    return {
        "instructions": len(code),
        "opcodes": dict(sorted(ops.items())),
        "loops": sorted(loops, reverse=True),
        "digest": hashlib.sha256(stream.encode()).hexdigest(),
    }


def _cache_value(cache: Path, key: str) -> str | None:
    for line in cache.read_text(errors="replace").splitlines():
        m = re.match(rf"{re.escape(key)}:[A-Z]+=(.*)$", line)
        if m:
            return m.group(1)
    return None


def _first_line(argv: list[str], last: bool = False) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None
    lines = [x for x in out.splitlines() if x.strip()]
    return (lines[-1] if last else lines[0]) if lines else None


def _commit(source: Path) -> str | None:
    head = _first_line(["git", "-C", str(source), "rev-parse", "HEAD"])
    if head is None:
        return None
    dirty = _first_line(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"])
    return head + ("+dirty" if dirty else "")


def provenance(objects: list[Path], escher: bool) -> dict:
    """What a report reads: the objects, their build trees, the commit, the tools."""
    objs = []
    for obj in objects:
        entry: dict[str, object] = {
            "path": str(obj.resolve()),
            "sha256": hashlib.sha256(obj.read_bytes()).hexdigest(),
        }
        cache = next(
            (
                d / "CMakeCache.txt"
                for d in obj.resolve().parents
                if (d / "CMakeCache.txt").is_file()
            ),
            None,
        )
        if cache is not None:
            source = _cache_value(cache, "CMAKE_HOME_DIRECTORY")
            nvcc = _cache_value(cache, "CMAKE_CUDA_COMPILER")
            entry.update(
                {
                    "build_tree": str(cache.parent),
                    "source_tree": source,
                    "build_type": _cache_value(cache, "CMAKE_BUILD_TYPE"),
                    "cuda_compiler": nvcc,
                    "cuda_compiler_version": _first_line([nvcc, "--version"], last=True)
                    if nvcc
                    else None,
                    "commit": _commit(Path(source)) if source else None,
                }
            )
        objs.append(entry)
    return {
        "what": "ESCHER-GPU's unpatched build objects" if escher else "dynG's counting kernels",
        "objects": objs,
        "cuobjdump": _first_line([cuobjdump(), "--version"], last=True),
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def report(objects: list[Path], escher: bool) -> dict:
    out: dict[str, dict] = {}
    for obj in objects:
        res = resource_usage(obj)
        code = sass(obj)
        for label, (port_re, a_re) in KERNELS.items():
            if label in out:
                continue
            pattern = a_re if escher else port_re
            for name, body in code.items():
                if re.search(pattern, name):
                    entry = analyse(name, body)
                    entry.update({"function": name, "object": obj.name})
                    entry["resources"] = res.get(name, {})
                    out[label] = entry
                    break
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("report")
    r.add_argument("--object", type=Path, action="append", default=[])
    r.add_argument("--escher", action="store_true", help="ESCHER-GPU's unpatched build objects")
    r.add_argument("--json", type=Path)
    c = sub.add_parser("compare")
    c.add_argument("a", type=Path)
    c.add_argument("b", type=Path)
    args = ap.parse_args()
    if args.cmd == "report":
        objs = [ESCHER_BUILD / o for o in ESCHER_OBJECTS] if args.escher else args.object
        if not objs:
            ap.error("--object or --escher")
        rep = report(objs, args.escher)
        prov = provenance(objs, args.escher)
        for o in prov["objects"]:
            print(f"object {o['path']} (commit {o.get('commit')}, {o.get('build_type')})")
        for label, e in rep.items():
            r_ = e["resources"]
            print(
                f"{label:18} REG {r_.get('REG')} STACK {r_.get('STACK')} SHARED {r_.get('SHARED')} "
                f"LOCAL {r_.get('LOCAL')}  instructions {e['instructions']}  loops {e['loops']}  "
                f"{e['digest'][:12]}"
            )
        if args.json:
            out = {"provenance": prov, **rep}
            args.json.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        return 0
    a = json.loads(args.a.read_text())
    b = json.loads(args.b.read_text())
    for name, rec in (("a", a), ("b", b)):
        prov = rec.pop("provenance", None)
        if prov is None:
            print(f"{name}: no provenance recorded")
            continue
        for o in prov.get("objects", []):
            print(
                f"{name}: {o.get('path')} commit {o.get('commit')} ({o.get('build_type')}, "
                f"{o.get('cuda_compiler_version')}), {prov.get('cuobjdump')}, {prov.get('date')}"
            )
    status = 0
    for label in sorted(set(a) | set(b)):
        x, y = a.get(label), b.get(label)
        if x is None or y is None:
            print(f"{label}: only in {'b' if x is None else 'a'}")
            status = 1
            continue
        same_code = x["digest"] == y["digest"]
        same_res = x["resources"] == y["resources"]
        same_loops = x["loops"] == y["loops"] and x["instructions"] == y["instructions"]
        verdict = (
            "identical"
            if same_code and same_res
            else ("same loops and resources" if same_loops and same_res else "DIFFERENT")
        )
        if verdict == "DIFFERENT":
            status = 1
        print(
            f"{label:18} {verdict}: instructions {x['instructions']} / {y['instructions']}, loops "
            f"{x['loops']} / {y['loops']}, REG {x['resources'].get('REG')} / "
            f"{y['resources'].get('REG')}, STACK {x['resources'].get('STACK')} / "
            f"{y['resources'].get('STACK')}, LOCAL {x['resources'].get('LOCAL')} / "
            f"{y['resources'].get('LOCAL')}"
        )
    return status


if __name__ == "__main__":
    sys.exit(main())
