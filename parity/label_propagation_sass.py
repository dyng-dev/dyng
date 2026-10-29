#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The build-flags go/no-go of label_propagation (M10a step 1; PLAN 6.4.7, 7.4, 8.6; ADR 0066):
O's six pipeline translation units compiled from a fresh `git archive a276a3a` twice, with O's
Makefile line and with the nvcc line dynG generates for a FAST_MATH CUDA TU under the parity-cuda
preset, and the SASS of every function compared.

    parity/label_propagation_sass.py compile --archive DIR --dyng-commands compile_commands.json
                                             --probe <path of the FAST_MATH probe .cu> --out DIR
    parity/label_propagation_sass.py compare --out DIR [--json FILE]

`compile` takes O's line from `make -n -B obj/<tu>.o` in the archive (the Makefile's defaults) and
dynG's from the compile_commands.json of a scratch configure (`cmake --preset parity-cuda` with a
throw-away `dyng_add_algorithm(... FAST_MATH)` target, whose .cu entry is --probe): every flag of
dynG's line is kept, its source and object replaced by O's, and O's include directory (-Iheaders)
added (O's sources need it; it changes no code generation). `compare` reads `cuobjdump -sass` and
`cuobjdump -res-usage` of both objects of each TU and compares every function: the instruction
text with only the address comments removed (the strictest normalization: a function is equal
only when its machine code is), plus the registers, stack, shared and local memory. It records
both command lines, the per-function instruction counts and digests, the cuobjdump version, and
exits 1 when any function differs or exists in one object only.
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import re
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
# O's six pipeline translation units (PLAN 6.4.7, the step's list).
TUS = (
    "dynamicGraph",
    "sparsifySubgraph",
    "connectedComponents",
    "labelInitialization",
    "iterativePropagation",
    "validation",
)


def cuobjdump() -> str:
    for c in [shutil.which("cuobjdump"), "/usr/local/cuda-13.1/bin/cuobjdump"]:
        if c and Path(c).exists():
            return c
    raise SystemExit("cuobjdump not found")


def original_line(archive: Path, tu: str) -> list[str]:
    out = subprocess.run(
        ["make", "-n", "-B", f"obj/{tu}.o"], cwd=archive, capture_output=True, text=True, check=True
    ).stdout
    line = next((x for x in out.splitlines() if f"-o obj/{tu}.o" in x), None)
    if line is None:
        raise SystemExit(f"no compile line for obj/{tu}.o in `make -n`")
    return shlex.split(line)


def dyng_line(commands: Path, probe: str) -> list[str]:
    for e in json.loads(commands.read_text()):
        if e["file"].endswith(probe):
            return shlex.split(e["command"])
    raise SystemExit(f"{probe}: not in {commands}")


def retarget(line: list[str], source: str, obj: str, include: str) -> list[str]:
    """dynG's line for O's TU: its flags kept, the source and object replaced, -I<headers>
    added before the first -I."""
    out: list[str] = []
    skip = False
    added = False
    for i, tok in enumerate(line):
        if skip:
            skip = False
            continue
        if tok == "-o":
            out += ["-o", obj]
            skip = True
            continue
        if tok == "-c":
            out += ["-c", source]
            skip = True
            continue
        if tok.startswith("-I") and not added:
            out.append(f"-I{include}")
            added = True
        out.append(tok)
        if i == 0:
            continue
    return out


def compile_cmd(args: argparse.Namespace) -> int:
    archive = args.archive.resolve()
    out = args.out.resolve()
    (out / "original").mkdir(parents=True, exist_ok=True)
    (out / "dyng").mkdir(parents=True, exist_ok=True)
    lines = {"original": {}, "dyng": {}}
    base = dyng_line(args.dyng_commands, args.probe)
    for tu in TUS:
        o = original_line(archive, tu)
        o = [x if x != f"obj/{tu}.o" else str(out / "original" / f"{tu}.o") for x in o]
        o = [x for x in o if x not in ("-MMD", "-MP")]  # dependency files only
        subprocess.run(o, cwd=archive, check=True)
        d = retarget(
            base,
            str(archive / "src" / f"{tu}.cu"),
            str(out / "dyng" / f"{tu}.o"),
            str(archive / "headers"),
        )
        subprocess.run(d, cwd=archive, check=True)
        lines["original"][tu] = shlex.join(o)
        lines["dyng"][tu] = shlex.join(d)
        print(f"{tu}: compiled both ways", flush=True)
    (out / "lines.json").write_text(json.dumps(lines, indent=1) + "\n")
    return 0


def sass(obj: Path) -> dict[str, list[str]]:
    text = subprocess.run(
        [cuobjdump(), "-sass", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    funcs: dict[str, list[str]] = {}
    cur = None
    for line in text.splitlines():
        m = re.match(r"\s*Function : (\S+)", line)
        if m:
            cur = m.group(1)
            funcs[cur] = []
            continue
        m = re.match(r"\s*/\*[0-9a-f]{4,}\*/\s+(.*?)\s*;", line)
        if cur and m:
            funcs[cur].append(re.sub(r"\s+", " ", m.group(1)).strip())
    return funcs


def resources(obj: Path) -> dict[str, dict]:
    text = subprocess.run(
        [cuobjdump(), "-res-usage", str(obj)], capture_output=True, text=True, check=True
    ).stdout
    res: dict[str, dict] = {}
    name = None
    for line in text.splitlines():
        m = re.match(r"\s*Function (\S+):", line)
        if m:
            name = m.group(1)
            continue
        if name and "REG:" in line:
            res[name] = {k: int(v) for k, v in re.findall(r"(\w+):(\d+)", line)}
            name = None
    return res


def compare_cmd(args: argparse.Namespace) -> int:
    out = args.out.resolve()
    lines = json.loads((out / "lines.json").read_text())
    record: dict = {
        "schema": 1,
        "what": "O's six pipeline TUs compiled with O's Makefile line and with dynG's FAST_MATH "
        "nvcc line under parity-cuda; every function's SASS (address comments removed) and "
        "resources compared (M10a step 1, ADR 0066)",
        "date": datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "cuobjdump": subprocess.run([cuobjdump(), "--version"], capture_output=True, text=True)
        .stdout.strip()
        .splitlines()[-1],
        "lines": lines,
        "tus": {},
    }
    differ = []
    for tu in TUS:
        a, b = out / "original" / f"{tu}.o", out / "dyng" / f"{tu}.o"
        sa, sb = sass(a), sass(b)
        ra, rb = resources(a), resources(b)
        funcs = {}
        for name in sorted(set(sa) | set(sb)):
            same = sa.get(name) == sb.get(name) and ra.get(name) == rb.get(name)
            funcs[name] = {
                "identical": same,
                "instructions": [len(sa.get(name, [])), len(sb.get(name, []))],
                "sha256": [
                    hashlib.sha256("\n".join(x.get(name, [])).encode()).hexdigest()
                    for x in (sa, sb)
                ],
                "resources": [ra.get(name), rb.get(name)],
            }
            if not same:
                differ.append(f"{tu}: {name}")
        record["tus"][tu] = {"functions": funcs, "count": len(funcs)}
        print(
            f"{tu}: {len(funcs)} functions, "
            f"{sum(1 for f in funcs.values() if not f['identical'])} differ",
            flush=True,
        )
    record["identical"] = not differ
    record["differ"] = differ
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(record, indent=1) + "\n")
    print("SASS identical in every function" if not differ else f"DIFFER: {differ}")
    return 0 if not differ else 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="command", required=True)
    c = sub.add_parser("compile")
    c.add_argument("--archive", type=Path, required=True)
    c.add_argument("--dyng-commands", type=Path, required=True)
    c.add_argument("--probe", required=True)
    c.add_argument("--out", type=Path, required=True)
    m = sub.add_parser("compare")
    m.add_argument("--out", type=Path, required=True)
    m.add_argument("--json", type=Path)
    args = ap.parse_args(argv)
    return compile_cmd(args) if args.command == "compile" else compare_cmd(args)


if __name__ == "__main__":
    sys.exit(main())
