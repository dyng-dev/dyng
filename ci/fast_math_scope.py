#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The scope of --use_fast_math (M10a criterion 12; PLAN 7.2, 7.4; ADRs 0066, 0075): in a
compile_commands.json, --use_fast_math is on cpp/src/algorithms/label_propagation/*.cu and on no
other translation unit, and those TUs carry the parity-cuda preset's global -fmad=true exactly once
(the target adds no -fmad flag of its own; --use_fast_math implies it anyway).

    python3 ci/fast_math_scope.py <compile_commands.json> [--require-fmad]

--require-fmad: the database is a parity-cuda build (CMAKE_CUDA_FLAGS_RELEASE holds -fmad=true);
without it the -fmad check is only that the LP TUs carry no second -fmad. Exit 0 when the scope
holds, 1 with the offending entries listed.
"""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path

SCOPE = "cpp/src/algorithms/label_propagation/"


def words(entry: dict) -> list[str]:
    if "arguments" in entry:
        return list(entry["arguments"])
    return shlex.split(entry.get("command", ""))


def relative(path: str) -> str:
    p = Path(path).as_posix()
    at = p.find("/cpp/")
    return p[at + 1 :] if at >= 0 else p


def check(entries: list[dict], require_fmad: bool) -> list[str]:
    problems = []
    lp_tus = 0
    for e in entries:
        f = relative(e["file"])
        w = words(e)
        fast = "--use_fast_math" in w or "-use_fast_math" in w
        in_scope = f.startswith(SCOPE) and f.endswith(".cu")
        if fast and not in_scope:
            problems.append(f"{f}: --use_fast_math outside {SCOPE}*.cu")
        if in_scope:
            lp_tus += 1
            if not fast:
                problems.append(f"{f}: no --use_fast_math")
            fmad = [x for x in w if x.startswith("-fmad") or x.startswith("--fmad")]
            if len(fmad) > 1:
                problems.append(f"{f}: {len(fmad)} -fmad flags (the target must add none)")
            if require_fmad and fmad != ["-fmad=true"]:
                problems.append(f"{f}: -fmad=true missing (parity-cuda's global flag): {fmad}")
    if lp_tus == 0:
        problems.append(f"no {SCOPE}*.cu entry (a CUDA build with label_propagation?)")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("database", type=Path)
    ap.add_argument("--require-fmad", action="store_true")
    args = ap.parse_args(argv)
    problems = check(json.loads(args.database.read_text()), args.require_fmad)
    for p in problems:
        print(f"fast_math_scope: {p}", file=sys.stderr)
    if not problems:
        print("fast_math_scope: --use_fast_math only on label_propagation's CUDA TUs")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
