#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The reproducible transplant of ESCHER-GPU@abcf9b2's CBST store (M8a step 3, commit c1; ADR 0038).

The first commit of the store port ("c1, transplant") is ESCHER-GPU's code with mechanical
changes only. This script regenerates it from the pinned original so that a reviewer can diff the
port against the original instead of reading it:

1. every target file of rename_map.toml is assembled from its parts (whole files of the original
   or line ranges of them), read with `git show <commit>:<path>` from the original's clone (read
   only; nothing in the clone changes);
2. the includes are rewritten ([includes]), the context rules applied ([[context]]), and every
   whole-token name of [names] replaced (identifiers, and the same words in comments);
3. include guards become `#pragma once`, the code after the includes is wrapped in the namespace,
   and the SPDX header, the provenance line `// Derived from ESCHER-GPU@abcf9b2:<path>` and a
   Doxygen @file block are added;
4. clang-format (the repository's .clang-format) formats the files;
5. the manual hunks (manual_hunks.patch: what the rules cannot express, each one listed and
   explained in ADR 0038) are applied with `patch -p1`.

    python3 parity/escher_port/transplant.py --out DIR            # write the transplant into DIR
    python3 parity/escher_port/transplant.py --check --rev <c1>   # compare with the c1 commit
    python3 parity/escher_port/transplant.py --mechanical DIR     # steps 1-4 only (no hunks)

The original's clone defaults to $DYNG_ORIGINALS/ESCHER-GPU or ~/Projects/ESCHER-GPU
(--original). Exit status: 0 identical (--check) or written, 1 different, 2 usage or input error.
"""

from __future__ import annotations

import argparse
import difflib
import os
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import tomllib
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
MAP = HERE / "rename_map.toml"
HUNKS = HERE / "manual_hunks.patch"
TOKEN = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*\b")
INCLUDE = re.compile(r'^(\s*#\s*include\s+)("[^"]+"|<[^>]+>)(.*)$')


def load_map(path: Path = MAP) -> dict:
    with path.open("rb") as f:
        return tomllib.load(f)


def git_show(original: Path, commit: str, path: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(original), "show", f"{commit}:{path}"], text=True
        )
    except subprocess.CalledProcessError as e:
        raise SystemExit(f"transplant: cannot read {commit}:{path} from {original}: {e}") from e


def assemble(original: Path, commit: str, parts: list[dict]) -> str:
    cache: dict[str, list[str]] = {}
    out: list[str] = []
    for part in parts:
        path = part["path"]
        if path not in cache:
            cache[path] = git_show(original, commit, path).splitlines()
        lines = cache[path]
        if "lines" in part:
            first, last = part["lines"]
            if not 1 <= first <= last <= len(lines):
                raise SystemExit(f"transplant: {path} has {len(lines)} lines, not {first}-{last}")
            out.extend(lines[first - 1 : last])
        else:
            out.extend(lines)
    return "\n".join(out) + "\n"


def rewrite_includes(text: str, includes: dict[str, str]) -> str:
    out = []
    for line in text.splitlines():
        m = INCLUDE.match(line)
        if m and m.group(2) in includes:
            repl = includes[m.group(2)]
            if repl == "":
                continue
            line = f"{m.group(1)}{repl}{m.group(3)}"
        out.append(line)
    return "\n".join(out) + "\n"


def rename(text: str, contexts: list[dict], names: dict[str, str]) -> str:
    for rule in contexts:
        text = re.sub(rule["pattern"], rule["replacement"], text)
    return TOKEN.sub(lambda m: names.get(m.group(0), m.group(0)), text)


def pragma_once(text: str) -> str:
    lines = text.splitlines()
    # An include guard: the first two non-blank lines are #ifndef X / #define X and the last
    # non-blank line is #endif.
    idx = [i for i, line in enumerate(lines) if line.strip()]
    if len(idx) >= 3:
        a, b, z = lines[idx[0]].split(), lines[idx[1]].split(), lines[idx[-1]].strip()
        if (
            len(a) == 2
            and a[0] == "#ifndef"
            and len(b) == 2
            and b[0] == "#define"
            and a[1] == b[1]
            and z.startswith("#endif")
        ):
            lines[idx[0]] = "#pragma once"
            del lines[idx[-1]]
            del lines[idx[1]]
    return "\n".join(lines) + "\n"


def wrap_namespace(text: str, namespace: str) -> str:
    lines = text.splitlines()
    last = -1
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("#include") or s == "#pragma once":
            last = i
    head, body = lines[: last + 1], lines[last + 1 :]
    while body and not body[0].strip():
        body.pop(0)
    while body and not body[-1].strip():
        body.pop()
    return "\n".join(
        head + ["", f"namespace {namespace} {{", ""] + body + ["", f"}}  // namespace {namespace}"]
    ) + "\n"


def header(cfg: dict, entry: dict) -> str:
    short = cfg["commit"][:7]
    sources = []
    for part in entry["parts"]:
        if part["path"] not in sources:
            sources.append(part["path"])
    lines = [
        # Split so that REUSE does not read these lines as this file's own header.
        "// SPDX-" "FileCopyrightText: 2026 The dynG Authors",
        "// SPDX-" "License-Identifier: Apache-2.0",
    ]
    lines += [f"// Derived from {cfg['origin']}@{short}:{p}" for p in sources]
    lines += [
        "/**",
        f" * @file {entry['target']}",
        *textwrap.wrap(
            entry["brief"], width=100, initial_indent=" * @brief ", subsequent_indent=" *        "
        ),
        " *",
        f" * Ported from {cfg['origin']}@{short} with the renames of",
        " * parity/escher_port/rename_map.toml (M8a, ADR 0038; the encoding is described in",
        " * cpp/src/hypergraph/escher/README.md).",
        " */",
    ]
    return "\n".join(lines) + "\n"


def mechanical(cfg: dict, original: Path, out: Path) -> list[str]:
    targets = []
    for entry in cfg["file"]:
        text = assemble(original, cfg["commit"], entry["parts"])
        text = rewrite_includes(text, cfg.get("includes", {}))
        text = rename(text, cfg.get("context", []), cfg.get("names", {}))
        text = pragma_once(text)
        text = wrap_namespace(text, cfg["namespace"])
        text = header(cfg, entry) + text
        dest = out / cfg["target_dir"] / entry["target"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(text, encoding="utf-8")
        targets.append(str(dest))
    clang_format = shutil.which("clang-format")
    if clang_format is None:
        raise SystemExit("transplant: clang-format not found (source scripts/dev_env.sh)")
    shutil.copy(REPO / ".clang-format", out / ".clang-format")
    subprocess.check_call([clang_format, "-i", "--style=file", *targets], cwd=out)
    (out / ".clang-format").unlink()
    return targets


def apply_hunks(out: Path) -> None:
    if HUNKS.exists() and HUNKS.stat().st_size > 0:
        subprocess.check_call(
            ["patch", "-p1", "--quiet", "--no-backup-if-mismatch", "-i", str(HUNKS)], cwd=out
        )


def default_original() -> Path:
    base = os.environ.get("DYNG_ORIGINALS", str(Path.home() / "Projects"))
    return Path(base) / "ESCHER-GPU"


def check(cfg: dict, generated: Path, rev: str) -> int:
    status = 0
    target_dir = cfg["target_dir"]
    produced = sorted(p for p in (generated / target_dir).rglob("*") if p.is_file())
    for p in produced:
        rel = p.relative_to(generated).as_posix()
        try:
            committed = subprocess.check_output(
                ["git", "-C", str(REPO), "show", f"{rev}:{rel}"], text=True
            )
        except subprocess.CalledProcessError:
            print(f"{rel}: not in {rev}")
            status = 1
            continue
        mine = p.read_text(encoding="utf-8")
        if mine != committed:
            status = 1
            sys.stdout.writelines(
                difflib.unified_diff(
                    committed.splitlines(True), mine.splitlines(True), f"{rev}:{rel}", "transplant"
                )
            )
    print(f"transplant: {len(produced)} files, {'identical' if status == 0 else 'DIFFERENT'}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--original", type=Path, default=default_original())
    ap.add_argument("--out", type=Path, help="write the transplant (with the manual hunks) here")
    ap.add_argument("--mechanical", type=Path, help="write the mechanical part only here")
    ap.add_argument("--check", action="store_true", help="compare with --rev")
    ap.add_argument("--rev", default="HEAD", help="the commit of the transplant (c1)")
    args = ap.parse_args()
    cfg = load_map()
    if args.mechanical:
        mechanical(cfg, args.original, args.mechanical)
        return 0
    if args.out:
        mechanical(cfg, args.original, args.out)
        apply_hunks(args.out)
        return 0
    if args.check:
        with tempfile.TemporaryDirectory(prefix="dyng-transplant-") as tmp:
            out = Path(tmp)
            mechanical(cfg, args.original, out)
            apply_hunks(out)
            return check(cfg, out, args.rev)
    ap.print_usage()
    return 2


if __name__ == "__main__":
    sys.exit(main())
