#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Convert a timestamped simplicial dataset of Austin R. Benson's collection to a `.hg` file.

    datasets/preprocess/benson_to_hg.py PREFIX OUT.hg [--variant escher|hsosp]

PREFIX names the dataset's files: PREFIX-nverts.txt holds the vertex count of every simplex and
PREFIX-simplices.txt their vertex ids, in file order (e.g. coauth-DBLP-full/coauth-DBLP-full).
The output has one hyperedge per line, its vertex ids sorted and separated by single spaces:

1. every simplex's vertex ids are deduplicated and sorted;
2. only the first occurrence of every distinct vertex set is kept (a simplex repeats once per
   timestamp in these datasets);
3. the vertices are renumbered 1..V in increasing order of their original id.

These are the preprocessing steps of the two originals' converters, which datasets/registry.toml
pins: ESCHER-GPU@abcf9b2's bench/datasets/convert_coauth.py (`--variant escher`, the default)
and MOSP_ESCHER@4b86159's scripts/benson_to_hg.py (`--variant hsosp`). This script is dynG's
own implementation of them; on coauth-DBLP-full both variants write the same 2,467,389 lines,
byte for byte the file the registry pins for both originals (SHA-256
65fc83b22dfb3ecd14b15436f4aaee54631657f2f4afec8260150679387b7cb6). The variants differ only on
inputs that the registered datasets do not have: `escher` drops simplices without vertices and
reads the files as whitespace-separated tokens; `hsosp` keeps an empty simplex as an empty line,
reads one id per line, and rejects a simplices file with more ids than nverts lists. How the
files are LOADED differs more (dyng::io::hypergraph_read_options::escher_gpu() and ::hsosp():
H-SOSP keeps hyperedges of at most 25 vertices and numbers vertices by first appearance).

io::read_hypergraph() reads the output with either option set (docs/api/file_formats.md, `.hg`).
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator
from pathlib import Path


def _tokens(path: Path) -> list[int]:
    """Every whitespace-separated integer of a file (the escher variant)."""
    with path.open() as f:
        return [int(t) for t in f.read().split()]


def _lines(path: Path) -> Iterator[int]:
    """One integer per non-blank line (the hsosp variant)."""
    with path.open() as f:
        for line in f:
            text = line.strip()
            if text:
                yield int(text)


def convert(prefix: str, out: Path, variant: str) -> tuple[int, int, int]:
    """Write OUT; returns (simplices, distinct hyperedges written, vertices)."""
    nverts_path = Path(prefix + "-nverts.txt")
    simplices_path = Path(prefix + "-simplices.txt")
    distinct: list[tuple[int, ...]] = []
    seen: set[tuple[int, ...]] = set()
    if variant == "escher":
        nverts = _tokens(nverts_path)
        ids = _tokens(simplices_path)
        if sum(nverts) != len(ids):
            raise SystemExit("benson_to_hg: the nverts and simplices files do not match")
        position = 0
        simplices = len(nverts)
        for count in nverts:
            members = tuple(sorted(set(ids[position : position + count])))
            position += count
            if members and members not in seen:
                seen.add(members)
                distinct.append(members)
    else:
        nverts = list(_lines(nverts_path))
        pins = _lines(simplices_path)
        simplices = len(nverts)
        try:
            for count in nverts:
                members = tuple(sorted({next(pins) for _ in range(count)}))
                if members not in seen:
                    seen.add(members)
                    distinct.append(members)
        except StopIteration:
            raise SystemExit(
                "benson_to_hg: the simplices file has fewer ids than nverts lists"
            ) from None
        if next(pins, None) is not None:
            raise SystemExit("benson_to_hg: the simplices file has more ids than nverts lists")
    vertices = sorted({v for members in distinct for v in members})
    number = {v: i + 1 for i, v in enumerate(vertices)}
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for members in distinct:
            f.write(" ".join(str(number[v]) for v in members) + "\n")
    return simplices, len(distinct), len(vertices)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("prefix", help="the dataset's file prefix (PREFIX-nverts.txt, ...)")
    p.add_argument("out", type=Path, help="the .hg file to write")
    p.add_argument(
        "--variant",
        choices=["escher", "hsosp"],
        default="escher",
        help="whose converter to follow (ESCHER-GPU's or MOSP_ESCHER's; default escher)",
    )
    args = p.parse_args(argv)
    simplices, hyperedges, vertices = convert(args.prefix, args.out, args.variant)
    print(
        f"{simplices} simplices -> {hyperedges} distinct hyperedges, {vertices} vertices "
        f"-> {args.out}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
