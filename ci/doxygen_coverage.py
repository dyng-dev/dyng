#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Check the Doxygen conventions that Doxygen itself does not enforce (PLAN Section 9.1).

Doxygen (run by ci/docs.sh with WARN_AS_ERROR) already fails on undocumented entities and on
missing @param / @return. This script reads its XML output and additionally requires:

  * every public header has an @file comment with a @brief;
  * every class, struct, function, enumeration, typedef and variable at namespace scope belongs
    to a group (@ingroup / @defgroup); members of a class inherit the group of their class;
  * every documented entity has a one-line @brief (a non-empty brief description);
  * compute() and update() of every algorithm namespace carry @backends and @determinism, and
    @paper for the published algorithms (all algorithms ported so far are published).

Usage: ci/doxygen_coverage.py <doxygen-xml-dir>
"""

from __future__ import annotations

import sys
import xml.etree.ElementTree as ET
from pathlib import Path

ALGORITHM_NAMESPACES = {"dyng::sssp"}
REQUIRED_ALGO_PARS = ("Backends:", "Determinism:", "Paper:")
IGNORED_NAMESPACES = {"std"}


def text_of(node: ET.Element | None) -> str:
    return "" if node is None else "".join(node.itertext()).strip()


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    xml_dir = Path(argv[1])
    index = ET.parse(xml_dir / "index.xml").getroot()
    problems: list[str] = []

    grouped_classes: set[str] = set()
    compounds: dict[str, ET.Element] = {}
    for entry in index.findall("compound"):
        refid = entry.get("refid")
        root = ET.parse(xml_dir / f"{refid}.xml").getroot().find("compounddef")
        assert root is not None
        compounds[refid] = root
        if entry.get("kind") == "group":
            grouped_classes.update(c.get("refid") for c in root.findall("innerclass"))

    private_classes = {
        inner.get("refid")
        for c in compounds.values()
        for inner in c.findall("innerclass")
        if inner.get("prot") != "public"
    }
    class_names = {
        text_of(c.find("compoundname"))
        for c in compounds.values()
        if c.get("kind") in ("class", "struct", "union")
    }

    for refid, cdef in compounds.items():
        kind = cdef.get("kind")
        name = text_of(cdef.find("compoundname"))
        where = cdef.find("location")
        loc = where.get("file") if where is not None else refid
        if kind == "file":
            if not text_of(cdef.find("briefdescription")):
                problems.append(f"{loc}: header has no @file / @brief")
            continue
        if kind == "namespace":
            if name in IGNORED_NAMESPACES:
                continue
            for member in cdef.iter("memberdef"):
                problems.append(
                    f"{member.find('location').get('file')}:{member.find('location').get('line')}:"
                    f" {name}::{text_of(member.find('name'))} is not in any group (@ingroup)"
                )
            continue
        if refid in private_classes:
            continue
        if kind in ("class", "struct", "union"):
            outer = name.rsplit("::", 1)[0]
            if refid not in grouped_classes and outer not in class_names:
                problems.append(f"{loc}: {kind} {name} is not in any group (@ingroup)")
            if not text_of(cdef.find("briefdescription")):
                problems.append(f"{loc}: {kind} {name} has no @brief")
        for member in cdef.iter("memberdef"):
            if member.get("prot") not in (None, "public"):
                continue
            mname = text_of(member.find("name"))
            mloc = member.find("location")
            at = f"{mloc.get('file')}:{mloc.get('line')}" if mloc is not None else loc
            if member.get("kind") == "friend":
                continue
            if text_of(member.find("argsstring")).replace(" ", "").endswith("=delete"):
                continue  # deleted special members (non-copyable, non-movable) need no text
            if not text_of(member.find("briefdescription")):
                problems.append(f"{at}: {name}::{mname} has no @brief")
            if kind == "group" and member.get("kind") == "function" and mname in (
                "compute",
                "update",
            ):
                qualified = text_of(member.find("qualifiedname"))
                namespace = qualified.rsplit("::", 1)[0]
                if namespace in ALGORITHM_NAMESPACES:
                    titles = {text_of(t) for t in member.iter("title")}
                    for required in REQUIRED_ALGO_PARS:
                        if required not in titles:
                            tag = "@" + required.rstrip(":").lower()
                            problems.append(f"{at}: {qualified} has no {tag}")

    for problem in sorted(set(problems)):
        print(f"doxygen-coverage: {problem}")
    if problems:
        print(f"doxygen-coverage: {len(set(problems))} problem(s)")
        return 1
    print(f"doxygen-coverage: OK ({len(compounds)} compounds checked)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
