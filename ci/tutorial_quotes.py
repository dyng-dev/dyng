#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The drift check of the tutorials' quoted code (PLAN 9.2-9.3: tutorials quote tested code, and
CI breaks on drift; M9b, ADR 0058).

    ci/tutorial_quotes.py --check     # ci/docs.sh: exit 1 on a moved, changed or missing quote
    ci/tutorial_quotes.py --update    # after re-reading the prose: record the new digests

Every `literalinclude` of a tutorial page (docs/tutorials/*.md) quotes a file of the repository by
markers (`:start-after:` / `:end-before:`, the `// [tutorial: <name>]` comments of the reference
code; `# [tutorial: <name>]` in a Python, CMake or shell file), by the first and last quoted lines
(`:start-at:` / `:end-at:`, anchor `at-<digest of the two strings>`) or by line numbers
(`:lines: a-b`), and is preceded by a MyST comment that pins what it quotes:

    % quote: examples/tutorial_algorithms/my_bfs/problem.hpp#workspace 3f2a9c01d4e7

(the path from the repository root, `#<marker>` or `#L<a>-L<b>`, and the first 12 hex digits of
the SHA-256 of the quoted text). The check fails when a quoted file or marker is missing or
repeated, when a quote has no pin, and when the quoted text differs from its pin: a line moved into
or out of a marked range, a quoted line range that now holds other lines, or an edited line. The
Sphinx build (-W) already fails on a missing marker; this check also catches the drift that keeps
the markers, so the prose that explains a quote is re-read whenever the quote changes (then
`--update` records the new digest).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PAGES = "docs/tutorials/*.md"

_PIN = re.compile(r"^% quote: (?P<path>[^#\s]+)#(?P<anchor>.+) (?P<digest>[0-9a-f]{12})\s*$")
_INCLUDE = re.compile(r"^(`{3,}|~{3,})\{literalinclude\}\s+(?P<target>\S+)\s*$")
_OPTION = re.compile(r"^:(?P<key>[a-z-]+):\s*(?P<value>.*?)\s*$")


class QuoteError(Exception):
    """A quote that cannot be resolved."""


# The `:start-at:` / `:end-at:` strings of the anchors "at-<digest>" read so far.
at_bounds: dict[str, tuple[str, str]] = {}


@dataclass
class Quote:
    """One literalinclude of a page."""

    page: Path
    line: int  # 1-based line of the literalinclude
    path: str  # the quoted file, from the repository root
    anchor: str  # "<marker>" or "L<a>-L<b>"
    pin_line: int | None  # 1-based line of its `% quote:` pin, if any
    pinned: str | None  # the pinned digest


# The line-comment prefix of the markers by the quoted file's suffix (`// [tutorial: <name>]` in
# C++ and CUDA, `# [tutorial: <name>]` in Python, CMake and shell files).
HASH_COMMENT_SUFFIXES = (".py", ".cmake", ".sh", ".txt", ".toml", ".yaml", ".yml")


def comment_of(path: str) -> str:
    return "#" if path.endswith(HASH_COMMENT_SUFFIXES) else "//"


def unquote(value: str) -> str:
    return value[1:-1] if len(value) >= 2 and value[0] == value[-1] == '"' else value


def quotes(page: Path) -> list[Quote]:
    lines = page.read_text(encoding="utf-8").splitlines()
    out: list[Quote] = []
    for i, line in enumerate(lines):
        m = _INCLUDE.match(line)
        if not m:
            continue
        options: dict[str, str] = {}
        for opt in lines[i + 1 :]:
            o = _OPTION.match(opt)
            if not o:
                break
            options[o["key"]] = unquote(o["value"])
        target = (page.parent / m["target"]).resolve()
        try:
            path = target.relative_to(REPO).as_posix()
        except ValueError as e:
            raise QuoteError(f"{page}:{i + 1}: {m['target']} is outside the repository") from e
        if "start-after" in options:
            c = comment_of(path)
            marker = re.match(
                rf"^{re.escape(c)} \[tutorial: (?P<name>[^\]]+)\]$", options["start-after"]
            )
            if not marker or options.get("end-before") != f"{c} [tutorial: {marker['name']} end]":
                raise QuoteError(
                    f"{page}:{i + 1}: quote by the markers `{c} [tutorial: <name>]` and "
                    f"`{c} [tutorial: <name> end]`"
                )
            anchor = marker["name"]
        elif "start-at" in options and "end-at" in options:
            # A quote from the line holding `start-at` through the next holding `end-at` (both
            # included, as Sphinx does); its anchor names the two strings by a digest.
            key = options["start-at"] + "\0" + options["end-at"]
            anchor = "at-" + hashlib.sha256(key.encode("utf-8")).hexdigest()[:8]
            at_bounds[anchor] = (options["start-at"], options["end-at"])
        elif "lines" in options:
            r = re.match(r"^(\d+)-(\d+)$", options["lines"])
            if not r:
                raise QuoteError(f"{page}:{i + 1}: `:lines:` must be one range a-b")
            anchor = f"L{r[1]}-L{r[2]}"
        else:
            raise QuoteError(f"{page}:{i + 1}: a quote needs markers or a line range")
        pin = _PIN.match(lines[i - 1]) if i > 0 else None
        if pin and (pin["path"] != path or pin["anchor"] != anchor):
            raise QuoteError(
                f"{page}:{i}: the pin names {pin['path']}#{pin['anchor']}, the quote "
                f"{path}#{anchor}"
            )
        out.append(
            Quote(page, i + 1, path, anchor, i if pin else None, pin["digest"] if pin else None)
        )
    return out


def quoted_text(path: str, anchor: str) -> str:
    file = REPO / path
    if not file.is_file():
        raise QuoteError(f"{path}: the quoted file does not exist")
    text = file.read_text(encoding="utf-8")
    if anchor.startswith("at-"):
        if anchor not in at_bounds:
            raise QuoteError(f"{path}: unknown quote {anchor}")
        first, last = at_bounds[anchor]
        lines = text.splitlines(keepends=True)
        a = next((i for i, ln in enumerate(lines) if first in ln), None)
        b = next((i for i, ln in enumerate(lines) if a is not None and i >= a and last in ln), None)
        if a is None or b is None:
            raise QuoteError(f"{path}: the quoted lines {first!r} .. {last!r} are not found")
        return "".join(lines[a : b + 1])
    r = re.match(r"^L(\d+)-L(\d+)$", anchor)
    if r:
        lines = text.splitlines(keepends=True)
        a, b = int(r[1]), int(r[2])
        if not 1 <= a <= b <= len(lines):
            raise QuoteError(f"{path}: lines {a}-{b} are outside the file ({len(lines)} lines)")
        return "".join(lines[a - 1 : b])
    c = comment_of(path)
    begin, end = f"{c} [tutorial: {anchor}]", f"{c} [tutorial: {anchor} end]"
    if text.count(begin) != 1 or text.count(end) != 1:
        raise QuoteError(f"{path}: the marker {begin!r} or its end is missing or repeated")
    start = text.index("\n", text.index(begin)) + 1
    stop = text.index(end)
    if stop <= start:
        raise QuoteError(f"{path}: the range of {begin!r} is empty or its end comes first")
    # The end marker's own indentation is not part of the quote.
    body = text[start:stop]
    return body[: body.rfind("\n") + 1]


def digest(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def check(pages: list[Path]) -> list[str]:
    problems: list[str] = []
    for page in pages:
        try:
            for q in quotes(page):
                where = f"{page.relative_to(REPO)}:{q.line}"
                try:
                    now = digest(quoted_text(q.path, q.anchor))
                except QuoteError as e:
                    problems.append(f"{where}: {e}")
                    continue
                if q.pinned is None:
                    problems.append(
                        f"{where}: the quote of {q.path}#{q.anchor} has no `% quote:` pin (run "
                        "ci/tutorial_quotes.py --update)"
                    )
                elif q.pinned != now:
                    problems.append(
                        f"{where}: {q.path}#{q.anchor} changed since the page was written "
                        f"(pinned {q.pinned}, now {now}): re-read the prose around the quote, "
                        "then run ci/tutorial_quotes.py --update"
                    )
        except QuoteError as e:
            problems.append(str(e))
    return problems


def update(pages: list[Path]) -> int:
    changed = 0
    for page in pages:
        lines = page.read_text(encoding="utf-8").splitlines(keepends=True)
        for q in reversed(quotes(page)):
            pin = f"% quote: {q.path}#{q.anchor} {digest(quoted_text(q.path, q.anchor))}\n"
            if q.pin_line is None:
                lines.insert(q.line - 1, pin)
                changed += 1
            elif lines[q.pin_line - 1] != pin:
                lines[q.pin_line - 1] = pin
                changed += 1
        page.write_text("".join(lines), encoding="utf-8")
    return changed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true", help="fail on drift (ci/docs.sh)")
    mode.add_argument("--update", action="store_true", help="record the current digests")
    parser.add_argument("pages", nargs="*", type=Path, help=f"pages (default: {PAGES})")
    args = parser.parse_args(argv)
    pages = [p.resolve() for p in args.pages] or sorted(REPO.glob(PAGES))
    if args.update:
        print(f"tutorial-quotes: {update(pages)} pin(s) written")
        return 0
    problems = check(pages)
    for p in problems:
        print(f"tutorial-quotes: {p}", file=sys.stderr)
    if problems:
        return 1
    count = sum(len(quotes(p)) for p in pages)
    print(f"tutorial-quotes: OK ({count} quotes in {len(pages)} pages match their pins)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
