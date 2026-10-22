# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Tests of ci/tutorial_quotes.py: a quoted line that moves, changes or disappears fails the
drift check; --update records the new pins; the repository's tutorials pass."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "tutorial_quotes", Path(__file__).resolve().parent.parent / "tutorial_quotes.py"
)
tutorial_quotes = importlib.util.module_from_spec(_SPEC)
sys.modules["tutorial_quotes"] = tutorial_quotes
_SPEC.loader.exec_module(tutorial_quotes)

CODE = """\
#pragma once
// [tutorial: op]
struct op {
  int k = 2;
};
// [tutorial: op end]
int helper();
int other();
"""

PAGE = """\
# A tutorial

```{literalinclude} ../../examples/code.hpp
:language: cpp
:start-after: "// [tutorial: op]"
:end-before: "// [tutorial: op end]"
```

and two lines:

```{literalinclude} ../../examples/code.hpp
:language: cpp
:lines: 7-8
```
"""


@pytest.fixture
def repo(tmp_path: Path, monkeypatch) -> Path:
    (tmp_path / "examples").mkdir()
    (tmp_path / "examples" / "code.hpp").write_text(CODE)
    (tmp_path / "docs" / "tutorials").mkdir(parents=True)
    (tmp_path / "docs" / "tutorials" / "page.md").write_text(PAGE)
    monkeypatch.setattr(tutorial_quotes, "REPO", tmp_path)
    return tmp_path


def run(*args: str) -> int:
    return tutorial_quotes.main(list(args))


def test_a_quote_without_a_pin_fails_and_update_pins_it(repo: Path, capsys) -> None:
    assert run("--check") == 1
    assert "has no `% quote:` pin" in capsys.readouterr().err
    assert run("--update") == 0
    page = (repo / "docs" / "tutorials" / "page.md").read_text()
    assert "% quote: examples/code.hpp#op " in page
    assert "% quote: examples/code.hpp#L7-L8 " in page
    assert run("--check") == 0


def test_a_line_moved_out_of_a_marked_range_fails(repo: Path, capsys) -> None:
    assert run("--update") == 0
    code = repo / "examples" / "code.hpp"
    text = code.read_text()
    # The markers stay; the quoted struct loses a line (moved below the range).
    code.write_text(
        text.replace("  int k = 2;\n", "").replace("int helper();\n", "int helper();\nint k;\n")
    )
    assert run("--check") == 1
    err = capsys.readouterr().err
    assert "examples/code.hpp#op changed since the page was written" in err


def test_a_quoted_line_range_that_moves_fails(repo: Path, capsys) -> None:
    assert run("--update") == 0
    code = repo / "examples" / "code.hpp"
    code.write_text("// a new first line\n" + code.read_text())
    assert run("--check") == 1
    assert "examples/code.hpp#L7-L8 changed" in capsys.readouterr().err


def test_a_missing_or_repeated_marker_fails(repo: Path, capsys) -> None:
    assert run("--update") == 0
    code = repo / "examples" / "code.hpp"
    code.write_text(code.read_text().replace("// [tutorial: op end]\n", ""))
    assert run("--check") == 1
    assert "missing or repeated" in capsys.readouterr().err


def test_the_repository_tutorials_match_their_pins() -> None:
    assert run("--check") == 0


def test_a_python_file_is_quoted_by_hash_markers(repo: Path, capsys) -> None:
    # The Python examples mark their ranges with `# [tutorial: <name>]` (M9b: the "Hypergraph
    # triads" tutorial quotes examples/python/triad_count.py).
    (repo / "examples" / "code.py").write_text(
        "import dyng\n# [tutorial: run]\nx = 1\n# [tutorial: run end]\n"
    )
    page = repo / "docs" / "tutorials" / "page.md"
    page.write_text(
        PAGE
        + "\n```{literalinclude} ../../examples/code.py\n:language: python\n"
        + ':start-after: "# [tutorial: run]"\n:end-before: "# [tutorial: run end]"\n```\n'
    )
    assert run("--update") == 0
    assert "% quote: examples/code.py#run " in page.read_text()
    assert run("--check") == 0
    code = repo / "examples" / "code.py"
    code.write_text(code.read_text().replace("x = 1", "x = 2"))
    assert run("--check") == 1
    # `//` markers do not quote a Python file.
    page.write_text(page.read_text().replace('"# [tutorial: run]"', '"// [tutorial: run]"'))
    with pytest.raises(tutorial_quotes.QuoteError):
        tutorial_quotes.quotes(page)
