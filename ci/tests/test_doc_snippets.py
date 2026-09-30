# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""ci/doc_snippets.py finds the marked snippets, counts their lines and rejects broken markers."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import doc_snippets  # noqa: E402

PY = "\n".join(["import dyng", "", *[f"x{i} = {i}" for i in range(8)]])
CPP = "\n".join(
    [
        "#include <dyng/dyng.hpp>",
        "",
        "int main() {",
        *[f"  int x{i} = {i};" for i in range(10)],
        "}",
    ]
)


def _tree(tmp_path: Path, readme: str, page: str = "") -> Path:
    (tmp_path / "docs").mkdir()
    (tmp_path / "README.md").write_text(readme, encoding="utf-8")
    (tmp_path / "docs" / "page.md").write_text(page, encoding="utf-8")
    return tmp_path


def _readme(py: str = PY, cpp: str = CPP) -> str:
    return (
        "# Title\n\n<!-- snippet: quickstart-python -->\n```python\n" + py + "\n```\n\n"
        "<!-- snippet-output: quickstart-python -->\n```text\nhello\n```\n\n"
        "<!-- snippet: quickstart-cpp -->\n```cpp\n" + cpp + "\n```\n"
    )


def test_the_repository_passes() -> None:
    assert doc_snippets.check() == []


def test_snippets_and_outputs_are_found(tmp_path: Path) -> None:
    root = _tree(tmp_path, _readme())
    snippets = doc_snippets.all_snippets(root)
    assert [s.name for s in snippets] == ["quickstart-python", "quickstart-cpp"]
    assert snippets[0].expected == "hello\n"
    assert snippets[1].expected is None
    assert doc_snippets.code_lines(snippets[0]) == 9
    assert doc_snippets.code_lines(snippets[1]) == 10
    assert doc_snippets.check(root) == []


def test_a_long_quickstart_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path, _readme(py=PY + "\ny = 1\nz = 2"))
    assert any("PLAN 9.6 allows 10" in p for p in doc_snippets.check(root))


def test_a_missing_quickstart_is_reported(tmp_path: Path) -> None:
    root = _tree(tmp_path, "# Title\n")
    problems = doc_snippets.check(root)
    assert any("quickstart-python" in p for p in problems)
    assert any("quickstart-cpp" in p for p in problems)


def test_other_cpp_snippets_and_unknown_languages_are_reported(tmp_path: Path) -> None:
    page = "<!-- snippet: other -->\n```cpp\nint x;\n```\n<!-- snippet: sh -->\n```bash\nls\n```\n"
    problems = doc_snippets.check(_tree(tmp_path, _readme(), page))
    assert any("only README.md's quickstart-cpp" in p for p in problems)
    assert any("'bash'" in p for p in problems)


@pytest.mark.parametrize(
    "text",
    [
        "<!-- snippet-output: nothing -->\n```text\nx\n```\n",  # output of an unknown snippet
        "<!-- snippet: a -->\nno fence here\n",  # a marker without a block
        "<!-- snippet: a -->\n```python\nx = 1\n",  # an unterminated block
        "<!-- snippet: a -->\n```python\nx\n```\n<!-- snippet: a -->\n```python\ny\n```\n",
    ],
)
def test_broken_markers_fail(tmp_path: Path, text: str) -> None:
    with pytest.raises(SystemExit):
        doc_snippets.all_snippets(_tree(tmp_path, text))
