# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The readers reached from Python on arbitrary text (Hypothesis; the Python side of the reader
fuzzers of cpp/fuzz, docs/developer/robustness.md).

The command line's text batches (``dyng cycle_count update --batch``) are parsed in Python, so
libFuzzer does not reach them; ``dyng.io``'s readers wrap the C++ readers the libFuzzer targets
cover, plus the Python conversions around them. Every input must either be read or be rejected
with the documented exception (``dyng.FileFormatError``; the command line's ``CliError`` for a
usage problem), never with another exception.
"""

from __future__ import annotations

from pathlib import Path

import pytest

hypothesis = pytest.importorskip("hypothesis")

import dyng  # noqa: E402
from dyng.cli._algorithms import batch_text, read_text_batch  # noqa: E402
from dyng.cli._common import CliError  # noqa: E402
from hypothesis import example, given  # noqa: E402
from hypothesis import strategies as st  # noqa: E402

# Lines that look like the formats (operators, integers of every size, separators, comments)
# mixed with arbitrary text, so most inputs get past the first token.
TOKENS = st.one_of(
    st.sampled_from(["+", "-", "+e", "-e", "%batch", "%dgt", "#", "1", "0", "-1", ",", "x"]),
    st.integers(-(2**70), 2**70).map(str),
    st.text(max_size=4),
)
LINES = st.lists(TOKENS, max_size=6).map(" ".join)
TEXTS = st.lists(LINES, max_size=12).map("\n".join) | st.text(max_size=60)
GRAPHS = {
    "unweighted": dyng.Graph.from_edges([0, 1], [1, 2]),
    "weighted": dyng.Graph.from_edges([0, 1], [1, 2], [[3, 4], [5, 6]]),
}


@given(TEXTS, st.sampled_from(sorted(GRAPHS)))
@example("+ 0 99999999999999999999999\n", "unweighted")
@example("- 1 2\n+ 0 1 7 8\n", "weighted")
@example("+ 0 1\n\udcff", "unweighted")
def test_cli_text_batch(tmp_path_factory: pytest.TempPathFactory, text: str, graph: str) -> None:
    path = tmp_path_factory.mktemp("batch") / "batch.txt"
    path.write_bytes(text.encode("utf-8", "surrogateescape"))
    g = GRAPHS[graph]
    try:
        batch = read_text_batch(str(path), g)
    except (dyng.FileFormatError, CliError):
        return
    # Accepted: the deletions and insertions survive the generator's text form.
    again = path.with_suffix(".again")
    again.write_text(batch_text(batch))
    back = read_text_batch(str(again), g)
    assert back.delete_src.tolist() == batch.delete_src.tolist()
    assert back.delete_dst.tolist() == batch.delete_dst.tolist()
    assert back.insert_src.tolist() == batch.insert_src.tolist()
    assert back.insert_dst.tolist() == batch.insert_dst.tolist()


@given(TEXTS, st.sampled_from([None, 0, 1, 2]))
@example("%dgt 1\n%batch 0\n+e 0 1 5\n", None)
def test_dgt_batches(tmp_path_factory: pytest.TempPathFactory, text: str, k: int | None) -> None:
    path = tmp_path_factory.mktemp("dgt") / "b.dgt"
    path.write_bytes(text.encode("utf-8", "surrogateescape"))
    try:
        batches = dyng.io.read_batches(str(path), num_weights=k)
    except dyng.FileFormatError:
        return
    out = Path(str(path) + ".copy.dgt")
    dyng.io.write_batches(out, batches)
    back = dyng.io.read_batches(str(out), num_weights=k)
    assert [b.insert_src.tolist() for b in back] == [b.insert_src.tolist() for b in batches]
    assert [b.delete_dst.tolist() for b in back] == [b.delete_dst.tolist() for b in batches]


@given(TEXTS)
@example("%%MatrixMarket matrix coordinate pattern general\n2 2 1\n1 2\n")
def test_edge_list(tmp_path_factory: pytest.TempPathFactory, text: str) -> None:
    path = tmp_path_factory.mktemp("el") / "edges.txt"
    path.write_bytes(text.encode("utf-8", "surrogateescape"))
    try:
        g = dyng.io.read_edge_list(str(path))
    except dyng.FileFormatError:
        return
    assert g.num_edges >= 0
