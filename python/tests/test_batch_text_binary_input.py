# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The .dgt batch readers given a binary file raise FileFormatError (R040's review).

A ``.dgb`` file (the kNN recipe writes one) given to ``read_batches`` or
``read_hyperedge_batches`` is named from its magic, with the readers to use instead; any other
input with bytes that are not UTF-8 is rejected as malformed with those bytes escaped in the
message, never with a UnicodeDecodeError raised in place of the error.
"""

from __future__ import annotations

from pathlib import Path

import dyng
import pytest

READERS = {
    "read_batches": dyng.io.read_batches,
    "read_hyperedge_batches": dyng.io.read_hyperedge_batches,
}


@pytest.mark.parametrize("kind", ["graph", "hypergraph"])
@pytest.mark.parametrize("name", sorted(READERS))
def test_a_dgb_file_is_named_from_its_magic(tmp_path: Path, name: str, kind: str) -> None:
    path = tmp_path / "b.dgb"
    if kind == "graph":
        batches = [dyng.EdgeBatch(insert=([14, 0], [2, 1]))]
    else:
        batches = [dyng.HyperedgeBatch([0], [[2, 0]])]
    dyng.io.write_binary_batches(path, batches)
    with pytest.raises(dyng.FileFormatError) as e:
        READERS[name](path)
    message = str(e.value)
    assert "DYNGBAT1" in message and "BatchReader" in message and name in message
    assert e.value.path == str(path)


@pytest.mark.parametrize("name", sorted(READERS))
def test_bytes_that_are_not_utf8_are_escaped(tmp_path: Path, name: str) -> None:
    path = tmp_path / "b.dgt"
    path.write_bytes(b"%dgt 1\n%batch 0\n\xbd\xef 1 2\n")
    with pytest.raises(dyng.FileFormatError) as e:
        READERS[name](path)
    assert "\\xbd" in str(e.value)


def test_arbitrary_bytes_are_read_or_rejected(tmp_path: Path) -> None:
    hypothesis = pytest.importorskip("hypothesis")
    st = hypothesis.strategies
    path = tmp_path / "b.dgt"

    @hypothesis.given(st.binary(max_size=80))
    @hypothesis.settings(deadline=None)
    def check(data: bytes) -> None:
        path.write_bytes(data)
        for reader in READERS.values():
            try:
                reader(path)
            except dyng.FileFormatError:
                pass

    check()
