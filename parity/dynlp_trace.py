# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""Reader of the DynLP stage traces (.dlt) of the LabelPropagation-CUDA@a276a3a recording twins
(parity/exporters/dynlp; ADR 0065; the format is documented in
parity/exporters/dynlp/dynlp_trace.hpp).

    read(path)            -> Trace (every record, in order; refuses a trace without its end record)
    Trace.records(tag)    -> the records of a tag
    Trace.batches()       -> {batch: {tag: record}} for the records of the batch loop
    Record.fields[name]   -> Field(dtype, digest, count, raw) with raw the little-endian bytes,
                             or None in the digest mode (Field.sha256 is the hex digest then);
                             Field.values() is an array.array of them (floats as float32 values),
                             Field.numpy() a numpy view (numpy is imported only there)

Every array is also given its SHA-256 (Field.sha256: of the raw little-endian bytes, the same
value the digest mode stores), so a raw trace and a digest trace of the same run compare field
by field.
"""

from __future__ import annotations

import array
import hashlib
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path

MAGIC = b"DYNGDLT1"
RECORD_MAGIC = 0x5250444C
# dtype -> (element size, array.array type code of the values, numpy dtype of the values)
DTYPES = {
    1: (4, "i", "<i4"),
    2: (8, "q", "<i8"),
    3: (8, "Q", "<u8"),
    4: (4, "f", "<f4"),  # stored as the IEEE-754 bits; values() gives the floats
    5: (1, "B", "u1"),
    6: (8, "d", "<f8"),
    7: (1, "B", "u1"),  # UTF-8 text
}
BITS = {4: "I", 6: "Q"}  # the bit patterns of the float dtypes
assert sys.byteorder == "little", "the traces are little-endian; this reader assumes a host that is"
DTYPE_NAMES = {1: "i32", 2: "i64", 3: "u64", 4: "f32", 5: "u8", 6: "f64", 7: "text"}
# The stages of one batch, in the order runDynLP calls them (absent ones are skipped).
BATCH_STAGES = (
    "apply_vertex_changes",
    "append_edges",
    "sparsify",
    "connected_components",
    "label_init",
    "propagation_begin",
    "propagation_end",
    "irlp_snapshot",
)


@dataclass
class Field:
    name: str
    dtype: int
    digest: bool
    count: int
    raw: bytes | None
    sha256: str

    @property
    def type_name(self) -> str:
        return DTYPE_NAMES[self.dtype]

    def _raw(self) -> bytes:
        if self.raw is None:
            raise ValueError(f"{self.name}: stored as a digest")
        return self.raw

    def values(self) -> array.array:
        """The elements (floats as Python floats of the stored float32 values)."""
        a = array.array(DTYPES[self.dtype][1])
        a.frombytes(self._raw())
        return a

    def bits(self) -> array.array:
        """The bit patterns of a float field (u32 for f32), else the values."""
        a = array.array(BITS.get(self.dtype, DTYPES[self.dtype][1]))
        a.frombytes(self._raw())
        return a

    def numpy(self):
        import numpy as np  # only the heavy tools need it

        return np.frombuffer(self._raw(), dtype=np.dtype(DTYPES[self.dtype][2]))

    def scalar(self):
        if self.count != 1:
            raise ValueError(f"{self.name}: not a scalar")
        v = self.values()[0]
        return float(v) if self.dtype in (4, 6) else int(v)

    def text(self) -> str:
        assert self.dtype == 7
        return self._raw().decode()


@dataclass
class Record:
    tag: str
    batch: int
    sequence: int
    fields: dict[str, Field] = field(default_factory=dict)

    def __getitem__(self, name: str) -> Field:
        return self.fields[name]

    def scalar(self, name: str):
        return self.fields[name].scalar()


@dataclass
class Trace:
    path: Path
    digest_mode: bool
    all: list[Record]

    def records(self, tag: str) -> list[Record]:
        return [r for r in self.all if r.tag == tag]

    def one(self, tag: str) -> Record:
        rs = self.records(tag)
        if len(rs) != 1:
            raise ValueError(f"{self.path}: {len(rs)} records '{tag}', expected one")
        return rs[0]

    def batches(self) -> dict[int, dict[str, Record]]:
        """The records of the batch loop by batch and stage; a stage appears at most once per
        batch (a second one is an error: the batch marker would be wrong)."""
        out: dict[int, dict[str, Record]] = {}
        for r in self.all:
            if r.tag not in BATCH_STAGES:
                continue
            per = out.setdefault(r.batch, {})
            if r.tag in per:
                raise ValueError(f"{self.path}: batch {r.batch} has two '{r.tag}' records")
            per[r.tag] = r
        return out


def read(path: Path | str) -> Trace:
    path = Path(path)
    return read_bytes(path.read_bytes(), path)


def read_bytes(buf: bytes, name: Path | str = "<trace>") -> Trace:
    """A trace from its bytes (e.g. a decompressed golden); `name` names it in errors."""
    try:
        return _read(buf, Path(name))
    except struct.error as e:
        raise ValueError(f"{name}: truncated or malformed ({e})") from e


def _read(buf: bytes, path: Path) -> Trace:
    if buf[:8] != MAGIC:
        raise ValueError(f"{path}: not a DynLP trace")
    version, flags = struct.unpack_from("<II", buf, 8)
    if version != 1:
        raise ValueError(f"{path}: format version {version} (this reader knows 1)")
    pos = 16
    records: list[Record] = []
    end_seen = False
    while pos < len(buf):
        if end_seen:
            raise ValueError(f"{path}: data after the end record")
        if pos + 6 > len(buf):
            raise ValueError(f"{path}: truncated record at byte {pos}")
        (magic,) = struct.unpack_from("<I", buf, pos)
        if magic != RECORD_MAGIC:
            raise ValueError(f"{path}: bad record magic at byte {pos}")
        (tag_len,) = struct.unpack_from("<H", buf, pos + 4)
        tag = buf[pos + 6 : pos + 6 + tag_len].decode()
        pos += 6 + tag_len
        batch, seq, nfields = struct.unpack_from("<iiI", buf, pos)
        pos += 12
        rec = Record(tag, batch, seq)
        for _ in range(nfields):
            (name_len,) = struct.unpack_from("<H", buf, pos)
            name = buf[pos + 2 : pos + 2 + name_len].decode()
            pos += 2 + name_len
            dtype, storage, count = struct.unpack_from("<BBQ", buf, pos)
            pos += 10
            if dtype not in DTYPES:
                raise ValueError(f"{path}: {tag}.{name}: unknown dtype {dtype}")
            if storage == 1:
                digest = buf[pos : pos + 32].hex()
                pos += 32
                rec.fields[name] = Field(name, dtype, True, count, None, digest)
            elif storage == 0:
                size = count * DTYPES[dtype][0]
                raw = buf[pos : pos + size]
                if len(raw) != size:
                    raise ValueError(f"{path}: {tag}.{name}: truncated")
                pos += size
                rec.fields[name] = Field(
                    name, dtype, False, count, raw, hashlib.sha256(raw).hexdigest()
                )
            else:
                raise ValueError(f"{path}: {tag}.{name}: unknown storage {storage}")
        if tag == "end":
            if rec.scalar("records") != len(records):
                raise ValueError(f"{path}: the end record counts {rec.scalar('records')} records, "
                                 f"the file has {len(records)}")  # fmt: skip
            end_seen = True
        else:
            records.append(rec)
    if not end_seen:
        raise ValueError(f"{path}: no end record (the recording process did not close the trace)")
    return Trace(path, bool(flags & 1), records)


def encode(records: list[Record], digest_mode: bool = False) -> bytes:
    """A trace file of `records` in the recorder's format (with its end record): what the
    goldens store as a canonical trace (a recording with its run-varying fields left out)."""
    out = [MAGIC, struct.pack("<II", 1, 1 if digest_mode else 0)]

    def record(tag: str, batch: int, seq: int, fields: list[Field]) -> bytes:
        parts = [struct.pack("<IH", RECORD_MAGIC, len(tag)), tag.encode(),
                 struct.pack("<iiI", batch, seq, len(fields))]  # fmt: skip
        for f in fields:
            parts.append(struct.pack("<H", len(f.name)) + f.name.encode())
            if f.digest:
                parts.append(struct.pack("<BBQ", f.dtype, 1, f.count) + bytes.fromhex(f.sha256))
            else:
                parts.append(struct.pack("<BBQ", f.dtype, 0, f.count) + f._raw())
        return b"".join(parts)

    for r in records:
        out.append(record(r.tag, r.batch, r.sequence, list(r.fields.values())))
    n = struct.pack("<Q", len(records))
    end = Field("records", 3, False, 1, n, hashlib.sha256(n).hexdigest())
    out.append(record("end", -1, 0, [end]))
    return b"".join(out)


def without(records: list[Record], drop: set[tuple[str, str]]) -> list[Record]:
    """The records with the fields (tag, name) of `drop` left out."""
    return [Record(r.tag, r.batch, r.sequence,
                   {k: f for k, f in r.fields.items() if (r.tag, k) not in drop})
            for r in records]  # fmt: skip
