#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
"""The fast path of `region_push_back` is `push_back`'s (M11b, ADR 0094; criterion 1).

    parity/thread_list_fast_path.py probe [--cxx c++] [--flags "-O3"] [--plant] [--json OUT]
    parity/thread_list_fast_path.py objects --base <openmp.cpp.o> --head <openmp.cpp.o>
        [--json OUT]

sssp's OpenMP engine appends to its per-thread lists inside parallel regions with
`util/thread_list.hpp`'s `region_push_back(list, value, errors)` (ADR 0094: an allocation failure
of a growth must not leave the region), where it called `list.push_back(value)` before. The
promise is that the call costs nothing when the list has room: its fast path is push_back's own
capacity compare and store.

probe    compiles a probe translation unit with the given compiler and flags against cpp/src (the
         library's flags of the parity preset by default: -O3) and disassembles it (objdump). The
         probe holds, for 32- and 64-bit elements, `push_back` and `region_push_back` as
         non-inlined functions (and, with --plant, a planted variant whose fast path also reads
         the carrier, `errors.failed()`, to show the check fails). The fast path of a function is
         the shortest path of instructions from its entry to a `ret` that makes no call (a
         branch to a relocated target, the cold section, is not followed). Its signature: the
         accesses to the list's header (loads and stores through the first argument's register,
         by offset), the number of other memory accesses that are not on the stack (the element
         store), the conditional branches and the calls. The check passes when the signatures of
         `region_push_back` and `push_back` are equal for both element types, and the planted
         variant's differs. Exit 1 otherwise (2 if the probe cannot be built or the target is not
         x86-64: then nothing is checked).
objects  compares the instructions of two builds' `sssp/openmp.cpp.o` (the same preset): per
         function, the instruction count, the calls, and the capacity compares (a load of
         offset 8 of a register followed by a compare with offset 16 of the same register and a
         conditional branch: a std::vector's `finish != end_of_storage`). Recorded, not gated: it
         shows that every push site keeps its compare (the head has the same compares per
         function) and where the cold paths moved.

The record of M11b (parity/results/M11b.md section 1) has both outputs.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PROBE = r"""
#include "util/thread_list.hpp"

#include <cstdint>

using dyng::detail::region_exceptions;
using dyng::detail::thread_list;

namespace {
// --plant: a fast path that also reads the carrier (one more load and branch per push).
template <typename value_t>
inline void planted_push_back(thread_list<value_t>& list, value_t value,
                              region_exceptions& errors) noexcept {
  if (!errors.failed() && list.size() != list.capacity()) {
    list.push_back(value);
  } else {
    dyng::detail::region_grow_push_back(list, value, errors);
  }
}
}  // namespace

extern "C" {
[[gnu::noinline]] void probe_push_back_32(thread_list<std::int32_t>& list, std::int32_t value) {
  list.push_back(value);
}
[[gnu::noinline]] void probe_region_push_back_32(thread_list<std::int32_t>& list,
                                                 std::int32_t value, region_exceptions& errors) {
  dyng::detail::region_push_back(list, value, errors);
}
[[gnu::noinline]] void probe_push_back_64(thread_list<std::int64_t>& list, std::int64_t value) {
  list.push_back(value);
}
[[gnu::noinline]] void probe_region_push_back_64(thread_list<std::int64_t>& list,
                                                 std::int64_t value, region_exceptions& errors) {
  dyng::detail::region_push_back(list, value, errors);
}
#ifdef DYNG_PROBE_PLANT
[[gnu::noinline]] void probe_planted_push_back_32(thread_list<std::int32_t>& list,
                                                  std::int32_t value, region_exceptions& errors) {
  planted_push_back(list, value, errors);
}
#endif
}
"""

FIRST_ARG = "%rdi"  # System V x86-64: the list
LINE = re.compile(r"^\s*([0-9a-f]+):\s+(\S+)\s*(.*)$")
RELOC = re.compile(r"^\s*([0-9a-f]+):\s+R_X86_64_\S+\s+(\S+)")
SYMBOL = re.compile(r"^([0-9a-f]+) <(.+)>:$")
MEM = re.compile(r"(-?0x[0-9a-f]+)?\((%[a-z0-9]+)(?:,[^)]*)?\)")


def disassemble(obj: Path, section: str = ".text") -> dict[str, list[dict]]:
    """Every function of `section`: its instructions (address, mnemonic, operands, relocated)."""
    out = subprocess.run(
        ["objdump", "-dr", "--no-show-raw-insn", "-j", section, str(obj)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    functions: dict[str, list[dict]] = {}
    current = None
    for line in out.splitlines():
        m = SYMBOL.match(line)
        if m:
            current = functions.setdefault(m.group(2), [])
            continue
        if current is None:
            continue
        r = RELOC.match(line)
        if r and current:
            current[-1]["reloc"] = r.group(2)
            continue
        m = LINE.match(line)
        if m:
            mnemonic = m.group(2)
            operands = m.group(3).split("#")[0].strip()
            current.append(
                {"addr": int(m.group(1), 16), "op": mnemonic, "args": operands, "reloc": None}
            )
    return functions


def is_conditional(op: str) -> bool:
    return op.startswith("j") and op not in ("jmp", "jmpq")


def branch_target(ins: dict) -> int | None:
    if ins["reloc"] is not None:
        return None  # a relocated target (another section or function): not followed
    m = re.match(r"([0-9a-f]+)", ins["args"])
    return int(m.group(1), 16) if m else None


def fast_path(instructions: list[dict]) -> list[dict] | None:
    """The shortest instruction path from the entry to a `ret` without a call."""
    index = {ins["addr"]: i for i, ins in enumerate(instructions)}

    start = 0
    queue = deque([(start, [start])])
    seen = {start}
    while queue:
        i, path = queue.popleft()
        ins = instructions[i]
        op = ins["op"]
        if op.startswith("ret"):
            return [instructions[j] for j in path]
        if op.startswith("call"):
            continue
        successors = []
        if op in ("jmp", "jmpq"):
            target = branch_target(ins)
            if target is not None and target in index:
                successors.append(index[target])
        else:
            if i + 1 < len(instructions):
                successors.append(i + 1)
            if is_conditional(op):
                target = branch_target(ins)
                if target is not None and target in index:
                    successors.append(index[target])
        for s in successors:
            if s not in seen:
                seen.add(s)
                queue.append((s, [*path, s]))
    return None


def signature(path: list[dict]) -> dict:
    """The header accesses, other non-stack memory accesses, conditional branches and calls."""
    header = []
    others = 0
    branches = 0
    calls = 0
    for ins in path:
        op, args = ins["op"], ins["args"]
        if op.startswith("call"):
            calls += 1
        if is_conditional(op):
            branches += 1
        operands = [a.strip() for a in re.split(r",(?![^(]*\))", args)] if args else []
        for position, operand in enumerate(operands):
            m = MEM.search(operand)
            if not m or op.startswith("lea") or op.startswith("nop"):
                continue
            base = m.group(2)
            if base in ("%rsp", "%rbp", "%rip"):
                continue  # the stack (spills) and constants
            # AT&T: the last operand is the destination of a mov.
            kind = "store" if op.startswith("mov") and position == len(operands) - 1 else "load"
            if base == FIRST_ARG:
                header.append(f"{kind}+{int(m.group(1) or '0', 16)}")
            else:
                others += 1
    return {
        "header": sorted(header),
        "other_memory": others,
        "conditional_branches": branches,
        "calls": calls,
    }


def probe(args: argparse.Namespace) -> int:
    if platform.machine() not in ("x86_64", "AMD64") or shutil.which("objdump") is None:
        print("probe: needs x86-64 and objdump; nothing checked", file=sys.stderr)
        return 2
    with tempfile.TemporaryDirectory(prefix="dyng-fastpath-") as tmp:
        src = Path(tmp) / "probe.cpp"
        obj = Path(tmp) / "probe.o"
        src.write_text(PROBE)
        cmd = [
            *shlex.split(args.cxx),
            "-std=c++17",
            *shlex.split(args.flags),
            "-DNDEBUG",
            f"-I{REPO / 'cpp' / 'src'}",
            f"-I{REPO / 'cpp' / 'include'}",
            *[f"-I{d}" for d in args.include],
            "-c",
            str(src),
            "-o",
            str(obj),
        ]
        if args.plant:
            cmd.insert(-4, "-DDYNG_PROBE_PLANT")
        build = subprocess.run(cmd, capture_output=True, text=True)
        if build.returncode != 0:
            print(build.stderr, file=sys.stderr)
            print("probe: the probe does not compile; nothing checked", file=sys.stderr)
            return 2
        functions = disassemble(obj)
    sigs = {}
    for name, body in functions.items():
        if name.startswith("probe_"):
            path = fast_path(body)
            sigs[name] = None if path is None else signature(path)
    failures = []
    for width in ("32", "64"):
        old, new = sigs.get(f"probe_push_back_{width}"), sigs.get(f"probe_region_push_back_{width}")
        if old is None or new is None:
            failures.append(f"{width}-bit: no fast path found (push_back {old}, region {new})")
        elif old != new:
            failures.append(f"{width}-bit: region_push_back {new} != push_back {old}")
    planted = None
    if args.plant:
        planted = sigs.get("probe_planted_push_back_32")
        caught = planted is not None and planted != sigs.get("probe_push_back_32")
    record = {
        "compiler": subprocess.run(
            [*shlex.split(args.cxx), "--version"], capture_output=True, text=True
        ).stdout.splitlines()[0],
        "flags": args.flags,
        "signatures": sigs,
        "equal": not failures,
        "failures": failures,
    }
    if args.plant:
        record["planted_caught"] = caught
        if not caught:
            failures.append("the planted fast path (one more load and branch) was not caught")
    for name, sig in sorted(sigs.items()):
        print(f"{name}: {sig}")
    if args.json:
        args.json.write_text(json.dumps(record, indent=2) + "\n")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        return 1
    print(
        "region_push_back's fast path is push_back's (header accesses, element store, one "
        "conditional branch, no call)" + ("; the planted change is caught" if args.plant else "")
    )
    return 0


CAPACITY = re.compile(r"^0x8\((%r[a-z0-9]+)\),(%r[a-z0-9]+)$")


def capacity_compares(body: list[dict]) -> int:
    """Loads of 8(%rA) into %rB followed (within 4 instructions) by a compare of %rB with
    16(%rA) and a conditional branch."""
    count = 0
    for i, ins in enumerate(body):
        m = CAPACITY.match(ins["args"]) if ins["op"].startswith("mov") else None
        if not m:
            continue
        base, reg = m.groups()
        for j in range(i + 1, min(i + 5, len(body))):
            nxt = body[j]
            if nxt["op"].startswith("cmp") and {f"0x10({base})", reg} == set(
                a.strip() for a in nxt["args"].split(",")
            ):
                if j + 1 < len(body) and is_conditional(body[j + 1]["op"]):
                    count += 1
                break
    return count


def objects(args: argparse.Namespace) -> int:
    sides = {}
    for label, obj in (("base", args.base), ("head", args.head)):
        functions = {}
        for section in (".text", ".text.unlikely"):
            try:
                for name, body in disassemble(obj, section).items():
                    entry = functions.setdefault(
                        name.removesuffix(".cold"),
                        {"instructions": 0, "calls": 0, "capacity_compares": 0, "cold": 0},
                    )
                    entry["instructions"] += len(body)
                    entry["calls"] += sum(1 for i in body if i["op"].startswith("call"))
                    entry["capacity_compares"] += capacity_compares(body)
                    if section != ".text":
                        entry["cold"] += len(body)
            except subprocess.CalledProcessError:
                continue
        sides[label] = functions
    rows = []
    for name in sorted(set(sides["base"]) | set(sides["head"])):
        b, h = sides["base"].get(name), sides["head"].get(name)
        if b is None or h is None or b != h:
            rows.append({"function": name, "base": b, "head": h})
    for r in rows:
        if "omp_fn" in r["function"] or "sssp_openmp_engine" in r["function"]:
            print(f"{r['function']}\n  base {r['base']}\n  head {r['head']}")
    compares = {
        label: sum(f["capacity_compares"] for f in side.values()) for label, side in sides.items()
    }
    print(f"capacity compares: base {compares['base']}, head {compares['head']}")
    if args.json:
        args.json.write_text(
            json.dumps(
                {
                    "base": str(args.base),
                    "head": str(args.head),
                    "compares": compares,
                    "changed": rows,
                },
                indent=2,
            )
            + "\n"
        )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("probe")
    p.add_argument("--cxx", default="c++")
    p.add_argument("--flags", default="-O3")
    p.add_argument("--include", action="append", default=[], help="more include directories")
    p.add_argument("--plant", action="store_true", help="also check that a planted change fails")
    p.add_argument("--json", type=Path)
    o = sub.add_parser("objects")
    o.add_argument("--base", type=Path, required=True)
    o.add_argument("--head", type=Path, required=True)
    o.add_argument("--json", type=Path)
    args = parser.parse_args()
    return probe(args) if args.command == "probe" else objects(args)


if __name__ == "__main__":
    sys.exit(main())
