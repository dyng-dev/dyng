#!/bin/sh
# SPDX-FileCopyrightText: 2026 The dynG Authors
# SPDX-License-Identifier: Apache-2.0
#
# The compiler launcher of a kernel group's CUDA sources (cmake/kernel_groups.cmake; ADR 0109
# part B). Usage (set by CMake as CUDA_COMPILER_LAUNCHER):
#
#   kernel_group_launcher.sh <objcopy> <nm> <group.syms> <compiler command ...>
#
# Runs the compiler command, then, in the object it wrote (its -o argument), renames the CUDA
# runtime's registration entry points and the runtime functions that take a kernel or a device
# symbol to the group's functions (objcopy --redefine-syms <group.syms>), and fails if the object
# still calls one of them, or another runtime function of that kind, unrenamed: such a call would
# reach the runtime before the group is registered (a new one goes into DYNG_KERNEL_GROUP_SYMBOLS
# and kernel_group.cpp.in).
set -e
objcopy=$1
nm=$2
syms=$3
shift 3
"$@"
obj=
prev=
for arg in "$@"; do
  if [ "$prev" = "-o" ]; then
    obj=$arg
  fi
  prev=$arg
done
if [ -z "$obj" ] || [ ! -f "$obj" ]; then
  exit 0
fi
"$objcopy" --redefine-syms="$syms" "$obj"
unrenamed=$("$nm" -u "$obj" | awk '{ print $NF }' |
  grep -E '^__cuda|^cuda(Func|Launch|Occupancy|GetSymbol|GetFuncBySymbol|GetKernel|Kernel|Library|Graph[A-Za-z]*Kernel)|Symbol' |
  grep -vxE '__cudaPushCallConfiguration|__cudaPopCallConfiguration|__cudaLaunchKernel|__cudaLaunchKernel_ptsz' || true)
if [ -n "$unrenamed" ]; then
  echo "kernel_group_launcher.sh: $obj calls CUDA runtime functions that take a kernel or a" \
    "device symbol and are not intercepted for its kernel group (cmake/kernel_groups.cmake):" \
    $unrenamed >&2
  rm -f "$obj"
  exit 1
fi
