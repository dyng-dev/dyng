// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// SOSP exporter of the parity harness. parity/build_reference.sh compiles it into the PATCHED
// scratch copies of MOSP-OpenMP@c352151 and MOSP-CUDA@e220ee2
// (parity/export_patches/<repo>/build.sh), against the copy's unchanged sources, and it calls the
// original file-based SOSP updates unchanged:
//
//   sequential <csrPrefix> <distIn> <treeIn> <insert> <delete> <objective> <source> <distOut>
//              <treeOut>
//              sequentialSOSPUpdate (the legacy re-scan reference)
//   parallel   <same arguments>
//              parallelSOSPUpdate (the file-based parallel update: sospUpdateCpu inside in
//              MOSP-OpenMP, sospUpdateGpu in MOSP-CUDA)
//
// parity/export_goldens.py and parity/fixtures/sssp/make_sssp_fixtures.sh compare both outputs
// with the in-memory driver `mosp` (mospUpdate) and with `mospPrep expected` (Dijkstra on the
// updated graph) before they write any expected file: the original implementations must agree
// byte for byte (PLAN Section 6.3, step 2).

#if __has_include("parallelSOSPUpdate.cuh")  // MOSP-CUDA's headers
#include "parallelSOSPUpdate.cuh"
#include "sequentialSOSPUpdate.cuh"
#else
#include "parallelSOSPUpdate.h"
#include "sequentialSOSPUpdate.h"
#endif

#include <cstdlib>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
  if (argc != 11) {
    std::cerr << "usage: export_sssp sequential|parallel <csrPrefix> <distIn> <treeIn> <insert> "
                 "<delete> <objective> <source> <distOut> <treeOut>\n";
    return 2;
  }
  const std::string command = argv[1];
  const int objective = static_cast<int>(std::strtol(argv[7], nullptr, 10));
  const int source = static_cast<int>(std::strtol(argv[8], nullptr, 10));
  bool ok = false;
  if (command == "sequential") {
    ok = sequentialSOSPUpdate(argv[2], argv[3], argv[4], argv[5], argv[6], objective, source,
                              argv[9], argv[10]);
  } else if (command == "parallel") {
    ok = parallelSOSPUpdate(argv[2], argv[3], argv[4], argv[5], argv[6], objective, source, argv[9],
                            argv[10]);
  } else {
    std::cerr << "unknown command: " << command << "\n";
    return 2;
  }
  return ok ? 0 : 1;
}
