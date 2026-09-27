// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fixture exporter for the sssp tests. It is compiled against the sources of the PINNED original
// MOSP-OpenMP@c352151 (a `git archive` copy under $DYNG_SCRATCH; never the original repository)
// by make_sssp_fixtures.sh, and calls the original file-based SOSP updates unchanged:
//
//   sequential <csrPrefix> <distIn> <treeIn> <insert> <delete> <objective> <source> <distOut>
//              <treeOut>
//              sequentialSOSPUpdate (the legacy re-scan reference)
//   parallel   <same arguments>
//              parallelSOSPUpdate (the file-based OpenMP update, sospUpdateCpu inside)
//
// The script compares both outputs with the in-memory driver `mosp` (mospUpdate) and with
// `mospPrep expected` (Dijkstra on the updated graph) before it commits any expected file: the
// three original implementations must agree byte for byte (PLAN Section 6.3, step 2).

#include "parallelSOSPUpdate.h"
#include "sequentialSOSPUpdate.h"

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
