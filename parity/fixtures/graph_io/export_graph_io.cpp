// SPDX-FileCopyrightText: 2026 The dynG Authors
// SPDX-License-Identifier: Apache-2.0
//
// Fixture exporter for the graph/io tests. It is compiled against the sources of the PINNED
// original MOSP-OpenMP@c352151 (a `git archive` copy under $DYNG_SCRATCH; never the original
// repository) by make_graph_io_fixtures.sh, and calls the original functions unchanged:
//
//   gen-graph   <prefix> <n> <m> <K> <wmin> <wmax> <seed>
//               generateGraphCSR (directed)
//   gen-changes <csrPrefix> <insert> <delete> <K> <n> <count> <ins%> <del%> <wmin> <wmax>
//               <exist> <duplicate> <selfLoop> <seed>
//               generateChangedEdges (directed)
//   apply       <csrPrefix> <insert> <delete> <outPrefix>
//               readCsrGraph + readChangeBatch + applyChangeBatch, then writes
//               <outPrefix>{RowPtr,ColInd,Values}.txt (writeCsrGraph),
//               <outPrefix>WeightIncrease.txt (one line per insertion: K flags 0/1, the bits of
//               weightIncreaseMask) and the reverse graph of the updated graph
//               <outPrefix>Transposed{RowPtr,ColInd,Values}.txt (transposeCsrGraph).
//
// The program is not part of the dynG build; it exists so that the committed expected outputs
// can be regenerated from the pinned original at any time.

#include "csrGraph.h"
#include "generateChangedEdges.h"
#include "generateGraphCSR.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>

namespace {

int usage() {
  std::cerr << "usage: export_graph_io gen-graph|gen-changes|apply ...\n";
  return 2;
}

int to_int(const char* text) {
  return static_cast<int>(std::strtol(text, nullptr, 10));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 2) {
    return usage();
  }
  const std::string command = argv[1];
  if (command == "gen-graph" && argc == 9) {
    return generateGraphCSR(to_int(argv[3]), to_int(argv[4]), true, argv[2], to_int(argv[5]),
                            to_int(argv[6]), to_int(argv[7]),
                            static_cast<unsigned>(std::strtoul(argv[8], nullptr, 10)))
               ? 0
               : 1;
  }
  if (command == "gen-changes" && argc == 16) {
    return generateChangedEdges(to_int(argv[10]), to_int(argv[11]), to_int(argv[5]),
                                to_int(argv[6]), to_int(argv[7]), std::atof(argv[8]),
                                std::atof(argv[9]), true, to_int(argv[12]) != 0,
                                to_int(argv[13]) != 0, to_int(argv[14]) != 0, argv[2], argv[3],
                                argv[4], static_cast<unsigned>(std::strtoul(argv[15], nullptr, 10)))
               ? 0
               : 1;
  }
  if (command == "apply" && argc == 6) {
    CsrGraph original;
    if (!readCsrGraph(argv[2], original)) {
      return 1;
    }
    ChangeBatch batch;
    if (!readChangeBatch(argv[3], argv[4], original.numberOfObjectives, original.numberOfNodes,
                         batch)) {
      return 1;
    }
    CsrGraph updated;
    if (!applyChangeBatch(original, batch, updated)) {
      return 1;
    }
    const std::string out = argv[5];
    if (!writeCsrGraph(out, updated)) {
      return 1;
    }
    std::ofstream mask(out + "WeightIncrease.txt");
    for (int i = 0; i < batch.numberOfInserts(); ++i) {
      for (int k = 0; k < batch.numberOfObjectives; ++k) {
        mask << ((batch.weightIncreaseMask[i] >> k) & 1u)
             << (k + 1 < batch.numberOfObjectives ? " " : "");
      }
      mask << "\n";
    }
    CsrGraph reverse;
    transposeCsrGraph(updated, reverse);
    return mask.good() && writeCsrGraph(out + "Transposed", reverse) ? 0 : 1;
  }
  return usage();
}
