// k2/csrc/fsa_renderer.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_renderer.h"

#include <iomanip>
#include <sstream>
#include <string>

namespace {

std::string GeneratePrologue() {
  // TODO(fangjun): the following options can be passed from outside
  std::string header = R"header(
digraph FSA {
  rankdir = LR;
  size = "8.5,11";
  label = "";
  center = 1;
  orientation = Portrait;
  ranksep = "0.4"
  nodesep = "0.25"
)header";
  return header;
}

std::string GenerateEpilogue() { return "}"; }

using k2::Arc;
using k2::Fsa;

std::string ProcessState(const Fsa &fsa, int32_t state,
                         const float *arc_weights = nullptr) {
  std::ostringstream os;
  os << "  " << state << " [label = \"" << state
     << "\", shape = circle, style = bold, fontsize = 14]"
     << "\n";

  int32_t arc_begin_index = fsa.indexes[0];  // it may be greater than 0
  int32_t begin = fsa.indexes[state];
  int32_t end = fsa.indexes[state + 1];

  for (; begin != end; ++begin) {
    const auto &arc = fsa.data[begin];
    int32_t src = arc.src_state;
    int32_t dest = arc.dest_state;
    int32_t label = arc.label;
    os << "          " << src << " -> " << dest << " [label = \"" << label;
    if (arc_weights != nullptr)
      os << "/" << std::fixed << std::setprecision(1)
         << arc_weights[begin - arc_begin_index];
    os << "\", fontsize = 14];"
       << "\n";
  }

  return os.str();
}

}  // namespace

namespace k2 {

std::string FsaRenderer::Render() const {
  int32_t num_states = fsa_.NumStates();
  if (num_states == 0) return "";

  std::ostringstream os;
  os << GeneratePrologue();

  int32_t final_state = fsa_.FinalState();
  for (int32_t i = 0; i != final_state; ++i) {
    os << ProcessState(fsa_, i, arc_weights_);
  }

  // now for the final state
  os << "  " << final_state << " [label = \"" << final_state
     << "\", shape = doublecircle, style = solid, fontsize = 14]"
     << "\n";

  os << GenerateEpilogue() << "\n";

  return os.str();
}

}  // namespace k2
