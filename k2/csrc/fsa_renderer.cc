// k2/csrc/fsa_renderer.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_renderer.h"

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
using k2::Label;
using k2::StateId;

std::string ProcessState(const Fsa &fsa, int32_t state) {
  std::ostringstream os;
  os << "  " << state << " [label = \"" << state
     << "\", shape = circle, style = bold, fontsize=14]"
     << "\n";

  int32_t begin = fsa.leaving_arcs[state].begin;
  int32_t end = fsa.leaving_arcs[state].end;

  for (; begin != end; ++begin) {
    const auto &arc = fsa.arcs[begin];
    StateId src = arc.src_state;
    StateId dest = arc.dest_state;
    Label label = arc.label;
    os << "          " << src << " -> " << dest << " [label = \"" << label
       << "\", fontsize = 14];"
       << "\n";
  }

  return os.str();
}

}  // namespace

namespace k2 {

FsaRenderer::FsaRenderer(const Fsa &fsa) : fsa_(fsa) {}

std::string FsaRenderer::Render() const {
  int32_t num_states = fsa_.NumStates();
  if (num_states == 0) return "";

  std::ostringstream os;
  os << GeneratePrologue();

  for (int32_t i = 0; i != num_states - 1; ++i) {
    os << ProcessState(fsa_, i);
  }

  // now for the final state
  os << "  " << (num_states - 1) << " [label = \"" << (num_states - 1)
     << "\", shape = doublecircle, style = solid, fontsize = 14]"
     << "\n";

  os << GenerateEpilogue() << "\n";

  return os.str();
}

}  // namespace k2
