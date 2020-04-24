// k2/csrc/properties.cc

// Copyright (c)  2020 Haowen Qiu
//                     Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include <unordered_set>

#include "k2/csrc/properties.h"
#include "k2/csrc/fsa.h"

namespace k2 {

bool IsTopSorted(const Fsa &fsa) {
  for (auto &arc : fsa.arcs) {
    if (arc.dest_state < arc.src_state) {
      return false;
    }
  }
  return true;
}

bool HasSelfLoops(const Fsa &fsa) {
  for (auto &arc : fsa.arcs) {
    if (arc.dest_state == arc.src_state) {
      return true;
    }
  }
  return false;
}

bool IsDeterministic(const Fsa &fsa) {
  std::unordered_set<Label> labels;
  StateId state = 0;
  for (auto &arc : fsa.arcs) {
    if (arc.src_state == state) {
      if (labels.find(arc.label) != labels.end()) {
        return false;
      }
      labels.insert(arc.label);
    } else {
      state = arc.src_state;
      labels.clear();
      labels.insert(arc.label);
    }
  }
  return true;
}

bool IsEpsilonFree(const Fsa &fsa) {
  for (auto &arc : fsa.arcs) {
    if (arc.label == kEpsilon) {
      return false;
    }
  }
  return true;
}

}  // namespace k2
