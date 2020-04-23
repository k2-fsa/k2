// k2/csrc/properties.cc

// Copyright (c)  2020  Daniel Povey

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/properties.h"

#include "k2/csrc/fsa.h"

namespace k2 {

bool IsTopSorted(const Fsa& fsa) {
  for (const auto& range : fsa.leaving_arcs) {
    for (auto arc_idx = range.begin; arc_idx < range.end; ++arc_idx) {
      const Arc& arc = fsa.arcs[arc_idx];
      if (arc.dest_state < arc.src_state) {
        return false;
      }
    }
  }
  return true;
}

bool HasSelfLoops(const Fsa& fsa) {
  // TODO(haowen): refactor code below as we have
  // so many for-for-loop structures
  for (const auto& range : fsa.leaving_arcs) {
    for (auto arc_idx = range.begin; arc_idx < range.end; ++arc_idx) {
      const Arc& arc = fsa.arcs[arc_idx];
      if (arc.dest_state == arc.src_state) {
        return false;
      }
    }
  }
  return true;
}

}  // namespace k2
