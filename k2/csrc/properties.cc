// k2/csrc/properties.cc

// Copyright     2020  Haowen Qiu

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABILITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.

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
