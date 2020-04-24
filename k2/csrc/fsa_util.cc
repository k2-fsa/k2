// k2/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <utility>
#include <vector>

namespace k2 {

void GetEnteringArcs(const Fsa &fsa, VecOfVec *entering_arcs) {
  // CHECK(CheckProperties(fsa, KTopSorted));

  int num_states = fsa.NumStates();
  std::vector<std::vector<std::pair<Label, StateId>>> vec(num_states);
  int num_arcs = 0;
  for (const auto &arc : fsa.arcs) {
    auto src_state = arc.src_state;
    auto dest_state = arc.dest_state;
    auto label = arc.label;
    vec[dest_state].emplace_back(label, src_state);
    ++num_arcs;
  }

  auto &ranges = entering_arcs->ranges;
  auto &values = entering_arcs->values;
  ranges.reserve(num_states);
  values.reserve(num_arcs);

  int32_t start = 0;
  int32_t end = 0;
  for (const auto &label_state : vec) {
    values.insert(values.end(), label_state.begin(), label_state.end());
    start = end;
    end += static_cast<int32_t>(label_state.size());
    ranges.push_back({start, end});
  }
}

}  // namespace k2
