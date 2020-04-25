// k2/csrc/fsa_util.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/fsa_util.h"

#include <utility>
#include <vector>

namespace k2 {

void GetEnteringArcs(const Fsa &fsa, std::vector<int32_t> *arc_index,
                     std::vector<int32_t> *end_index) {
  // CHECK(CheckProperties(fsa, KTopSorted));

  auto num_states = fsa.NumStates();
  std::vector<std::vector<int32_t>> vec(num_states);
  int32_t k = 0;
  for (const auto &arc : fsa.arcs) {
    auto dest_state = arc.dest_state;
    vec[dest_state].push_back(k);
    ++k;
  }
  arc_index->clear();
  end_index->clear();

  arc_index->reserve(fsa.arcs.size());
  end_index->reserve(num_states);

  for (const auto &indices : vec) {
    arc_index->insert(arc_index->end(), indices.begin(), indices.end());
    auto end = static_cast<int32_t>(arc_index->size());
    end_index->push_back(end);
  }
}

}  // namespace k2
