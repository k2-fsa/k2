// k2/csrc/arcsort.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/arcsort.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "glog/logging.h"
#include "k2/csrc/fsa.h"

namespace k2 {
void ArcSorter::GetSizes(Array2Size<int32_t> *fsa_size) {
  CHECK_NOTNULL(fsa_size);
  fsa_size->size1 = fsa_in_.size1;
  fsa_size->size2 = fsa_in_.size2;
}

void ArcSorter::GetOutput(Fsa *fsa_out, int32_t *arc_map /*= nullptr*/) {
  CHECK_NOTNULL(fsa_out);
  CHECK_EQ(fsa_out->size1, fsa_in_.size1);
  CHECK_EQ(fsa_out->size2, fsa_in_.size2);

  using ArcWithIndex = std::pair<Arc, int32_t>;
  std::vector<int32_t> indexes(fsa_in_.size2);  // arc index mapping
  std::iota(indexes.begin(), indexes.end(), 0);
  const auto arc_begin_iter = fsa_in_.data;
  const auto index_begin_iter = indexes.begin();
  int32_t num_states = fsa_in_.NumStates();
  int32_t num_arcs = 0;
  for (int32_t state = 0; state != num_states; ++state) {
    fsa_out->indexes[state] = num_arcs;
    int32_t begin = fsa_in_.indexes[state];
    int32_t end = fsa_in_.indexes[state + 1];
    std::vector<ArcWithIndex> arc_range_to_be_sorted;
    arc_range_to_be_sorted.reserve(end - begin);
    std::transform(arc_begin_iter + begin, arc_begin_iter + end,
                   index_begin_iter + begin,
                   std::back_inserter(arc_range_to_be_sorted),
                   [](const Arc &arc, int32_t index) -> ArcWithIndex {
                     return std::make_pair(arc, index);
                   });
    std::sort(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
              [](const ArcWithIndex &left, const ArcWithIndex &right) {
                return left.first < right.first;  // sort on arc
              });
    // copy index mappings back to `indexes`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   index_begin_iter + begin,
                   [](const ArcWithIndex &v) { return v.second; });
    // move-copy sorted arcs to `fsa_out`
    std::transform(arc_range_to_be_sorted.begin(), arc_range_to_be_sorted.end(),
                   fsa_out->data + num_arcs,
                   [](ArcWithIndex &v) { return v.first; });
    num_arcs += end - begin;
  }
  fsa_out->indexes[num_states] = num_arcs;

  if (arc_map != nullptr) std::copy(indexes.begin(), indexes.end(), arc_map);
}

}  // namespace k2
