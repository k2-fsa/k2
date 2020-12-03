/**
 * @brief
 * arcsort
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/arcsort.h"

#include <algorithm>
#include <numeric>
#include <utility>
#include <vector>

#include "k2/csrc/host/fsa.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {
void ArcSorter::GetSizes(Array2Size<int32_t> *fsa_size) const {
  K2_CHECK_NE(fsa_size, nullptr);
  fsa_size->size1 = fsa_in_.size1;
  fsa_size->size2 = fsa_in_.size2;
}

void ArcSorter::GetOutput(Fsa *fsa_out, int32_t *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa_out, nullptr);
  K2_CHECK_EQ(fsa_out->size1, fsa_in_.size1);
  K2_CHECK_EQ(fsa_out->size2, fsa_in_.size2);

  std::vector<int32_t> indexes(fsa_in_.size2);
  // After arc sorting, indexes[i] = j means we mapped arc-index `i` of
  // `fsa_out` to arc-index `j` of `fsa_in`
  std::iota(indexes.begin(), indexes.end(), 0);
  const int32_t num_states = fsa_in_.NumStates();
  int32_t num_arcs = 0;
  const int32_t arc_begin_index =
      fsa_in_.indexes[0];  // it may be greater than 0
  const auto &arcs_in = fsa_in_.data + arc_begin_index;
  for (int32_t state = 0; state != num_states; ++state) {
    fsa_out->indexes[state] = num_arcs;
    int32_t begin = fsa_in_.indexes[state] - arc_begin_index;
    int32_t end = fsa_in_.indexes[state + 1] - arc_begin_index;
    std::sort(
        indexes.begin() + begin, indexes.begin() + end,
        [&arcs_in](int32_t i, int32_t j) { return arcs_in[i] < arcs_in[j]; });
    // copy sorted arcs to `fsa_out`
    std::transform(indexes.begin() + begin, indexes.begin() + end,
                   fsa_out->data + num_arcs,
                   [&arcs_in](int32_t i) { return arcs_in[i]; });
    num_arcs += end - begin;
  }
  fsa_out->indexes[num_states] = num_arcs;

  if (arc_map != nullptr) std::copy(indexes.begin(), indexes.end(), arc_map);
}

void ArcSort(Fsa *fsa, int32_t *arc_map /*= nullptr*/) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_NE(fsa, nullptr);

  std::vector<int32_t> indexes(fsa->size2);
  // After arc sorting, indexes[i] = j means we mapped arc-index `i` of
  // original FSa to arc-index `j` of output arc-sorted FSA
  std::iota(indexes.begin(), indexes.end(), 0);
  const int32_t num_states = fsa->NumStates();
  const int32_t arc_begin_index = fsa->indexes[0];  // it may be greater than 0
  const auto &arcs_in = fsa->data + arc_begin_index;
  for (int32_t state = 0; state != num_states; ++state) {
    int32_t begin = fsa->indexes[state] - arc_begin_index;
    int32_t end = fsa->indexes[state + 1] - arc_begin_index;
    std::sort(indexes.begin() + begin, indexes.begin() + end,
              [&arcs_in, arc_begin_index](int32_t i, int32_t j) {
                return arcs_in[i] < arcs_in[j];
              });
    std::sort(arcs_in + begin, arcs_in + end,
              [](const Arc &left, const Arc &right) { return left < right; });
  }

  if (arc_map != nullptr) std::copy(indexes.begin(), indexes.end(), arc_map);
}

}  // namespace k2host
