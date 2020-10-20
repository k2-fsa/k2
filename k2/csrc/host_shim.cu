/**
 * @brief
 * host_shim
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host_shim.h"

namespace k2 {

k2host::Fsa FsaToHostFsa(Fsa &fsa) {
  K2_CHECK_EQ(fsa.NumAxes(), 2);
  K2_CHECK_EQ(fsa.Context()->GetDeviceType(), kCpu);
  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).
  return k2host::Fsa(fsa.shape.Dim0(), fsa.shape.TotSize(1),
                     fsa.shape.RowSplits(1).Data(),
                     reinterpret_cast<k2host::Arc *>(fsa.values.Data()));
}

k2host::Fsa FsaVecToHostFsa(FsaVec &fsa_vec, int32_t index) {
  K2_CHECK_EQ(fsa_vec.NumAxes(), 3);
  K2_CHECK_LT(static_cast<uint32_t>(index),
              static_cast<uint32_t>(fsa_vec.Dim0()));
  K2_CHECK_EQ(fsa_vec.Context()->GetDeviceType(), kCpu);

  // reinterpret_cast works because the arcs have the same members
  // (except our 'score' is called 'weight' there).

  int32_t *row_splits1_data = fsa_vec.RowSplits(1).Data(),
          *row_splits2_data = fsa_vec.RowSplits(2).Data();
  Arc *arcs_data = fsa_vec.values.Data();
  int32_t start_state_idx01 = row_splits1_data[index],
          end_state_idx01 = row_splits1_data[index + 1],
          size1 = end_state_idx01 - start_state_idx01,
          start_arc_idx012 = row_splits2_data[start_state_idx01],
          end_arc_idx012 = row_splits2_data[end_state_idx01],
          size2 = end_arc_idx012 - start_arc_idx012;

  return k2host::Fsa(size1, size2, row_splits2_data + start_state_idx01,
                     reinterpret_cast<k2host::Arc *>(arcs_data));
}

void FsaVecCreator::Init(
    const std::vector<k2host::Array2Size<int32_t>> &sizes) {
  int32_t num_fsas = static_cast<int32_t>(sizes.size());
  K2_CHECK_GT(num_fsas, 0);
  ContextPtr c = GetCpuContext();

  row_splits1_ = Array1<int32_t>(c, num_fsas + 1);
  row_splits12_ = Array1<int32_t>(c, num_fsas + 1);
  int32_t *row_splits1_data = row_splits1_.Data(),
          *row_splits12_data = row_splits12_.Data();
  for (int32_t i = 0; i < num_fsas; i++) {
    row_splits1_data[i] = sizes[i].size1;   // num_states
    row_splits12_data[i] = sizes[i].size2;  // num_arcs
  }
  ExclusiveSum(row_splits1_, &row_splits1_);
  ExclusiveSum(row_splits12_, &row_splits12_);

  int32_t tot_states = row_splits1_[num_fsas],
          tot_arcs = row_splits12_[num_fsas];
  row_splits2_ = Array1<int32_t>(c, tot_states + 1);
  arcs_ = Array1<Arc>(c, tot_arcs);

  finalized_row_splits2_ = false;
  next_fsa_idx_ = 0;
}

void FsaVecCreator::FinalizeRowSplits2() {
  int32_t num_fsas = row_splits1_.Dim() - 1;
  K2_CHECK_EQ(next_fsa_idx_, num_fsas);

  const int32_t *row_splits1_data = row_splits1_.Data(),
                *row_splits12_data = row_splits12_.Data();
  int32_t *row_splits2_data = row_splits2_.Data();

  for (int32_t i = 0; i < num_fsas; i++) {
    int32_t num_states = row_splits1_data[i + 1] - row_splits1_data[i],
            begin_state = row_splits1_data[i], begin_arc = row_splits12_data[i];
    K2_CHECK(row_splits2_data[begin_state] == 0 || num_states == 0);
    for (int32_t j = 0; j < num_states; j++)
      row_splits2_data[begin_state + j] += begin_arc;
  }
}

k2host::Fsa FsaVecCreator::GetHostFsa(int32_t i) {
  K2_CHECK_EQ(i, next_fsa_idx_);  // make sure they are called in order.
  next_fsa_idx_++;

  const int32_t *row_splits1_data = row_splits1_.Data(),
                *row_splits12_data = row_splits12_.Data();
  int32_t *row_splits2_data = row_splits2_.Data();
  int32_t num_states = row_splits1_data[i + 1] - row_splits1_data[i],
          num_arcs = row_splits12_data[i + 1] - row_splits12_data[i];
  k2host::Arc *arcs_data = reinterpret_cast<k2host::Arc *>(arcs_.Data());
  return k2host::Fsa(num_states, num_arcs,
                     row_splits2_data + row_splits1_data[i],
                     arcs_data + row_splits12_data[i]);
}

FsaVec FsaVecCreator::GetFsaVec() {
  FinalizeRowSplits2();
  return Ragged<Arc>(
      RaggedShape3(&row_splits1_, nullptr, -1, &row_splits2_, nullptr, -1),
      arcs_);
}

}  // namespace k2
