/**
 * @brief
 * ragged
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey, Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cub/cub.cuh>
#include <vector>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/math.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_utils.h"

namespace k2 {


void CheckAxisEqual(int32_t num_srcs,
                    int32_t axis,
                    RaggedShape **src) {
  if (num_srcs <= 1)
    return;
  K2_CHECK(axis >= 0 && axis + 1 < src[0]->NumAxes());
  std::vector<int32_t*> row_splits_data_vec;
  row_splits_data_vec.reserve(num_srcs);
  int32_t row_splits_dim, row_ids_dim;
  for (int32_t s = 0; s < num_srcs; s++) {
    // RowSplits(1) .. is the lowest numbered row-splits...
    int32_t *data = src[s]->RowSplits(axis + 1).Data();
    if (s == 0 || data != row_splits_data_vec[0])
      row_splits_data_vec.push_back(data);
    if (s == 0) {
      row_splits_dim = src[s]->RowSplits(axis + 1).Dim();
      row_ids_dim = src[s]->RowIds(axis + 1).Dim();
    } else {
      K2_CHECK_EQ(row_splits_dim, src[s]->RowSplits(axis + 1).Dim());
      K2_CHECK_EQ(row_ids_dim, src[s]->RowIds(axis + 1).Dim());
    }
  }
  if (row_splits_data_vec.size() <= 1)
    return;
  ContextPtr c = src[0]->Context();
#ifndef NDEBUG
  Array1<int32_t> is_bad(c, 1, 0);
  Array1<int32_t*> row_splits_ptrs(c, row_splits_data_vec);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();
  int32_t *is_bad_data = is_bad.Data();
  K2_EVAL2(c, row_splits_ptrs.Dim() - 1,
           row_splits_dim, lambda_check_row_splits,
           (int32_t i, int32_t j) -> void {
             if (row_splits_ptrs_data[i+1][j] !=
                 row_splits_ptrs_data[0][j])
               is_bad_data[0] = 1;
           });
  if (!(is_bad[0] == 0)) {
    std::ostringstream arrays_os;
    for (int32_t i = 0; i < num_srcs; i++)
      arrays_os << "Shape " << i << " = " << *(src[i]) << "; ";
    K2_LOG(FATAL) << "Axes were expected to be equal: "
                  << arrays_os.str();
  }
#endif
}


RaggedShape AppendRaggedAxisAfter(int32_t num_srcs,
                                  int32_t axis,
                                  RaggedShape **src,
                                  Array1<int32_t> *merge_map) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK_GE(axis, 0);
  K2_CHECK_LT(axis + 1, src[0]->NumAxes());
  if (num_srcs == 1) {
    if (merge_map)
      *merge_map = Range(src[0]->Context(), src[0]->TotSize(axis + 1), 0);
    return *src[0];
  }
  int32_t row_splits_dim,
      tot_row_ids_dim;
  std::vector<int32_t> row_ids_dims_vec(num_srcs);
  std::vector<int32_t*> row_splits_ptrs_vec(num_srcs);
  std::vector<int32_t*> row_ids_ptrs_vec(num_srcs);
  // set these up...

  // TODO(dan).


}



}  // namespace k2
