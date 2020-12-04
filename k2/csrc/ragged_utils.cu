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
  std::vector<int32_t*> row_splits_data_vec(num_srcs);
  int32_t row_splits_dim, row_ids_dim;
  for (int32_t s = 0; s < num_srcs; s++) {
    // RowSplits(1) .. is the lowest numbered row-splits...
    row_splits_data_vec[s] = src[s]->RowSplits(axis + 1).Data();
    if (s == 0) {
      row_splits_dim = src[s]->RowSplits(axis + 1).Dim();
      row_ids_dim = src[s]->RowIds(axis + 1).Dim();
    } else {
      K2_CHECK_EQ(row_splits_dim, src[s]->RowSplits(axis + 1).Dim());
      K2_CHECK_EQ(row_ids_dim, src[s]->RowIds(axis + 1).Dim());
    }
  }
  ContextPtr c = src[0]->Context();
#ifndef NDEBUG
  Array1<int32_t> is_bad(c, 1, 0);
  Array1<int32_t*> row_splits_ptrs(c, row_splits_data_vec);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();
  int32_t *is_bad_data = is_bad.Data();
  K2_EVAL2(c, num_srcs - 1, row_splits_dim, lambda_check_row_splits,
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



}  // namespace k2
