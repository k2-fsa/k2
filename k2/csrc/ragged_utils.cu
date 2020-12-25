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
#include "k2/csrc/ragged_ops.h"

namespace k2 {


void CheckLayerEqual(int32_t layer,
                     int32_t num_srcs,
                     RaggedShape **src) {
  if (num_srcs <= 1)
    return;
  K2_CHECK(layer >= 0 && layer + 1 < src[0]->NumAxes());
  std::vector<const int32_t*> row_splits_data_vec;
  row_splits_data_vec.reserve(num_srcs);
  int32_t row_splits_dim, row_ids_dim;
  for (int32_t s = 0; s < num_srcs; s++) {
    // RowSplits(1) .. is the lowest numbered row-splits...
    const int32_t *data = src[s]->RowSplits(layer + 1).Data();
    if (s == 0 || data != row_splits_data_vec[0])
      row_splits_data_vec.push_back(data);
    if (s == 0) {
      row_splits_dim = src[s]->TotSize(layer) + 1;
      row_ids_dim = src[s]->TotSize(layer + 1);
    } else {
      K2_CHECK_EQ(row_splits_dim, src[s]->TotSize(layer) + 1);
      K2_CHECK_EQ(row_ids_dim, src[s]->TotSize(layer + 1));
    }
  }
  if (row_splits_data_vec.size() <= 1) {
    // No point in checking because the row_splits all had the same address.
    return;
  }
  ContextPtr &c = src[0]->Context();
#ifndef NDEBUG
  Array1<int32_t> is_bad(c, 1, 0);
  Array1<const int32_t*> row_splits_ptrs(c, row_splits_data_vec);
  const int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();
  int32_t *is_bad_data = is_bad.Data();
  K2_EVAL2(c, row_splits_ptrs.Dim() - 1,
           row_splits_dim, lambda_check_row_splits,
           (int32_t i, int32_t j) -> void {
             if (row_splits_ptrs_data[i+1][j] !=
                 row_splits_ptrs_data[0][j])
               is_bad_data[0] = 1;
           });
  if (is_bad[0] == 1) {
    std::ostringstream arrays_os;
    for (int32_t i = 0; i < num_srcs; i++)
      arrays_os << "Shape " << i << " = " << *(src[i]) << "; ";
    K2_LOG(FATAL) << "Shapes were expected to be equal: "
                  << arrays_os.str();
  }
#endif
}


RaggedShape IntersperseRaggedLayer(int32_t layer,
                                   int32_t num_srcs,
                                   RaggedShape **src,
                                   Array1<uint32_t> *merge_map) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK_GE(layer, 0);
  K2_CHECK_LT(layer + 1, src[0]->NumAxes());
  if (num_srcs == 1) {
    if (merge_map)
      *(reinterpret_cast<Array1<int32_t>*>(merge_map)) =
          Range(src[0]->Context(), src[0]->TotSize(layer + 1), 0);
    return *src[0];
  }

  std::vector<int32_t*> row_splits_ptrs_vec(num_srcs);

  int32_t num_axes = src[0]->NumAxes(),
          num_rows = src[0]->TotSize(layer),
       tot_elems = 0;
  for (int32_t i = 0; i < num_srcs; ++i) {
    if (i > 0) {
      K2_CHECK_EQ(src[i]->NumAxes(), num_axes);
      K2_CHECK_EQ(src[i]->TotSize(layer), num_rows);
    }
    Array1<int32_t> &row_splits = src[i]->RowSplits(layer + 1);
    tot_elems += src[i]->TotSize(layer + 1);
    row_splits_ptrs_vec[i] = row_splits.Data();
  }
  ContextPtr &c = src[0]->Context();

  int32_t new_num_rows = num_rows * num_srcs;
  Array1<int32_t> row_ids(c, tot_elems),
      row_splits(c, new_num_rows + 1);
  int32_t *row_splits_data = row_splits.Data();
  Array1<int32_t*> row_splits_ptrs(c, row_splits_ptrs_vec);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();

  if (c->GetDeviceType() == kCpu) {
    int32_t row_splits_sum = 0;
    row_splits_data[0] = 0;
    for (int32_t i = 0; i < num_rows; i++) {
      for (int32_t j = 0; j < num_srcs; j++) {
        int32_t row_len = row_splits_ptrs_data[j][i+1] -
                          row_splits_ptrs_data[j][i];
        row_splits_sum += row_len;
        row_splits_data[i * num_srcs + j + 1] = row_splits_sum;
      }
    }
  } else {
    if (num_srcs <= 16) {
      // If num_srcs is not too large, we do an optimization.  Instead
      // of computing the length of each row (as row_splits[i+1] -
      // row_splits[i]) and doing exclusive-sum to get the row_splits, we sum up
      //  `num_srcs` row_splits numbered `i, i+1, .. i+num_srcs-1.`
      // (These numberings map to source i % num_srcs at position i / num_srcs).
      // This gives us the same answer, with less latency.
      auto lambda_get_row_splits = [=] __device__(int32_t i) -> void {
        int32_t sum = 0;
        for (int32_t j = i; j < i + num_srcs; j++) {
          int32_t src = j % num_srcs,
                  pos = j / num_srcs;
          int32_t this_row_split = row_splits_ptrs_data[src][pos];
          sum += this_row_split;
        }
        row_splits_data[i] = sum;
      };
      EvalDevice(c, new_num_rows + 1, lambda_get_row_splits);
    } else {
      // Set the row_splits initially to the sizes, then do exclusive-sum.
      auto lambda_get_sizes = [=] __device__(int32_t i) -> void {
        int32_t src = i % num_srcs, pos = i / num_srcs;
        int32_t this_size = row_splits_ptrs_data[src][pos + 1] -
             row_splits_ptrs_data[src][pos];
        row_splits_data[i] = this_size;
      };
      EvalDevice(c, new_num_rows + 1, lambda_get_sizes);
      ExclusiveSum(row_splits, &row_splits);
    }
  }
  RowSplitsToRowIds(row_splits, &row_ids);

  if (merge_map != nullptr) {
    *merge_map = Array1<uint32_t>(c, tot_elems);
    const int32_t *row_ids_data = row_ids.Data();
    uint32_t *merge_map_data = merge_map->Data();

    K2_EVAL(c, tot_elems, lambda_set_merge_map, (int32_t idx01) -> void {
        int32_t idx0 = row_ids_data[idx01],
               idx0x = row_splits_data[idx0],
                idx1 = idx01 - idx0x,
                 src = idx0 % num_srcs,
            src_idx0 = idx0 / num_srcs,
           src_idx0x = row_splits_ptrs_data[src][src_idx0],
           src_idx01 = src_idx0x + idx1;
        // We multiply the src_idx01 by num_srcs as a way of encoding it and the
        // src into a single integer.
        merge_map_data[idx01] =
            uint32_t(src) + ((uint32_t)num_srcs * uint32_t(src_idx01));
      });
  }

  return RaggedShape2(&row_splits, &row_ids, tot_elems);
}


RaggedShape MergeRaggedLayer(int32_t layer,
                             int32_t num_srcs,
                             RaggedShape **src,
                             const Array1<uint32_t> &merge_map,
                             Array1<uint32_t> *merge_map_out /*= nullptr*/) {
  K2_CHECK_GT(num_srcs, 0);
  K2_CHECK_GE(layer, 0);
  K2_CHECK_LT(layer + 1, src[0]->NumAxes());

  ContextPtr &c = src[0]->Context();
  std::vector<int32_t*> row_splits_ptrs_vec(num_srcs);

  int32_t tot_rows = 0, tot_elems = 0;
  for (int32_t i = 0; i < num_srcs; i++) {
    tot_rows += src[i]->TotSize(layer);
    tot_elems += src[i]->TotSize(layer + 1);
    row_splits_ptrs_vec[i] = src[i]->RowSplits(layer + 1).Data();
  }
  K2_CHECK_EQ(tot_rows, merge_map.Dim());

  Array1<int32_t> row_splits_out(c, merge_map.Dim() + 1);
  Array1<int32_t> row_ids_out(c, tot_elems);

  const uint32_t *merge_map_data = merge_map.Data();
  Array1<int32_t*> row_splits_ptrs(c, row_splits_ptrs_vec);
  int32_t **row_splits_ptrs_data = row_splits_ptrs.Data();
  int32_t *sizes_data = row_splits_out.Data();

  K2_EVAL(c, tot_rows, lambda_set_sizes, (int32_t i) -> void {
      uint32_t m = merge_map_data[i],
             src = m % num_srcs,
             pos = m / num_srcs;
      int32_t size = row_splits_ptrs_data[src][pos + 1] -
                     row_splits_ptrs_data[src][pos];
      sizes_data[i] = size;
    });
  ExclusiveSum(row_splits_out, &row_splits_out);
  RowSplitsToRowIds(row_splits_out, &row_ids_out);

  if (merge_map_out != nullptr) {
    *merge_map_out = Array1<uint32_t>(c, tot_elems);
    const int32_t *row_ids_data = row_ids_out.Data(),
               *row_splits_data = row_splits_out.Data();
    uint32_t *merge_map_out_data = merge_map_out->Data();

    K2_EVAL(c, tot_elems, lambda_set_merge_map, (int32_t idx01) -> void {
        int32_t idx0 = row_ids_data[idx01],
               idx0x = row_splits_data[idx0],
                idx1 = idx01 - idx0x,
                   m = merge_map_data[idx0],
                 src = m % num_srcs,
            src_idx0 = m / num_srcs,
           src_idx0x = row_splits_ptrs_data[src][src_idx0],
           src_idx01 = src_idx0x + idx1;
        // We multiply the src_idx01 by num_srcs as a way of encoding it and the
        // src into a single integer.
        merge_map_out_data[idx01] = uint32_t(src) +
                                    ((uint32_t)num_srcs * uint32_t(src_idx01));
      });
  }
  return RaggedShape2(&row_splits_out, &row_ids_out, tot_elems);
}


RaggedShape SubsampleRaggedLayer(RaggedShape &src, int32_t layer,
                                 int32_t subsample_factor) {
  K2_CHECK_GE(layer, 0);
  K2_CHECK_LT(layer, src.NumAxes() - 1);
  int32_t num_rows = src.TotSize(layer),
         num_elems = src.TotSize(layer + 1);
  K2_CHECK_EQ(src.TotSize(layer) % subsample_factor, 0);

  ContextPtr &c = src.Context();

  int32_t new_num_rows = num_rows / subsample_factor;

  Array1<int32_t> new_row_splits(c, new_num_rows + 1),
      new_row_ids(c, num_elems);

  const int32_t *row_splits_data = src.RowSplits(layer + 1).Data(),
                   *row_ids_data = src.RowIds(layer + 1).Data();
  int32_t *new_row_splits_data = new_row_splits.Data(),
             *new_row_ids_data = new_row_ids.Data();
  if (c->GetDeviceType() == kCpu) {
    for (int32_t i = 0; i <= new_num_rows; i++)
      new_row_splits_data[i] = row_splits_data[i * subsample_factor];
    for (int32_t i = 0; i < num_elems; i++)
      new_row_ids_data[i] = row_ids_data[i] / subsample_factor;
  } else {
    int32_t block_size = 32;
    auto lambda_round_up = [=] (int32_t n) -> int32_t {
      return block_size * ((n + block_size - 1) / block_size);
    };
    // this rounding is to avoid one warp having to do 2 jobs, which would slow
    // down the code due to warp divergence.
    int32_t num_elems_plus = lambda_round_up(num_elems);

    auto lambda_set_row_splits_and_ids = [=] __device__(int32_t i) -> void {
      if (i >= num_elems_plus) {
        int32_t r = i - num_elems_plus;
        new_row_splits_data[r] = row_splits_data[r * subsample_factor];
      } else if (i < num_elems) {
        new_row_ids_data[i] = row_ids_data[i] / subsample_factor;
      }
    };
    EvalDevice(c, num_elems_plus + new_num_rows + 1,
               lambda_set_row_splits_and_ids);
  }
  return RaggedShape2(&new_row_splits, &new_row_ids, num_elems);
}




}  // namespace k2
