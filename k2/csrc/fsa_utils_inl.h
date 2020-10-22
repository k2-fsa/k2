/**
 * @brief
 * fsa_utils_inl
 *
 * @note
 * Don't include this file directly; it is included by fsa_utils.h.
 * It contains implementation code.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_FSA_UTILS_INL_H_
#define K2_CSRC_FSA_UTILS_INL_H_

#ifndef IS_IN_K2_CSRC_FSA_UTILS_H_
#error "this file is supposed to be included only by fsa_utils.h"
#endif

#include <algorithm>
#include <limits>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/ragged.h"

namespace k2 {
template <typename FloatType>
Array1<FloatType> GetForwardScores(FsaVec &fsas, Ragged<int32_t> &state_batches,
                                   Ragged<int32_t> &entering_arc_batches,
                                   bool log_semiring) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK(IsCompatible(fsas, entering_arc_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(entering_arc_batches.NumAxes(), 4);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  // just using DCHECK below to save time in production code
  K2_DCHECK_EQ((state_batches.TotSize(1) / num_batches), num_fsas);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(entering_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(entering_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(entering_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(entering_arc_batches.NumElements(), num_arcs);

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> state_scores(c, num_states, negative_infinity);
  FloatType *state_scores_data = state_scores.Data();
  // set the score of start state in each fsa to be 0
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  auto lambda_set_start_state_score = [=] __host__ __device__(int32_t fsa_idx) {
    int32_t start_state = fsa_row_splits1[fsa_idx],
            start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
    if (start_state_next_fsa - start_state > 0)
      state_scores_data[start_state] = 0;
  };
  Eval(c, num_fsas, lambda_set_start_state_score);

  // compute entering arc numbers in each batch
  Array1<int32_t> entering_arc_nums(c, num_batches);
  Array1<int32_t> entering_arc_start_index(c, num_batches);
  int32_t *entering_arc_nums_data = entering_arc_nums.Data();
  int32_t *entering_arc_start_index_data = entering_arc_start_index.Data();
  const int32_t *fsa_in_batch_row_splits =
      entering_arc_batches.RowSplits(1).Data();
  const int32_t *state_in_batch_row_splits =
      entering_arc_batches.RowSplits(2).Data();
  const int32_t *entering_arc_row_splits =
      entering_arc_batches.RowSplits(3).Data();
  auto lambda_set_entering_arc_nums =
      [=] __host__ __device__(int32_t batch_idx) {
        int32_t first_state_idx_in_batch =
                    state_in_batch_row_splits[batch_idx * num_fsas],
                next_last_state_idx_in_batch =
                    state_in_batch_row_splits[(batch_idx + 1) * num_fsas];
        int32_t first_entering_arc_idx =
                    entering_arc_row_splits[first_state_idx_in_batch],
                next_last_entering_arc_idx =
                    entering_arc_row_splits[next_last_state_idx_in_batch];
        entering_arc_start_index_data[batch_idx] = first_entering_arc_idx;
        entering_arc_nums_data[batch_idx] =
            next_last_entering_arc_idx - first_entering_arc_idx;
      };
  Eval(c, num_batches, lambda_set_entering_arc_nums);

  const int32_t *fsa_in_batch_row_ids = entering_arc_batches.RowIds(1).Data();
  const int32_t *state_in_batch_row_ids = entering_arc_batches.RowIds(2).Data();
  const int32_t *entering_arc_row_ids = entering_arc_batches.RowIds(3).Data();
  const int32_t *entering_arc_ids = entering_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> entering_arc_score_values(
      c, num_arcs);  // entering arc_scores in batches
  FloatType *arc_scores_data = entering_arc_score_values.Data();
  // convert entering_arc_nums to cpu as we will access its elements in below
  // Eval function for `lambda_set_entering_arc_scores`
  entering_arc_nums = entering_arc_nums.To(GetCpuContext());
  const int32_t *cpu_entering_arc_nums = entering_arc_nums.Data();
  Array1<int32_t> cpu_entering_arc_start_index =
      entering_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_entering_arc_start = cpu_entering_arc_start_index.Data();
  // copy row_splits2 to cpu
  Array1<int32_t> cpu_row_splits2 =
      entering_arc_batches.RowSplits(2).To(GetCpuContext());
  const int32_t *cpu_row_splits2_data = cpu_row_splits2.Data();
  Array1<int32_t> arc_sum_row_splits(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);
  // process batch sequentially.
  for (int32_t i = 0; i != num_batches; ++i) {
    // get the range we would call Max/LogSum per sub list
    int32_t row_begin =
        cpu_row_splits2_data[i * num_fsas];  // row_start is idx01x
    int32_t row_end = cpu_row_splits2_data[(i + 1) * num_fsas];
    K2_CHECK_LT(row_begin, num_states);
    K2_CHECK_LE(row_end, num_states);
    int32_t num_rows =
        row_end - row_begin;  // num_rows is num_states in this batch
    K2_CHECK_LT(num_rows, arc_sum_row_splits.Dim());
    // we always use the first num_rows elements in arc_sum_row_splits.
    Array1<int32_t> sum_sub_range =
        arc_sum_row_splits.Range(0, num_rows + 1);  // +1 for the last element
    {
      ParallelRunner pr(c);
      // get entering arc scores
      {
        cudaStream_t stream = pr.NewStream();
        With w(stream);
        auto lambda_set_entering_arc_score = [=] __host__ __device__(
                                                 int32_t idx3) {
          // all idx** in below code are the indexes to entering_arc_batches
          int32_t idx0123 = entering_arc_start_index_data[i] + idx3;
          int32_t idx012 = entering_arc_row_ids[idx0123];
          int32_t idx01 = state_in_batch_row_ids[idx012];
          K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
          int32_t fsa_id = idx01 % num_fsas;

          int32_t entering_arc_id = entering_arc_ids[idx0123];
          float curr_arc_score = arcs[entering_arc_id].score;
          int32_t src_state_idx1 = arcs[entering_arc_id].src_state;
          int32_t src_state_idx01 = fsa_row_splits1[fsa_id] + src_state_idx1;
          arc_scores_data[idx0123] =
              state_scores_data[src_state_idx01] + curr_arc_score;
        };
        Eval(stream, cpu_entering_arc_nums[i], lambda_set_entering_arc_score);
      }
      {
        cudaStream_t stream = pr.NewStream();
        With w(stream);
        // make entering arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = sum_sub_range.Data();
        auto lambda_set_row_splits_for_sum =
            [=] __host__ __device__(int32_t idx) {
              sum_splits_data[idx] = entering_arc_row_splits[idx + row_begin] -
                                     entering_arc_row_splits[row_begin];
            };
        Eval(stream, num_rows + 1, lambda_set_row_splits_for_sum);
      }
    }
    int32_t arc_start = cpu_entering_arc_start[i];
    int32_t num_arcs_this_batch = cpu_entering_arc_nums[i];
    Array1<FloatType> sub_scores_values =
        entering_arc_score_values.Range(arc_start, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&sum_sub_range, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores = score_cache.Range(0, num_rows);
    // get scores per state in this batch
    if (log_semiring)
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    else
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // copy those scores to corresponding state in state_scores
    auto lambda_copy_state_scores = [=] __host__ __device__(int32_t idx2) {
      int32_t idx012 = row_begin + idx2;
      int32_t state_idx012 = states_data[idx012];
      int32_t idx01 = state_in_batch_row_ids[idx012];
      int32_t fsa_id = idx01 % num_fsas;
      int32_t start_state_idx = fsa_row_splits1[fsa_id];
      // don't override score 0 in the start state in each fsa.
      if (state_idx012 != start_state_idx)
        state_scores_data[state_idx012] = sub_state_scores_data[idx2];
    };
    Eval(c, num_rows, lambda_copy_state_scores);
  }

  return state_scores;
}

template <typename FloatType>
Array1<FloatType> GetBackwardScores(
    FsaVec &fsas, Ragged<int32_t> &state_batches,
    Ragged<int32_t> &leaving_arc_batches,
    const Array1<FloatType> *tot_scores /*= nullptr*/,
    bool log_semiring /*= true*/) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(IsCompatible(fsas, state_batches));
  K2_CHECK(IsCompatible(fsas, leaving_arc_batches));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  K2_CHECK_EQ(state_batches.NumAxes(), 3);
  K2_CHECK_EQ(leaving_arc_batches.NumAxes(), 4);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  int32_t num_batches = state_batches.Dim0();
  // just using DCHECK below to save time in production code
  K2_DCHECK_EQ((state_batches.TotSize(1) / num_batches), num_fsas);
  K2_DCHECK_EQ(state_batches.NumElements(), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.Dim0(), num_batches);
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(1), state_batches.TotSize(1));
  K2_DCHECK_EQ(leaving_arc_batches.TotSize(2), num_states);
  K2_DCHECK_EQ(leaving_arc_batches.NumElements(), num_arcs);

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> state_scores(c, num_states, negative_infinity);
  FloatType *state_scores_data = state_scores.Data();
  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  if (tot_scores != nullptr) {
    K2_CHECK(IsCompatible(fsas, *tot_scores));
    K2_CHECK_EQ(tot_scores->Dim(), num_fsas);
    const FloatType *tot_scores_data = tot_scores->Data();
    // set the score of final state in fsa i to be negative of tot_scores[i]
    auto lambda_set_final_state_score = [=] __host__ __device__(
                                            int32_t fsa_idx) {
      int32_t start_state = fsa_row_splits1[fsa_idx],
              start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
      if (start_state_next_fsa - start_state > 0)
        state_scores_data[start_state_next_fsa - 1] = -tot_scores_data[fsa_idx];
    };
    Eval(c, num_fsas, lambda_set_final_state_score);
  } else {
    // set the score of final state in each fsa to be 0
    auto lambda_set_final_state_score =
        [=] __host__ __device__(int32_t fsa_idx) {
          int32_t start_state = fsa_row_splits1[fsa_idx],
                  start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
          if (start_state_next_fsa - start_state > 0)
            state_scores_data[start_state_next_fsa - 1] = 0;
        };
    Eval(c, num_fsas, lambda_set_final_state_score);
  }

  // compute leaving arc numbers in each batch
  Array1<int32_t> leaving_arc_nums(c, num_batches);
  Array1<int32_t> leaving_arc_start_index(c, num_batches);
  int32_t *leaving_arc_nums_data = leaving_arc_nums.Data();
  int32_t *leaving_arc_start_index_data = leaving_arc_start_index.Data();
  const int32_t *fsa_in_batch_row_splits =
      leaving_arc_batches.RowSplits(1).Data();
  const int32_t *state_in_batch_row_splits =
      leaving_arc_batches.RowSplits(2).Data();
  const int32_t *leaving_arc_row_splits =
      leaving_arc_batches.RowSplits(3).Data();
  auto lambda_set_leaving_arc_nums =
      [=] __host__ __device__(int32_t batch_idx) {
        int32_t first_state_idx_in_batch =
                    state_in_batch_row_splits[batch_idx * num_fsas],
                next_last_state_idx_in_batch =
                    state_in_batch_row_splits[(batch_idx + 1) * num_fsas];
        int32_t first_leaving_arc_idx =
                    leaving_arc_row_splits[first_state_idx_in_batch],
                next_last_leaving_arc_idx =
                    leaving_arc_row_splits[next_last_state_idx_in_batch];
        leaving_arc_start_index_data[batch_idx] = first_leaving_arc_idx;
        leaving_arc_nums_data[batch_idx] =
            next_last_leaving_arc_idx - first_leaving_arc_idx;
      };
  Eval(c, num_batches, lambda_set_leaving_arc_nums);

  const int32_t *fsa_in_batch_row_ids = leaving_arc_batches.RowIds(1).Data();
  const int32_t *state_in_batch_row_ids = leaving_arc_batches.RowIds(2).Data();
  const int32_t *leaving_arc_row_ids = leaving_arc_batches.RowIds(3).Data();
  const int32_t *leaving_arc_ids = leaving_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> leaving_arc_score_values(
      c, num_arcs);  // leaving arc_scores in batches
  FloatType *arc_scores_data = leaving_arc_score_values.Data();
  // convert leaving_arc_nums to cpu as we will access its elements in below
  // Eval function for `lambda_set_leaving_arc_scores`
  leaving_arc_nums = leaving_arc_nums.To(GetCpuContext());
  const int32_t *cpu_leaving_arc_nums = leaving_arc_nums.Data();
  Array1<int32_t> cpu_leaving_arc_start_index =
      leaving_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_leaving_arc_start = cpu_leaving_arc_start_index.Data();
  // copy row_splits2 to cpu
  Array1<int32_t> cpu_row_splits2 =
      leaving_arc_batches.RowSplits(2).To(GetCpuContext());
  const int32_t *cpu_row_splits2_data = cpu_row_splits2.Data();
  Array1<int32_t> arc_sum_row_splits(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);
  // process batch sequentially.
  for (int32_t i = num_batches - 1; i >= 0; --i) {
    // get the range we would call Max/LogSum per sub list
    int32_t row_begin =
        cpu_row_splits2_data[i * num_fsas];  // row_start is idx01x
    int32_t row_end = cpu_row_splits2_data[(i + 1) * num_fsas];
    K2_CHECK_LT(row_begin, num_states);
    K2_CHECK_LE(row_end, num_states);
    int32_t num_rows =
        row_end - row_begin;  // num_rows is num_states in this batch
    K2_CHECK_LT(num_rows, arc_sum_row_splits.Dim());
    // we always use the first num_rows elements in arc_sum_row_splits.
    Array1<int32_t> sum_sub_range =
        arc_sum_row_splits.Range(0, num_rows + 1);  // +1 for the last element
    {
      ParallelRunner pr(c);
      // get leaving arc scores
      {
        cudaStream_t stream = pr.NewStream();
        With w(stream);
        auto lambda_set_leaving_arc_score = [=] __host__ __device__(
                                                int32_t idx3) {
          // all idx** in below code are the indexes to leaving_arc_batches
          int32_t idx0123 = leaving_arc_start_index_data[i] + idx3;
          int32_t idx012 = leaving_arc_row_ids[idx0123];
          int32_t idx01 = state_in_batch_row_ids[idx012];
          K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
          int32_t fsa_id = idx01 % num_fsas;

          int32_t leaving_arc_id = leaving_arc_ids[idx0123];
          float curr_arc_score = arcs[leaving_arc_id].score;
          int32_t dest_state_idx1 = arcs[leaving_arc_id].dest_state;
          int32_t dest_state_idx01 = fsa_row_splits1[fsa_id] + dest_state_idx1;
          arc_scores_data[idx0123] =
              state_scores_data[dest_state_idx01] + curr_arc_score;
        };
        Eval(stream, cpu_leaving_arc_nums[i], lambda_set_leaving_arc_score);
      }
      {
        cudaStream_t stream = pr.NewStream();
        With w(stream);
        // make leaving arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = sum_sub_range.Data();
        auto lambda_set_row_splits_for_sum =
            [=] __host__ __device__(int32_t idx) {
              sum_splits_data[idx] = leaving_arc_row_splits[idx + row_begin] -
                                     leaving_arc_row_splits[row_begin];
            };
        Eval(stream, num_rows + 1, lambda_set_row_splits_for_sum);
      }
    }
    int32_t arc_start = cpu_leaving_arc_start[i];
    int32_t num_arcs_this_batch = cpu_leaving_arc_nums[i];
    Array1<FloatType> sub_scores_values =
        leaving_arc_score_values.Range(arc_start, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&sum_sub_range, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores = score_cache.Range(0, num_rows);
    // get scores per state in this batch
    if (log_semiring)
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    else
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // copy those scores to corresponding state in state_scores
    auto lambda_copy_state_scores = [=] __host__ __device__(int32_t idx2) {
      int32_t idx012 = row_begin + idx2;
      int32_t state_idx012 = states_data[idx012];
      int32_t idx01 = state_in_batch_row_ids[idx012];
      int32_t fsa_id = idx01 % num_fsas;
      int32_t start_state = fsa_row_splits1[fsa_id],
              start_state_next_fsa = fsa_row_splits1[fsa_id + 1];
      if (start_state_next_fsa - start_state > 0) {  // non-empty fsa
        int32_t final_state_idx = start_state_next_fsa - 1;
        // don't override score in the final state in each fsa.
        if (state_idx012 != final_state_idx)
          state_scores_data[state_idx012] = sub_state_scores_data[idx2];
      }
    };
    Eval(c, num_rows, lambda_copy_state_scores);
  }

  return state_scores;
}

template <typename FloatType>
Array1<FloatType> GetTotScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(IsCompatible(fsas, forward_scores));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1);
  K2_CHECK_EQ(num_states, forward_scores.Dim());

  FloatType negative_infinity = -std::numeric_limits<FloatType>::infinity();
  Array1<FloatType> tot_scores(c, num_fsas, negative_infinity);
  FloatType *tot_scores_data = tot_scores.Data();

  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  auto lambda_copy_tot_scores = [=] __host__ __device__(int32_t fsa_idx) {
    int32_t start_state = fsa_row_splits1[fsa_idx],
            start_state_next_fsa = fsa_row_splits1[fsa_idx + 1];
    if (start_state_next_fsa - start_state > 0) {  // non-empty fsa
      int32_t final_state_idx = start_state_next_fsa - 1;
      tot_scores_data[fsa_idx] = forward_scores_data[final_state_idx];
    }
  };
  Eval(c, num_fsas, lambda_copy_tot_scores);

  return tot_scores;
}

// TODO(haowen): implement below functions
template <typename FloatType>
Array1<FloatType> GetArcScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores,
                               const Array1<FloatType> &backward_scores,
                               bool log_semiring) {
  ContextPtr &c = fsas.Context();
  K2_LOG(INFO) << "Not Implemented!";
  return Array1<FloatType>(c, 0);
}

}  // namespace k2

#endif  // K2_CSRC_FSA_UTILS_INL_H_
