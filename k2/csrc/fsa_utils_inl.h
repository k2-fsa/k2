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

  // get the 1st entering arc index in each batch, +1 so we can get the number
  // of entering arcs in each batch by taking the difference of adjacent
  // elements
  Array1<int32_t> entering_arc_start_index(c, num_batches + 1);
  int32_t *entering_arc_start_index_data = entering_arc_start_index.Data();
  const int32_t *arc_batches_row_splits1 =
      entering_arc_batches.RowSplits(1).Data();
  const int32_t *arc_batches_row_splits2 =
      entering_arc_batches.RowSplits(2).Data();
  const int32_t *arc_batches_row_splits3 =
      entering_arc_batches.RowSplits(3).Data();
  auto lambda_set_entering_arc_start_index = [=] __host__ __device__(
                                                 int32_t batch_idx) {
    int32_t this_state_idx0xx = arc_batches_row_splits2[batch_idx * num_fsas];
    int32_t this_arc_idx0xxx = arc_batches_row_splits3[this_state_idx0xx];
    entering_arc_start_index_data[batch_idx] = this_arc_idx0xxx;
    if (batch_idx == num_batches - 1) {
      // process the last element
      int32_t next_state_idx0xx =
          arc_batches_row_splits2[num_batches * num_fsas];
      int32_t next_arc_idx0xxx = arc_batches_row_splits3[next_state_idx0xx];
      entering_arc_start_index_data[num_batches] = next_arc_idx0xxx;
    }
  };
  Eval(c, num_batches, lambda_set_entering_arc_start_index);

  const int32_t *arc_batches_row_ids1 = entering_arc_batches.RowIds(1).Data();
  const int32_t *arc_batches_row_ids2 = entering_arc_batches.RowIds(2).Data();
  const int32_t *arc_batches_row_ids3 = entering_arc_batches.RowIds(3).Data();
  const int32_t *entering_arc_ids = entering_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> entering_arc_score_values(
      c, num_arcs);  // entering arc_scores in batches
  FloatType *arc_scores_data = entering_arc_score_values.Data();
  // copy entering_arc_start_index to cpu as we will access its elements in
  // below Eval function for `lambda_set_entering_arc_scores`
  Array1<int32_t> cpu_entering_arc_start_index =
      entering_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_entering_arc_start = cpu_entering_arc_start_index.Data();
  // copy the index of start state in each fsa to CPU
  Array1<int32_t> arc_batches_row_splits1_array =
      entering_arc_batches.RowSplits(1);
  Array1<int32_t> cpu_state_idx0xx =
      entering_arc_batches.RowSplits(2)[arc_batches_row_splits1_array].To(
          GetCpuContext());
  K2_CHECK_EQ(cpu_state_idx0xx.Dim(), num_batches + 1);
  const int32_t *cpu_state_idx0xx_data = cpu_state_idx0xx.Data();
  Array1<int32_t> arc_sum_row_splits(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);
  // process batch sequentially.
  for (int32_t i = 0; i < num_batches; ++i) {
    // get the range we would call Max/LogSum per sub list
    int32_t this_state_idx0xx = cpu_state_idx0xx[i];
    int32_t next_state_idx0xx =
        cpu_state_idx0xx_data[i + 1];  // the 1st state idx in the next batch
    K2_CHECK_LT(this_state_idx0xx, num_states);
    K2_CHECK_LE(next_state_idx0xx, num_states);
    int32_t num_states_this_batch = next_state_idx0xx - this_state_idx0xx;
    K2_CHECK_LT(num_states_this_batch, arc_sum_row_splits.Dim());
    // we always use the first `num_states_this_batch` elements in
    // arc_sum_row_splits.
    Array1<int32_t> sum_sub_range = arc_sum_row_splits.Range(
        0, num_states_this_batch + 1);  // +1 for the last element
    int32_t num_arcs_this_batch =
        cpu_entering_arc_start[i + 1] - cpu_entering_arc_start[i];
    {
      ParallelRunner pr(c);
      // get entering arc scores
      {
        With w(pr.NewStream());
        auto lambda_set_entering_arc_score = [=] __host__ __device__(
                                                 int32_t idx3) {
          // all idx** in below code are the indexes to entering_arc_batches
          int32_t idx0123 = entering_arc_start_index_data[i] + idx3;
          int32_t idx012 = arc_batches_row_ids3[idx0123];
          int32_t idx01 = arc_batches_row_ids2[idx012];
          K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
          int32_t fsa_id = idx01 % num_fsas;

          int32_t entering_arc_id = entering_arc_ids[idx0123];
          float curr_arc_score = arcs[entering_arc_id].score;
          int32_t src_state_idx1 = arcs[entering_arc_id].src_state;
          int32_t src_state_idx01 = fsa_row_splits1[fsa_id] + src_state_idx1;
          arc_scores_data[idx0123] =
              state_scores_data[src_state_idx01] + curr_arc_score;
        };
        Eval(c, num_arcs_this_batch, lambda_set_entering_arc_score);
      }
      {
        With w(pr.NewStream());
        // make entering arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = sum_sub_range.Data();
        auto lambda_set_row_splits_for_sum =
            [=] __host__ __device__(int32_t idx) {
              sum_splits_data[idx] =
                  arc_batches_row_splits3[idx + this_state_idx0xx] -
                  arc_batches_row_splits3[this_state_idx0xx];
            };
        Eval(c, num_states_this_batch + 1, lambda_set_row_splits_for_sum);
      }
    }
    int32_t this_arc_idx0xxx = cpu_entering_arc_start[i];
    Array1<FloatType> sub_scores_values =
        entering_arc_score_values.Range(this_arc_idx0xxx, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&sum_sub_range, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores =
        score_cache.Range(0, num_states_this_batch);
    // get scores per state in this batch
    if (log_semiring)
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    else
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // copy those scores to corresponding state in state_scores
    auto lambda_copy_state_scores = [=] __host__ __device__(int32_t idx2) {
      int32_t idx012 = this_state_idx0xx + idx2;
      int32_t state_idx012 = states_data[idx012];
      int32_t idx01 = arc_batches_row_ids2[idx012];
      int32_t fsa_id = idx01 % num_fsas;
      int32_t start_state_idx = fsa_row_splits1[fsa_id];
      // don't override score 0 in the start state in each fsa.
      if (state_idx012 != start_state_idx)
        state_scores_data[state_idx012] = sub_state_scores_data[idx2];
    };
    Eval(c, num_states_this_batch, lambda_copy_state_scores);
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

  // get the 1st leaving arc index in each batch, +1 so we can get the number of
  // leaving arcs in each batch by taking the difference of adjacent elements
  Array1<int32_t> leaving_arc_start_index(c, num_batches + 1);
  int32_t *leaving_arc_start_index_data = leaving_arc_start_index.Data();
  const int32_t *arc_batches_row_splits1 =
      leaving_arc_batches.RowSplits(1).Data();
  const int32_t *arc_batches_row_splits2 =
      leaving_arc_batches.RowSplits(2).Data();
  const int32_t *arc_batches_row_splits3 =
      leaving_arc_batches.RowSplits(3).Data();
  auto lambda_set_leaving_arc_start_index = [=] __host__ __device__(
                                                int32_t batch_idx) {
    int32_t this_state_idx0xx = arc_batches_row_splits2[batch_idx * num_fsas];
    int32_t this_arc_idx0xxx = arc_batches_row_splits3[this_state_idx0xx];
    leaving_arc_start_index_data[batch_idx] = this_arc_idx0xxx;
    if (batch_idx == num_batches - 1) {
      // process the last element
      int32_t next_state_idx0xx =
          arc_batches_row_splits2[num_batches * num_fsas];
      int32_t next_arc_idx0xxx = arc_batches_row_splits3[next_state_idx0xx];
      leaving_arc_start_index_data[num_batches] = next_arc_idx0xxx;
    }
  };
  Eval(c, num_batches, lambda_set_leaving_arc_start_index);

  const int32_t *arc_batches_row_ids1 = leaving_arc_batches.RowIds(1).Data();
  const int32_t *arc_batches_row_ids2 = leaving_arc_batches.RowIds(2).Data();
  const int32_t *arc_batches_row_ids3 = leaving_arc_batches.RowIds(3).Data();
  const int32_t *leaving_arc_ids = leaving_arc_batches.values.Data();
  const int32_t *states_data = state_batches.values.Data();
  const Arc *arcs = fsas.values.Data();
  Array1<FloatType> leaving_arc_score_values(
      c, num_arcs);  // leaving arc_scores in batches
  FloatType *arc_scores_data = leaving_arc_score_values.Data();
  // copy leaving_arc_start_index to cpu as we will access its elements in below
  // Eval function for `lambda_set_leaving_arc_scores`
  Array1<int32_t> cpu_leaving_arc_start_index =
      leaving_arc_start_index.To(GetCpuContext());
  const int32_t *cpu_leaving_arc_start = cpu_leaving_arc_start_index.Data();
  // copy the index of start state in each fsa to CPU
  Array1<int32_t> arc_batches_row_splits1_array =
      leaving_arc_batches.RowSplits(1);
  Array1<int32_t> cpu_state_idx0xx =
      leaving_arc_batches.RowSplits(2)[arc_batches_row_splits1_array].To(
          GetCpuContext());
  K2_CHECK_EQ(cpu_state_idx0xx.Dim(), num_batches + 1);
  const int32_t *cpu_state_idx0xx_data = cpu_state_idx0xx.Data();
  Array1<int32_t> arc_sum_row_splits(c, num_states + 1);
  Array1<FloatType> score_cache(c, num_states + 1);
  // process batch sequentially.
  for (int32_t i = num_batches - 1; i >= 0; --i) {
    // get the range we would call Max/LogSum per sub list
    int32_t this_state_idx0xx = cpu_state_idx0xx[i];
    int32_t next_state_idx0xx =
        cpu_state_idx0xx_data[i + 1];  // the 1st state idx in the next batch
    K2_CHECK_LT(this_state_idx0xx, num_states);
    K2_CHECK_LE(next_state_idx0xx, num_states);
    int32_t num_states_this_batch = next_state_idx0xx - this_state_idx0xx;
    K2_CHECK_LT(num_states_this_batch, arc_sum_row_splits.Dim());
    // we always use the first `num_states_this_batch` elements in
    // arc_sum_row_splits.
    Array1<int32_t> sum_sub_range = arc_sum_row_splits.Range(
        0, num_states_this_batch + 1);  // +1 for the last element
    int32_t num_arcs_this_batch =
        cpu_leaving_arc_start[i + 1] - cpu_leaving_arc_start[i];
    {
      ParallelRunner pr(c);
      // get leaving arc scores
      {
        With w(pr.NewStream());
        auto lambda_set_leaving_arc_score = [=] __host__ __device__(
                                                int32_t idx3) {
          // all idx** in below code are the indexes to leaving_arc_batches
          int32_t idx0123 = leaving_arc_start_index_data[i] + idx3;
          int32_t idx012 = arc_batches_row_ids3[idx0123];
          int32_t idx01 = arc_batches_row_ids2[idx012];
          K2_CHECK_EQ(idx01 / num_fsas, i);  // idx01/num_fsas is batch_id
          int32_t fsa_id = idx01 % num_fsas;

          int32_t leaving_arc_id = leaving_arc_ids[idx0123];
          float curr_arc_score = arcs[leaving_arc_id].score;
          int32_t dest_state_idx1 = arcs[leaving_arc_id].dest_state;
          int32_t dest_state_idx01 = fsa_row_splits1[fsa_id] + dest_state_idx1;
          arc_scores_data[idx0123] =
              state_scores_data[dest_state_idx01] + curr_arc_score;
        };
        Eval(c, num_arcs_this_batch, lambda_set_leaving_arc_score);
      }
      {
        With w(pr.NewStream());
        // make leaving arc row splits info in each batch starting from zero,
        // we will use it to call MaxPerSublist or LogSumPerSubList
        int32_t *sum_splits_data = sum_sub_range.Data();
        auto lambda_set_row_splits_for_sum =
            [=] __host__ __device__(int32_t idx) {
              sum_splits_data[idx] =
                  arc_batches_row_splits3[idx + this_state_idx0xx] -
                  arc_batches_row_splits3[this_state_idx0xx];
            };
        Eval(c, num_states_this_batch + 1, lambda_set_row_splits_for_sum);
      }
    }
    int32_t this_arc_idx0xxx = cpu_leaving_arc_start[i];
    Array1<FloatType> sub_scores_values =
        leaving_arc_score_values.Range(this_arc_idx0xxx, num_arcs_this_batch);
    RaggedShape sub_scores_shape =
        RaggedShape2(&sum_sub_range, nullptr, sub_scores_values.Dim());
    Ragged<FloatType> sub_scores(sub_scores_shape, sub_scores_values);
    // we always use the first num_rows elements in score_cache.
    Array1<FloatType> sub_state_scores =
        score_cache.Range(0, num_states_this_batch);
    // get scores per state in this batch
    if (log_semiring)
      LogSumPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    else
      MaxPerSublist(sub_scores, negative_infinity, &sub_state_scores);
    const FloatType *sub_state_scores_data = sub_state_scores.Data();
    // copy those scores to corresponding state in state_scores
    auto lambda_copy_state_scores = [=] __host__ __device__(int32_t idx2) {
      int32_t idx012 = this_state_idx0xx + idx2;
      int32_t state_idx012 = states_data[idx012];
      int32_t idx01 = arc_batches_row_ids2[idx012];
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
    Eval(c, num_states_this_batch, lambda_copy_state_scores);
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

template <typename FloatType>
Array1<FloatType> GetArcScores(FsaVec &fsas,
                               const Array1<FloatType> &forward_scores,
                               const Array1<FloatType> &backward_scores) {
  K2_STATIC_ASSERT((std::is_same<float, FloatType>::value ||
                    std::is_same<double, FloatType>::value));
  K2_CHECK(IsCompatible(fsas, forward_scores));
  K2_CHECK(IsCompatible(fsas, backward_scores));
  K2_CHECK_EQ(fsas.NumAxes(), 3);
  ContextPtr &c = fsas.Context();
  int32_t num_fsas = fsas.Dim0(), num_states = fsas.TotSize(1),
          num_arcs = fsas.TotSize(2);
  K2_CHECK_EQ(num_states, forward_scores.Dim());
  K2_CHECK_EQ(num_states, backward_scores.Dim());

  Array1<FloatType> arc_scores(c, num_arcs);
  FloatType *arc_scores_data = arc_scores.Data();

  const int32_t *fsa_row_splits1 = fsas.RowSplits(1).Data();
  const int32_t *fsa_row_ids1 = fsas.RowIds(1).Data();
  const int32_t *fsa_row_ids2 = fsas.RowIds(2).Data();
  const Arc *arcs = fsas.values.Data();
  const FloatType *forward_scores_data = forward_scores.Data();
  const FloatType *backward_scores_data = backward_scores.Data();
  auto lambda_get_arc_scores = [=] __host__ __device__(int32_t arc_idx012) {
    int32_t src_state_idx1 = arcs[arc_idx012].src_state;
    int32_t dest_state_idx1 = arcs[arc_idx012].dest_state;
    float arc_score = arcs[arc_idx012].score;

    int32_t idx01 = fsa_row_ids2[arc_idx012];
    int32_t idx0 = fsa_row_ids1[idx01];
    int32_t idx0x = fsa_row_splits1[idx0];
    int32_t src_state_idx01 = idx0x + src_state_idx1;
    int32_t dest_state_idx01 = idx0x + dest_state_idx1;
    arc_scores_data[arc_idx012] = arc_score +
                                  forward_scores_data[src_state_idx01] +
                                  backward_scores_data[dest_state_idx01];
  };
  Eval(c, num_arcs, lambda_get_arc_scores);

  return arc_scores;
}

}  // namespace k2

#endif  // K2_CSRC_FSA_UTILS_INL_H_
