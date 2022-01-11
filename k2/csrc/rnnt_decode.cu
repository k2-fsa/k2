/**
 * Copyright      2021  Xiaomi Corporation (authors: Daniel Povey, Wei Kang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "k2/csrc/context.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/csrc/rnnt_decode.h"

namespace k2 {

void PruneStreams(const RaggedShape &shape, const Array1<double> &scores,
                  const Array1<int32_t> &categories, float beam,
                  int32_t max_per_stream, int32_t max_per_category,
                  Renumbering *renumbering, Ragged<int32_t> *kept_states,
                  Array1<int32_t> *kept_categories) {
  NVTX_RANGE(K2_FUNC);
  K2_CHECK_EQ(shape.NumAxes(), 2);
  int32_t num_streams = shape.Dim0(), num_total_states = shape.TotSize(1);
  K2_CHECK_EQ(num_total_states, scores.Dim());
  K2_CHECK_EQ(num_total_states, categories.Dim());
  K2_CHECK(renumbering != nullptr);
  K2_CHECK(kept_states != nullptr);
  K2_CHECK(kept_categories != nullptr);

  ContextPtr c = GetContext(shape, scores, categories);

  bool prune_with_max_per_stream =
      max_per_stream > 0 && max_per_stream < num_total_states;
  bool prune_with_max_per_category =
      max_per_category > 0 && max_per_category < num_total_states;

  // ragged_scores with shape [stream][value]
  Ragged<double> ragged_scores;
  Array1<int32_t> scores_order;
  if (prune_with_max_per_stream) {
    scores_order = Array1<int32_t>(c, num_total_states);
    // Would be sorted in place, using `Clone` here to avoid modifying scores.
    ragged_scores = Ragged<double>(shape, scores.Clone());
    SortSublists<double, GreaterThan<double>>(&ragged_scores, &scores_order);
  } else {
    ragged_scores = Ragged<double>(shape, scores);
  }

  // max_scores is needed by beam pruning
  auto max_scores = Array1<double>(c, num_streams);
  MaxPerSublist(ragged_scores, -std::numeric_limits<double>::infinity(),
                &max_scores);

  // Will sort categories in place, using `Clone` here to avoid modifying
  // categories.
  auto ragged_categories = Ragged<int32_t>(shape, categories.Clone());
  Array1<int32_t> cates_order(c, num_total_states);
  SortSublists(&ragged_categories, &cates_order);

  // Segment items in each stream by categories
  // We will use `tail concept` to get row_ids, see docs in utils.h for more
  // details of `tail concept`.
  Array1<int32_t> cates_tail(c, num_total_states);
  // rc is short for ragged categories
  const int32_t *rc_row_ids1_data = ragged_categories.RowIds(1).Data(),
                *rc_data = ragged_categories.values.Data();
  int32_t *cates_tail_data = cates_tail.Data();
  K2_EVAL(
      c, num_total_states - 1, lambda_get_cates_tail, (int32_t idx01)->void {
        int32_t idx0 = rc_row_ids1_data[idx01],
                next_idx0 = rc_row_ids1_data[idx01 + 1];
        if (idx0 == next_idx0 && rc_data[idx01] != rc_data[idx01 + 1] ||
            idx0 != next_idx0)
          cates_tail_data[idx01] = 1;
        else
          cates_tail_data[idx01] = 0;
      });
  auto cates_row_ids = Array1<int32_t>(c, num_total_states);
  ExclusiveSum(cates_tail, &cates_row_ids);
  // shape : [cate][value]
  RaggedShape cates_shape =
      RaggedShape2(nullptr, &cates_row_ids, cates_row_ids.Dim());

  // stc is short for stream to categories
  auto stc_row_ids = Array1<int32_t>(c, cates_shape.Dim0());
  int32_t *stc_row_ids_data = stc_row_ids.Data();
  const int32_t *cates_shape_row_splits1_data = cates_shape.RowSplits(1).Data(),
                *shape_row_ids1_data = shape.RowIds(1).Data();
  K2_EVAL(
      c, cates_shape.Dim0(), lambda_set_stc_row_ids, (int32_t i)->void {
        int32_t idx0x = cates_shape_row_splits1_data[i],
                idx0 = shape_row_ids1_data[idx0x];
        stc_row_ids_data[i] = idx0;
      });
  // shape : [stream][cate][value]
  RaggedShape stream_cates_shape = ComposeRaggedShapes(
      RaggedShape2(nullptr, &stc_row_ids, stc_row_ids.Dim()), cates_shape);

  *renumbering = Renumbering(c, num_total_states);
  char *keep_state_data = renumbering->Keep().Data();
  const double *scores_data = ragged_scores.values.Data(),
               *max_scores_data = max_scores.Data();
  const int32_t *shape_row_splits1_data = shape.RowSplits(1).Data(),
                *rs_row_splits1_data = ragged_scores.RowSplits(1).Data(),
                *scores_order_data;
  if (prune_with_max_per_stream) scores_order_data = scores_order.Data();

  // prune with beam and max_per_stream
  K2_EVAL(
      c, num_total_states, lambda_prune_with_beam_and_max_per_stream,
      (int32_t idx01)->void {
        // we will sort the scores if pruning with max_per_stream,
        // getting its original index by scores_order array
        // we sort the scores within the stream, so idx0 (i.e. stream idx)
        // won't change.
        int32_t idx0 = shape_row_ids1_data[idx01], original_idx01;

        bool pruned_by_max_per_stream = false;
        if (prune_with_max_per_stream) {
          original_idx01 = scores_order_data[idx01];
          int32_t idx0x = rs_row_splits1_data[idx0];
          pruned_by_max_per_stream = idx01 - idx0x >= max_per_stream;
        } else {
          original_idx01 = idx01;
        }

        bool pruned_by_beam = scores_data[idx01] < max_scores_data[idx0] - beam;

        if (pruned_by_beam || pruned_by_max_per_stream)
          keep_state_data[original_idx01] = 0;
        else
          keep_state_data[original_idx01] = 1;
      });

  const int32_t *cates_order_data = cates_order.Data(),
                *stream_cates_shape_row_ids2_data =
                    stream_cates_shape.RowIds(2).Data(),
                *stream_cates_shape_row_splits2_data =
                    stream_cates_shape.RowSplits(2).Data();
  // prune with max_per_category
  if (prune_with_max_per_category) {
    // shape : [stream][cate][value]
    Ragged<double> stream_cates_scores(stream_cates_shape, scores[cates_order]);
    Array1<int32_t> cates_scores_order(c, num_total_states);
    SortSublists<double, GreaterThan<double>>(&stream_cates_scores,
                                              &cates_scores_order);
    // Now we have sorted the scores twice, first by categories on each stram
    // second by scores on each stream each category
    const int32_t *cates_scores_order_data = cates_scores_order.Data();
    K2_EVAL(
        c, num_total_states, lambda_prune_with_max_per_category,
        (int32_t idx012)->void {
          int32_t original_idx01 =
                      cates_order_data[cates_scores_order_data[idx012]],
                  idx01 = stream_cates_shape_row_ids2_data[idx012],
                  idx01x = stream_cates_shape_row_splits2_data[idx01];
          if (idx012 - idx01x >= max_per_category)
            keep_state_data[original_idx01] = 0;
        });
  }

  // here sorted means sorted by categories on each stream
  // The returned kept_states should be sorted by categories, so we have to do
  // renumbering on sorted states.
  Renumbering renumber_sorted_states(c, num_total_states);
  char *keep_sorted_state_data = renumber_sorted_states.Keep().Data();

  K2_EVAL(
      c, num_total_states, lambda_renumber_sorted_states,
      (int32_t idx01)->void {
        int32_t original_idx01 = cates_order_data[idx01];
        if (keep_state_data[original_idx01])
          keep_sorted_state_data[idx01] = 1;
        else
          keep_sorted_state_data[idx01] = 0;
      });

  Array1<int32_t> kept_sorted_state_ids = renumber_sorted_states.New2Old();

  Array1<int32_t> kept_states_value(c, kept_sorted_state_ids.Dim());
  Array1<int32_t> kept_states_row_ids(c, kept_sorted_state_ids.Dim());
  const int32_t *kept_sorted_state_ids_data = kept_sorted_state_ids.Data();
  int32_t *kept_states_value_data = kept_states_value.Data(),
          *kept_states_row_ids_data = kept_states_row_ids.Data();

  *kept_categories = Array1<int32_t>(c, stream_cates_shape.TotSize(1));
  int32_t *kept_categories_data = kept_categories->Data();

  K2_EVAL(
      c, kept_sorted_state_ids.Dim(), lambda_set_kept_state_value_and_row_ids,
      (int32_t i)->void {
        int32_t idx01 = kept_sorted_state_ids_data[i],
                original_idx01 = cates_order_data[idx01],
                // idx01 of sorted ragged_categories is the same as idx012
                // of stream_cates_shape. ragged_categories has shape
                // [stream][value], stream_cates_shape has shape
                // [stream][cate][value]
                // csc is short for stream cates shape
            scs_idx01 = stream_cates_shape_row_ids2_data[idx01],
                scs_idx01x = stream_cates_shape_row_splits2_data[scs_idx01];

        kept_categories_data[scs_idx01] = rc_data[scs_idx01x];

        kept_states_value_data[i] = original_idx01;
        kept_states_row_ids_data[i] = scs_idx01;
      });

  RaggedShape kept_states_shape = ComposeRaggedShapes(
      GetLayer(stream_cates_shape, 0),
      RaggedShape2(nullptr, &kept_states_row_ids, kept_states_row_ids.Dim()));

  Renumbering renumber_kept_states;
  kept_states_shape =
      RemoveEmptyLists(kept_states_shape, 1, &renumber_kept_states);

  *kept_states = Ragged<int32_t>(kept_states_shape, kept_states_value);
  *kept_categories = (*kept_categories)[renumber_kept_states.New2Old()];
}

}  // namespace k2
