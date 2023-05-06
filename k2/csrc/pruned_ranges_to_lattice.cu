/**
 * Copyright      2022  Xiaomi Corporation (authors: Liyong Guo)
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

#ifndef K2_CSRC_PRUNE_RANGE_TO_LATTICE_H_
#define K2_CSRC_PRUNE_RANGE_TO_LATTICE_H_

#include <torch/extension.h>

#include <vector>

#include "k2/python/csrc/torch.h"

namespace k2 {

FsaVec PrunedRangesToLattice(
    // Normally, ranges is with shape [B][S][T+1] if !modified, [B][S][T] if modified.
    // Currently, only [B][S][T] is supported.
    torch::Tensor ranges,
    torch::Tensor x_lens,  // [B][T]
    // torch::Tensor blank_connections,
    torch::Tensor y,
    // const Ragged<int32_t> &y,
    torch::Tensor logits,
    Array1<int32_t> *arc_map) {
    ContextPtr context;
    if (ranges.device().type() == torch::kCPU) {
      context = GetCpuContext();
    } else if (ranges.is_cuda()) {
      context = GetCudaContext(ranges.device().index());

      TORCH_CHECK(ranges.get_device() == x_lens.get_device(), "x_lens is on a different device");
      TORCH_CHECK(ranges.get_device() == y.get_device(), "y device");
      TORCH_CHECK(ranges.get_device() == logits.get_device(), "logits device");

    } else {
      K2_LOG(FATAL) << "Unsupported device: " << ranges.device()
                    << "\nOnly CPU and CUDA are supported";
    }
 
    TORCH_CHECK(ranges.dim() == 3, "ranges must be 3-dimensional");
    // U is always 5
    const int32_t B = ranges.size(0), T = ranges.size(1), U = ranges.size(2);


    Dtype t = ScalarTypeToDtype(ranges.scalar_type());
    K2_CHECK_EQ(torch::kInt, ranges.scalar_type());
    K2_CHECK_EQ(torch::kInt, x_lens.scalar_type()); // int32_t
    const float *logits_data = logits.data_ptr<float>();
    const int32_t *ranges_data = ranges.data_ptr<int32_t>();
    const int32_t *x_lens_data = x_lens.data_ptr<int32_t>();
    const int32_t *y_data = y.data_ptr<int32_t>();
    const int32_t rng_stride_0 = ranges.stride(0),
                  rng_stride_1 = ranges.stride(1),
                  rng_stride_2 = ranges.stride(2);
    const int32_t lg_stride_0 = logits.stride(0),
                  lg_stride_1 = logits.stride(1),
                  lg_stride_2 = logits.stride(2),
                  lg_stride_3 = logits.stride(3);
    K2_CHECK_EQ(x_lens.numel(), B);
    Array1<int32_t> f2s_row_splits(context, B + 1);
    int32_t * f2s_row_splits_data = f2s_row_splits.Data();
    K2_EVAL(context, B, lambda_set_f2s_row_splits, (int32_t fsa_idx0) {
        int32_t t = x_lens_data[fsa_idx0];
        K2_CHECK_LE(t, T);
        // + 1 in "t * U + 1" is for super-final state.
        f2s_row_splits_data[fsa_idx0] = t * U + 1;
    });
    ExclusiveSum(f2s_row_splits, &f2s_row_splits);

    RaggedShape fsa_to_states =
            RaggedShape2(&f2s_row_splits, nullptr, -1);
    int32_t num_states = fsa_to_states.NumElements();
    Array1<int32_t> s2c_row_splits(context, num_states + 1);
    int32_t *s2c_row_splits_data = s2c_row_splits.Data();
    const int32_t *fts_row_splits1_data = fsa_to_states.RowSplits(1).Data(),
                  *fts_row_ids1_data = fsa_to_states.RowIds(1).Data();

    // set the arcs number for each state
    K2_EVAL(
        context, num_states, lambda_set_num_arcs, (int32_t state_idx01)->void {
          int32_t fsa_idx0 = fts_row_ids1_data[state_idx01],
                  state_idx0x = fts_row_splits1_data[fsa_idx0],
                  state_idx0x_next = fts_row_splits1_data[fsa_idx0 + 1],
                  state_idx1 = state_idx01 - state_idx0x,
                  t = state_idx1 / U,
                  token_index = state_idx1 % U;

          K2_CHECK_LE(t, x_lens_data[fsa_idx0]);
          if (state_idx1 == x_lens_data[fsa_idx0] * U - 1) {
            // x_lens[fsa_idx0] * U is the state_idx1 of super final-state.
            // x_lens[fsa_idx0] * U - 1 is the state pointing to super final-state.
            // final arc to super final state.
            s2c_row_splits_data[state_idx01] = 1;
            return;
          }
          if (state_idx1 == x_lens_data[fsa_idx0] * U) {
            // x_lens[fsa_idx0] * U is the state_idx1 of super final-state.
            // final state has no leaving arcs.
            s2c_row_splits_data[state_idx01] = 0;
            return;
          }
          int32_t range_offset = fsa_idx0 * rng_stride_0 + t * rng_stride_1 + token_index * rng_stride_2;

          int32_t next_state_idx1 = -1;
          // blank connections of last frame is -1
          // So we need process t == x_lens_data[fsa_idx0]
          if (t < x_lens_data[fsa_idx0] - 1) {
            int32_t range_offset_of_lower_bound_of_next_time_step = fsa_idx0 * rng_stride_0 + (t + 1) * rng_stride_1;
            next_state_idx1 = ranges_data[range_offset] - ranges_data[range_offset_of_lower_bound_of_next_time_step];
          }
          // K2_CHECK_EQ(next_state_idx1_tmp, next_state_idx1);
          if (token_index < U - 1) {
            // Typically, U == 5,
            // the [0, 1, 2, 3] states for each time step may have a vertial arc plus an optional horizontal blank arc.
            s2c_row_splits_data[state_idx01] = 1;
            if (next_state_idx1 >= 0) {
              s2c_row_splits_data[state_idx01] = 2;
            }
          } else {
            // Typically, U == 5,
            // the [4] state for each time step have and only have an horizontal blank arc.
            // K2_CHECK_GE(next_state_idx1, 0);
            s2c_row_splits_data[state_idx01] = 1;
          }
      });

    ExclusiveSum(s2c_row_splits, &s2c_row_splits);
    RaggedShape states_to_arcs =
            RaggedShape2(&s2c_row_splits, nullptr, -1);

    RaggedShape ofsa_shape = ComposeRaggedShapes(fsa_to_states, states_to_arcs);
    int32_t num_arcs = ofsa_shape.NumElements();
    Array1<Arc> arcs(context, num_arcs);
    Array1<int32_t> out_map(context, num_arcs);
    int32_t* out_map_data = out_map.Data();
    Arc *arcs_data = arcs.Data();
    const int32_t *row_splits1_data = ofsa_shape.RowSplits(1).Data(),
                  *row_ids1_data = ofsa_shape.RowIds(1).Data(),
                  *row_splits2_data = ofsa_shape.RowSplits(2).Data(),
                  *row_ids2_data = ofsa_shape.RowIds(2).Data();

    // auto y_shape = y.shape;
    // const int32_t * y_data = y.values.Data();
    int32_t y_stride_0 = y.stride(0),
            y_stride_1 = y.stride(1);
    K2_EVAL(
          context, num_arcs, lambda_set_arcs, (int32_t arc_idx012)->void {
            int32_t state_idx01 = row_ids2_data[arc_idx012],
                    fsa_idx0 = row_ids1_data[state_idx01],
                    state_idx0x = row_splits1_data[fsa_idx0],
                    state_idx0x_next = row_splits1_data[fsa_idx0 + 1],
                    arc_idx01x = row_splits2_data[state_idx01],
                    state_idx1 = state_idx01 - state_idx0x,
                    arc_idx2 = arc_idx012 - arc_idx01x,
                    t = state_idx1 / U,
                    token_index = state_idx1 % U;  // token_index is belong to [0, U)
            Arc arc;
            if (state_idx1 == x_lens_data[fsa_idx0] * U - 1) {
              arc.src_state = state_idx1;
              arc.dest_state = state_idx1 + 1;
              arc.label = -1;
              arc.score = 0.0;
              arcs_data[arc_idx012] = arc;
              out_map_data[arc_idx012] = -1;
              return;
            }
            int32_t range_offset = fsa_idx0 * rng_stride_0 + t * rng_stride_1 + token_index * rng_stride_2;
            int32_t range_offset_of_lower_bound_of_next_time_step = fsa_idx0 * rng_stride_0 + (t + 1) * rng_stride_1;
            int32_t next_state_idx1, logits_offset;
            arc.src_state = state_idx1;
            // arc.
            if (token_index < U - 1) {
              int32_t actual_u = ranges_data[range_offset];
              int32_t y_offset = fsa_idx0 * y_stride_0 + actual_u * y_stride_1;
              int32_t arc_label =  y_data[y_offset];
              switch (arc_idx2) {
                case 0:
                  arc.dest_state = state_idx1 + 1;
                  arc.label = arc_label;
                  logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2 + arc_label * lg_stride_3;
                  // K2_CHECK_LE(logits_offset, 135);
                  arc.score = logits_data[logits_offset];
                  out_map_data[arc_idx012] = logits_offset;
                  // arc.score = 0;
                  break;
                case 1:
                  next_state_idx1 = ranges_data[range_offset] - ranges_data[range_offset_of_lower_bound_of_next_time_step];
                  // blank_connections_data_offset = fsa_idx0 * blk_stride_0 + t * blk_stride_1 + token_index * blk_stride_2;
                  // next_state_idx1 = blank_connections_data[blank_connections_data_offset];
                  // blank connections of last frame is always -1,
                  // So states with num_arcs > 2 (i.e. with arc_idx2==1) could not belong to last frame, i.e. x_lens_data[fsa_idx0] - 1.
                  // K2_CHECK_LE(t, x_lens_data[fsa_idx0] - 1);
                  // K2_CHECK_GE(next_state_idx1, 0);
                  arc.dest_state = next_state_idx1 + (t + 1) * U;
                  arc.label = 0;
                  logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2;
                  arc.score = logits_data[logits_offset];
                  out_map_data[arc_idx012] = logits_offset;
                  // arc.score = 0.0;
                  break;
                default:
                   K2_LOG(FATAL) << "Arc index must be less than 3";
              }
            } else {
              K2_CHECK_EQ(arc_idx2, 0);
              // blank_connections_data_offset = fsa_idx0 * blk_stride_0 + t * blk_stride_1 + token_index * blk_stride_2;
              // next_state_idx1 = blank_connections_data[blank_connections_data_offset];
              next_state_idx1 = ranges_data[range_offset] - ranges_data[range_offset_of_lower_bound_of_next_time_step];
              // K2_CHECK_GE(next_state_idx1, 0);
              arc.dest_state = next_state_idx1 + (t + 1) * U;
              arc.label = 0;
              logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2;
              arc.score = logits_data[logits_offset];
              out_map_data[arc_idx012] = logits_offset;
              // arc.score = 0.0;
            }
            arcs_data[arc_idx012] = arc;
    });
    *arc_map = std::move(out_map);
    return Ragged<Arc>(ofsa_shape, arcs);

}

}  // namespace k2

#endif  // K2_CSRC_PRUNE_RANGE_TO_LATTICE_H_
