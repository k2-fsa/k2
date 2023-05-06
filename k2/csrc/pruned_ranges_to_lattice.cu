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

#include <torch/extension.h>

#include <vector>

#include "k2/python/csrc/torch.h"

namespace k2 {

FsaVec PrunedRangesToLattice(
    torch::Tensor ranges,   // [B][T][s_range]
    torch::Tensor frames,   // [B]
    torch::Tensor symbols,  // [B][S]
    torch::Tensor logits,   // [B][S][s_range][C]
    Array1<int32_t> *arc_map) {
    ContextPtr context;
    if (ranges.device().type() == torch::kCPU) {
      context = GetCpuContext();
    } else if (ranges.is_cuda()) {
      context = GetCudaContext(ranges.device().index());

      TORCH_CHECK(ranges.get_device() == frames.get_device());
      TORCH_CHECK(ranges.get_device() == symbols.get_device());
      TORCH_CHECK(ranges.get_device() == logits.get_device());

    } else {
      K2_LOG(FATAL) << "Unsupported device: " << ranges.device()
                    << "\nOnly CPU and CUDA are verified";
    }

    TORCH_CHECK(ranges.dim() == 3, "ranges should be 3-dimensional");
    TORCH_CHECK(frames.dim() == 1, "frames should be 1-dimensional");
    TORCH_CHECK(symbols.dim() == 2, "symbols should be 2-dimensional");
    TORCH_CHECK(logits.dim() == 4, "logits should be 4-dimensional");

    TORCH_CHECK(torch::kInt == ranges.scalar_type());
    TORCH_CHECK(torch::kInt == frames.scalar_type());
    TORCH_CHECK(torch::kInt == symbols.scalar_type());

    // TODO: Support double and half.
    // currently only float type logits is verified.
    TORCH_CHECK(torch::kFloat == logits.scalar_type());

    // AT_DISPATCH_FLOATING_TYPES(
        // logits.scalar_type(), "pruned_ranges_to_lattice", ([&] {
          auto ranges_a = ranges.accessor<int32_t, 3>();
          auto frames_a = frames.accessor<int32_t, 1>();
          auto symbols_a = symbols.accessor<int32_t, 2>();
          // auto logits_a = logits.accessor<scalar_t, 4>();
          auto logits_a = logits.accessor<float, 4>();
    // }));

    // Typically, s_range is 5.
    const int32_t B = ranges.size(0), T = ranges.size(1), s_range = ranges.size(2);
    const float *logits_data = logits.data_ptr<float>();
    const int32_t *ranges_data = ranges.data_ptr<int32_t>();
    const int32_t *frames_data = frames.data_ptr<int32_t>();
    const int32_t *symbols_data = symbols.data_ptr<int32_t>();
    const int32_t lg_stride_0 = logits.stride(0),
                  lg_stride_1 = logits.stride(1),
                  lg_stride_2 = logits.stride(2),
                  lg_stride_3 = logits.stride(3);
    K2_CHECK_EQ(frames.numel(), B);
    // f2s is short for fsa_to_state.
    Array1<int32_t> f2s_row_splits(context, B + 1);
    int32_t * f2s_row_splits_data = f2s_row_splits.Data();
    K2_EVAL(context, B, lambda_set_f2s_row_splits, (int32_t fsa_idx0) {
        int32_t t = frames_data[fsa_idx0];
        K2_CHECK_LE(t, T);
        // + 1 in "t * U + 1" is for super-final state.
        f2s_row_splits_data[fsa_idx0] = t * s_range + 1;
    });
    ExclusiveSum(f2s_row_splits, &f2s_row_splits);

    RaggedShape f2s_shape =
            RaggedShape2(&f2s_row_splits, nullptr, -1);
    int32_t num_states = f2s_shape.NumElements();
    Array1<int32_t> s2c_row_splits(context, num_states + 1);
    int32_t *s2c_row_splits_data = s2c_row_splits.Data();
    const int32_t *fts_row_splits1_data = f2s_shape.RowSplits(1).Data(),
                  *fts_row_ids1_data = f2s_shape.RowIds(1).Data();

    // set the arcs number for each state
    K2_EVAL(
        context, num_states, lambda_set_num_arcs, (int32_t state_idx01)->void {
          int32_t fsa_idx0 = fts_row_ids1_data[state_idx01],
                  state_idx0x = fts_row_splits1_data[fsa_idx0],
                  state_idx1 = state_idx01 - state_idx0x,
                  t = state_idx1 / s_range,
                  token_index = state_idx1 % s_range;

          K2_CHECK_LE(t, frames_data[fsa_idx0]);
          if (state_idx1 == frames_data[fsa_idx0] * s_range - 1) {
            // frames[fsa_idx0] * s_range is the state_idx1 of super final-state.
            // frames[fsa_idx0] * s_range - 1 is the state pointing to super final-state.
            // final arc to super final state.
            s2c_row_splits_data[state_idx01] = 1;
            return;
          }
          if (state_idx1 == frames_data[fsa_idx0] * s_range) {
            // frames[fsa_idx0] * U is the state_idx1 of super final-state.
            // final state has no leaving arcs.
            s2c_row_splits_data[state_idx01] = 0;
            return;
          }

          bool has_horizontal_blank_arc = false;
          if (t < frames_data[fsa_idx0] - 1) {
            has_horizontal_blank_arc = ranges_a[fsa_idx0][t][token_index] >= ranges_a[fsa_idx0][t + 1][0];
          }
          if (token_index < s_range - 1) {
            // Typically, s_range == 5,
            // the [0, 1, 2, 3] states for each time step may have a vertial arc plus an optional horizontal blank arc.
            s2c_row_splits_data[state_idx01] = 1;
            if (has_horizontal_blank_arc) {
              s2c_row_splits_data[state_idx01] = 2;
            }
          } else {
            // Typically, s_range == 5,
            // the [4] state for each time step have and only have a horizontal blank arc.
            s2c_row_splits_data[state_idx01] = 1;
          }
      });

    ExclusiveSum(s2c_row_splits, &s2c_row_splits);
    RaggedShape s2a_shape =
            RaggedShape2(&s2c_row_splits, nullptr, -1);

    RaggedShape ofsa_shape = ComposeRaggedShapes(f2s_shape, s2a_shape);
    int32_t num_arcs = ofsa_shape.NumElements();
    Array1<Arc> arcs(context, num_arcs);
    Array1<int32_t> out_map(context, num_arcs);
    int32_t* out_map_data = out_map.Data();
    Arc *arcs_data = arcs.Data();
    const int32_t *row_splits1_data = ofsa_shape.RowSplits(1).Data(),
                  *row_ids1_data = ofsa_shape.RowIds(1).Data(),
                  *row_splits2_data = ofsa_shape.RowSplits(2).Data(),
                  *row_ids2_data = ofsa_shape.RowIds(2).Data();

    int32_t symbols_stride_0 = symbols.stride(0),
            symbols_stride_1 = symbols.stride(1);
    K2_EVAL(
          context, num_arcs, lambda_set_arcs, (int32_t arc_idx012)->void {
            int32_t state_idx01 = row_ids2_data[arc_idx012],
                    fsa_idx0 = row_ids1_data[state_idx01],
                    state_idx0x = row_splits1_data[fsa_idx0],
                    arc_idx01x = row_splits2_data[state_idx01],
                    state_idx1 = state_idx01 - state_idx0x,
                    arc_idx2 = arc_idx012 - arc_idx01x,
                    t = state_idx1 / s_range,
                    token_index = state_idx1 % s_range;  // token_index is belong to [0, U)
            Arc arc;
            if (state_idx1 == frames_data[fsa_idx0] * s_range - 1) {
              arc.src_state = state_idx1;
              arc.dest_state = state_idx1 + 1;
              arc.label = -1;
              arc.score = 0.0;
              arcs_data[arc_idx012] = arc;
              out_map_data[arc_idx012] = -1;
              return;
            }
            int32_t next_state_idx1, logits_offset;
            arc.src_state = state_idx1;
            // arc.
            if (token_index < s_range - 1) {
              int32_t actual_u = ranges_a[fsa_idx0][t][token_index];
              int32_t symbols_offset = fsa_idx0 * symbols_stride_0 + actual_u * symbols_stride_1;
              int32_t arc_label =  symbols_data[symbols_offset];
              switch (arc_idx2) {
                case 0:
                  arc.dest_state = state_idx1 + 1;
                  arc.label = arc_label;
                  logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2 + arc_label * lg_stride_3;
                  arc.score = logits_data[logits_offset];
                  out_map_data[arc_idx012] = logits_offset;
                  break;
                case 1:
                  next_state_idx1 = ranges_a[fsa_idx0][t][token_index] - ranges_a[fsa_idx0][t + 1][0];
                  arc.dest_state = next_state_idx1 + (t + 1) * s_range;
                  arc.label = 0;
                  logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2;
                  arc.score = logits_data[logits_offset];
                  out_map_data[arc_idx012] = logits_offset;
                  break;
                default:
                   K2_LOG(FATAL) << "Arc index must be less than 3";
              }
            } else {
              K2_CHECK_EQ(arc_idx2, 0);
              next_state_idx1 = ranges_a[fsa_idx0][t][token_index] - ranges_a[fsa_idx0][t + 1][0];
              arc.dest_state = next_state_idx1 + (t + 1) * s_range;
              arc.label = 0;
              logits_offset = fsa_idx0 * lg_stride_0 + t * lg_stride_1 + token_index * lg_stride_2;
              arc.score = logits_data[logits_offset];
              out_map_data[arc_idx012] = logits_offset;
            }
            arcs_data[arc_idx012] = arc;
    });
    *arc_map = std::move(out_map);
    return Ragged<Arc>(ofsa_shape, arcs);
}

}  // namespace k2
