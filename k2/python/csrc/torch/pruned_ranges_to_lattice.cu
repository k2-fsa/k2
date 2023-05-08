/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Liyong Guo)
 *
 * @copyright
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

#include <utility>

#include "k2/csrc/device_guard.h"
#include "k2/csrc/fsa.h"
#include "k2/csrc/torch_util.h"
#include "k2/python/csrc/torch/pruned_ranges_to_lattice.h"
#include "k2/python/csrc/torch/v2/ragged_any.h"


namespace k2 {

FsaVec PrunedRangesToLattice(
    torch::Tensor ranges,   // [B][T][s_range]
    torch::Tensor frames,   // [B]
    torch::Tensor symbols,  // [B][S]
    torch::Tensor logits,   // [B][S][s_range][C]
    Array1<int32_t> *arc_map) {

    TORCH_CHECK(ranges.get_device() == frames.get_device());
    TORCH_CHECK(ranges.get_device() == symbols.get_device());
    TORCH_CHECK(ranges.get_device() == logits.get_device());

    TORCH_CHECK(ranges.dim() == 3, "ranges should be 3-dimensional");
    TORCH_CHECK(frames.dim() == 1, "frames should be 1-dimensional");
    TORCH_CHECK(symbols.dim() == 2, "symbols should be 2-dimensional");
    TORCH_CHECK(logits.dim() == 4, "logits should be 4-dimensional");

    TORCH_CHECK(torch::kInt == ranges.scalar_type());
    TORCH_CHECK(torch::kInt == frames.scalar_type());
    TORCH_CHECK(torch::kLong == symbols.scalar_type());

    ContextPtr context;
    if (ranges.device().type() == torch::kCPU) {
      context = GetCpuContext();
    } else if (ranges.is_cuda()) {
      context = GetCudaContext(ranges.device().index());
    } else {
      K2_LOG(FATAL) << "Unsupported device: " << ranges.device()
                    << "\nOnly CPU and CUDA are verified";
    }

    // "_a" is short for accessor.
    auto ranges_a = ranges.accessor<int32_t, 3>();
    auto frames_a = frames.accessor<int32_t, 1>();
    auto symbols_a = symbols.accessor<int64_t, 2>();

    // Typically, s_range is 5.
    const int32_t B = ranges.size(0),
                  T = ranges.size(1),
                  s_range = ranges.size(2);

    // Compute f2s_shape: fsa_to_state_shape.
    Array1<int32_t> f2s_row_splits(context, B + 1);
    int32_t * f2s_row_splits_data = f2s_row_splits.Data();
    K2_EVAL(context, B, lambda_set_f2s_row_splits, (int32_t fsa_idx0) {
        int32_t t = frames_a[fsa_idx0];
        K2_CHECK_LE(t, T);
        // + 1 in "t * s_range + 1" is for super-final state.
        f2s_row_splits_data[fsa_idx0] = t * s_range + 1;
    });

    ExclusiveSum(f2s_row_splits, &f2s_row_splits);
    RaggedShape f2s_shape =
            RaggedShape2(&f2s_row_splits, nullptr, -1);

    // Compute s2a_shape: state_to_arc_shape.
    int32_t num_states = f2s_shape.NumElements();
    Array1<int32_t> s2c_row_splits(context, num_states + 1);
    int32_t *s2c_row_splits_data = s2c_row_splits.Data();
    const int32_t *f2s_row_splits1_data = f2s_shape.RowSplits(1).Data(),
                  *f2s_row_ids1_data = f2s_shape.RowIds(1).Data();
    // Compute number of arcs for each state.
    K2_EVAL(
        context, num_states, lambda_set_num_arcs, (int32_t state_idx01)->void {
          int32_t fsa_idx0 = f2s_row_ids1_data[state_idx01],
                  state_idx0x = f2s_row_splits1_data[fsa_idx0],
                  state_idx1 = state_idx01 - state_idx0x,
                  t = state_idx1 / s_range,
                  token_idx = state_idx1 % s_range;

          K2_CHECK_LE(t, frames_a[fsa_idx0]);

          // The state doesn't have leaving arc: super final_state.
          if (state_idx1 == frames_a[fsa_idx0] * s_range) {
            s2c_row_splits_data[state_idx01] = 0;
            return;
          }

          // States have a leaving arc if no specially processed.
          s2c_row_splits_data[state_idx01] = 1;

          // States have two leaving arcs.
          bool has_horizontal_blank_arc = false;
          if (t < frames_a[fsa_idx0] - 1) {
            has_horizontal_blank_arc =
              ranges_a[fsa_idx0][t][token_idx] >= ranges_a[fsa_idx0][t + 1][0];
          }
          // Typically, s_range == 5, i.e. 5 states for each time step.
          // the 4th state(0-based index) ONLY has a horizontal blank arc.
          // While state [0, 1, 2, 3] have a vertial arc
          // and MAYBE a horizontal blank arc.
          if (token_idx != s_range - 1 && has_horizontal_blank_arc) {
              s2c_row_splits_data[state_idx01] = 2;
          }
    });
    ExclusiveSum(s2c_row_splits, &s2c_row_splits);
    RaggedShape s2a_shape =
            RaggedShape2(&s2c_row_splits, nullptr, -1);

    // ofsa_shape: output_fsa_shape.
    RaggedShape ofsa_shape = ComposeRaggedShapes(f2s_shape, s2a_shape);

    int32_t num_arcs = ofsa_shape.NumElements();
    Array1<Arc> arcs(context, num_arcs);

    Arc *arcs_data = arcs.Data();
    const int32_t *row_splits1_data = ofsa_shape.RowSplits(1).Data(),
                  *row_ids1_data = ofsa_shape.RowIds(1).Data(),
                  *row_splits2_data = ofsa_shape.RowSplits(2).Data(),
                  *row_ids2_data = ofsa_shape.RowIds(2).Data();

    Array1<int32_t> out_map(context, num_arcs);
    int32_t* out_map_data = out_map.Data();
    // Used to populate out_map.
    const int32_t lg_stride_0 = logits.stride(0),
                  lg_stride_1 = logits.stride(1),
                  lg_stride_2 = logits.stride(2),
                  lg_stride_3 = logits.stride(3);

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(
        logits.scalar_type(), "pruned_ranges_to_lattice", ([&] {
          auto logits_a = logits.accessor<scalar_t, 4>();

          K2_EVAL(
              context, num_arcs, lambda_set_arcs, (int32_t arc_idx012)->void {
                const int32_t state_idx01 = row_ids2_data[arc_idx012],
                              fsa_idx0 = row_ids1_data[state_idx01],
                              state_idx0x = row_splits1_data[fsa_idx0],
                              arc_idx01x = row_splits2_data[state_idx01],
                              state_idx1 = state_idx01 - state_idx0x,
                              arc_idx2 = arc_idx012 - arc_idx01x,
                              t = state_idx1 / s_range,
                              // token_idx lies within interval [0, s_range)
                              // but does not include s_range.
                              token_idx = state_idx1 % s_range;
                Arc arc;
                arc.src_state = state_idx1;
                // The penultimate state only has a leaving arc
                // pointing to super final state.
                if (state_idx1 == frames_a[fsa_idx0] * s_range - 1) {
                  arc.src_state = state_idx1;
                  arc.dest_state = state_idx1 + 1;
                  arc.label = -1;
                  arc.score = 0.0;
                  arcs_data[arc_idx012] = arc;
                  out_map_data[arc_idx012] = -1;
                  return;
                }

                if (token_idx < s_range - 1) {
                  // States have a vertal arc with non-blank label and
                  // MAYBE a horizontal arc with blank label.
                  const int32_t symbol_idx = ranges_a[fsa_idx0][t][token_idx],
                                arc_label =  symbols_a[fsa_idx0][symbol_idx];
                  K2_CHECK_LE(arc_idx2, 2);
                  switch (arc_idx2) {
                    // For vertial arc with non-blank label.
                    case 0:
                      arc.dest_state = state_idx1 + 1;
                      arc.label = arc_label;
                      arc.score = logits_a[fsa_idx0][t][token_idx][arc_label];

                      out_map_data[arc_idx012] =
                        fsa_idx0 * lg_stride_0 + t * lg_stride_1 +
                        token_idx * lg_stride_2 + arc_label * lg_stride_3;
                      break;
                    // For horizontal arc with blank label.
                    case 1:
                      const int32_t dest_state_token_idx =
                        ranges_a[fsa_idx0][t][token_idx] -
                        ranges_a[fsa_idx0][t + 1][0];
                      K2_CHECK_GE(dest_state_token_idx, 0);
                      arc.dest_state = dest_state_token_idx + (t + 1) * s_range;
                      arc.label = 0;
                      arc.score = logits_a[fsa_idx0][t][token_idx][0];

                      out_map_data[arc_idx012] =
                        fsa_idx0 * lg_stride_0 + t * lg_stride_1 +
                        token_idx * lg_stride_2;
                      break;
                  }
                } else {
                  // States only have a horizontal arc with blank label.
                  K2_CHECK_EQ(arc_idx2, 0);
                  const int32_t dest_state_token_idx =
                    ranges_a[fsa_idx0][t][token_idx] -
                    ranges_a[fsa_idx0][t + 1][0];
                  arc.dest_state =
                    dest_state_token_idx + (t + 1) * s_range;
                  arc.label = 0;
                  arc.score = logits_a[fsa_idx0][t][token_idx][0];

                  out_map_data[arc_idx012] =
                    fsa_idx0 * lg_stride_0 + t * lg_stride_1 +
                    token_idx * lg_stride_2;
                }
                arcs_data[arc_idx012] = arc;
          });
          *arc_map = std::move(out_map);
    }));

    return Ragged<Arc>(ofsa_shape, arcs);
}

}  // namespace k2

void PybindPrunedRangesToLattice(py::module &m) {
  m.def(
      "pruned_ranges_to_lattice",
      [](torch::Tensor ranges,
         torch::Tensor frames,
         torch::Tensor symbols,
         torch::Tensor logits) -> std::pair<k2::FsaVec, torch::Tensor> {
        k2::DeviceGuard guard(k2::GetContext(ranges));
        k2::Array1<int32_t> arc_to_logit_map;
        k2::FsaVec ofsa = k2::PrunedRangesToLattice(
            ranges, frames, symbols, logits, &arc_to_logit_map);
        return std::make_pair(ofsa, ToTorch(arc_to_logit_map));
      },
      py::arg("ranges"), py::arg("frames"),
      py::arg("symbols"), py::arg("logits"));
}
