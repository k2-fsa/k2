/**
 * Copyright      2021  Xiaomi Corporation (authors: Fangjun Kuang)
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

#include <algorithm>
#include <limits>
#include <vector>

#include "k2/csrc/log.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/utils.h"
#include "torch/script.h"

namespace k2 {

DenseFsaVec CreateDenseFsaVec(torch::Tensor log_probs,
                              torch::Tensor supervision_segments,
                              int32_t allow_truncate /*=0*/) {
  K2_CHECK_EQ(log_probs.dtype(), torch::kFloat32);
  K2_CHECK_EQ(log_probs.dim(), 3);

  K2_CHECK_EQ(supervision_segments.dtype(), torch::kInt);
  K2_CHECK_EQ(supervision_segments.dim(), 2);
  K2_CHECK_EQ(supervision_segments.size(1), 3);
  K2_CHECK_EQ(supervision_segments.device().type(), torch::kCPU);
  K2_CHECK_GE(allow_truncate, 0);

  int32_t N = log_probs.size(0);
  int32_t T = log_probs.size(1);
  int32_t C = log_probs.size(2);

  // iterate the supervision_segments to get number of frames for each segment
  int32_t num_utt = supervision_segments.size(0);
  int32_t stride = supervision_segments.stride(0);
  const int32_t *p_sup = supervision_segments.data_ptr<int32_t>();

  // linear_indexes contains indexes along axis 0
  // for the tensor obtained from log_probs.view(-1, C)
  std::vector<int64_t> linear_indexes;
  linear_indexes.reserve(num_utt * (T + 1));  // the worse case

  // It contains the index of the extra frame of each utterance
  // that will be set to [0, -inf, -inf, -inf, ... ]
  std::vector<int64_t> extra_frame_indexes;
  extra_frame_indexes.reserve(num_utt);

  int32_t duration_in_total = 0;

  for (int32_t i = 0; i != num_utt; ++i) {
    const int32_t *this_row = p_sup + i * stride;
    int32_t utt_index = this_row[0];
    int32_t start_frame = this_row[1];
    int32_t duration = this_row[2];

    K2_CHECK_GE(utt_index, 0);
    K2_CHECK_LT(utt_index, N);

    K2_CHECK_GE(start_frame, 0);
    K2_CHECK_LT(start_frame, T);

    K2_CHECK_GE(duration, 0);
    K2_CHECK_LE(start_frame + duration, T + allow_truncate);

    int32_t end_frame = std::min(start_frame + duration, T);  // exclusive
    duration = end_frame - start_frame;

    int32_t offset = utt_index * T;
    std::vector<int32_t> this_utt_frames(duration);
    std::iota(this_utt_frames.begin(), this_utt_frames.end(),
              start_frame + offset);
    linear_indexes.insert(linear_indexes.end(), this_utt_frames.begin(),
                          this_utt_frames.end());

    // a placeholder for the extra frame that will be set to
    // [ 0, -inf, -inf, ...]
    linear_indexes.push_back(0);

    duration_in_total += duration;
    extra_frame_indexes.push_back(duration_in_total);
    duration_in_total += 1;  // plus one for the extra frame
  }

  torch::Tensor indexes =
      torch::from_blob(
          linear_indexes.data(), /*sizes*/ {int64_t(linear_indexes.size())},
          /*options*/ torch::device(torch::kCPU).dtype(torch::kLong))
          .to(log_probs.device());

  torch::Tensor extra_frame_indexes_tensor =
      torch::from_blob(
          extra_frame_indexes.data(),
          /*sizes*/ {int64_t(extra_frame_indexes.size())},
          /*options*/ torch::device(torch::kCPU).dtype(torch::kLong))
          .to(log_probs.device());

  torch::Tensor scores =
      torch::empty({duration_in_total, C + 1}, log_probs.options());

  using namespace torch::indexing;  // NOLINT
  // scores[:, 1:] = log_probs.reshape(-1, C).index_select(0, indexes)
  scores.index({"...", Slice(1, None, None)}) =
      log_probs.reshape({-1, C}).index_select(0, indexes);

  // now set extra frames to [0, -inf, -inf. -inf, ... ]
  //
  // `scores` contains -infinity in certain locations: in scores[j,0] where
  // j is not the last row-index for a given FSA-index, and scores[j,k]
  // where j is the last row-index for a given FSA-index and k>0.
  // The remaining locations contain the neural net output, except
  // scores[j,0] where j is the last row-index for a given FSA-index;
  // this contains zero.
  //
  // scores[:, 0] = float('-inf');
  // scores[last_frame_indexes] = torch.tensor([0] + [float('-inf')] * C,
  //                                           device=device);
  constexpr float kNegInf = -1.0f * std::numeric_limits<float>::infinity();
  scores.index({"...", 0}) = kNegInf;
  std::vector<float> tmp(C + 1);
  tmp[0] = 0;
  std::fill_n(tmp.begin() + 1, C, kNegInf);
  torch::Tensor extra_frame =
      torch::from_blob(
          tmp.data(), /*sizes*/ {int64_t(tmp.size())},
          /*options*/ torch::device(torch::kCPU).dtype(torch::kFloat32))
          .to(log_probs.device());

  scores.index_put_({extra_frame_indexes_tensor}, extra_frame);

  // Now compute row splits so that we can create a ragged shape
  std::vector<int32_t> row_splits(num_utt + 1);
  row_splits[0] = 0;
  std::transform(extra_frame_indexes.begin(), extra_frame_indexes.end(),
                 &row_splits[1], [](int64_t i) { return i + 1; });

  ContextPtr ctx = ContextFromTensor(log_probs);
  Array1<int32_t> row_splits_array(ctx, row_splits);
  Array2<float> scores_array = Array2FromTorch<float>(scores);
  RaggedShape shape =
      RaggedShape2(&row_splits_array, nullptr, row_splits.back());

  return {shape, scores_array};
}

torch::Tensor GetSupervisionSegments(torch::IValue supervisions,
                                     int32_t subsampling_factor) {
  torch::Dict<torch::IValue, torch::IValue> dict = supervisions.toGenericDict();
  torch::Tensor sequence_idx = dict.at("sequence_idx").toTensor();
  torch::Tensor start_frame = torch::floor_divide(
      dict.at("start_frame").toTensor(), subsampling_factor);

  torch::Tensor num_frames =
      torch::floor_divide(dict.at("num_frames").toTensor(), subsampling_factor);

  torch::Tensor supervision_segments =
      torch::stack({sequence_idx, start_frame, num_frames}, 1).to(torch::kCPU);
  return supervision_segments;
}

}  // namespace k2
