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

#ifndef K2_TORCH_CSRC_DENSE_FSA_VEC_H_
#define K2_TORCH_CSRC_DENSE_FSA_VEC_H_

#include "k2/csrc/fsa.h"
#include "torch/script.h"

namespace k2 {

/** Construct a DenseFsaVec from neural net log-softmax outputs.

  @params log_probs  A 3-D tensor of dtype torch.float32. It has shape
                     (N, T, C), where `N` is the number of utterances,
                     `T` the maximum input length, and `C` the number of
                     output classes. This is usually the output of the
                     log-softmax layer of a neural network.

  @param supervision_segments  A 2-D tensor of dtype torch.int32 with 3 columns.
            It has be to on CPU.
            Each row contains information for a supervision segment. Column 0
            is the `utterance_index` indicating which utterance this segment
            comes from; column 1 specifies the `start_frame` of this segment
            within the utterance; column 2 contains the `duration` of this
            segment (in number of frames).

            Note:
              - `0 < start_frame + duration <= T + allow_truncate`
              - `0 <= start_frame < T`
              - `duration > 0`

            Caution:
              If the resulting dense fsa vec is used as an input to
              `k2::IntersectDense`, then the last column, i.e., the duration
              column, has to be sorted in **decreasing** order.
              That is, the first supervision_segment (the first row) has the
              largest duration.
              Otherwise, you don't need to sort the last column.

              `k2::IntersectDense` is often used in the training stage, so
              you should usually sort dense fsa vecs by its duration
              in training. `k2::IntersectDensePruned` is usually used in the
              decoding stage, so you don't need to sort dense fsa vecs in
              decoding.

  @param allow_truncate  If not zero, it truncates at most this number of frames
                         from `duration` in case `start_frame + duration > T`.

  @param Return a DenseFsaVec.
 */
DenseFsaVec CreateDenseFsaVec(torch::Tensor log_probs,
                              torch::Tensor supervision_segments,
                              int32_t allow_truncate = 0);

// See
// https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32
// for the format of "supervisions"
//
// @param supervisions A dict containing keys and values shown in the following:
//                     - sequence_idx: torch.Tensor
//                     - start_frame: torch.Tensor
//                     - num_frames: torch.Tensor
// @return Return a 2-D torch.int32 tensor that can be used to construct a
//  DenseFsaVec. See `k2::CreateDenseFsaVec()`
torch::Tensor GetSupervisionSegments(torch::IValue supervisions,
                                     int32_t subsampling_factor);

}  // namespace k2

#endif  // K2_TORCH_CSRC_DENSE_FSA_VEC_H_
