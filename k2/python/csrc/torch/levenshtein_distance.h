/**
 * @copyright
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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

#ifndef K2_PYTHON_CSRC_TORCH_LEVENSHTEIN_DISTANCE_H_
#define K2_PYTHON_CSRC_TORCH_LEVENSHTEIN_DISTANCE_H_

#include <torch/extension.h>

#include <vector>

#include "k2/python/csrc/torch.h"

namespace k2 {

/*
  Compute the levenshtein distance between sequences in batches.

    @param px  A two-dimensional tensor with the shape of ``[B][S]`` containing
               sequences. It's data type MUST be ``torch.int32``.
    @param py  A two-dimensional tensor with the shape of ``[B][U]`` containing
               sequences. It's data type MUST be ``torch.int32``.
               ``py`` and ``px`` should have the same batch size.
    @param boundary  If supplied, a torch.LongTensor of shape ``[B][4]``, where
                     each row contains ``[s_begin, u_begin, s_end, u_end]``,
                     with ``0 <= s_begin <= s_end <= S`` and
                     ``0 <= u_begin <= u_end <= U``
                     (this implies that empty sequences are allowed).
                     If not supplied, the values ``[0, 0, S, U]`` will be
                     assumed. These are the beginning and one-past-the-last
                     positions in the ``px`` and ``py`` sequences respectively,
                     and can be used if not all sequences are of the same
                     length.
    @return  A tensor with shape ``[B][S + 1][U + 1]`` containing the
             levenshtein distance between the sequences. Each element
             ``[b][s][u]`` means the levenshtein distance between ``px[b][:s]``
             and ``py[b][:u]``.  If `boundary` is set, the values in the
             positions out of the range of boundary are uninitialized, can be
             any random values.
*/
torch::Tensor LevenshteinDistanceCpu(
    torch::Tensor px,                          // [B][S]
    torch::Tensor py,                          // [B][U]
    torch::optional<torch::Tensor> boundary);  // [B][4], int64_t.

/*
  The same as above function, but it runs on GPU.
 */
torch::Tensor LevenshteinDistanceCuda(
    torch::Tensor px,                          // [B][S]
    torch::Tensor py,                          // [B][U]
    torch::optional<torch::Tensor> boundary);  // [B][4], int64_t.

}  // namespace k2

void PybindLevenshteinDistance(py::module &m);

#endif  // K2_PYTHON_CSRC_TORCH_LEVENSHTEIN_DISTANCE_H_
