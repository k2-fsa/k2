/**
 * Copyright      2022  Xiaomi Corporation (authors: Fangjun Kuang)
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

#ifndef K2_TORCH_CSRC_BEAM_SEARCH_H_
#define K2_TORCH_CSRC_BEAM_SEARCH_H_

#include <vector>

#include "torch/all.h"

namespace k2 {

/** RNN-T Greedy search decoding by limiting the max symol per frame to one.
 *
 * @param model The transducer model. See pruned_transducer_stateless2/model.py
 *               for the methods and properties it has.
 *
 * @param encoder_out Output from the encoder network. Its shape is
 *                    (batch_size, T, encoder_out_dim) and its dtype is
 *                    torch::kFloat.
 *
 * @param encoder_out_lens A 1-D tensor containing the valid frames before
 *                         padding in `encoder_out`. Its dtype is torch.kLong
 *                         and its shape is (batch_size,). Also, it must be
 *                         on CPU.
 *
 * @return Return A list-of-list of token IDs containing the decoding results.
 * The returned vector has size `batch_size` and each entry contains the
 * decoding results for the corresponding input in encoder_out.
 */
std::vector<std::vector<int32_t>> GreedySearch(
    const torch::jit::Module &model, const torch::Tensor &encoder_out,
    const torch::Tensor &encoder_out_lens);

std::vector<std::vector<int32_t>> ModifiedBeamSearch(
    const torch::jit::Module &model, const torch::Tensor &encoder_out,
    const torch::Tensor &encoder_out_lens, int32_t num_acitve_paths = 4);

}  // namespace k2

#endif  // K2_TORCH_CSRC_BEAM_SEARCH_H_
