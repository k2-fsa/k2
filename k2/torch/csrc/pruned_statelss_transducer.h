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

#ifndef K2_TORCH_CSRC_PRUNED_STATELESS_TRANSDUCER_H_
#define K2_TORCH_CSRC_PRUNED_STATELESS_TRANSDUCER_H_

#include "k2/torch/csrc/parse_options.h"
#include "torch/all.h"

namespace k2 {

struct ModifiedBeamSearchOptions {
  void Regsiter(ParseOptions *po);
};

struct PrunedStatelessTransducerOptions {
  std::string model_filename;
  ModifiedBeamSearchOptions modified_beam_search_opts;

  // TODO(fangjun): Add more options
  void Regsiter(ParseOptions *po) {
    po->Register("model-filename", &model_filename,
                 "Path to the torch jit model filename");

    modified_beam_search_opts.Register(po);
  }
};

class PrunedStatelessTransducer {
 public:
  explicit PrunedStatelessTransducer(
      const PrunedStatelessTransducerOptions &opts);

  /**
   * Run the encoder network (for offline use).
   *
   * @param features  A 3-D tensor of shape (N, T, C) containing the features.
   * @param feature_lens  A 1-D tensor of shape (N,) containing number of valid
   *                      frames before padding in features.
   *
   * @return Return a pair containing two tensors
   *  - output from the encoder network; its shape is (N, T, C)
   *  - output length of the encoder network; its shape is (N,)
   */
  std::pair<torch::Tensor, torch::Tensor> ForwardEncoder(
      const torch::Tensor &features, const torch::Tensor &feature_lens);

  /** Run the encoder network (for streaming decoding)
   *
   * @params features A 3-D tensor containing the features.
   * @params prev_states  The previous state of the encoder model
   *
   * @return Return a pair containing
   *  - encoder output
   *  - next state
   */
  std::pair<torch::Tensor, torch::IValue> StreamingForwardEncoder(
      const torch::Tensor &features,
      torch::optional<torch::Ivalue> prev_states = torch::nullopt);

  // For offline use
  std::vector<std::vector<int32_t>> GreedySearch(
      const torch::Tensor &encoder_out, const torch::Tensor &encoder_out_lens);

  // for streaming decoding
  std::vector<std::vector<int32_t>> StreamingGreedySearch(
      const torch::Tensor &encoder_out, const torch::Tensor &encoder_out_lens,
      const std::vector<std::vector<int32_t>> &prev_partial_results);

  // add similar interface for modified_beam_search and fast_beam_search

 private:
  torch::jit::Module model_;
};

}  // namespace k2

#endif  // // K2_TORCH_CSRC_PRUNED_STATELESS_TRANSDUCER_H_
