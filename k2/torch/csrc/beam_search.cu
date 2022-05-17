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

#include "k2/csrc/log.h"
#include "k2/torch/csrc/beam_search.h"
#include "torch/all.h"

namespace k2 {

/**
 * Construct the decoder input from the current hypothesis.
 *
 * @param hyps  A list-of-list of token IDs containing the current decoding
 *              results. Its length is `batch_size`
 * @param decoder_input A 2-D tensor of shape (batch_size, context_size).
 */
static void BuildDecoderInput(const std::vector<std::vector<int32_t>> &hyps,
                              torch::Tensor *decoder_input) {
  int32_t batch_size = decoder_input->size(0);
  int32_t context_size = decoder_input->size(1);
  int64_t *p = decoder_input->data_ptr<int64_t>();
  for (int32_t i = 0; i != batch_size; ++i) {
    auto start = hyps[i].end() - context_size;
    auto end = hyps[i].end();
    std::copy(start, end, p);
    p += context_size;
  }
}

std::vector<std::vector<int32_t>> GreedySearch(
    const torch::jit::Module &model, const torch::Tensor &encoder_out,
    const torch::Tensor &encoder_out_lens) {
  K2_CHECK_EQ(encoder_out.dim(), 3);
  K2_CHECK_EQ(encoder_out.scalar_type(), torch::kFloat);

  K2_CHECK_EQ(encoder_out_lens.dim(), 1);
  K2_CHECK_EQ(encoder_out_lens.scalar_type(), torch::kLong);
  K2_CHECK(encoder_out_lens.is_cpu());

  torch::nn::utils::rnn::PackedSequence packed_seq =
      torch::nn::utils::rnn::pack_padded_sequence(encoder_out, encoder_out_lens,
                                                  /*batch_first*/ true,
                                                  /*enforce_sorted*/ false);
  torch::jit::Module decoder = model.attr("decoder").toModule();
  torch::jit::Module joiner = model.attr("joiner").toModule();
  torch::jit::Module decoder_proj = joiner.attr("decoder_proj").toModule();

  auto projected_encoder_out = joiner.attr("encoder_proj")
                                   .toModule()
                                   .run_method("forward", packed_seq.data())
                                   .toTensor();

  int32_t blank_id = decoder.attr("blank_id").toInt();

  int32_t unk_id = blank_id;
  if (decoder.hasattr("unk_id")) {
    unk_id = decoder.attr("unk_id").toInt();
  }

  int32_t context_size = decoder.attr("context_size").toInt();
  int32_t batch_size = encoder_out_lens.size(0);

  torch::Device device = encoder_out.device();

  std::vector<int32_t> blanks(context_size, blank_id);
  std::vector<std::vector<int32_t>> hyps(batch_size, blanks);

  auto decoder_input =
      torch::full({batch_size, context_size}, blank_id,
                  torch::dtype(torch::kLong)
                      .memory_format(torch::MemoryFormat::Contiguous));
  auto decoder_out =
      decoder
          .run_method("forward", decoder_input.to(device), /*need_pad*/ false)
          .toTensor();
  decoder_out = decoder_proj.run_method("forward", decoder_out).toTensor();
  // decoder_out's shape is (batch_size, 1, joiner_dim)

  using torch::indexing::Slice;
  auto batch_sizes_accessor = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t num_batches = packed_seq.batch_sizes().numel();
  int32_t offset = 0;
  for (int32_t i = 0; i != num_batches; ++i) {
    int32_t cur_batch_size = batch_sizes_accessor[i];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = projected_encoder_out.index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (cur_batch_size, 1, 1, joiner_dim)
    if (cur_batch_size < decoder_out.size(0)) {
      decoder_out = decoder_out.index({Slice(0, cur_batch_size)});
    }

    auto logits =
        joiner
            .run_method("forward", cur_encoder_out, decoder_out.unsqueeze(1),
                        /*project_input*/ false)
            .toTensor();
    // logits' shape is (cur_batch_size, 1, 1, vocab_size)
    logits = logits.squeeze(1).squeeze(1);
    auto max_indices = logits.argmax(/*dim*/ -1).cpu();
    auto max_indices_accessor = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t k = 0; k != cur_batch_size; ++k) {
      auto index = max_indices_accessor[k];
      if (index != blank_id && index != unk_id) {
        emitted = true;
        hyps[k].push_back(index);
      }
    }

    if (emitted) {
      if (cur_batch_size < decoder_input.size(0)) {
        decoder_input = decoder_input.index({Slice(0, cur_batch_size)});
      }
      BuildDecoderInput(hyps, &decoder_input);
      decoder_out = decoder
                        .run_method("forward", decoder_input.to(device),
                                    /*need_pad*/ false)
                        .toTensor();
      decoder_out = decoder_proj.run_method("forward", decoder_out).toTensor();
    }
  }

  auto unsorted_indices_accessor =
      packed_seq.unsorted_indices().accessor<int64_t, 1>();

  std::vector<std::vector<int32_t>> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    torch::ArrayRef<int32_t> arr(hyps[unsorted_indices_accessor[i]]);
    ans[i] = arr.slice(context_size).vec();
  }

  return ans;
}

}  // namespace k2

#endif  // K2_TORCH_CSRC_BEAM_SEARCH_H_
