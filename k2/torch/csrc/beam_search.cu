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

#include <algorithm>
#include <deque>
#include <utility>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/log.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/beam_search.h"
#include "k2/torch/csrc/hypothesis.h"
#include "torch/all.h"

namespace k2 {

static inline torch::Tensor FloorDivide(torch::Tensor a, int32_t b) {
#if K2_TORCH_VERSION_MAJOR > 1 || \
    (K2_TORCH_VERSION_MAJOR == 1 && K2_TORCH_VERSION_MINOR > 7)
  return torch::div(a, b, /*rounding_mode*/ "trunc");
#else
  return torch::floor_divide(a, b);
#endif
}

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

static torch::Tensor BuildDecoderInput(const std::vector<Hypothesis> hyps,
                                       int32_t context_size) {
  int32_t num_hyps = hyps.size();
  torch::Tensor decoder_input =
      torch::empty({num_hyps, context_size},
                   torch::dtype(torch::kLong)
                       .memory_format(torch::MemoryFormat::Contiguous));

  int64_t *p = decoder_input.data_ptr<int64_t>();
  for (const auto &h : hyps) {
    auto start = h.ys.end() - context_size;
    auto end = h.ys.end();

    std::copy(start, end, p);
    p += context_size;
  }

  return decoder_input;
}

/** Return a ragged shape with axes [utt][num_hyps].
 *
 * @param hyps hyps.size() == batch_size. Each entry contains the active
 *              hypotheses of an utterance.
 * @return Return a ragged shape with 2 axes [utt][num_hyps]. Note that the
 *         shape is on CPU.
 */
static RaggedShape GetHypsShape(const std::vector<Hypotheses> &hyps) {
  int32_t num_utt = hyps.size();
  Array1<int32_t> row_splits(GetCpuContext(), num_utt + 1);
  int32_t *row_splits_data = row_splits.Data();

  for (int32_t i = 0; i != num_utt; ++i) {
    row_splits_data[i] = hyps[i].Size();
  }

  ExclusiveSum(row_splits, &row_splits);

  return RaggedShape2(&row_splits, nullptr, row_splits.Back());
}

std::vector<std::vector<int32_t>> GreedySearch(
    const torch::jit::Module &model, const torch::Tensor &encoder_out,
    const torch::Tensor &encoder_out_lens) {
  K2_CHECK_EQ(encoder_out.dim(), 3);
  K2_CHECK_EQ(encoder_out.scalar_type(), torch::kFloat);

  K2_CHECK_EQ(encoder_out_lens.dim(), 1);
  K2_CHECK_EQ(encoder_out_lens.scalar_type(), torch::kLong);
  K2_CHECK(encoder_out_lens.device().is_cpu());

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
  auto batch_sizes_acc = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t num_batches = packed_seq.batch_sizes().numel();
  int32_t offset = 0;
  for (int32_t i = 0; i != num_batches; ++i) {
    int32_t cur_batch_size = batch_sizes_acc[i];
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
    auto max_indices_acc = max_indices.accessor<int64_t, 1>();
    bool emitted = false;
    for (int32_t k = 0; k != cur_batch_size; ++k) {
      auto index = max_indices_acc[k];
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

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<std::vector<int32_t>> ans(batch_size);

  for (int32_t i = 0; i != batch_size; ++i) {
    torch::ArrayRef<int32_t> arr(hyps[unsorted_indices_accessor[i]]);
    ans[i] = arr.slice(context_size).vec();
  }

  return ans;
}

std::vector<std::vector<int32_t>> ModifiedBeamSearch(
    const torch::jit::Module &model, const torch::Tensor &encoder_out,
    const torch::Tensor &encoder_out_lens, int32_t num_acitve_paths /*=4*/) {
  K2_CHECK_EQ(encoder_out.dim(), 3);
  K2_CHECK_EQ(encoder_out.scalar_type(), torch::kFloat);

  K2_CHECK_EQ(encoder_out_lens.dim(), 1);
  K2_CHECK_EQ(encoder_out_lens.scalar_type(), torch::kLong);
  K2_CHECK(encoder_out_lens.device().is_cpu());

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
  Hypotheses blank_hyp({{blanks, 0}});

  std::deque<Hypotheses> finalized;
  std::vector<Hypotheses> cur(batch_size, blank_hyp);
  std::vector<Hypothesis> prev;

  using torch::indexing::Slice;
  auto batch_sizes_acc = packed_seq.batch_sizes().accessor<int64_t, 1>();
  int32_t num_batches = packed_seq.batch_sizes().numel();
  int32_t offset = 0;
  for (int32_t i = 0; i != num_batches; ++i) {
    int32_t cur_batch_size = batch_sizes_acc[i];
    int32_t start = offset;
    int32_t end = start + cur_batch_size;
    auto cur_encoder_out = projected_encoder_out.index({Slice(start, end)});
    offset = end;

    cur_encoder_out = cur_encoder_out.unsqueeze(1).unsqueeze(1);
    // Now cur_encoder_out's shape is (cur_batch_size, 1, 1, joiner_dim)

    if (cur_batch_size < cur.size()) {
      for (int32_t k = static_cast<int32_t>(cur.size()) - 1;
           k >= cur_batch_size; --k) {
        finalized.push_front(std::move(cur[k]));
      }
      cur.erase(cur.begin() + cur_batch_size, cur.end());
    }

    // Due to merging paths with identical token sequences,
    // not all utterances have "num_acitve_paths" paths.
    auto hyps_shape = GetHypsShape(cur);

    prev.clear();
    prev.reserve(hyps_shape.TotSize(1));
    for (auto &hyps : cur) {
      for (auto &h : hyps) {
        prev.push_back(std::move(h.second));
      }
    }
    cur.clear();
    cur.reserve(cur_batch_size);

    torch::Tensor ys_log_probs =
        torch::empty({hyps_shape.TotSize(1), 1}, torch::dtype(torch::kFloat));

    auto ys_log_probs_acc = ys_log_probs.accessor<float, 2>();
    for (int32_t k = 0; k != prev.size(); ++k) {
      ys_log_probs_acc[k][0] = prev[k].log_prob;
    }

    auto decoder_input = BuildDecoderInput(prev, context_size).to(device);

    auto decoder_out =
        decoder.run_method("forward", decoder_input, /*need_pad*/ false)
            .toTensor();

    decoder_out = decoder_proj.run_method("forward", decoder_out).toTensor();
    // decoder_out is of shape (num_hyps, 1, joiner_dim)

    auto row_ids = hyps_shape.RowIds(1);

    auto index =
        torch::from_blob(row_ids.Data(), {row_ids.Dim()}, torch::kInt32)
            .to(torch::device(device).dtype(torch::kLong));

    cur_encoder_out = cur_encoder_out.index_select(/*dim*/ 0, /*index*/ index);
    // cur_encoder_out is of shape (num_hyps, 1, 1, joiner_dim)

    auto logits =
        joiner
            .run_method("forward", cur_encoder_out, decoder_out.unsqueeze(1),
                        /*project_input*/ false)
            .toTensor();
    // logits' shape is (cur_batch_size, 1, 1, vocab_size)
    logits = logits.squeeze(1).squeeze(1);
    // now logits' shape is (cur_batch_size, vocab_size)

    auto log_probs = logits.log_softmax(-1).cpu();

    log_probs.add_(ys_log_probs);

    int32_t vocab_size = log_probs.size(1);
    log_probs = log_probs.reshape(-1);
    auto row_splits = hyps_shape.RowSplits(1);
    const int32_t *row_splits_data = row_splits.Data();

    for (int32_t k = 0; k != cur_batch_size; ++k) {
      int32_t start = row_splits_data[k];
      int32_t end = row_splits_data[k + 1];

      torch::Tensor values, indexes;
      std::tie(values, indexes) =
          log_probs.slice(/*dim*/ 0, start * vocab_size, end * vocab_size)
              .topk(/*k*/ num_acitve_paths, /*dim*/ 0,
                    /*largest*/ true, /*sorted*/ true);

      auto topk_hyp_indexes = FloorDivide(indexes, vocab_size);
      auto topk_token_indexes = torch::remainder(indexes, vocab_size);

      auto values_acc = values.accessor<float, 1>();
      auto topk_hyp_indexes_acc = topk_hyp_indexes.accessor<int64_t, 1>();
      auto topk_token_indexes_acc = topk_token_indexes.accessor<int64_t, 1>();

      Hypotheses hyps;
      for (int32_t j = 0; j != values.numel(); ++j) {
        int32_t hyp_idx = topk_hyp_indexes_acc[j];
        Hypothesis new_hyp = prev[start + hyp_idx];  // note: hyp_idx is 0 based

        int32_t new_token = topk_token_indexes_acc[j];
        if (new_token != blank_id && new_token != unk_id) {
          new_hyp.ys.push_back(new_token);
        }

        // We already added log_prob of the path to log_probs before, so
        // we use values_acc[j] here directly.
        new_hyp.log_prob = values_acc[j];
        hyps.Add(std::move(new_hyp));
      }
      cur.push_back(std::move(hyps));
    }
  }

  for (auto &h : finalized) {
    cur.push_back(std::move(h));
  }

  auto unsorted_indices = packed_seq.unsorted_indices().cpu();
  auto unsorted_indices_accessor = unsorted_indices.accessor<int64_t, 1>();

  std::vector<std::vector<int32_t>> ans(batch_size);
  for (int32_t i = 0; i != batch_size; ++i) {
    Hypothesis hyp = cur[unsorted_indices_accessor[i]].GetMostProbable(true);
    torch::ArrayRef<int32_t> arr(hyp.ys);
    ans[i] = arr.slice(context_size).vec();
  }

  return ans;
}

}  // namespace k2

#endif  // K2_TORCH_CSRC_BEAM_SEARCH_H_
