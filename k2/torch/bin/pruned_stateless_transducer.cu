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

#include "k2/csrc/log.h"
#include "k2/torch/csrc/beam_search.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/parse_options.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/wave_reader.h"
#include "torch/all.h"

static constexpr const char *kUsageMessage = R"(
This file implements RNN-T decoding for pruned stateless transducer models
that are trained using pruned_transducer_statelessX (X>=2) from icefall.

Usage:
  ./bin/pruned_stateless_transducer --help

  ./bin/pruned_stateless_transducer \
    --nn-model=/path/to/cpu_jit.pt \
    --tokens=/path/to/tokens.txt \
    --use-gpu=true \
    --decoding-method=modified_beam_search \
    /path/to/foo.wav \
    /path/to/bar.wav
)";

static void RegisterFrameExtractionOptions(
    k2::ParseOptions *po, kaldifeat::FrameExtractionOptions *opts) {
  po->Register("sample-frequency", &opts->samp_freq,
               "Waveform data sample frequency (must match the waveform file, "
               "if specified there)");

  po->Register("frame-length", &opts->frame_length_ms,
               "Frame length in milliseconds");

  po->Register("frame-shift", &opts->frame_shift_ms,
               "Frame shift in milliseconds");

  po->Register("dither", &opts->dither,
               "Dithering constant (0.0 means no dither).");
}

static void RegisterMelBanksOptions(k2::ParseOptions *po,
                                    kaldifeat::MelBanksOptions *opts) {
  po->Register("num-mel-bins", &opts->num_bins,
               "Number of triangular mel-frequency bins");
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  k2::ParseOptions po(kUsageMessage);

  std::string nn_model;  // path to the torch jit model file
  std::string tokens;    // path to tokens.txt
  bool use_gpu = false;  // true to use GPU for decoding; false to use CPU.
  std::string decoding_method = "greedy_search";  // Supported methods are:
                                                  // greedy_search,
                                                  // modified_beam_search

  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.dither = 0;
  RegisterFrameExtractionOptions(&po, &fbank_opts.frame_opts);
  fbank_opts.mel_opts.num_bins = 80;
  RegisterMelBanksOptions(&po, &fbank_opts.mel_opts);

  po.Register("nn-model", &nn_model, "Path to the torch jit model file");

  po.Register("tokens", &tokens, "Path to the tokens.txt");

  po.Register("use-gpu", &use_gpu,
              "true to use GPU for decoding; false to use CPU. "
              "If GPU is enabled, it always uses GPU 0. You can use "
              "the environment variable CUDA_VISIBLE_DEVICES to control "
              "which GPU device to use.");

  po.Register(
      "decoding-method", &decoding_method,
      "Decoding method to use."
      "Currently implemented methods are: greedy_search, modified_beam_search");

  po.Read(argc, argv);

  K2_CHECK(decoding_method == "greedy_search" ||
           decoding_method == "modified_beam_search")
      << "Currently supported decoding methods are: "
         "greedy_search, modified_beam_search. "
      << "Given: " << decoding_method;

  torch::Device device(torch::kCPU);
  if (use_gpu) {
    K2_LOG(INFO) << "Use GPU";
    device = torch::Device(torch::kCUDA, 0);
  }

  K2_LOG(INFO) << "Device: " << device;

  int32_t num_waves = po.NumArgs();
  K2_CHECK_GT(num_waves, 0) << "Please provide at least one wave file";

  std::vector<std::string> wave_filenames(num_waves);
  for (int32_t i = 0; i < num_waves; ++i) {
    wave_filenames[i] = po.GetArg(i + 1);
  }

  K2_LOG(INFO) << "Loading wave files";
  std::vector<torch::Tensor> wave_data =
      k2::ReadWave(wave_filenames, fbank_opts.frame_opts.samp_freq);
  for (auto &w : wave_data) {
    w = w.to(device);
  }

  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  K2_LOG(INFO) << "Computing features";
  std::vector<int64_t> num_frames;
  std::vector<torch::Tensor> features_vec =
      k2::ComputeFeatures(fbank, wave_data, &num_frames);

  // Note: math.log(1e-10) is -23.025850929940457
  torch::Tensor features = torch::nn::utils::rnn::pad_sequence(
      features_vec, /*batch_first*/ true,
      /*padding_value*/ -23.025850929940457f);
  torch::Tensor feature_lens = torch::tensor(num_frames, device);

  K2_LOG(INFO) << "Loading neural network model from " << nn_model;
  torch::jit::Module module = torch::jit::load(nn_model);
  module.eval();
  module.to(device);

  K2_LOG(INFO) << "Computing output of the encoder network";

  auto outputs = module.attr("encoder")
                     .toModule()
                     .run_method("forward", features, feature_lens)
                     .toTuple();
  assert(outputs->elements().size() == 2u);

  auto encoder_out = outputs->elements()[0].toTensor();
  auto encoder_out_lens = outputs->elements()[1].toTensor();

  K2_LOG(INFO) << "Using " << decoding_method;

  std::vector<std::vector<int32_t>> hyp_tokens;
  if (decoding_method == "greedy_search") {
    hyp_tokens = k2::GreedySearch(module, encoder_out, encoder_out_lens.cpu());
  } else {
    hyp_tokens =
        k2::ModifiedBeamSearch(module, encoder_out, encoder_out_lens.cpu());
  }

  k2::SymbolTable symbol_table(tokens);

  std::vector<std::string> texts;
  for (const auto &ids : hyp_tokens) {
    std::string text;
    for (auto id : ids) {
      text.append(symbol_table[id]);
    }
    texts.emplace_back(std::move(text));
  }

  std::ostringstream os;
  os << "\nDecoding result:\n\n";
  for (int32_t i = 0; i != num_waves; ++i) {
    os << wave_filenames[i] << "\n";
    os << texts[i];
    os << "\n\n";
  }
  K2_LOG(INFO) << os.str();
};
