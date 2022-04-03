/**
 * Copyright      2022  Xiaomi Corporation (authors: Wei Kang)
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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/utils.h"
#include "k2/torch/csrc/wave_reader.h"
#include "kaldifeat/csrc/feature-fbank.h"
#include "sentencepiece_processor.h"  // NOLINT
#include "torch/all.h"
#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_bool(use_gpu, false, "True to use GPU. False to use CPU");
C10_DEFINE_string(jit_pt, "", "Path to exported jit file.");
C10_DEFINE_bool(
    use_lg, false,
    "True to use an LG decoding graph. False to use a trivial decoding graph");
C10_DEFINE_string(
    bpe_model, "",
    "Path to a pretrained BPE model. Needed if --use_lg is false");
C10_DEFINE_string(lg, "", "Path to LG.pt. Needed if --use_lg is true");
C10_DEFINE_string(word_table, "",
                  "Path to words.txt. Needed if --use_lg is true");
// Rnnt decoding related
C10_DEFINE_double(beam, 4, "beam in RnntDecodingStreams");
C10_DEFINE_int(max_states, 32, "max_states in RnntDecodingStreams");
C10_DEFINE_int(max_contexts, 4, "max_contexts in RnntDecodingStreams");
// fbank related
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");

static void CheckArgs() {
#if !defined(K2_WITH_CUDA)
  if (FLAGS_use_gpu) {
    std::cerr << "k2 was not compiled with CUDA"
              << "\n";
    std::cerr << "Please use --use_gpu 0"
              << "\n";
    exit(EXIT_FAILURE);
  }
#endif

  if (FLAGS_jit_pt.empty()) {
    std::cerr << "Please provide --jit_pt"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_lg == false && FLAGS_bpe_model.empty()) {
    std::cout << "Please provide --bpe_model"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_lg && FLAGS_lg.empty()) {
    std::cout << "Please provide --lg"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_lg && FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string usage = R"(
  (1) Decoding without LG graph
    ./bin/rnnt_demo \
      --use_lg false \
      --jit_pt <path to exported torch script pt file> \
      --bpe_model <path to pretrained BPE model> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
  (2) Decoding with LG graph
    ./bin/rnnt_demo \
      --use_lg true \
      --jit_pt <path to exported torch script pt file> \
      --lg <path to LG.pt> \
      --word_table <path to words.txt> \
      --beam 8 \
      --max_contexts 8 \
      --max_states 64 \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
   --use_gpu false to use CPU
   --use_gpu true to use GPU
  )";
  torch::SetUsageMessage(usage);

  torch::ParseCommandLineFlags(&argc, &argv);
  CheckArgs();

  torch::Device device(torch::kCPU);
  if (FLAGS_use_gpu) {
    device = torch::Device(torch::kCUDA, 0);
  }

  K2_LOG(INFO) << "Device: " << device;

  int32_t num_waves = argc - 1;
  K2_CHECK_GE(num_waves, 1) << "You have to provide at least one wave file";
  std::vector<std::string> wave_filenames(num_waves);
  for (int32_t i = 0; i != num_waves; ++i) {
    wave_filenames[i] = argv[i + 1];
  }

  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = FLAGS_sample_rate;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = FLAGS_frame_shift_ms;
  fbank_opts.frame_opts.frame_length_ms = FLAGS_frame_length_ms;
  fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  K2_LOG(INFO) << "Load wave files";
  auto wave_data = k2::ReadWave(wave_filenames, FLAGS_sample_rate);

  for (auto &w : wave_data) {
    w = w.to(device);
  }

  K2_LOG(INFO) << "Compute features";
  std::vector<int64_t> num_frames;
  auto features_vec = k2::ComputeFeatures(fbank, wave_data, &num_frames);

  // Note: math.log(1e-10) is -23.025850929940457
  auto features = torch::nn::utils::rnn::pad_sequence(features_vec, true,
                                                      -23.025850929940457f);

  K2_LOG(INFO) << "Load neural network model";
  torch::jit::script::Module module = torch::jit::load(FLAGS_jit_pt);
  module.eval();
  module.to(device);

  k2::FsaClass decoding_graph;
  if (FLAGS_use_lg) {
    K2_LOG(INFO) << "Load LG.pt";
    decoding_graph = k2::LoadFsa(FLAGS_lg, device);
    K2_CHECK(decoding_graph.HasTensorAttr("aux_labels") ||
             decoding_graph.HasRaggedTensorAttr("aux_labels"));
  } else {
    K2_LOG(INFO) << "Build Trivial graph";
    decoding_graph = k2::TrivialGraph(logits.size(2) - 1, device);
  }

  K2_LOG(INFO) << "Decoding";

  auto decoding_fsa = std::make_shared<k2::Fsa>(decoding_graph.fsa);
  int32_t vocab_size = module.attr("vocab_size").toInt();
  int32_t context_size = module.attr("context_size").toInt();

  k2::rnnt_decoding::RnntDecodingConfig config(vocab_size, context_size,
                                               FLAGS_beam, FLAGS_max_states,
                                               FLAGS_max_contexts);

  std::vector<std::shared_ptr<k2::rnnt_decoding::RnntDecodingStream>>
      individual_streams;
  for (int32_t i = 0; i < num_waves; ++i) {
    individual_streams.emplace_back(
        k2::rnnt_decoding::CreateStream(decoding_fsa));
  }

  int32_t subsampling_factor = module.attr("subsampling_factor").toInt();
  auto input_lengths =
      torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
          .to(torch::kInt);

  K2_LOG(INFO) << "Compute encoder outs";
  // the output for module.encoder.forward() is a tuple of 2 tensors
  auto outputs =
      module.run_method("encoder_forward", features, input_lengths).toTuple();
  assert(outputs->elements().size() == 2u);

  auto logits = outputs->elements()[0].toTensor();
  auto logit_lengths = outputs->elements()[1].toTensor();

  k2::rnnt_decoding::RnntDecodingStreams streams(individual_streams, config);

  return 0;
}
