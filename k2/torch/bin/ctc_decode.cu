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

#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/wave_reader.h"
#include "torch/all.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
This file implements decoding with a CTC topology, without any
kinds of LM or lexicons.

Usage:
  ./bin/ctc_decode \
    --use_gpu true \
    --nn_model <path to torch scripted pt file> \
    --tokens <path to tokens.txt> \
    <path to foo.wav> \
    <path to bar.wav> \
    <more waves if any>

To see all possible options, use
  ./bin/ctc_decode --help

Caution:
 - Only sound files (*.wav) with single channel are supported.
 - It assumes the model is conformer_ctc/transformer.py from icefall.
   If you use a different model, you have to change the code
   related to `model.forward` in this file.
)";

C10_DEFINE_bool(use_gpu, false, "true to use GPU; false to use CPU");
C10_DEFINE_string(nn_model, "", "Path to the model exported by torch script.");
C10_DEFINE_string(tokens, "", "Path to the tokens.txt");

// Fsa decoding related
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 8, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 10000,
               "max_activate_states in IntersectDensePruned");
// Fbank related
// NOTE: These parameters must match those used in training
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");

static void CheckArgs() {
#if !defined(K2_WITH_CUDA)
  if (FLAGS_use_gpu) {
    std::cerr << "k2 was not compiled with CUDA. "
                 "Please use --use_gpu false";
    exit(EXIT_FAILURE);
  }
#endif

  if (FLAGS_nn_model.empty()) {
    std::cerr << "Please provide --nn_model\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_tokens.empty()) {
    std::cerr << "Please provide --tokens\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);
  torch::NoGradGuard no_grad;

  torch::SetUsageMessage(kUsageMessage);
  torch::ParseCommandLineFlags(&argc, &argv);
  CheckArgs();

  torch::Device device(torch::kCPU);
  if (FLAGS_use_gpu) {
    K2_LOG(INFO) << "Use GPU";
    device = torch::Device(torch::kCUDA, 0);
  }

  K2_LOG(INFO) << "Device: " << device;

  int32_t num_waves = argc - 1;
  K2_CHECK_GE(num_waves, 1) << "You have to provide at least one wave file";
  std::vector<std::string> wave_filenames(num_waves);
  for (int32_t i = 0; i != num_waves; ++i) {
    wave_filenames[i] = argv[i + 1];
  }

  K2_LOG(INFO) << "Load wave files";
  auto wave_data = k2::ReadWave(wave_filenames, FLAGS_sample_rate);

  for (auto &w : wave_data) {
    w = w.to(device);
  }

  K2_LOG(INFO) << "Build Fbank computer";
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = FLAGS_sample_rate;
  fbank_opts.frame_opts.dither = 0;
  fbank_opts.frame_opts.frame_shift_ms = FLAGS_frame_shift_ms;
  fbank_opts.frame_opts.frame_length_ms = FLAGS_frame_length_ms;
  fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  K2_LOG(INFO) << "Compute features";
  std::vector<int64_t> num_frames;
  auto features_vec = k2::ComputeFeatures(fbank, wave_data, &num_frames);

  // Note: math.log(1e-10) is -23.025850929940457
  auto features = torch::nn::utils::rnn::pad_sequence(features_vec, true,
                                                      -23.025850929940457f);

  K2_LOG(INFO) << "Load neural network model";
  torch::jit::script::Module module = torch::jit::load(FLAGS_nn_model);
  module.eval();
  module.to(device);

  int32_t subsampling_factor = module.attr("subsampling_factor").toInt();
  torch::Dict<std::string, torch::Tensor> sup;
  sup.insert("sequence_idx", torch::arange(num_waves, torch::kInt));
  sup.insert("start_frame", torch::zeros({num_waves}, torch::kInt));
  sup.insert("num_frames",
             torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
                 .to(torch::kInt));

  torch::IValue supervisions(sup);

  K2_LOG(INFO) << "Compute nnet_output";
  // the output for module.forward() is a tuple of 3 tensors
  // See the definition of the model in conformer_ctc/transformer.py
  // from icefall.
  // If you use a model that has a different signature for `forward`,
  // you can change the following line.
  auto outputs = module.run_method("forward", features, supervisions).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();

  torch::Tensor supervision_segments =
      k2::GetSupervisionSegments(supervisions, subsampling_factor);

  K2_LOG(INFO) << "Build CTC topo";
  auto decoding_graph = k2::CtcTopo(nnet_output.size(2) - 1, false, device);

  K2_LOG(INFO) << "Decoding";
  k2::FsaClass lattice = k2::GetLattice(
      nnet_output, decoding_graph, supervision_segments, FLAGS_search_beam,
      FLAGS_output_beam, FLAGS_min_activate_states, FLAGS_max_activate_states,
      subsampling_factor);

  lattice = k2::ShortestPath(lattice);

  auto ragged_aux_labels = k2::GetTexts(lattice);
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  k2::SymbolTable symbol_table(FLAGS_tokens);

  std::vector<std::string> texts;
  for (const auto &ids : aux_labels_vec) {
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

  return 0;
}
