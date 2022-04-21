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
  ./bin/sp_ctc_decode \
    --use_gpu true \
    --nn_model <path to torch scripted pt file> \
    --word_table <path to words.txt> \
    --batch_size <number of waves in each batch> \
    --wav_scp <path to wav.scp> \
    --result_file <path to result file>

Example:
  ./bin/sp_ctc_decode \
    --use_gpu true \
    --nn_model ./final.zip \
    --word_table ./lang_char.txt \
    --batch_size 300 \
    --wav_scp ./wavs \
    --result_file result2.txt

To see all possible options, use
  ./bin/sp_ctc_decode --help

Caution:
 - Only sound files (*.wav) with single channel are supported.
)";

C10_DEFINE_bool(use_gpu, false, "true to use GPU; false to use CPU");
C10_DEFINE_bool(use_modified_ctc_topo, true,
                "true to use modified ctc_topo; "
                "false to use standard ctc_topo");
C10_DEFINE_string(nn_model, "", "Path to the model exported by torch script.");
C10_DEFINE_string(word_table, "", "Path to word_table.");
// each line in wav_scp contains at least two fields, separated by spaces.
// The first field is the utterance ID and the second is the path to the
// wave file.
C10_DEFINE_string(wav_scp, "", "Path to wav.scp.");
C10_DEFINE_string(result_file, "results.txt",
                  "Path to save the decoding results.");
C10_DEFINE_int(batch_size, 10, "Process this number of batch each time");

// Fsa decoding related
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 8, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 1000,
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

  if (FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_wav_scp.empty()) {
    std::cerr << "Please provide --wav_scp\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_result_file.empty()) {
    std::cerr << "Please provide --result_file\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  K2_CHECK_GT(FLAGS_batch_size, 0);
}

namespace k2 {

/// Return a list of pair. pairs.first is the utterance ID, while
// pair.second is the path to the wave file.
static void ReadWavFilenames(const std::string &wav_scp,
                             std::vector<std::string> *utt_ids,
                             std::vector<std::string> *wav_filenames) {
  utt_ids->clear();
  wav_filenames->clear();

  auto lines = ReadLines(wav_scp);
  for (auto line : lines) {
    auto fields = SplitStringToVector(line, "\t ");
    K2_CHECK_GE(fields.size(), 2u) << line;
    if (fields[0][0] == '#') {
      K2_LOG(WARNING) << "Skip " << line;
      continue;
    }

    utt_ids->push_back(std::move(fields[0]));
    wav_filenames->push_back(std::move(fields[1]));
  }
}

static Ragged<int32_t> Decode(torch::jit::Module &module, torch::Device device,
                              kaldifeat::Fbank &fbank, FsaClass &decoding_graph,
                              const std::vector<std::string> &wave_filenames) {
  torch::jit::Module encoder_module = module.attr("encoder").toModule();
  torch::jit::Module ctc_module = module.attr("ctc").toModule();

  int32_t num_waves = wave_filenames.size();
  float normalizer = 1;  // Kaldi uses no normalizers.
  auto wave_data = ReadWave(wave_filenames, FLAGS_sample_rate, normalizer);

  for (auto &w : wave_data) {
    w = w.to(device);
  }

  std::vector<int64_t> num_frames;
  auto features_vec = ComputeFeatures(fbank, wave_data, &num_frames);

  // Note: math.log(1e-10) is -23.025850929940457
  auto features = torch::nn::utils::rnn::pad_sequence(features_vec, true,
                                                      -23.025850929940457f);

  torch::Tensor feature_lengths =
      torch::from_blob(num_frames.data(), {num_waves}, torch::kLong).to(device);

  int32_t decoding_chunk_size = -1;
  int32_t num_decoding_left_chunks = -1;

  // encoder.forward accepts 4 inputs, see
  // see
  // https://github.com/wenet-e2e/wenet/blob/main/wenet/transformer/encoder.py#L123
  auto outputs = encoder_module
                     .run_method("forward", features, feature_lengths,
                                 decoding_chunk_size, num_decoding_left_chunks)
                     .toTuple();
  K2_CHECK_EQ(outputs->elements().size(), 2u);

  auto encoder_out = outputs->elements()[0].toTensor();   // (N, T, C)
  auto encoder_mask = outputs->elements()[1].toTensor();  // (N, 1, C)
  K2_CHECK_EQ(encoder_mask.dim(), 3);
  K2_CHECK_EQ(encoder_mask.size(1), 1);

  auto encoder_out_lengths =
      encoder_mask.squeeze(1).sum(1).cpu().to(torch::kInt);

  torch::Tensor sequence_idx = torch::arange(num_waves, torch::kInt);
  torch::Tensor start_frames = torch::zeros({num_waves}, torch::kInt);
  torch::Tensor supervision_segments =
      torch::stack({sequence_idx, start_frames, encoder_out_lengths})
          .t()
          .contiguous();

  torch::Tensor nnet_output =
      ctc_module.run_method("log_softmax", encoder_out).toTensor();

  FsaClass lattice =
      GetLattice(nnet_output, decoding_graph, supervision_segments,
                 FLAGS_search_beam, FLAGS_output_beam,
                 FLAGS_min_activate_states, FLAGS_max_activate_states, 1);

  lattice = ShortestPath(lattice);

  auto ragged_aux_labels = GetTexts(lattice);
  return ragged_aux_labels;
}

}  // namespace k2

int main(int argc, char *argv[]) {
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

  K2_LOG(INFO) << "Build Fbank computer";
  kaldifeat::FbankOptions fbank_opts;
  fbank_opts.frame_opts.samp_freq = FLAGS_sample_rate;
  fbank_opts.frame_opts.dither = 0;  // TODO(fangjun): Change it to 1
  fbank_opts.frame_opts.frame_shift_ms = FLAGS_frame_shift_ms;
  fbank_opts.frame_opts.frame_length_ms = FLAGS_frame_length_ms;
  fbank_opts.mel_opts.num_bins = FLAGS_num_bins;
  fbank_opts.device = device;

  kaldifeat::Fbank fbank(fbank_opts);

  K2_LOG(INFO) << "Load neural network model";
  torch::jit::script::Module module = torch::jit::load(FLAGS_nn_model);
  module.eval();
  module.to(device);

  K2_CHECK(module.hasattr("encoder"));
  K2_CHECK(module.attr("encoder").isModule());

  K2_CHECK(module.hasattr("ctc"));
  K2_CHECK(module.attr("ctc").isModule());
  K2_CHECK(module.hasattr("vocab_size"));
  int32_t vocab_size = module.attr("vocab_size").toInt();

  K2_LOG(INFO) << "use_modified_ctc_topo: " << std::boolalpha
               << FLAGS_use_modified_ctc_topo;

  K2_LOG(INFO) << "max token id: " << vocab_size - 1;

  k2::FsaClass decoding_graph =
      k2::CtcTopo(vocab_size - 1, FLAGS_use_modified_ctc_topo, device);

  k2::SymbolTable symbol_table(FLAGS_word_table);

  std::vector<std::string> utt_ids;
  std::vector<std::string> wav_filenames;
  k2::ReadWavFilenames(FLAGS_wav_scp, &utt_ids, &wav_filenames);

  int32_t num_waves = wav_filenames.size();

  std::vector<std::vector<std::string>> batches;
  int32_t i = 0;
  // TODO(fangjun): Use max_duration to group batches
  for (; i + FLAGS_batch_size < num_waves; i += FLAGS_batch_size) {
    auto start = wav_filenames.begin() + i;
    auto end = start + FLAGS_batch_size;
    batches.emplace_back(start, end);
  }
  batches.emplace_back(wav_filenames.begin() + i, wav_filenames.end());

  K2_LOG(INFO) << "num_wavs: " << wav_filenames.size();
  K2_LOG(INFO) << "num_batches: " << batches.size();

  std::vector<k2::Ragged<int32_t>> aux_labels;
  for (int32_t idx = 0; idx != batches.size(); ++idx) {
    if (idx % 2 == 0) {
      K2_LOG(INFO) << "Processing " << idx << "/" << batches.size();
    }
    auto tmp = k2::Decode(module, device, fbank, decoding_graph, batches[idx]);
    aux_labels.push_back(std::move(tmp));
  }

  k2::Ragged<int32_t> ragged_aux_labels =
      k2::Cat(0, aux_labels.size(), aux_labels.data());
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  std::vector<std::string> texts;
  for (const auto &ids : aux_labels_vec) {
    std::string text;
    std::string sep = "";
    for (auto id : ids) {
      text.append(sep);
      text.append(symbol_table[id]);
      // sep = " ";
    }
    texts.emplace_back(std::move(text));
  }

  std::ostringstream os;
  std::string sep = "";
  for (size_t i = 0; i != wav_filenames.size(); ++i) {
    os << sep << utt_ids[i] << " " << texts[i];
    sep = "\n";
  }
  std::ofstream of(FLAGS_result_file);
  of << os.str();

  return 0;
}
