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
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "k2/csrc/fsa_algo.h"
#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
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

// TODO(fangjun):
// Refactor this file.
//
// Create a binary for each decoding method.
// Don't put all decoding methods in a single binary.

enum class DecodingMethod {
  kInvalid,
  kCtcDecoding,
  kHLG,
  kNgramRescroing,
  kAttentionRescoring,
};

C10_DEFINE_bool(use_gpu, false, "True to use GPU. False to use CPU");
C10_DEFINE_string(jit_pt, "", "Path to exported jit file.");
C10_DEFINE_string(
    bpe_model, "",
    "Path to a pretrained BPE model. Needed if --method is 'ctc-decoding'");
C10_DEFINE_string(method, "", R"(Decoding method.
      Supported values are:
        - ctc-decoding. Use CTC topology for decoding. You have to
                        provide --bpe_model.
        - hlg. Use HLG graph for decoding.
        - ngram-rescoring. Use HLG for decoding and an n-gram LM for rescoring.
                         You have to provide --G.
        - attention-rescoring. Use HLG for decoding, an n-gram LM and a
                               attention decoder for rescoring.
)");
C10_DEFINE_string(hlg, "",
                  "Path to HLG.pt. Needed if --method is not 'ctc-decoding'");
C10_DEFINE_string(g, "",
                  "Path to an ngram LM, e.g, G_4gram.pt. Needed "
                  "if --method is 'ngram-rescoring' or 'attention-rescoring'");
C10_DEFINE_double(ngram_lm_scale, 1.0,
                  "Used only when method is ngram-rescoring");
C10_DEFINE_string(
    word_table, "",
    "Path to words.txt. Needed if --method is not 'ctc-decoding'");
// Fsa decoding related
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 8, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 10000,
               "max_activate_states in IntersectDensePruned");
// fbank related
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");

static void CheckArgs(DecodingMethod method) {
#if !defined(K2_WITH_CUDA)
  if (FLAGS_use_gpu) {
    std::cerr << "k2 was not compiled with CUDA. "
                 "Please use --use_gpu false";
    exit(EXIT_FAILURE);
  }
#endif

  if (FLAGS_jit_pt.empty()) {
    std::cerr << "Please provide --jit_pt\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (method == DecodingMethod::kCtcDecoding && FLAGS_bpe_model.empty()) {
    std::cerr << "Please provide --bpe_model\n"
              << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (method != DecodingMethod::kCtcDecoding && FLAGS_hlg.empty()) {
    std::cerr << "Please provide --hlg\n" << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (method != DecodingMethod::kCtcDecoding && FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table\n"
              << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if ((method == DecodingMethod::kNgramRescroing ||
       method == DecodingMethod::kAttentionRescoring) &&
      FLAGS_g.empty()) {
    std::cerr << "Please provide --g\n" << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string usage = R"(
  (1) CTC decoding
    ./bin/decode \
      --method ctc-decoding \
      --jit_pt <path to exported torch script pt file> \
      --bpe_model <path to pretrained BPE model> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
  (2) HLG decoding
    ./bin/decode \
      --method hlg \
      --jit_pt <path to exported torch script pt file> \
      --hlg <path to HLG.pt> \
      --word_table <path to words.txt> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
  (3) HLG decoding + ngram LM rescoring
    ./bin/decode \
      --method ngram-rescoring \
      --g <path to G.pt> \
      --jit_pt <path to exported torch script pt file> \
      --hlg <path to HLG.pt> \
      --word_table <path to words.txt> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>

   --use_gpu false to use CPU
   --use_gpu true to use GPU

   ./bin/decode --help
      to view all possible options.
  )";
  torch::SetUsageMessage(usage);

  DecodingMethod method = DecodingMethod::kInvalid;
  torch::ParseCommandLineFlags(&argc, &argv);
  if (FLAGS_method == "ctc-decoding") {
    method = DecodingMethod::kCtcDecoding;
  } else if (FLAGS_method == "hlg") {
    method = DecodingMethod::kHLG;
  } else if (FLAGS_method == "ngram-rescoring") {
    method = DecodingMethod::kNgramRescroing;
  } else if (FLAGS_method == "attention-rescoring") {
    // method = DecodingMethod::kAttentionRescoring;
    K2_LOG(FATAL) << "Not implemented yet for: " << FLAGS_method;
  } else {
    K2_LOG(FATAL) << "Unsupported method: " << FLAGS_method << "\n"
                  << torch::UsageMessage();
  }

  CheckArgs(method);

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
  // from icefall
  auto outputs = module.run_method("forward", features, supervisions).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();
  auto memory = outputs->elements()[1].toTensor();

  // memory_key_padding_mask is used in attention decoder rescoring
  // auto memory_key_padding_mask = outputs->elements()[2].toTensor();

  torch::Tensor supervision_segments =
      k2::GetSupervisionSegments(supervisions, subsampling_factor);

  k2::FsaClass decoding_graph;

  if (method == DecodingMethod::kCtcDecoding) {
    K2_LOG(INFO) << "Build CTC topo";
    decoding_graph = k2::CtcTopo(nnet_output.size(2) - 1, false, device);
  } else {
    K2_LOG(INFO) << "Load " << FLAGS_hlg;
    decoding_graph = k2::LoadFsa(FLAGS_hlg, device);
    K2_CHECK(decoding_graph.HasAttr("aux_labels"));
  }

  if (method == DecodingMethod::kNgramRescroing ||
      method == DecodingMethod::kAttentionRescoring) {
    // Add `lm_scores` so that we can separate acoustic scores and lm scores
    // later in the rescoring stage.
    decoding_graph.SetTensorAttr("lm_scores", decoding_graph.Scores().clone());
  }

  K2_LOG(INFO) << "Decoding";
  k2::FsaClass lattice = k2::GetLattice(
      nnet_output, decoding_graph, supervision_segments, FLAGS_search_beam,
      FLAGS_output_beam, FLAGS_min_activate_states, FLAGS_max_activate_states,
      subsampling_factor);

  if (method == DecodingMethod::kNgramRescroing) {
    // rescore with an n-gram LM
    K2_LOG(INFO) << "Load n-gram LM: " << FLAGS_g;
    k2::FsaClass G = k2::LoadFsa(FLAGS_g, device);
    G.fsa = k2::FsaToFsaVec(G.fsa);

    K2_CHECK_EQ(G.NumAttrs(), 0) << "G is expected to be an acceptor.";
    k2::AddEpsilonSelfLoops(G.fsa, &G.fsa);
    k2::ArcSort(&G.fsa);
    G.SetTensorAttr("lm_scores", G.Scores().clone());

    WholeLatticeRescoring(G, FLAGS_ngram_lm_scale, &lattice);
  }

  lattice = k2::ShortestPath(lattice);

  auto ragged_aux_labels = k2::GetTexts(lattice);

  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  std::vector<std::string> texts;
  if (method == DecodingMethod::kCtcDecoding) {
    sentencepiece::SentencePieceProcessor processor;
    auto status = processor.Load(FLAGS_bpe_model);
    if (!status.ok()) {
      K2_LOG(FATAL) << status.ToString();
    }
    for (const auto &ids : aux_labels_vec) {
      std::string text;
      status = processor.Decode(ids, &text);
      if (!status.ok()) {
        K2_LOG(FATAL) << status.ToString();
      }
      texts.emplace_back(std::move(text));
    }
  } else {
    k2::SymbolTable symbol_table(FLAGS_word_table);
    for (const auto &ids : aux_labels_vec) {
      std::string text;
      std::string sep = "";
      for (auto id : ids) {
        text.append(sep);
        text.append(symbol_table[id]);
        sep = " ";
      }
      texts.emplace_back(std::move(text));
    }
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
