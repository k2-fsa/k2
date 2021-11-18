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

#include "k2/csrc/online_intersect_dense_pruned.h"
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

C10_DEFINE_bool(use_gpu, false, "True to use GPU. False to use CPU");
C10_DEFINE_string(jit_pt, "", "Path to exported jit file.");
C10_DEFINE_string(
    bpe_model, "",
    "Path to a pretrained BPE model. Needed if --use_ctc_decoding is true");
C10_DEFINE_bool(use_ctc_decoding, true, "True to use CTC decoding");
C10_DEFINE_string(hlg, "",
                  "Path to HLG.pt. Needed if --use_ctc_decoding is false");
C10_DEFINE_string(word_table, "",
                  "Path to words.txt. Needed if --use_ctc_decoding is false");
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

  if (FLAGS_use_ctc_decoding && FLAGS_bpe_model.empty()) {
    std::cout << "Please provide --bpe_model"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_hlg.empty()) {
    std::cerr << "Please provide --hlg"
              << "\n";
    std::cerr << torch::UsageMessage() << "\n";
    exit(EXIT_FAILURE);
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_word_table.empty()) {
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
  (1) CTC decoding
    ./bin/decode \
      --use_ctc_decoding true \
      --jit_pt <path to exported torch script pt file> \
      --bpe_model <path to pretrained BPE model> \
      /path/to/foo.wav \
      /path/to/bar.wav \
      <more wave files if any>
  (2) HLG decoding
    ./bin/decode \
      --use_ctc_decoding false \
      --jit_pt <path to exported torch script pt file> \
      --hlg <path to HLG.pt> \
      --word_table <path to words.txt> \
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

  int32_t subsampling_factor = module.attr("subsampling_factor").toInt();
  torch::Dict<std::string, torch::Tensor> sup;
  sup.insert("sequence_idx", torch::arange(num_waves, torch::kInt));
  sup.insert("start_frame", torch::zeros({num_waves}, torch::kInt));
  sup.insert("num_frames",
             torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
                 .to(torch::kInt));

  torch::IValue supervisions(sup);

  std::vector<torch::IValue> inputs;
  inputs.emplace_back(std::move(features));
  inputs.emplace_back(supervisions);

  K2_LOG(INFO) << "Compute nnet_output";
  // the output for module.forward() is a tuple of 3 tensors
  auto outputs = module.forward(inputs).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();
  auto memory = outputs->elements()[1].toTensor();

  // memory_key_padding_mask is used in attention decoder rescoring
  // auto memory_key_padding_mask = outputs->elements()[2].toTensor();

  k2::FsaClass decoding_graph;

  if (FLAGS_use_ctc_decoding) {
    K2_LOG(INFO) << "Build CTC topo";
    decoding_graph =
        k2::CtcTopo(nnet_output.size(2) - 1, /*modified*/ false, device);
  } else {
    K2_LOG(INFO) << "Load HLG.pt";
    decoding_graph = k2::LoadFsa(FLAGS_hlg);
    decoding_graph = decoding_graph.To(device);
  }

  K2_LOG(INFO) << "Decoding";

  auto decoding_fsa = k2::FsaToFsaVec(decoding_graph.fsa);
  k2::OnlineIntersectDensePruned decoder(
      decoding_fsa, num_waves, FLAGS_search_beam, FLAGS_output_beam,
      FLAGS_min_activate_states, FLAGS_max_activate_states);

  std::vector<std::string> texts(num_waves, "");
  int32_t T = nnet_output.size(1);
  int32_t chunk_size = 21;
  int32_t chunk_num = (T / chunk_size) + ((T % chunk_size) ? 1 : 0);
  K2_LOG(INFO) << T << " " << chunk_num;
  for (int32_t c = 0; c < chunk_num; ++c) {
    int32_t start = c * chunk_size;
    int32_t end = (c + 1) * chunk_size >= T ? T : (c + 1) * chunk_size;
    K2_LOG(INFO) << "start : " << start << " end : " << end;
    std::vector<int64_t> num_frame;
    for (auto &frame : num_frames) {
      K2_LOG(INFO) << "frame : " << frame;
      if (frame < chunk_size * subsampling_factor) {
        num_frame.push_back(frame);
        frame = 0;
      } else {
        num_frame.push_back(chunk_size * subsampling_factor);
        frame -= chunk_size * subsampling_factor;
      }
    }
    torch::Dict<std::string, torch::Tensor> sup;
    sup.insert("sequence_idx", torch::arange(num_waves, torch::kInt));
    sup.insert("start_frame", torch::zeros({num_waves}, torch::kInt));
    sup.insert("num_frames",
               torch::from_blob(num_frame.data(), {num_waves}, torch::kLong)
                   .to(torch::kInt));
    torch::IValue supervision(sup);

    using namespace torch::indexing;
    auto sub_nnet_output =
        nnet_output.index({Slice(), Slice(start, end), Slice()});
    K2_LOG(INFO) << "nnet_output : " << nnet_output.sizes();
    K2_LOG(INFO) << "sub_nnet_output : " << sub_nnet_output.sizes();

    torch::Tensor supervision_segments =
        k2::GetSupervisionSegments(supervision, subsampling_factor);

    k2::DenseFsaVec dense_fsa_vec = k2::CreateDenseFsaVec(
        sub_nnet_output, supervision_segments, subsampling_factor - 1);

    K2_LOG(INFO) << "dense fsa shape : " << dense_fsa_vec.shape;

    auto dense_fsa = std::make_shared<k2::DenseFsaVec>(dense_fsa_vec);
    bool is_final = c == chunk_num - 1 ? true : false;
    decoder.Intersect(dense_fsa, is_final);

    k2::FsaVec tmp_fsa;
    k2::Array1<int32_t> tmp_graph_arc_map;
    decoder.FormatPartial2(&tmp_fsa, &tmp_graph_arc_map);
    K2_LOG(INFO) << "partial fsa : " << tmp_fsa;
    k2::FsaClass tmp_lattice(tmp_fsa);
  }

  k2::FsaVec fsa;
  k2::Array1<int32_t> graph_arc_map;
  decoder.FormatOutput(&fsa, &graph_arc_map, true);

  K2_LOG(INFO) << "Fsa : " << fsa;

  k2::FsaClass lattice(fsa);
  lattice.CopyAttrs(decoding_graph, k2::Array1ToTorch<int32_t>(graph_arc_map));

  lattice = k2::ShortestPath(lattice);

  // K2_LOG(INFO) << "Lattice : " << lattice.fsa;

  auto ragged_aux_labels = k2::GetTexts(lattice);

  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  if (FLAGS_use_ctc_decoding) {
    sentencepiece::SentencePieceProcessor processor;
    auto status = processor.Load(FLAGS_bpe_model);
    if (!status.ok()) {
      K2_LOG(FATAL) << status.ToString();
    }
    for (int32_t i = 0; i < aux_labels_vec.size(); ++i) {
      std::string text;
      status = processor.Decode(aux_labels_vec[i], &text);
      if (!status.ok()) {
        K2_LOG(FATAL) << status.ToString();
      }
      texts[i] += text + " ";
    }
  } else {
    k2::SymbolTable symbol_table(FLAGS_word_table);
    for (int32_t i = 0; i < aux_labels_vec.size(); ++i) {
      std::string text;
      std::string sep = "";
      for (auto id : aux_labels_vec[i]) {
        text.append(sep);
        text.append(symbol_table[id]);
        sep = " ";
      }
      texts[i] += text + " ";
    }
    std::ostringstream os;
    os << "\nPartial result:\n\n";
    for (int32_t i = 0; i != num_waves; ++i) {
      os << wave_filenames[i] << "\n";
      os << texts[i];
      os << "\n\n";
    }
    K2_LOG(INFO) << os.str();
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
