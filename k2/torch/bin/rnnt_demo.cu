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
#include "torch/all.h"
#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_bool(use_gpu, false, "True to use GPU. False to use CPU");
C10_DEFINE_string(jit_pt, "", "Path to exported jit file.");
C10_DEFINE_bool(
    use_lg, false,
    "True to use an LG decoding graph. False to use a trivial decoding graph");
C10_DEFINE_string(tokens, "",
                  "Path to a tokens.txt. Needed if --use_lg is false");
C10_DEFINE_string(lg, "", "Path to LG.pt. Needed if --use_lg is true");
C10_DEFINE_string(word_table, "",
                  "Path to words.txt. Needed if --use_lg is true");
// Rnnt decoding related
C10_DEFINE_double(beam, 8.0, "beam in RnntDecodingStreams");
C10_DEFINE_int(max_states, 64, "max_states in RnntDecodingStreams");
C10_DEFINE_int(max_contexts, 8, "max_contexts in RnntDecodingStreams");
// fbank related
C10_DEFINE_int(sample_rate, 16000, "Expected sample rate of wave files");
C10_DEFINE_double(frame_shift_ms, 10.0,
                  "Frame shift in ms for computing Fbank");
C10_DEFINE_double(frame_length_ms, 25.0,
                  "Frame length in ms for computing Fbank");
C10_DEFINE_int(num_bins, 80, "Number of triangular bins for computing Fbank");
C10_DEFINE_int(max_num_streams, 2, "Max number of decoding streams");
C10_DEFINE_bool(
    use_max, true,
    "True to use max operation to select the hypothesis with the largest "
    "log_prob when there are duplicate hypotheses; False to use log-add.");
C10_DEFINE_int(num_paths, 200,
               "Number of paths to sample when generating Nbest");
C10_DEFINE_double(nbest_scale, 0.8,
                  "The scale value applying to lattice.score before sampling");

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

  if (FLAGS_use_lg == false && FLAGS_tokens.empty()) {
    std::cout << "Please provide --tokens"
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
      --tokens <path to tokens.txt> \
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

  int32_t vocab_size = 500;
  int32_t context_size = 2;
  int32_t subsampling_factor = 4;

  k2::FsaClass decoding_graph;
  if (FLAGS_use_lg) {
    K2_LOG(INFO) << "Load LG.pt";
    decoding_graph = k2::LoadFsa(FLAGS_lg, device);
    K2_CHECK(decoding_graph.HasTensorAttr("aux_labels") ||
             decoding_graph.HasRaggedTensorAttr("aux_labels"));
  } else {
    K2_LOG(INFO) << "Build Trivial graph";
    decoding_graph = k2::TrivialGraph(vocab_size - 1, device);
  }

  K2_LOG(INFO) << "Decoding";

  k2::rnnt_decoding::RnntDecodingConfig config(vocab_size, context_size,
                                               FLAGS_beam, FLAGS_max_states,
                                               FLAGS_max_contexts);

  std::vector<std::shared_ptr<k2::rnnt_decoding::RnntDecodingStream>>
      individual_streams(num_waves);
  std::vector<k2::FsaClass> individual_graphs(num_waves);
  // suppose we are using same graph for all waves.
  for (int32_t i = 0; i < num_waves; ++i) {
    individual_graphs[i] = decoding_graph;
    individual_streams[i] = k2::rnnt_decoding::CreateStream(
        std::make_shared<k2::Fsa>(individual_graphs[i].fsa));
  }

  // we are not using a streaming model currently, so calculate encoder_outs
  // at a time.
  auto input_lengths =
      torch::from_blob(num_frames.data(), {num_waves}, torch::kLong)
          .to(torch::kInt)
          .to(device);

  K2_LOG(INFO) << "Compute encoder outs";
  // the output for module.encoder.forward() is a tuple of 2 tensors
  auto outputs = module.attr("encoder")
                     .toModule()
                     .run_method("forward", features, input_lengths)
                     .toTuple();
  assert(outputs->elements().size() == 2u);

  auto encoder_outs = outputs->elements()[0].toTensor();
  auto encoder_outs_lengths = outputs->elements()[1].toTensor();

  int32_t T = encoder_outs.size(1);
  int32_t chunk_size = 10;  // 10 frames per chunk
  std::vector<int32_t> decoded_frames(num_waves, 0);
  std::vector<int32_t> positions(num_waves, 0);

  // decocding results for each waves
  std::vector<std::string> texts(num_waves, "");

  // simulate asynchronous decoding
  while (true) {
    std::vector<std::shared_ptr<k2::rnnt_decoding::RnntDecodingStream>>
        current_streams;
    std::vector<torch::Tensor> current_encoder_outs;
    // which waves we are decoding now
    std::vector<int32_t> current_wave_ids;

    std::vector<int32_t> current_num_frames;
    std::vector<k2::FsaClass> current_graphs;

    for (int32_t i = 0; i < num_waves; ++i) {
      // this wave is done
      if (decoded_frames[i] * subsampling_factor >= num_frames[i]) continue;

      current_streams.emplace_back(individual_streams[i]);
      current_graphs.emplace_back(individual_graphs[i]);
      current_wave_ids.push_back(i);

      if ((num_frames[i] - decoded_frames[i]) <=
          chunk_size * subsampling_factor) {
        decoded_frames[i] = num_frames[i] / subsampling_factor;
      } else {
        decoded_frames[i] += chunk_size;
      }

      current_num_frames.emplace_back(decoded_frames[i]);

      int32_t start = positions[i],
              end = start + chunk_size >= T ? T : start + chunk_size;
      positions[i] = end;
      auto sub_output = encoder_outs.index(
          {i, torch::indexing::Slice(start, end), torch::indexing::Slice()});

      // padding T axis to chunk_size if needed
      namespace F = torch::nn::functional;
      sub_output = F::pad(sub_output,
                          F::PadFuncOptions({0, 0, 0, chunk_size - end + start})
                              .mode(torch::kConstant));

      current_encoder_outs.push_back(sub_output);

      // we can decode at most `FLAGS_max_num_streams` waves at a time
      if (static_cast<int32_t>(current_wave_ids.size()) >=
          FLAGS_max_num_streams)
        break;
    }
    if (current_wave_ids.size() == 0) break;  // finished

    auto sub_encoder_outs = torch::stack(current_encoder_outs);

    auto streams =
        k2::rnnt_decoding::RnntDecodingStreams(current_streams, config);
    k2::DecodeOneChunk(streams, module, sub_encoder_outs);

    k2::FsaVec ofsa;
    k2::Array1<int32_t> out_map;
    bool allow_partial = true;
    streams.FormatOutput(current_num_frames, allow_partial, &ofsa, &out_map);

    auto arc_map = k2::Ragged<int32_t>(ofsa.shape, out_map).RemoveAxis(1);
    k2::FsaClass lattice(ofsa);
    lattice.CopyAttrs(current_graphs, arc_map);

    lattice = k2::GetBestPaths(lattice, FLAGS_use_max, FLAGS_num_paths,
                               FLAGS_nbest_scale);

    auto ragged_aux_labels = k2::GetTexts(lattice);

    auto aux_labels_vec = ragged_aux_labels.ToVecVec();

    if (!FLAGS_use_lg) {
      k2::SymbolTable symbol_table(FLAGS_tokens);
      for (size_t i = 0; i < current_wave_ids.size(); ++i) {
        std::string text;
        for (auto id : aux_labels_vec[i]) {
          text.append(symbol_table[id]);
        }
        texts[current_wave_ids[i]] = std::move(text);
      }
    } else {
      k2::SymbolTable symbol_table(FLAGS_word_table);
      for (size_t i = 0; i < current_wave_ids.size(); ++i) {
        std::string text;
        std::string sep = "";
        for (auto id : aux_labels_vec[i]) {
          text.append(sep);
          text.append(symbol_table[id]);
          sep = " ";
        }
        texts[current_wave_ids[i]] = text;
      }
    }
    std::ostringstream os;
    os << "\nPartial result:\n";
    for (size_t i = 0; i != current_wave_ids.size(); ++i) {
      os << wave_filenames[current_wave_ids[i]] << "\n";
      os << texts[current_wave_ids[i]];
      os << "\n\n";
    }
    K2_LOG(INFO) << os.str();
  }

  std::ostringstream os;
  os << "\nDecoding result:\n";
  for (int32_t i = 0; i != num_waves; ++i) {
    os << wave_filenames[i] << "\n";
    os << texts[i];
    os << "\n\n";
  }
  K2_LOG(INFO) << os.str();
  return 0;
}
