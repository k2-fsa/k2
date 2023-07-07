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

#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/decode.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/features.h"
#include "k2/torch/csrc/fsa_algo.h"
#include "k2/torch/csrc/nbest.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/wave_reader.h"
#include "torch/all.h"
#include "torch/script.h"

static constexpr const char *kUsageMessage = R"(
This file implements decoding with an HLG decoding graph, using
an n-gram LM and an attention decoder for rescoring.

Usage:
  ./bin/attention_rescore \
    --use_gpu true \
    --nn_model <path to torch scripted pt file> \
    --hlg <path to HLG.pt> \
    --g <path to G.pt> \
    --ngram_lm_scale 1.0 \
    --attention_scale 1.0 \
    --num_paths 100 \
    --nbest_scale 0.5 \
    --word_table <path to words.txt> \
    --sos_id <ID of start of sentence symbol> \
    --eos_id <ID of end of sentence symbol> \
    <path to foo.wav> \
    <path to bar.wav> \
    <more waves if any>

To see all possible options, use
  ./bin/attention_rescore --help

Caution:
 - Only sound files (*.wav) with single channel are supported.
 - It assumes the model is conformer_ctc/transformer.py from icefall.
   If you use a different model, you have to change the code
   related to `model.forward` in this file.
)";

C10_DEFINE_bool(use_gpu, false, "true to use GPU; false to use CPU");
C10_DEFINE_string(nn_model, "", "Path to the model exported by torch script.");
C10_DEFINE_string(hlg, "", "Path to HLG.pt.");
C10_DEFINE_string(g, "", "Path to an ngram LM, e.g, G_4gram.pt");
C10_DEFINE_double(ngram_lm_scale, 1.0, "Scale for ngram LM scores");
C10_DEFINE_double(attention_scale, 1.0, "Scale for attention scores");
C10_DEFINE_int(num_paths, -1, "Number of paths to sample for rescoring");
C10_DEFINE_double(nbest_scale, 0.5,
                  "Scale for lattice.scores by this value before sampling.");
C10_DEFINE_string(word_table, "", "Path to words.txt.");
C10_DEFINE_int(sos_id, -1, "ID of start of sentence symbol.");
C10_DEFINE_int(eos_id, -1, "ID of end of sentence symbol.");

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

  if (FLAGS_hlg.empty()) {
    std::cerr << "Please provide --hlg\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_g.empty()) {
    std::cerr << "Please provide --g\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_word_table.empty()) {
    std::cerr << "Please provide --word_table\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_sos_id == -1) {
    std::cerr << "Please provide --sos_id\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  if (FLAGS_eos_id == -1) {
    std::cerr << "Please provide --eos_id\n" << torch::UsageMessage();
    exit(EXIT_FAILURE);
  }

  K2_CHECK_GT(FLAGS_num_paths, 0);
  K2_CHECK_GT(FLAGS_nbest_scale, 0);
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

  auto nnet_output = outputs->elements()[0].toTensor();  // shape (N, T, C)
  auto memory = outputs->elements()[1].toTensor();       // shape (T, N, C)
  auto memory_key_padding_mask =
      outputs->elements()[2].toTensor();  // shape (N, T)

  torch::Tensor supervision_segments =
      k2::GetSupervisionSegments(supervisions, subsampling_factor);

  K2_LOG(INFO) << "Load " << FLAGS_hlg;
  k2::FsaClass decoding_graph = k2::LoadFsa(FLAGS_hlg, device);
  K2_CHECK(decoding_graph.HasTensorAttr("aux_labels") ||
           decoding_graph.HasRaggedTensorAttr("aux_labels"));
  // Add `lm_scores` so that we can separate acoustic scores and lm scores
  // later in the rescoring stage.
  decoding_graph.SetTensorAttr("lm_scores", decoding_graph.Scores().clone());

  K2_LOG(INFO) << "Decoding";
  k2::FsaClass lattice = k2::GetLattice(
      nnet_output, decoding_graph, supervision_segments, FLAGS_search_beam,
      FLAGS_output_beam, FLAGS_min_activate_states, FLAGS_max_activate_states,
      subsampling_factor);

  K2_LOG(INFO) << "Load n-gram LM: " << FLAGS_g;
  k2::FsaClass G = k2::LoadFsa(FLAGS_g, device);
  G.fsa = k2::FsaToFsaVec(G.fsa);

  K2_CHECK_EQ(G.NumAttrs(), 0) << "G is expected to be an acceptor.";
  k2::AddEpsilonSelfLoops(G.fsa, &G.fsa);
  k2::ArcSort(&G.fsa);
  G.SetTensorAttr("lm_scores", G.Scores().clone());

  K2_LOG(INFO) << "Rescore with an n-gram LM";
  WholeLatticeRescoring(G, /*ngram_lm_scale*/ 1, &lattice);

  K2_LOG(INFO) << "Sample " << FLAGS_num_paths << " paths";
  k2::Nbest nbest =
      k2::Nbest::FromLattice(lattice, FLAGS_num_paths, FLAGS_nbest_scale);
  // nbest.fsa.Scores() are all 0s at this point

  nbest.Intersect(&lattice);
  // Caution: lattice is changed inside nbest, we don't need it after
  // this line
  //
  // Now nbest.fsa has its scores set.
  // Also, nbest.fsa inherits the attributes from `lattice`.
  K2_CHECK(nbest.fsa.HasTensorAttr("lm_scores"));
  torch::Tensor am_scores = nbest.ComputeAmScores();
  torch::Tensor ngram_lm_scores = nbest.ComputeLmScores();

  K2_CHECK(nbest.fsa.HasTensorAttr("tokens"));

  auto &path_to_utt_map_array = nbest.shape.RowIds(1);
  torch::Tensor path_to_utt_map =
      Array1ToTorch(path_to_utt_map_array).to(torch::kLong);

  // the shape of memory is (T, N, C), so we use axis=1 here
  torch::Tensor expanded_memory = memory.index_select(1, path_to_utt_map);

  // the shape of memory_key_padding_mask is (N, T), so we use axis=0 here
  torch::Tensor expanded_memory_key_padding_mask =
      memory_key_padding_mask.index_select(0, path_to_utt_map);

  k2::RaggedShape tokens_shape = k2::RemoveAxis(nbest.fsa.fsa.shape, 1);

  torch::Tensor tokens_value = nbest.fsa.GetTensorAttr("tokens");
  k2::Ragged<int32_t> tokens{tokens_shape,
                             k2::Array1FromTorch<int32_t>(tokens_value)};
  tokens = k2::RemoveValuesLeq(tokens, 0);

  std::vector<std::vector<int32_t>> token_ids = tokens.ToVecVec();
  // convert std::vector<std::vector<int32_t>>
  // to
  // torch::List<torch::IValue> where torch::IValue is torch::Tensor
  torch::List<torch::IValue> token_ids_list(torch::TensorType::get());

  token_ids_list.reserve(token_ids.size());
  for (const auto tids : token_ids) {
    torch::Tensor tids_tensor = torch::tensor(tids);
    token_ids_list.emplace_back(tids_tensor);
  }

  K2_LOG(INFO) << "Run attention decoder";
  torch::Tensor nll =
      module
          .run_method("decoder_nll", expanded_memory,
                      expanded_memory_key_padding_mask, token_ids_list,
                      FLAGS_sos_id, FLAGS_eos_id)
          .toTensor();
  K2_CHECK_EQ(nll.dim(), 2);
  K2_CHECK_EQ(nll.size(0), nbest.shape.TotSize(1));

  K2_LOG(INFO) << "Rescoring";

  torch::Tensor attention_scores = -1 * nll.sum(1);

  torch::Tensor tot_scores = am_scores +
                             FLAGS_ngram_lm_scale * ngram_lm_scores +
                             FLAGS_attention_scale * attention_scores;
  k2::Array1<float> tot_scores_array = k2::Array1FromTorch<float>(tot_scores);
  k2::Ragged<float> ragged_tot_scores(nbest.shape, tot_scores_array);
  k2::Array1<int32_t> argmax(ragged_tot_scores.Context(),
                             ragged_tot_scores.Dim0());

  k2::ArgMaxPerSublist(ragged_tot_scores, std::numeric_limits<float>::lowest(),
                       &argmax);
  k2::Array1<int32_t> value_indexes_out;
  k2::Fsa best_paths =
      k2::Index<k2::Arc>(nbest.fsa.fsa, 0, argmax, &value_indexes_out);

  lattice = k2::FsaClass(best_paths);

  if (nbest.fsa.HasTensorAttr("aux_labels")) {
    torch::Tensor in_aux_labels_tensor = nbest.fsa.GetTensorAttr("aux_labels");

    k2::Array1<int32_t> in_aux_labels =
        k2::Array1FromTorch<int32_t>(in_aux_labels_tensor);

    k2::Array1<int32_t> out_aux_labels =
        k2::Index(in_aux_labels, value_indexes_out,
                  false,  // allow_minus_one
                  0);     // default_value

    lattice.SetTensorAttr("aux_labels", k2::Array1ToTorch(out_aux_labels));
  } else {
    K2_CHECK(nbest.fsa.HasRaggedTensorAttr("aux_labels"));
    k2::Ragged<int32_t> in_aux_labels =
        nbest.fsa.GetRaggedTensorAttr("aux_labels");

    k2::Ragged<int32_t> out_aux_labels =
        k2::Index(in_aux_labels, 0, value_indexes_out);

    lattice.SetRaggedTensorAttr("aux_labels", out_aux_labels);
  }

  auto ragged_aux_labels = k2::GetTexts(lattice);
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  std::vector<std::string> texts;
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
