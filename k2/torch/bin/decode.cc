#include <cassert>
#include <cstdio>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/utils.h"
#include "sentencepiece_processor.h"
#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_string(jit_pt, "", "Path to exported jit filename.");
C10_DEFINE_string(feature_pt, "", "Path to pre-computed feature filename.");
C10_DEFINE_string(bpe_model, "", "Path to a pretrained BPE model.");
// TODO: add more options

// Load a file saved by torch.save(x, filename), where
// x can be
//   - a tensor
//   - a tuple of tensors
//   - a list of ints
//   - a tuple containing: a tensor, a list of ints, etc.
static torch::IValue Load(const std::string &filename) {
  FILE *fp = fopen(filename.c_str(), "rb");
  assert(fp != nullptr);

  fseek(fp, 0, SEEK_END);
  size_t size = ftell(fp);
  fseek(fp, 0, SEEK_SET);

  std::vector<char> f(size);
  size_t len = fread(f.data(), 1, size, fp);
  assert(len == size);

  fclose(fp);

  return torch::jit::pickle_load(f);
}

// See
// https://github.com/lhotse-speech/lhotse/blob/master/lhotse/dataset/speech_recognition.py#L32
// for the format of "supervisions"
//
// @param supervisions A dict containing keys and values shown in the following:
//                     - sequence_idx: torch.Tensor
//                     - start_frame: torch.Tensor
//                     - num_frames: torch.Tensor
// @return Return a 2-D torch.int32 tensor that can be used to construct a
//  DenseFsaVec. See `k2::CreateDenseFsaVec()`
static torch::Tensor GetSupervisionSegments(torch::IValue supervisions,
                                            int32_t subsampling_factor) {
  torch::Dict<torch::IValue, torch::IValue> dict = supervisions.toGenericDict();
  torch::Tensor sequence_idx = dict.at("sequence_idx").toTensor();
  torch::Tensor start_frame = torch::floor_divide(
      dict.at("start_frame").toTensor(), subsampling_factor);

  torch::Tensor num_frames =
      torch::floor_divide(dict.at("num_frames").toTensor(), subsampling_factor);

  torch::Tensor supervision_segments =
      torch::stack({sequence_idx, start_frame, num_frames}, 1).to(torch::kCPU);
  return supervision_segments;
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  torch::set_num_threads(1);
  torch::set_num_interop_threads(1);

  std::string usage = R"(
    ./bin/decode \
      --jit_pt <path to exported torch script pt file> \
      --feature_pt <path to precomputed feature pt file> \
      --bpe_model <path to pretrained BPE model>
  )";
  torch::SetUsageMessage(usage);

  torch::ParseCommandLineFlags(&argc, &argv);
  if (FLAGS_jit_pt.empty()) {
    std::cout << "Please provide --jit_pt"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  if (FLAGS_feature_pt.empty()) {
    std::cout << "Please provide --feature_pt"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  if (FLAGS_bpe_model.empty()) {
    std::cout << "Please provide --bpe_model"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  torch::jit::script::Module module;
  module = torch::jit::load(FLAGS_jit_pt);
  module.eval();

  int32_t subsampling_factor = module.attr("subsampling_factor").toInt();

  torch::IValue features = Load(FLAGS_feature_pt);
  assert(features.isTensor() == true);

  torch::Dict<std::string, torch::Tensor> sup;
  {
    torch::IValue _sup = Load("sup.pt");
    assert(_sup.isGenericDict() == true);

    torch::Dict<torch::IValue, torch::IValue> dict = _sup.toGenericDict();
    sup.insert("sequence_idx", dict.at("sequence_idx").toTensor());
    sup.insert("start_frame", dict.at("start_frame").toTensor());
    sup.insert("num_frames", dict.at("num_frames").toTensor());
  }

  torch::IValue supervisions(sup);

  std::vector<torch::IValue> inputs;
  inputs.emplace_back(std::move(features));
  inputs.emplace_back(supervisions);

  // the output for module.forward() is a tuple of 3 tensors
  auto outputs = module.forward(inputs).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();
  auto memory = outputs->elements()[1].toTensor();
  auto memory_key_padding_mask = outputs->elements()[2].toTensor();

  torch::Tensor supervision_segments =
      GetSupervisionSegments(supervisions, subsampling_factor);

  k2::DenseFsaVec dense_fsa_vec = k2::CreateDenseFsaVec(
      nnet_output, supervision_segments, subsampling_factor - 1);

  k2::ContextPtr ctx = k2::ContextFromTensor(nnet_output);

  // We first try CTC decoding
  k2::Array1<int32_t> aux_labels;
  k2::Fsa ctc_topo = k2::CtcTopo(ctx, nnet_output.size(2) - 1,
                                 /*modified*/ false, &aux_labels);
  ctc_topo = k2::FsaToFsaVec(ctc_topo);

  float search_beam = 20;
  float output_beam = 8;
  int32_t min_activate_states = 30;
  int32_t max_activate_states = 10000;
  k2::FsaVec lattice;
  k2::Array1<int32_t> arc_map_a;
  k2::Array1<int32_t> arc_map_b;
  k2::IntersectDensePruned(ctc_topo, dense_fsa_vec, search_beam, output_beam,
                           min_activate_states, max_activate_states, &lattice,
                           &arc_map_a, &arc_map_b);
  // see Index() in array_ops.h
  aux_labels = k2::Index(aux_labels, arc_map_a, /*allow_minus_one*/ false,
                         /*default_value*/ 0);

  k2::Ragged<int32_t> state_batches = k2::GetStateBatches(lattice, true);
  k2::Array1<int32_t> dest_states = k2::GetDestStates(lattice, true);
  k2::Ragged<int32_t> incoming_arcs = k2::GetIncomingArcs(lattice, dest_states);
  k2::Ragged<int32_t> entering_arc_batches =
      k2::GetEnteringArcIndexBatches(lattice, incoming_arcs, state_batches);

  bool log_semiring = false;
  k2::Array1<int32_t> entering_arcs;
  k2::GetForwardScores<float>(lattice, state_batches, entering_arc_batches,
                              log_semiring, &entering_arcs);

  k2::Ragged<int32_t> best_path_arc_indexes =
      k2::ShortestPath(lattice, entering_arcs);

  // See Index() in ragged_ops.h
  k2::Ragged<int32_t> ragged_aux_labels =
      k2::Index(aux_labels, best_path_arc_indexes);
  ragged_aux_labels = k2::RemoveValuesLeq(ragged_aux_labels, 0);

  // TODO: convert a Ragged<int32_t> to a std::vector<std::vector<int32_t>>

  std::vector<int32_t> aux_labels_vec(
      ragged_aux_labels.values.Data(),
      ragged_aux_labels.values.Data() + ragged_aux_labels.values.Dim());

  sentencepiece::SentencePieceProcessor processor;
  const auto status = processor.Load(FLAGS_bpe_model);
  if (!status.ok()) {
    K2_LOG(FATAL) << status.ToString();
  }
  std::string text;
  processor.Decode(aux_labels_vec, &text);
  // NOTE: text contains the concatenated transcripts from all utterances.
  K2_LOG(INFO) << text;

  // TODO:
  // Get aux_labels from the lattice and use sentence piece APIs to turn them
  // into words

  // clang-format off
  // TODO(fangjun):
  // [x] Use faked data to test "module" and compare its outputs with that from Python
  // [ ] Use some third party library to read wave files (or write our own)
  // [ ] Use kaldifeat to compute features
  // [ ] Inplement CTC decoding
  // [ ] Inplement HLG decoding
  // clang-format on
}
