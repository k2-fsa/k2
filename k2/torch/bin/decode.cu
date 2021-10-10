#include <cassert>
#include <cstdio>
#include <string>
#include <vector>

#include "k2/csrc/array.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/fsa_algo.h"
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged.h"
#include "k2/csrc/ragged_ops.h"
#include "k2/torch/csrc/dense_fsa_vec.h"
#include "k2/torch/csrc/deserialization.h"
#include "k2/torch/csrc/symbol_table.h"
#include "k2/torch/csrc/utils.h"
#include "sentencepiece_processor.h"
#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_string(jit_pt, "", "Path to exported jit filename.");
C10_DEFINE_string(feature_pt, "", "Path to pre-computed feature filename.");
C10_DEFINE_string(
    bpe_model, "",
    "Path to a pretrained BPE model. Needed if --use_ctc_decoding is true");
C10_DEFINE_bool(use_ctc_decoding, true, "True to use CTC decoding");
C10_DEFINE_string(hlg, "",
                  "Path to HLG.pt. Needed if --use_ctc_decoding is false");
C10_DEFINE_string(word_table, "",
                  "Path to wors.txt. Needed if --use_ctc_decoding is false");
//
C10_DEFINE_double(search_beam, 20, "search_beam in IntersectDensePruned");
C10_DEFINE_double(output_beam, 8, "output_beam in IntersectDensePruned");
C10_DEFINE_int(min_activate_states, 30,
               "min_activate_states in IntersectDensePruned");
C10_DEFINE_int(max_activate_states, 10000,
               "max_activate_states in IntersectDensePruned");
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

static k2::FsaVec GetLattice(
    torch::Tensor nnet_output, k2::FsaVec decoding_graph,
    torch::Tensor supervision_segments, float search_beam, float output_beam,
    int32_t min_activate_states, int32_t max_activate_states,
    int32_t subsampling_factor, k2::Array1<int32_t> &in_aux_labels,
    k2::Ragged<int32_t> &in_ragged_aux_labels,
    k2::Array1<int32_t> *out_aux_labels,
    k2::Ragged<int32_t> *out_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

  k2::DenseFsaVec dense_fsa_vec = k2::CreateDenseFsaVec(
      nnet_output, supervision_segments, subsampling_factor - 1);

  k2::FsaVec lattice;
  k2::Array1<int32_t> arc_map_a;
  k2::Array1<int32_t> arc_map_b;
  k2::IntersectDensePruned(decoding_graph, dense_fsa_vec, search_beam,
                           output_beam, min_activate_states,
                           max_activate_states, &lattice, &arc_map_a,
                           &arc_map_b);
  if (in_aux_labels.Dim() > 0) {
    // see Index() in array_ops.h
    *out_aux_labels =
        k2::Index(in_aux_labels, arc_map_a, /*allow_minus_one*/ false,
                  /*default_value*/ 0);
  } else {
    // See Index() in ragged_ops.h
    *out_ragged_aux_labels =
        k2::Index(in_ragged_aux_labels, /*axis*/ 0, arc_map_a);
  }
  return lattice;
}

static k2::FsaVec OneBestDecoding(k2::FsaVec &lattice,
                                  k2::Array1<int32_t> &in_aux_labels,
                                  k2::Ragged<int32_t> &in_ragged_aux_labels,
                                  k2::Array1<int32_t> *out_aux_labels,
                                  k2::Ragged<int32_t> *out_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

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

  if (in_aux_labels.Dim() > 0) {
    *out_aux_labels = k2::Index(in_aux_labels, best_path_arc_indexes.values,
                                /*allow_minus_one*/ false,
                                /*default_value*/ 0);
  } else {
    *out_ragged_aux_labels = k2::Index(in_ragged_aux_labels, /*axis*/ 0,
                                       best_path_arc_indexes.values);
  }

  k2::FsaVec out = k2::FsaVecFromArcIndexes(lattice, best_path_arc_indexes);
  return out;
}

static k2::Ragged<int32_t> GetTexts(k2::FsaVec &lattice,
                                    k2::Array1<int32_t> &in_aux_labels,
                                    k2::Ragged<int32_t> &in_ragged_aux_labels) {
  if (in_aux_labels.Dim() != 0) {
    K2_CHECK_EQ(in_ragged_aux_labels.values.Dim(), 0);
  } else {
    K2_CHECK_NE(in_ragged_aux_labels.values.Dim(), 0);
  }

  k2::Ragged<int32_t> ragged_aux_labels;
  if (in_aux_labels.Dim() != 0) {
    // [utt][state][arc] -> [utt][arc]
    k2::RaggedShape aux_labels_shape = k2::RemoveAxis(lattice.shape, 1);
    ragged_aux_labels = k2::Ragged<int32_t>(aux_labels_shape, in_aux_labels);
  } else {
    k2::RaggedShape aux_labels_shape =
        k2::ComposeRaggedShapes(lattice.shape, in_ragged_aux_labels.shape);
    aux_labels_shape = k2::RemoveAxis(aux_labels_shape, 1);
    aux_labels_shape = k2::RemoveAxis(aux_labels_shape, 1);
    ragged_aux_labels =
        k2::Ragged<int32_t>(aux_labels_shape, in_ragged_aux_labels.values);
  }
  ragged_aux_labels = k2::RemoveValuesLeq(ragged_aux_labels, 0);
  return ragged_aux_labels;
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
      --use_ctc_decoding <true|false> \
      --bpe_model <path to pretrained BPE model> \
      --hlg <path to HLG.pt> \
      --word_tabel <path to words.txt>
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

  if (FLAGS_use_ctc_decoding && FLAGS_bpe_model.empty()) {
    std::cout << "Please provide --bpe_model"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_hlg.empty()) {
    std::cout << "Please provide --hlg"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  if (FLAGS_use_ctc_decoding == false && FLAGS_word_table.empty()) {
    std::cout << "Please provide --word_table"
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

  k2::ContextPtr ctx = k2::ContextFromTensor(nnet_output);

  k2::Fsa decoding_graph;

  k2::Array1<int32_t> aux_labels;  // only one of the two aux_labels is used
  k2::Ragged<int32_t> ragged_aux_labels;

  if (FLAGS_use_ctc_decoding) {
    k2::Fsa ctc_topo = k2::CtcTopo(ctx, nnet_output.size(2) - 1,
                                   /*modified*/ false, &aux_labels);
    decoding_graph = k2::FsaToFsaVec(ctc_topo);
  } else {
    // TODO(fangjun): We will eventually use an FSA wrapper to
    // associate attributes with an FSA.
    decoding_graph = k2::LoadFsa(FLAGS_hlg, &ragged_aux_labels);
  }
  k2::FsaVec lattice = GetLattice(
      nnet_output, decoding_graph, supervision_segments, FLAGS_search_beam,
      FLAGS_output_beam, FLAGS_min_activate_states, FLAGS_max_activate_states,
      subsampling_factor, aux_labels, ragged_aux_labels, &aux_labels,
      &ragged_aux_labels);

  lattice = OneBestDecoding(lattice, aux_labels, ragged_aux_labels, &aux_labels,
                            &ragged_aux_labels);

  ragged_aux_labels = GetTexts(lattice, aux_labels, ragged_aux_labels);
  auto aux_labels_vec = ragged_aux_labels.ToVecVec();

  std::vector<std::string> texts;
  if (FLAGS_use_ctc_decoding) {
    sentencepiece::SentencePieceProcessor processor;
    const auto status = processor.Load(FLAGS_bpe_model);
    if (!status.ok()) {
      K2_LOG(FATAL) << status.ToString();
    }
    for (const auto &ids : aux_labels_vec) {
      std::string text;
      processor.Decode(ids, &text);
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

  for (const auto &text : texts) {
    K2_LOG(INFO) << text;
  }

  // clang-format off
  // TODO(fangjun):
  // [x] Use faked data to test "module" and compare its outputs with that from Python
  // [ ] Use some third party library to read wave files (or write our own)
  // [ ] Use kaldifeat to compute features
  // [ ] Inplement CTC decoding
  // [ ] Inplement HLG decoding
  // clang-format on
}
