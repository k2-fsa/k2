#include <cassert>
#include <cstdio>

#include "torch/script.h"
#include "torch/utils.h"

C10_DEFINE_string(jit_pt, "", "Path to exported jit filename.");
C10_DEFINE_string(feature_pt, "", "Path to pre-computed feature filename.");
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
// DenseFsaVec
static torch::Tensor GetSupervisionSegments(torch::IValue supervisions,
                                            int32_t subsampling_factor) {
  torch::Dict<torch::IValue, torch::IValue> dict = supervisions.toGenericDict();
  torch::Tensor sequence_idx = dict.at("sequence_idx").toTensor();
  torch::Tensor start_frame = torch::floor_divide(
      dict.at("start_frame").toTensor(), subsampling_factor);

  torch::Tensor num_frames =
      torch::floor_divide(dict.at("num_frames").toTensor(), subsampling_factor);

  torch::Tensor supervision_segments =
      torch::stack({sequence_idx, start_frame, num_frames}, 1);
  std::cout << "supervision_segments: " << supervision_segments << "\n";
  return supervision_segments;
}

int main(int argc, char *argv[]) {
  // see
  // https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html
  at::set_num_threads(1);
  at::set_num_interop_threads(1);

  std::string usage = R"(
    ./bin/decode.py \
      --jit_pt <path to exported torch script pt file> \
      --feature_pt <path to precomputed feature pt file>
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

  std::cout << nnet_output.sum() << "\n";
  std::cout << nnet_output.mean() << "\n";
  std::cout << nnet_output.sizes() << "\n";
  std::cout << memory.sum() << "\n";
  std::cout << memory.mean() << "\n";
  std::cout << memory_key_padding_mask.sum() << "\n";

  // clang-format off
  // TODO(fangjun):
  // [x] Use faked data to test "module" and compare its outputs with that from Python
  // [ ] Use some third party library to read wave files (or write our own)
  // [ ] Use kaldifeat to compute features
  // [ ] Inplement CTC decoding
  // [ ] Inplement HLG decoding
  // clang-format on
}
