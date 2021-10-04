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

int main(int argc, char *argv[]) {
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

  torch::IValue features = Load(FLAGS_feature_pt);
  std::vector<torch::IValue> inputs;
  inputs.emplace_back(std::move(features));

  // the output for module.forward() is a tuple of 3 tensors
  auto outputs = module.forward(inputs).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();
  auto memory = outputs->elements()[1].toTensor();
  assert(outputs->elements()[2].isNone() == true);

  std::cout << nnet_output.sum() << "\n";
  std::cout << nnet_output.mean() << "\n";
  std::cout << memory.sum() << "\n";
  std::cout << memory.mean() << "\n";

  // clang-format off
  // TODO(fangjun):
  // [x] Use faked data to test "module" and compare its outputs with that from Python
  // [ ] Use some third party library to read wave files (or write our own)
  // [ ] Use kaldifeat to compute features
  // [ ] Inplement CTC decoding
  // [ ] Inplement HLG decoding
  // clang-format on
}
