#include <cassert>

#include "torch/script.h"

C10_DEFINE_string(jit_pt, "", "Path to exported jit filename.");
// TODO: add more options

int main(int argc, char *argv[]) {
  std::string usage = R"(
    ./bin/decode.py --jit_pt <path to exported torch script pt file>
  )";
  torch::SetUsageMessage(usage);

  torch::ParseCommandLineFlags(&argc, &argv);
  if (FLAGS_jit_pt.empty()) {
    std::cout << "Please provide --jit_pt"
              << "\n";
    std::cout << torch::UsageMessage() << "\n";
    return -1;
  }

  std::cout << "jit pt filename: " << FLAGS_jit_pt << "\n";

  torch::jit::script::Module module;
  module = torch::jit::load(FLAGS_jit_pt);
  module.eval();
  torch::Tensor x = torch::ones({1, 100, 80}, torch::kFloat32);

  std::vector<torch::jit::IValue> inputs;
  inputs.emplace_back(std::move(x));

  // the output for module.forward() is a tuple of 3 tensors
  auto outputs = module.forward(inputs).toTuple();
  assert(outputs->elements().size() == 3u);

  auto nnet_output = outputs->elements()[0].toTensor();
  auto memory = outputs->elements()[1].toTensor();
  assert(outputs->elements()[2].isNone() == true);

  std::cout << nnet_output.sum() << "\n";
  std::cout << memory.sum() << "\n";

  // clang-format off
  // TODO(fangjun):
  // [x] Use faked data to test "module" and compare its outputs with that from Python
  // [ ] Use some third party library to read wave files (or write our own)
  // [ ] Use kaldifeat to compute features
  // [ ] Inplement CTC decoding
  // [ ] Inplement HLG decoding
  // clang-format on
}
