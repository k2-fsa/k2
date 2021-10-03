#include "torch/script.h"

C10_DEFINE_string(jit_pt, "", "Path to exported jit filename.");
// TODO: add more options

int main(int argc, char *argv[]) {
  std::string usage = R"(
    ./bin/decode.py --jit_pt <path to exported torch script pt file>
  )";
  torch::SetUsageMessage(usage);

  torch::ParseCommandLineFlags(&argc, &argv);
  std::cout << "jit pt filename: " << FLAGS_jit_pt << "\n";

  torch::jit::script::Module module;
  module = torch::jit::load(FLAGS_jit_pt);
  module.eval();
  std::cout << "is training: " << module.is_training() << "\n";

  // clang-format off
  // TODO(fangjun):
  // (1) Use faked data to test "module" and compare it outputs with that from Python
  // (2) Use some third party library to read wave files (or write our own)
  // (3) Use kaldifeat to compute features
  // clang-format on
}
