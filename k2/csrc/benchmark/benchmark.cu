/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <sstream>
#include <string>

#include "k2/csrc/benchmark/benchmark.h"

namespace k2 {

std::string BenchmarkRun::ToString() const {
  std::ostringstream os;
  os << name << "," << stat.num_iter << "," << std::fixed
     << stat.eplased_per_iter;
  return os.str();
}

std::vector<std::unique_ptr<BenchmarkInstance>> *GetRegisteredBenchmarks() {
  static std::vector<std::unique_ptr<BenchmarkInstance>> instances;
  return &instances;
}

void RegisterBenchmark(const std::string &name, BenchmarkFunc func) {
  auto benchmark_inst = std::make_unique<BenchmarkInstance>(name, func);
  GetRegisteredBenchmarks()->emplace_back(std::move(benchmark_inst));
}

void FilterRegisteredBenchmarks(const std::string &regex) {
  (void)regex;
  K2_LOG(INFO) << "Not implemented";
}

}  // namespace k2
