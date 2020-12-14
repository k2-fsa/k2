/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */
#include <regex>
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

std::vector<BenchmarkRun> RunBechmarks() {
  auto &registered_benchmarks = *GetRegisteredBenchmarks();
  std::vector<BenchmarkRun> results;
  for (const auto &b : registered_benchmarks) {
    BenchmarkRun run;
    run.name = b->name;
    run.stat = b->func();
    results.push_back(run);
  }
  return results;
}

void FilterRegisteredBenchmarks(const std::string &pattern) {
  std::regex regex(pattern);
  std::smatch match;
  auto &benchmarks = *GetRegisteredBenchmarks();

  std::vector<std::unique_ptr<BenchmarkInstance>> kept;
  for (auto &b : benchmarks) {
    if (std::regex_search(b->name, match, regex)) {
      kept.emplace_back(std::move(b));
    }
  }
  std::swap(kept, benchmarks);
}

}  // namespace k2
