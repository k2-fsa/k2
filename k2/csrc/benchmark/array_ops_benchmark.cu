/**
 * @brief Benchmarks for array_ops.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cstdlib>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/benchmark/benchmark.h"

namespace k2 {

template <typename T>
static BenchmarkStat BenchmarkExclusiveSum(int32_t dim) {
  ContextPtr context = GetCudaContext();
  int32_t num_iter = std::min(500, 1000000 / dim);
  Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);

  BenchmarkStat stat;
  stat.num_iter = num_iter;

  // there are overloads of ExclusiveSum, so we use an explicit conversion here.
  stat.eplased_per_iter =
      BenchmarkOp(num_iter, context,
                  (Array1<T>(*)(const Array1<T> &))(&ExclusiveSum<T>), src);
  return stat;
}

static void RegisterBenchmarkExclusiveSum() {
  std::vector<int32_t> problems_sizes = {100,  500,   1000,  2000,
                                         5000, 10000, 100000};
  for (auto s : problems_sizes) {
    std::string name = std::string("ExclusiveSum_") + std::to_string(s);
    RegisterBenchmark(name, [s]() -> BenchmarkStat {
      return BenchmarkExclusiveSum<int32_t>(s);
    });
  }
}

}  // namespace k2

int main() {
  std::cout << k2::GetCurrentDateTime() << "\n";
  std::cout << k2::GetDeviceInfo() << "\n";

  k2::RegisterBenchmarkExclusiveSum();

  // Users can set a regular expression via environment
  // variable `K2_BENCHMARK_FILTER` such that only benchmarks
  // with name matching the pattern are candidates to run.
  const char *filter = std::getenv("K2_BENCHMARK_FILTER");
  if (filter != nullptr) k2::FilterRegisteredBenchmarks(filter);

  std::vector<k2::BenchmarkRun> results = k2::RunBechmarks();
  for (const auto &r : results) {
    std::cout << r << "\n";
  }
  return 0;
}
