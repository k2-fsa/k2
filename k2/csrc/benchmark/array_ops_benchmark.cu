/**
 * @brief Benchmarks for array_ops.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/array_ops.h"
#include "k2/csrc/benchmark/benchmark.h"

namespace k2 {

template <typename T>
static BenchmarkRun BenchmarkExclusiveSum(int32_t dim) {
  ContextPtr context = GetCudaContext();
  int32_t num_iter = std::min(500, 1000000 / dim);
  Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);

  BenchmarkRun run;
  run.name = std::string("ExclusiveSum_") + std::to_string(dim);
  run.num_iter = num_iter;

  // there are overloads of ExclusiveSum, so we use an explicit conversion here.
  run.eplased_per_iter =
      BenchmarkOp(num_iter, context,
                  (Array1<T>(*)(const Array1<T> &))(&ExclusiveSum<T>), src);
  return run;
}

static void RunBenchmarks() {
  std::vector<int32_t> problems_sizes = {100, 500, 1000, 2000, 5000, 10000};
  std::vector<BenchmarkRun> results;
  for (auto s : problems_sizes) {
    auto r = BenchmarkExclusiveSum<int32_t>(s);
    results.push_back(r);
  }
  for (const auto &r : results) std::cout << r.ToString() << "\n";
}

}  // namespace k2

int main() {
  k2::RunBenchmarks();
  return 0;
}
