/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "benchmark/benchmark.h"
#include "k2/csrc/array_ops.h"
#include "k2/csrc/timer.h"

namespace k2 {

void SetProblemSizes(const std::vector<int32_t> &sizes,
                     benchmark::internal::Benchmark *benchmark) {
  for (auto s : sizes) benchmark->Arg(s);
}

template <typename T>
static void BM_ExclusiveSum(benchmark::State &state) {
  ContextPtr context = GetCudaContext();

  for (int32_t i = 0; i != 10; ++i) {  // warm up
    Array1<T> src = RandUniformArray1<T>(context, 100, -1000, 1000);
    Array1<T> dst = ExclusiveSum(src);
  }

  Timer timer(context);
  int32_t offset = RandInt(-3, 3);
  while (offset == 0) offset = RandInt(-2, 2);

  int32_t dim = state.range(0) + offset;
  for (auto _ : state) {
    Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);
    timer.Reset();
    Array1<T> dst = ExclusiveSum(src);
    state.SetIterationTime(timer.Elapsed());  // in seconds
  }
  state.SetLabel(std::to_string(offset));
}

static void RegisterBenchmarks() {
  const int32_t kNumIterations = 1e4;
  {
    benchmark::internal::Benchmark *b = benchmark::RegisterBenchmark(
        "ExclusiveSum_int32", BM_ExclusiveSum<int32_t>);
    b->Iterations(kNumIterations)
        ->RangeMultiplier(10)
        ->Range(10, 10 << 10)
        ->Unit(benchmark::kMillisecond);
  }

  {
    benchmark::internal::Benchmark *b = benchmark::RegisterBenchmark(
        "ExclusiveSum_float", BM_ExclusiveSum<float>);
    b->Iterations(kNumIterations)
        ->RangeMultiplier(10)
        ->Range(10, 10 << 10)
        ->Unit(benchmark::kMillisecond);
  }
}

}  // namespace k2

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return EXIT_FAILURE;
  k2::RegisterBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
}
