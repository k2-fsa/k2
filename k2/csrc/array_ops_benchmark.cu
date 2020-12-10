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

  for (int32_t i = 0; i != 3; ++i) {  // warm up
    Array1<T> src = RandUniformArray1<T>(context, 100, -1000, 1000);
    Array1<T> dst = ExclusiveSum(src);
  }

  Timer timer(context);
  for (auto _ : state) {
    int32_t dim = state.range(0);
    Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);
    timer.Reset();
    Array1<T> dst = ExclusiveSum(src);
    state.SetIterationTime(timer.Elapsed());  // in seconds
  }
}

static void RegisterBenchmarks() {
  {
    benchmark::internal::Benchmark *b = benchmark::RegisterBenchmark(
        "ExclusiveSum_int32", BM_ExclusiveSum<int32_t>);
    SetProblemSizes({100, 500, 1000, 2000, 5000, 10000}, b);
    b->Unit(benchmark::kMillisecond);
  }

  {
    benchmark::internal::Benchmark *b = benchmark::RegisterBenchmark(
        "ExclusiveSum_float", BM_ExclusiveSum<float>);

    SetProblemSizes({100, 500, 1000, 2000, 5000, 10000}, b);
    // b->Unit(benchmark::kMillisecond);
  }
}

}  // namespace k2

int main(int argc, char **argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) return EXIT_FAILURE;
  k2::RegisterBenchmarks();
  benchmark::RunSpecifiedBenchmarks();
}
