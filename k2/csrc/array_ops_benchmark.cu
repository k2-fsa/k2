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

template <typename T>
static void BM_ExclusiveSum(benchmark::State &state) {
  using namespace k2;  // TODO(fangjun): get rid of `using`  // NOLINT
  ContextPtr context = GetCudaContext();
  Timer timer(context);
  int32_t num_bytes_processed = 0;
  for (auto _ : state) {
    int32_t dim = state.range(0);
    Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);
    timer.Reset();
    Array1<T> dst = ExclusiveSum(src);
    state.SetIterationTime(timer.Elapsed());  // in seconds
    num_bytes_processed += dim * sizeof(T);
  }
  state.SetBytesProcessed(num_bytes_processed);
}

BENCHMARK_TEMPLATE(BM_ExclusiveSum, int32_t)
    ->Range(8, 8 << 10)
    ->UseManualTime();

BENCHMARK_MAIN();
