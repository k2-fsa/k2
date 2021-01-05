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
static BenchmarkStat BenchmarkExclusiveSum(int32_t dim,
                                           DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(500, 1000000 / dim);
  Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000, GetSeed());

  BenchmarkStat stat;
  stat.op_name = "ExclusiveSum";
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<T>::dtype).Name();
  stat.device_type = device_type;

  // there are overloads of ExclusiveSum, so we use an explicit conversion here.
  stat.eplased_per_iter =
      BenchmarkOp(num_iter, context,
                  (Array1<T>(*)(const Array1<T> &))(&ExclusiveSum<T>), src);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

static BenchmarkStat BenchmarkRowSplitsToRowIds(int32_t dim,
                                                DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(500, 1000000 / dim);
  Array1<int32_t> sizes =
      RandUniformArray1<int32_t>(context, dim, 0, 1000, GetSeed());
  Array1<int32_t> row_splits = ExclusiveSum(sizes);
  Array1<int32_t> row_ids(context, row_splits.Back());

  BenchmarkStat stat;
  stat.op_name = "RowSplitsToRowIds_" + std::to_string(row_ids.Dim());
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<int32_t>::dtype).Name();
  stat.device_type = device_type;

  // there are overloads of RowSplitsToRowIds,
  // so we use an explicit conversion here.
  stat.eplased_per_iter =
      BenchmarkOp(num_iter, context,
                  (void (*)(const Array1<int32_t> &, Array1<int32_t> *))(
                      &RowSplitsToRowIds),
                  row_splits, &row_ids);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

static BenchmarkStat BenchmarkRowIdsToRowSplits(int32_t dim,
                                                DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(500, 1000000 / dim);
  Array1<int32_t> sizes =
      RandUniformArray1<int32_t>(context, dim, 0, 1000, GetSeed());
  Array1<int32_t> row_splits = ExclusiveSum(sizes);
  Array1<int32_t> row_ids(context, row_splits.Back());
  RowSplitsToRowIds(row_splits, &row_ids);

  BenchmarkStat stat;
  stat.op_name = "RowIdsToRowSplits";
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<int32_t>::dtype).Name();
  stat.device_type = device_type;

  // there are overloads of RowIdsToRowSplits,
  // so we use an explicit conversion here.
  stat.eplased_per_iter =
      BenchmarkOp(num_iter, context,
                  (void (*)(const Array1<int32_t> &, Array1<int32_t> *))(
                      &RowIdsToRowSplits),
                  row_ids, &row_splits);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

template <typename T>
static void RegisterBenchmarkExclusiveSum(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {100,  500,   1000,  2000,
                                         5000, 10000, 100000};
  for (auto s : problems_sizes) {
    std::string name = GenerateBenchmarkName<T>("ExclusiveSum", device_type) +
                       "_" + std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkExclusiveSum<T>(s, device_type);
    });
  }
}

static void RegisterBenchmarkRowSplitsToRowIds(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {100,  500,   1000,  2000,
                                         5000, 10000, 100000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("RowSplitsToRowIds", device_type) + "_" +
        std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkRowSplitsToRowIds(s, device_type);
    });
  }
}

static void RegisterBenchmarkRowIdsToRowSplits(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {100,  500,   1000,  2000,
                                         5000, 10000, 100000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("RowIdsToRowSplits", device_type) + "_" +
        std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkRowIdsToRowSplits(s, device_type);
    });
  }
}

static void RunArrayOpsBenchmark() {
  PrintEnvironmentInfo();

  RegisterBenchmarkExclusiveSum<int32_t>(kCpu);
  RegisterBenchmarkExclusiveSum<int32_t>(kCuda);

  RegisterBenchmarkRowSplitsToRowIds(kCpu);
  RegisterBenchmarkRowSplitsToRowIds(kCuda);

  RegisterBenchmarkRowIdsToRowSplits(kCpu);
  RegisterBenchmarkRowIdsToRowSplits(kCuda);

  // Users can set a regular expression via environment
  // variable `K2_BENCHMARK_FILTER` such that only benchmarks
  // with name matching the pattern are candidates to run.
  const char *filter = std::getenv("K2_BENCHMARK_FILTER");
  if (filter != nullptr) FilterRegisteredBenchmarks(filter);

  std::vector<BenchmarkRun> results = RunBechmarks();
  std::cout << BenchmarkRun::GetFieldsName() << "\n";
  for (const auto &r : results) {
    std::cout << r << "\n";
  }
}

}  // namespace k2

int main() {
  k2::RunArrayOpsBenchmark();
  return 0;
}
