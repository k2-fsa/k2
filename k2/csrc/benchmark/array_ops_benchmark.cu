/**
 * Copyright      2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdlib>

#include "k2/csrc/array_ops.h"
#include "k2/csrc/benchmark/benchmark.h"
#include "k2/csrc/test_utils.h"

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
static BenchmarkStat BenchmarkCat(int32_t num_array, DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  std::vector<Array1<T>> arrays_vec(num_array);
  std::vector<const Array1<T> *> arrays(num_array);
  int32_t total_size = 0, max_size = 0;
  // notice `j != num_array - 1` below, we may push a very long array
  // after the loop
  for (int32_t j = 0; j != num_array - 1; ++j) {
    int32_t curr_array_size = RandInt(0, 10000);
    std::vector<T> data(curr_array_size);
    std::iota(data.begin(), data.end(), total_size);
    total_size += curr_array_size;
    arrays_vec[j] = Array1<T>(context, data);
    arrays[j] = &arrays_vec[j];
    if (curr_array_size > max_size) max_size = curr_array_size;
  }
  {
    // below we may generate an array with very large size depend on the value
    // of RandInt(0,1)
    int32_t average_size = total_size / num_array;
    int32_t curr_array_size =
        RandInt(0, 1) == 0 ? RandInt(0, 10000) : average_size * 10;
    std::vector<T> data(curr_array_size);
    std::iota(data.begin(), data.end(), total_size);
    total_size += curr_array_size;
    arrays_vec[num_array - 1] = Array1<T>(context, data);
    arrays[num_array - 1] = &arrays_vec[num_array - 1];
    if (curr_array_size > max_size) max_size = curr_array_size;
  }
  bool is_balanced = (max_size < 2 * (total_size / num_array) + 512);
  const Array1<T> **src = arrays.data();

  BenchmarkStat stat;
  stat.op_name = "Cat_" + std::to_string(num_array) + "_" +
                 std::to_string(total_size) + "_" +
                 std::to_string(total_size / num_array) + "_" +
                 std::to_string(max_size) + "_" + std::to_string(is_balanced);
  int32_t num_iter = 20;
  stat.num_iter = num_iter;
  stat.problem_size = num_array;
  stat.dtype_name = TraitsOf(DtypeOf<T>::dtype).Name();
  stat.device_type = device_type;

  stat.eplased_per_iter = BenchmarkOp(
      num_iter, context,
      (Array1<T>(*)(ContextPtr, int32_t, const Array1<T> **))(&Cat<T>),
      context, num_array, src);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

static BenchmarkStat BenchmarkSpliceRowSplits(int32_t num_array,
                                              DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  std::vector<Array1<int32_t>> arrays_vec(num_array);
  std::vector<const Array1<int32_t> *> arrays(num_array);
  int32_t total_size = 0, max_size = 0, total_num_elem = 0;
  // notice `j != num_array - 1` below, we may push a very long array
  // after the loop
  for (int32_t j = 0; j != num_array - 1; ++j) {
    int32_t num_elements = RandInt(0, 10000);
    total_num_elem += num_elements;
    RaggedShape shape =
        RandomRaggedShape(false, 2, 2, num_elements, num_elements);
    Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
    int32_t array_size = row_splits.Dim();
    total_size += array_size;
    arrays_vec[j] = row_splits;
    arrays[j] = &arrays_vec[j];
    if (array_size > max_size) max_size = array_size;
  }
  {
    // below we may generate an array with very large size depend on the value
    // of RandInt(0,1)
    int32_t average_size = total_num_elem / num_array;
    int32_t num_elements =
        RandInt(0, 1) == 0 ? RandInt(0, 10000) : average_size * 10;
    RaggedShape shape =
        RandomRaggedShape(false, 2, 2, num_elements, num_elements);
    Array1<int32_t> row_splits = shape.RowSplits(1).To(context);
    int32_t array_size = row_splits.Dim();
    total_size += array_size;
    arrays_vec[num_array - 1] = row_splits;
    arrays[num_array - 1] = &arrays_vec[num_array - 1];
    if (array_size > max_size) max_size = array_size;
  }
  bool is_balanced = (max_size < 2 * (total_size / num_array) + 512);
  const Array1<int32_t> **src = arrays.data();

  BenchmarkStat stat;
  stat.op_name = "SpliceRowSplits_" + std::to_string(num_array) + "_" +
                 std::to_string(total_size) + "_" +
                 std::to_string(total_size / num_array) + "_" +
                 std::to_string(max_size) + "_" + std::to_string(is_balanced);
  int32_t num_iter = 20;
  stat.num_iter = num_iter;
  stat.problem_size = num_array;
  stat.device_type = device_type;

  stat.eplased_per_iter = BenchmarkOp(
      num_iter, context,
      (Array1<int32_t>(*)(int32_t, const Array1<int32_t> **))(&SpliceRowSplits),
      num_array, src);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

static BenchmarkStat BenchmarkSizesToMergeMap(int32_t num_src,
                                              DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  std::vector<int32_t> sizes(num_src);
  int32_t tot_size = 0;
  for (int32_t n = 0; n != num_src; ++n) {
    int32_t cur_size = RandInt(0, 1000);
    sizes[n] = cur_size;
    tot_size += cur_size;
  }

  BenchmarkStat stat;
  stat.op_name = "SizesToMergeMap_" + std::to_string(num_src) + "_" +
                 std::to_string(tot_size) + "_" +
                 std::to_string(tot_size / num_src);
  int32_t num_iter = 20;
  stat.num_iter = num_iter;
  stat.problem_size = num_src;
  stat.device_type = device_type;

  stat.eplased_per_iter = BenchmarkOp(
      num_iter, context,
      (Array1<uint32_t>(*)(ContextPtr, const std::vector<int32_t> &))(
          &SizesToMergeMap),
      context, sizes);
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

template <typename T>
static void RegisterBenchmarkCat(DeviceType device_type) {
  // problem_sizes here is the number of arrays to concatenate
  std::vector<int32_t> problems_sizes = {10,   50,   100,  200,  500,
                                         1000, 2000, 5000, 10000};
  for (auto s : problems_sizes) {
    std::string name = GenerateBenchmarkName<T>("Cat", device_type) + "_" +
                       std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkCat<T>(s, device_type);
    });
  }
}

static void RegisterBenchmarkSpliceRowSplits(DeviceType device_type) {
  // problem_sizes here is the number of arrays that we feed into
  // `SpliceRowSplits`
  std::vector<int32_t> problems_sizes = {10,   50,   100,  200,  500,
                                         1000, 2000, 5000, 10000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("SpliceRowSplits", device_type) + "_" +
        std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkSpliceRowSplits(s, device_type);
    });
  }
}

static void RegisterBenchmarkSizesToMergeMap(DeviceType device_type) {
  // problem_sizes here is the `sizes.size()` in
  // SizesToMergeMap(ContextPtr c, const std::vector<int32_t> sizes).
  std::vector<int32_t> problems_sizes = {3,   5,   10,   20,   50,   100,
                                         200, 500, 1000, 2000, 5000, 10000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("SizesToMergeMap", device_type) + "_" +
        std::to_string(s);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkSizesToMergeMap(s, device_type);
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

  RegisterBenchmarkCat<int32_t>(kCuda);

  RegisterBenchmarkSpliceRowSplits(kCuda);

  RegisterBenchmarkSizesToMergeMap(kCuda);

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
