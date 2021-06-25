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

#include "k2/csrc/benchmark/benchmark.h"
#include "k2/csrc/tensor_ops.h"
#include "k2/csrc/test_utils.h"

namespace k2 {

template <typename T>
static BenchmarkStat BenchmarkIndexAdd1D(int32_t dim, DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(500, 10000000 / dim);
  int32_t src_dim = dim;
  int32_t dest_dim = src_dim / 10;
  bool allow_minus_one = RandInt(0, 100) & 1;

  Array1<T> src(context, src_dim);
  Array1<T> dest(context, dest_dim);
  Array1<int32_t> indexes =
      GenerateRandomIndexes(context, allow_minus_one, src_dim, dest_dim - 1);

  Tensor src_tensor = src.ToTensor();
  Tensor dest_tensor = dest.ToTensor();

  BenchmarkStat stat;
  stat.op_name = "IndexAdd1D";
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<T>::dtype).Name();
  stat.device_type = device_type;

  stat.eplased_per_iter = BenchmarkOp(
      num_iter, context,
      (void (*)(Tensor &, Array1<int32_t> &, bool, Tensor *))(&IndexAdd),
      src_tensor, indexes, allow_minus_one, &dest_tensor);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

template <typename T>
static void RegisterBenchmarkIndexAdd1D(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {10,   100,  500,   1000,
                                         2000, 5000, 10000, 50000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("IndexAdd1D", device_type);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkIndexAdd1D<T>(s, device_type);
    });
  }
}

static void RunTensorOpsBenchmark() {
  PrintEnvironmentInfo();

  RegisterBenchmarkIndexAdd1D<int32_t>(kCpu);
  RegisterBenchmarkIndexAdd1D<int32_t>(kCuda);

  RegisterBenchmarkIndexAdd1D<float>(kCpu);
  RegisterBenchmarkIndexAdd1D<float>(kCuda);

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
  k2::RunTensorOpsBenchmark();
  return 0;
}
