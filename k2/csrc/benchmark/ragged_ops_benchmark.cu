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
#include "k2/csrc/fsa_utils.h"
#include "k2/csrc/ragged_ops.h"

namespace k2 {

static BenchmarkStat BenchmarkGetTransposeReordering(int32_t dim,
                                                     DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(100, 10000 / dim);
  int32_t min_num_fsas = dim;
  int32_t max_num_fsas = dim * 2;
  bool acyclic = false;
  int32_t max_symbol = 100;
  int32_t min_num_arcs = min_num_fsas * 10;
  int32_t max_num_arcs = max_num_fsas * 20;

  FsaVec fsas = RandomFsaVec(min_num_fsas, max_num_fsas, acyclic, max_symbol,
                             min_num_arcs, max_num_arcs);
  fsas = fsas.To(context);

  Array1<int32_t> dest_states = GetDestStates(fsas, true);
  Ragged<int32_t> dest_states_tensor(fsas.shape, dest_states);
  int32_t num_fsas = fsas.TotSize(0);
  int32_t num_states = fsas.TotSize(1);
  int32_t num_arcs = fsas.TotSize(2);

  BenchmarkStat stat;
  stat.op_name = "GetTransposeReordering_" + std::to_string(num_fsas) + "_" +
                 std::to_string(num_states) + "_" + std::to_string(num_arcs);
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<int32_t>::dtype).Name();
  stat.device_type = device_type;

  stat.eplased_per_iter =
      BenchmarkOp(num_iter, context, &GetTransposeReordering,
                  dest_states_tensor, num_states);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

static void RegisterBenchmarkGetTransposeReordering(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {10, 20, 30, 50, 100, 200, 300, 500};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<int32_t>("GetTransposeReordering", device_type);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkGetTransposeReordering(s, device_type);
    });
  }
}

template <typename T>
static BenchmarkStat BenchmarkSegmentedExclusiveSum(int32_t dim,
                                                    DeviceType device_type) {
  ContextPtr context;
  if (device_type == kCpu) {
    context = GetCpuContext();
  } else {
    K2_CHECK_EQ(device_type, kCuda);
    context = GetCudaContext();
  }

  int32_t num_iter = std::min(100, 10000 / dim);
  int32_t min_num_elems = dim * 10;
  int32_t max_num_elems = dim * 20;

  Ragged<T> ragged =
      RandomRagged<T>(0, 100, 2, 2, min_num_elems, max_num_elems).To(context);
  int32_t num_elems = ragged.NumElements();
  Array1<T> dst(context, num_elems);

  BenchmarkStat stat;
  stat.op_name = "SegmentedExclusiveSum" + std::to_string(ragged.Dim0()) + "_" +
                 std::to_string(num_elems);
  stat.num_iter = num_iter;
  stat.problem_size = dim;
  stat.dtype_name = TraitsOf(DtypeOf<int32_t>::dtype).Name();
  stat.device_type = device_type;

  stat.eplased_per_iter = BenchmarkOp(
      num_iter, context,
      (void (*)(Ragged<T> &, Array1<int32_t> *))(&SegmentedExclusiveSum<T>),
      ragged, &dst);
  stat.eplased_per_iter *= 1e6;  // from seconds to microseconds
  return stat;
}

template <typename T>
static void RegisterBenchmarkSegmentedExclusiveSum(DeviceType device_type) {
  std::vector<int32_t> problems_sizes = {50, 100, 200, 500, 1000, 10000};
  for (auto s : problems_sizes) {
    std::string name =
        GenerateBenchmarkName<T>("SegmentedExclusiveSum", device_type);
    RegisterBenchmark(name, [s, device_type]() -> BenchmarkStat {
      return BenchmarkSegmentedExclusiveSum<T>(s, device_type);
    });
  }
}

static void RunRaggedOpsBenchmark() {
  PrintEnvironmentInfo();

  RegisterBenchmarkGetTransposeReordering(kCpu);
  RegisterBenchmarkGetTransposeReordering(kCuda);
  RegisterBenchmarkSegmentedExclusiveSum<int32_t>(kCpu);
  RegisterBenchmarkSegmentedExclusiveSum<int32_t>(kCuda);

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
  k2::RunRaggedOpsBenchmark();
  return 0;
}
