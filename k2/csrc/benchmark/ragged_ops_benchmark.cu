/**
 * @brief Benchmarks for ragged_ops.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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

static void RunRaggedOpsBenchmark() {
  PrintEnvironmentInfo();

  RegisterBenchmarkGetTransposeReordering(kCpu);
  RegisterBenchmarkGetTransposeReordering(kCuda);

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
