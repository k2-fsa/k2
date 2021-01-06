/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_BENCHMARK_BENCHMARK_H_
#define K2_CSRC_BENCHMARK_BENCHMARK_H_

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/log.h"
#include "k2/csrc/timer.h"

namespace k2 {

// When helper information, e.g., date time
// and device information, is printed,
// each line is prepended with `kPrefix`
constexpr const char *kPrefix = "# ";

/* Run an op for a given number of iterations.

  @param [in]  num_iter   Number iterations to run.
  @param [in]  context    The context for creating timer.
  @param [in]  op         The operation to be benchmarked.
  @param [in]  args       The arguments for `op`.

  @return Number of elapsed seconds per iteration on average.
 */
template <typename Op, typename... Args>
float BenchmarkOp(int32_t num_iter, ContextPtr context, Op &&op,
                  Args &&... args) {
  K2_CHECK_GT(num_iter, 0);

  for (int32_t i = 0; i != 30; ++i) {
    // warm up
    std::forward<Op>(op)(std::forward<Args>(args)...);
  }

  Timer timer(context);
  for (int32_t i = 0; i != num_iter; ++i) {
    std::forward<Op>(op)(std::forward<Args>(args)...);
  }

  return timer.Elapsed() / num_iter;
}

struct DeviceInfo {
  std::string device_name;
  int32_t compute_capability_major;
  int32_t compute_capability_minor;
  float gpu_clock_freq;  // in GHz
  int32_t driver_version_major;
  int32_t driver_version_minor;
  int32_t runtime_version_major;
  int32_t runtime_version_minor;
  int32_t warp_size;
  float l2_cache_size;               // in MB
  float total_global_mem;            // in GB
  float total_const_mem;             // in KB
  float total_shared_mem_per_block;  // in KB
  float total_shared_mem_per_mp;     // in KB
  int32_t ecc_enabled;
  int32_t num_multiprocessors;
  int32_t num_cuda_cores;

  std::string ToString() const;
};

std::ostream &operator<<(std::ostream &os, const DeviceInfo &info);

DeviceInfo GetDeviceInfo();

struct BenchmarkStat {
  std::string op_name;  // operator name (i.e., function name) of the benchmark
  int32_t num_iter;     // number of iterations of this run
  float eplased_per_iter;  // number of microseconds per iteration on average
  int32_t problem_size;
  std::string dtype_name;  // e.g., int32_t, float
  DeviceType device_type;  // e.g., kCpu, kCuda
};

// TODO(fangjun): Implement a reporter for formatted printing.
struct BenchmarkRun {
  std::string name;  // name of the benchmark
  BenchmarkStat stat;

  // Return a string representation in CSV of this object
  std::string ToString() const;

  // Keep in sync with ToString()
  //
  // Return the field name of CSV format returned by `ToString()`
  static std::string GetFieldsName();
};

std::ostream &operator<<(std::ostream &os, const BenchmarkRun &run);

using BenchmarkFunc = std::function<BenchmarkStat()>;

struct BenchmarkInstance {
  std::string name;
  BenchmarkFunc func;

  BenchmarkInstance(const std::string &name, BenchmarkFunc func)
      : name(name), func(func) {}
};

/* Register a benchmark.

   @param [in] name  The name of the benchmark.
   @param [in] func  The function to be run for the benchmark.
 */
void RegisterBenchmark(const std::string &name, BenchmarkFunc func);

/* Get a list of registered benchmarks.
 */
std::vector<std::unique_ptr<BenchmarkInstance>> *GetRegisteredBenchmarks();

/* Filter registered benchmarks whose name does not match
   the given regular expression.

   @param [in] pattern The regular expression. Benchmark names that
                       do not match the pattern will be excluded
                       and will not be run while invoking `RunBenchmarks()`.
 */
void FilterRegisteredBenchmarks(const std::string &pattern);

/* Run registered benchmarks.

   @return Return the benchmark results.
 */
std::vector<BenchmarkRun> RunBechmarks();

/* Return current date time as a string.
   E.g., Mon Dec 14 14:08:33 2020
 */
std::string GetCurrentDateTime();

template <typename T>
std::string GenerateBenchmarkName(const std::string &base_name,
                                  DeviceType device_type) {
  std::ostringstream os;
  os << base_name << '_' << TraitsOf(DtypeOf<T>::dtype).Name() << '_'
     << device_type;
  return os.str();
}

// for debugging only
void PrintRegisteredBenchmarks();

/* Print environment information of the current benchmark.

  Environment information includes:

    - current date time
    - device information
    - how k2 was built (information from version.h)
 */
void PrintEnvironmentInfo();

}  // namespace k2

#endif  // K2_CSRC_BENCHMARK_BENCHMARK_H_
