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

#include "k2/csrc/log.h"
#include "k2/csrc/timer.h"

namespace k2 {

// ------- from google/benchmark  ---- begin ----
#if defined(__GNUC__)
#define K2_ALWAYS_INLINE __attribute__((always_inline))
#else
#define K2_ALWAYS_INLINE
#endif

// The DoNotOptimize(...) function can be used to prevent a value or
// expression from being optimized away by the compiler. This function is
// intended to add little to no overhead.
// See: https://youtu.be/nXaxk27zwlk?t=2441
template <class Tp>
inline K2_ALWAYS_INLINE void DoNotOptimize(Tp const &value) {
  asm volatile("" : : "r,m"(value) : "memory");
}

template <class Tp>
inline K2_ALWAYS_INLINE void DoNotOptimize(Tp &value) {
#if defined(__clang__)
  asm volatile("" : "+r,m"(value) : : "memory");
#else
  asm volatile("" : "+m,r"(value) : : "memory");
#endif
}
// ------- from google/benchmark  ---- end ----

// If `Op(Args...)` returns void, then
// `ReturnVoid<Op, Args...>::value` is true.
// Otherwise it is false.
template <typename Op, typename... Args>
struct ReturnVoid
    : public std::is_same<void, typename std::result_of<Op(Args...)>::type> {};

/*
  For `Op(Args...)` that returns non-void.

  @param [in]  num_iter   Number iterations to run.
  @param [in]  context    The context for creating timer.
  @param [in]  op         The operation to be benchmarked.
  @param [in]  args       The arguments for `op`.

  @return Number of elapsed seconds per iteration on average.
 */
template <typename Op, typename... Args>
typename std::enable_if<!ReturnVoid<Op, Args...>::value, float>::type
BenchmarkOp(int32_t num_iter, ContextPtr context, Op &&op, Args &&... args) {
  K2_CHECK_GT(num_iter, 0);

  for (int32_t i = 0; i != 30; ++i) {
    // warm up
    DoNotOptimize(std::forward<Op>(op)(std::forward<Args>(args)...));
  }

  Timer timer(context);
  for (int32_t i = 0; i != num_iter; ++i) {
    DoNotOptimize(std::forward<Op>(op)(std::forward<Args>(args)...));
  }

  return timer.Elapsed() / num_iter;
}

struct BenchmarkStat {
  int32_t num_iter;        // number of iterations of this run
  float eplased_per_iter;  // number of seconds per iteration on average
};

// TODO(fangjun): Implement a reporter for formatted printing.
struct BenchmarkRun {
  std::string name;  // name of this run
  BenchmarkStat stat;
  // Return a string representation of this object
  std::string ToString() const;
};

using BenchmarkFunc = std::function<struct BenchmarkStat()>;

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
 */
void FilterRegisteredBenchmarks(const std::string &regex);

}  // namespace k2

#endif  // K2_CSRC_BENCHMARK_BENCHMARK_H_
