/**
 * @brief Benchmarks for k2 APIs.
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/array_ops.h"
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

/* Return the number of milliseconds per iteration on average.

   @param  [in]  dim  Number of elements in Array1<T> for testing.
   @return Return number of ms per iteration on avarage.
 */

template <typename T>
using UnaryOp = Array1<T> (*)(const Array1<T> &);

template <typename T>
static float BM_ExclusiveSum(int32_t dim, UnaryOp<T> op) {
  ContextPtr context = GetCudaContext();

  Array1<T> src = RandUniformArray1<T>(context, dim, -1000, 1000);
  for (int32_t i = 0; i != 30; ++i) {  // warm up
    DoNotOptimize(op(src));
  }

  int32_t num_iter = std::min(500, 1000000 / dim);
  K2_CHECK_GT(num_iter, 0);

  cudaEvent_t start;
  cudaEvent_t stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaStream_t stream = context->GetCudaStream();
  cudaEventRecord(start, stream);
  for (int32_t i = 0; i != num_iter; ++i) {
    DoNotOptimize(op(src));
  }
  cudaEventRecord(stop, stream);
  cudaEventSynchronize(stop);

  float ms = 0;
  cudaEventElapsedTime(&ms, start, stop);
  ms /= num_iter;

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return ms;
}

static void RunBenchmarks() {
  std::vector<int32_t> problems_sizes = {100, 500, 1000, 2000, 5000, 10000};
  for (auto s : problems_sizes) {
    float ms = BM_ExclusiveSum<int32_t>(s, ExclusiveSum<int32_t>);
    printf("%6d -->\t%.5f ms\n", s, ms);
  }
}

}  // namespace k2

int main() {
  k2::RunBenchmarks();
  return 0;
}
