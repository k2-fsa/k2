// k2/csrc/rand.cu
/**
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include <cmath>
#include <random>
#include <type_traits>

#include "curand.h"         // NOLINT
#include "curand_kernel.h"  // NOLINT
#include "k2/csrc/rand.h"

namespace k2 {

namespace {

// when calling curand_init() in kernels, its arguments
// seed and offset are from this struct. All kernels
// share the same seed and offset.
struct CudaRandState {
  // the default value for seed is from
  // https://github.com/pytorch/pytorch/blob/master/c10/core/GeneratorImpl.h#L56
  //
  // It has a good distribution of 0s and 1s in bit representation.
  uint64_t seed = 67280421310721u;
  uint64_t offset = 0;
};

struct CpuRandState {
  uint64_t seed = std::mt19937::default_seed;
  std::mt19937 generator;
};

static CudaRandState &GetCudaRandState(ContextPtr context) {
  int32_t device_id = context->GetDeviceId();
  K2_CHECK_LT(device_id, kMaxNumGpus);

  static CudaRandState rand_states[kMaxNumGpus];
  return rand_states[device_id];
}  // namespace

static CpuRandState &GetCpuRandState() {
  static thread_local CpuRandState state;
  return state;
}

template <typename FloatType>
static void RandCpu(int32_t dim, FloatType low, FloatType high,
                    FloatType *out) {
  static_assert(std::is_same<FloatType, float>::value ||
                    std::is_same<FloatType, double>::value,
                "");
  // std::uniform_real_distribution returns a number in
  // the interval [low, high), but we want (low, high].
  // That is, we want to exclude `low`.

  if (std::is_same<FloatType, float>::value) {
    low = std::nexttowardf(low, low + 1);
    high = std::nexttowardf(high, high + 1);
  } else {
    low = std::nexttoward(low, low + 1);
    high = std::nexttoward(high, high + 1);
  }

  std::uniform_real_distribution<FloatType> distribution(low, high);
  auto &generator = GetCpuRandState().generator;

  for (int32_t i = 0; i != dim; ++i) {
    out[i] = distribution(generator);
  }
}

}  // namespace

uint64_t GetSeed(ContextPtr context) {
  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCuda) return GetCudaRandState(context).seed;

  K2_CHECK_EQ(device_type, kCpu);
  return GetCpuRandState().seed;
}

void SetSeed(ContextPtr context, uint64_t seed) {
  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCuda) {
    // TODO(fangjun): we may need a lock here
    CudaRandState &state = GetCudaRandState(context);
    state.seed = seed;
    state.offset = 0;
    return;
  }

  K2_CHECK_EQ(device_type, kCpu);
  CpuRandState &state = GetCpuRandState();
  state.seed = seed;
  state.generator.seed(seed);
}

template <>
void Rand<float>(Array1<float> *array, float low, float high) {
  K2_CHECK_GT(high, low);
  ContextPtr &context = array->Context();
  int32_t dim = array->Dim();
  if (dim == 0) return;

  float *array_data = array->Data();
  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) {
    RandCpu(dim, low, high, array_data);
    return;
  }

  K2_CHECK_EQ(device_type, kCuda);
  CudaRandState &state = GetCudaRandState(context);
  auto generate_rand_lambda_float = [=] __device__(int32_t i) {
    curandStatePhilox4_32_10_t philox_state;
    curand_init(state.seed,
                i,  // sequence
                state.offset, &philox_state);

    float4 r = curand_uniform4(&philox_state);
    array_data[i] = r.x * (high - low) + low;
  };
  EvalDevice(context, dim, generate_rand_lambda_float);

  state.offset += 4;
}

template <>
void Rand<double>(Array1<double> *array, double low, double high) {
  K2_CHECK_GT(high, low);
  ContextPtr &context = array->Context();
  int32_t dim = array->Dim();
  if (dim == 0) return;

  double *array_data = array->Data();
  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) {
    RandCpu(dim, low, high, array_data);
    return;
  }

  K2_CHECK_EQ(device_type, kCuda);
  CudaRandState &state = GetCudaRandState(context);
  auto generate_rand_lambda_double = [=] __device__(int32_t i) {
    curandStatePhilox4_32_10_t philox_state;
    curand_init(state.seed,
                i,  // sequence
                state.offset, &philox_state);

    double2 r = curand_uniform2_double(&philox_state);
    array_data[i] = r.x * (high - low) + low;
  };
  EvalDevice(context, dim, generate_rand_lambda_double);

  state.offset += 4;
}

}  // namespace k2
