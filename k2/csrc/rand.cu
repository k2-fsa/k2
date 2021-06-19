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

#include <cmath>
#include <random>
#include <type_traits>

#ifdef K2_WITH_CUDA
#include "curand.h"         // NOLINT
#include "curand_kernel.h"  // NOLINT
#endif

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

template <typename T, typename Distribution>
static void RandCpu(int32_t dim, T low, T high, T *out) {
  Distribution distribution(low, high);
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
void Rand<float>(ContextPtr context, float low, float high, int32_t dim,
                 float *array_data) {
  K2_CHECK_LT(low, high);
  if (dim == 0) return;

  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) {
    RandCpu<float, std::uniform_real_distribution<float>>(dim, low, high,
                                                          array_data);
    return;
  }

  K2_CHECK_EQ(device_type, kCuda);
#ifdef K2_WITH_CUDA
  CudaRandState &state = GetCudaRandState(context);
  float range = high - low;
  auto generate_rand_lambda_float = [=] __device__(int32_t i) {
    curandStatePhilox4_32_10_t philox_state;
    curand_init(state.seed,
                i,  // sequence
                state.offset, &philox_state);

    float4 r = curand_uniform4(&philox_state);

    // curand_uniform4() returns a number in (0, 1],
    // we want to transform it to [0, 1)
    //
    // CAUTION: `1 - r.x` is not used here as it may be rounded up to 1
    // when `r.x` is close to 0
    float t = (r.x == 1.0f) ? 0.0f : r.x;
    array_data[i] = t * range + low;
  };
  EvalDevice(context, dim, generate_rand_lambda_float);
  state.offset += 4;
#else
  K2_LOG(FATAL) << "Unreachable code";
#endif
}

template <>
void Rand<double>(ContextPtr context, double low, double high, int32_t dim,
                  double *array_data) {
  K2_CHECK_LT(low, high);
  if (dim == 0) return;

  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) {
    RandCpu<double, std::uniform_real_distribution<double>>(dim, low, high,
                                                            array_data);
    return;
  }
#ifdef K2_WITH_CUDA
  K2_CHECK_EQ(device_type, kCuda);
  CudaRandState &state = GetCudaRandState(context);
  double range = high - low;
  auto generate_rand_lambda_double = [=] __device__(int32_t i) {
    curandStatePhilox4_32_10_t philox_state;
    curand_init(state.seed,
                i,  // sequence
                state.offset, &philox_state);

    double2 r = curand_uniform2_double(&philox_state);
    double t = (r.x == 1.0) ? 0.0 : r.x;

    array_data[i] = t * range + low;
  };
  EvalDevice(context, dim, generate_rand_lambda_double);
  state.offset += 4;
#else
  K2_LOG(FATAL) << "Unreachable code.";
#endif
}

template <>
void Rand<int32_t>(ContextPtr context, int32_t low, int32_t high, int32_t dim,
                   int32_t *array_data) {
  K2_CHECK_LT(low, high);

  if (dim == 0) return;
  DeviceType device_type = context->GetDeviceType();
  if (device_type == kCpu) {
    RandCpu<int32_t, std::uniform_int_distribution<int32_t>>(
        dim, low, high - 1,  // -1 since high is to be excluded
        array_data);
    return;
  }

#ifdef K2_WITH_CUDA
  K2_CHECK_EQ(device_type, kCuda);
  CudaRandState &state = GetCudaRandState(context);
  uint32_t range = high - low;
  auto generate_rand_lambda_double = [=] __device__(int32_t i) {
    curandStatePhilox4_32_10_t philox_state;
    curand_init(state.seed,
                i,  // sequence
                state.offset, &philox_state);

    uint4 r = curand4(&philox_state);
    int32_t t = static_cast<int32_t>(r.x % range + low);

    array_data[i] = t;
  };

  EvalDevice(context, dim, generate_rand_lambda_double);

  state.offset += 4;
#else
  K2_LOG(FATAL) << "Unreachable code.";
#endif
}

}  // namespace k2
