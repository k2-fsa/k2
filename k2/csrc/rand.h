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

#ifndef K2_CSRC_RAND_H_
#define K2_CSRC_RAND_H_

#include "k2/csrc/array.h"
#include "k2/csrc/context.h"

namespace k2 {

/* Get the current seed of the device associated with `context`.
 *
 * @param [in] context  It specifies the device whose seed is to be returned.
 *                      It can be either a CPU context or a CUDA context.
 *
 * @return  Return the seed of the device associated with the given `context`.
 *
 * TODO(fangjun): we may not need it.
 */
uint64_t GetSeed(ContextPtr context);

/* Set the seed of the device associated with the given `context`.
 *
 * @param [in] context  It specifies the device whose seed is to be set.
 *                      It can be either a CPU context or a CUDA context.
 *
 * @param [in] seed     The target seed.
 */
void SetSeed(ContextPtr context, uint64_t seed);

/* Fill the given array with random numbers from a uniform distribution on
 * the interval [low, high).
 *
 * low is inclusive and high is exclusive.
 *
 * `T` can be `float`, `double`, or `int32_t`.
 *
 * @param [in]  context      It specifies the device on which
 *                           `array_data` resides
 * @param [in]  low          The lower bound of the interval (inclusive).
 * @param [in]  high         The upper bound of the interval (exclusive).
 * @param [in]  dim          Number of elements in the output array.
 * @param [out] array_data   Pointer to the beginning of the output array.
 */
template <typename T>
void Rand(ContextPtr context, T low, T high, int32_t dim, T *array_data);

/* Fill the given array with random numbers from a uniform distribution on
 * the interval [low, high).
 *
 * low is inclusive and high is exclusive.
 *
 * `T` can be `float`, `double`, or `int32_t`.
 *
 * @param [in]  low       The lower bound of the interval (inclusive).
 * @param [in]  high      The upper bound of the interval (exclusive).
 * @param [out] array     The array is modified in-place.
 */
template <typename T>
void Rand(T low, T high, Array1<T> *array) {
  Rand(array->Context(), low, high, array->Dim(), array->Data());
}

/* Returns an array filled with random numbers from a uniform distribution on
 * the interval [low, high).
 *
 * low is inclusive and high is exclusive.
 *
 * `T` can be `float`, `double`, or `int32_t`.
 *
 * @param [in]  context  It specifies the device on which the random
 *                       numbers are generated.
 * @param [in]  low      The lower bound of the interval (inclusive).
 * @param [in]  high     The upper bound of the interval (exclusive).
 * @param [in]  dim      The dimension of the returned array.
 */
template <typename T>
Array1<T> Rand(ContextPtr context, T low, T high, int32_t dim) {
  Array1<T> ans(context, dim);
  Rand(low, high, &ans);
  return ans;
}

}  // namespace k2

#endif  // K2_CSRC_RAND_H_
