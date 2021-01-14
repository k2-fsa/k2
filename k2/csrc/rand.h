// k2/csrc/rand.h
/**
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
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
 * `FloatType` can be either `float` or `double`.
 *
 * @param [inout] array  The array is modified in-place.
 */
template <typename FloatType>
void Rand(Array1<FloatType> *array, FloatType low = FloatType(0),
          FloatType high = FloatType(1));

/* Returns an array filled with random numbers from a uniform distribution on
 * the interval [low, high).
 *
 * low is inclusive and high is exclusive.
 *
 * `FloatType` can be either `float` or `double`.
 *
 * @param [in]  context  It specifies the device on which the random
 *                       numbers are generated.
 * @param [in]  dim      The dimension of the returned array.
 */
template <typename FloatType>
Array1<FloatType> Rand(ContextPtr context, int32_t dim,
                       FloatType low = FloatType(0),
                       FloatType high = FloatType(1)) {
  Array1<FloatType> ans(context, dim);
  Rand(&ans, low, high);
  return ans;
}

}  // namespace k2

#endif  // K2_CSRC_RAND_H_
