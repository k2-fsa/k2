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
 * the interval (0.0, 1.0].
 *
 * 0 is exclusive and 1 is inclusive.
 *
 * `FloatType` can be either `float` or `double`.
 *
 * @param [inout] array  The array is modified in-place.
 */
template <typename FloatType>
void Rand(Array1<FloatType> *array);

/* Returns an array filled with random numbers from a uniform distribution on
 * the interval [0, 1).
 *
 * 0 is inclusive and 1 is exclusive.
 *
 * `FloatType` can be either `float` or `double`.
 *
 * @param [in]  context  It specifies the device on which the random
 *                       numbers are generated.
 * @param [in]  dim      The dimension of the returned array.
 */
template <typename FloatType>
Array1<FloatType> Rand(ContextPtr context, int32_t dim) {
  Array1<FloatType> ans(context, dim);
  Rand(&ans);
  return ans;
}

}  // namespace k2

#endif  // K2_CSRC_RAND_H_
