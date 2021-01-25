/**
 * k2/csrc/cudpp/cudpp.h
 *
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_CUDPP_CUDPP_H_
#define K2_CSRC_CUDPP_CUDPP_H_

namespace k2 {

/* Similar to ExclusiveSum, this function implements exclusive sum
 * per sub list, i.e., SegmentedExclusiveSum.
 *
 * @param [in] context A CUDA context. `in`, `out` and `flags` should be
 *                     allocated by this context.
 * @param [in] in  Pointer to the input array. CAUTION: the last element
 *                 does not contribute to the sum.
 * @param [in] num_elements
 *                 Number of elements in the input array.
 * @param [in] flags  Its size equals to num_elements and its entries are 0
 *                    and 1. An entry with 1 indicates the beginning of a
 *                    sublist. For example, if the input sublist is
 *                    [ [x x] [ x x x] ], then flags is [0 0 1 0 0].
 *                    If the input sublist is [ [x] [x x] [x x x] ],
 *                    then flags is [ 0 1 0 1 0 0].
 *                    If the input sublist is [ [] [x x x] ], then flags
 *                    is [1 0 0].
 * @param [out] out  Pointer to the output array. May be the same as `in`.
 *                   Its length is num_elements.
 *
 * NOTE: It supports `in == out`.
 *
 * Refer to
 * http://cudpp.github.io/cudpp/2.0/group__public_interface.html
 * for the documentation of `cudppSegmentedScan` for more info.
 */
template <typename T>
void SegmentedExclusiveSum(ContextPtr context, const T *in,
                           int32_t num_elements, const uint32_t *flags, T *out);

}  // namespace k2

#endif  // K2_CSRC_CUDPP_CUDPP_H_
