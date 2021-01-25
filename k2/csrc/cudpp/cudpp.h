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

void cudppSegmentedScan(void *d_out, const void *d_in,
                        const unsigned int *d_iflags, size_t numElements);

}  // namespace k2

#endif  // K2_CSRC_CUDPP_CUDPP_H_
