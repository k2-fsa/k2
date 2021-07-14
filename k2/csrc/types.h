/**
 * Copyright (c)  2021  Xiaomi Corporation (authors: Fangjun Kuang)
 *                      
 *
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_CSRC_TYPES_H_
#define K2_CSRC_TYPES_H_

#ifdef _MSC_VER
// See https://docs.microsoft.com/en-us/cpp/cpp/int8-int16-int32-int64
using uint16_t = unsigned __int16;
using uint32_t = unsigned __int32;
using uint64_t = unsigned __int64;

using int16_t = __int16;
using int32_t = __int32;
using int64_t = __int64;
#else
#include <cstdint>
#endif

#endif  // K2_CSRC_TYPES_H_
