/**
 * Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/dtype.h"

namespace k2 {

const DtypeTraits g_dtype_traits_array[] = {
    {kFloatBase, 4, "float"}, {kFloatBase, 8, "double"}, {kIntBase, 1, "int8"},
    {kIntBase, 2, "int16"},   {kIntBase, 4, "int32"},    {kIntBase, 8, "int64"},
    {kUintBase, 4, "uint32"}, {kUintBase, 8, "uint64"},
};

const Dtype DtypeOf<float>::dtype;
const Dtype DtypeOf<double>::dtype;
const Dtype DtypeOf<int8_t>::dtype;
const Dtype DtypeOf<int16_t>::dtype;
const Dtype DtypeOf<int32_t>::dtype;
const Dtype DtypeOf<int64_t>::dtype;
const Dtype DtypeOf<uint32_t>::dtype;
const Dtype DtypeOf<uint64_t>::dtype;

std::ostream &operator<<(std::ostream &os, Dtype dtype) {
  os << TraitsOf(dtype).Name();
  return os;
}

}  // namespace k2
