// k2/csrc/cuda/dtype.cc

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey )

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/cuda/dtype.h"

namespace k2 {

DtypeTraits g_dtype_traits_array[] = {{kFloatBase, 4}, {kFloatBase, 8},
                                      {kIntBase, 4},   {kIntBase, 8},
                                      {kUintBase, 4},  {kUintBase, 8}};

}  // namespace k2
