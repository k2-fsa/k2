/**
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *                      Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * See LICENSE for clarification regarding multiple authors
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <numeric>
#include <vector>

#include "k2/csrc/dtype.h"
#include "k2/csrc/fsa.h"

namespace k2 {

template <typename T> void CheckDtypes() {
  Dtype d = DtypeOf<T>::dtype;
  DtypeTraits t = TraitsOf(d);
  std::cout << "dtype_traitgs.Name() = " << t.Name() << "\n";
  if (std::is_floating_point<T>::value) {
    ASSERT_EQ(t.GetBaseType(), kFloatBase);
    EXPECT_EQ(t.NumScalars(), 1);
  } else if (std::is_integral<T>::value) {
    if (std::is_unsigned<T>::value) {
      ASSERT_EQ(t.GetBaseType(), kUintBase);
    } else {
      ASSERT_EQ(t.GetBaseType(), kIntBase);
    }
    EXPECT_EQ(t.NumScalars(), 1);
  } else {
    ASSERT_EQ(t.GetBaseType(), kUnknownBase);
  }
  if (!K2_TYPE_IS_ANY(T)) {
    EXPECT_EQ(t.NumBytes(), sizeof(T));
  }
}

TEST(DtypeTest, CheckDtypes) {
  // CheckDtypes<half>();
  CheckDtypes<float>();
  CheckDtypes<double>();
  CheckDtypes<int8_t>();
  CheckDtypes<int16_t>();
  CheckDtypes<int32_t>();
  CheckDtypes<int64_t>();
  CheckDtypes<uint8_t>();
  CheckDtypes<uint16_t>();
  CheckDtypes<uint32_t>();
  CheckDtypes<uint64_t>();
  CheckDtypes<Any>();
  CheckDtypes<Arc>();
}


}  // namespace k2
