// k2/csrc/array_test.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#include "k2/csrc/array.h"

#include <iterator>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"

namespace k2 {

template <typename Ptr, typename IndexType>
void TestArray2(int32_t stride) {
  using ValueType = typename std::iterator_traits<Ptr>::value_type;
  Array2Size<IndexType> array2_size = {4, 10};
  std::vector<IndexType> indexes = {0, 3, 5, 9, 10};
  std::vector<ValueType> data(array2_size.size2);
  std::iota(data.begin(), data.end(), 0);

  Array2Storage<Ptr, IndexType> array2_storage(array2_size, stride);
  array2_storage.FillIndexes(indexes);
  array2_storage.FillData(data);
  Array2<Ptr, IndexType> &array2 = array2_storage.GetArray2();

  EXPECT_EQ(array2.size1, array2_size.size1);
  EXPECT_EQ(array2.size2, array2_size.size2);
  EXPECT_FALSE(array2.Empty());

  // test indexes
  for (auto i = 0; i != array2.size1 + 1; ++i) {
    EXPECT_EQ(array2.indexes[i], indexes[i]);
  }

  // test operator []
  for (auto i = 0; i != array2.size2; ++i) {
    EXPECT_EQ(array2.data[i], data[i]);
  }

  auto data_ptr = array2.begin();

  // test operator*
  EXPECT_EQ(*data_ptr, 0);
  *data_ptr = -1;
  EXPECT_EQ(*data_ptr, -1);
  *data_ptr = 0;

  // test prefix increment
  ++data_ptr;
  EXPECT_EQ(*data_ptr, 1);

  // test postfix increment
  data_ptr = array2.begin();  // reset
  auto data_ptr1 = data_ptr++;
  EXPECT_EQ(*data_ptr, 1);
  EXPECT_EQ(*data_ptr1, 0);

  // test operator +=
  data_ptr = array2.begin();  // reset
  data_ptr += 3;
  EXPECT_EQ(*data_ptr, 3);

  // test operator +
  data_ptr = array2.begin();  // reset
  data_ptr1 = data_ptr + 3;
  EXPECT_EQ(*data_ptr, 0);
  EXPECT_EQ(*data_ptr1, 3);

  // test operator == operator !=
  data_ptr = array2.begin();  // reset
  data_ptr1 = data_ptr + 3;
  EXPECT_FALSE(data_ptr == data_ptr1);
  EXPECT_TRUE(data_ptr != data_ptr1);

  // test begin() and end()
  std::size_t n = 0;
  for (auto it = array2.begin(); it != array2.end(); ++it) {
    EXPECT_EQ(*it, data[n++]);
  }
  // test with it++
  n = 0;
  for (auto it = array2.begin(); it != array2.end(); it++) {
    EXPECT_EQ(*it, data[n++]);
  }

  // mutate data
  for (auto i = 0; i != array2.size2; ++i) {
    array2.data[i] += 1;
  }

  // test with indexes
  for (auto i = 0; i != array2.size1; ++i) {
    auto start = array2.indexes[i];
    auto end = array2.indexes[i + 1];
    for (auto j = start; j != end; ++j) {
      EXPECT_EQ(array2.data[j], data[j] + 1);
    }
  }
}

TEST(Array2Test, RawPointer) { TestArray2<int32_t *, int32_t>(1); }

TEST(Array2Test, StridedPtr) {
  TestArray2<StridedPtr<int32_t, int32_t>, int32_t>(2);
}

}  // namespace k2
