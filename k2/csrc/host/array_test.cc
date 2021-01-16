/**
 * @brief
 * array_test
 *
 * @copyright
 * Copyright (c)  2020  Xiaomi Corporation (authors: Haowen Qiu)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/host/array.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>

#include <iterator>
#include <memory>
#include <numeric>
#include <type_traits>
#include <utility>
#include <vector>

namespace k2host {

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

  // test range-based for loop
  n = 0;
  for (const auto &element : array2) {
    EXPECT_EQ(element, data[n++]);
  }

  // test range-based for loop: mutate data
  for (auto &element : array2) {
    ++element;
  }
  n = 0;
  for (auto &element : array2) {
    EXPECT_EQ(element, data[n++] + 1);
  }
  for (auto &element : array2) {
    --element;
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

  {
    // test `begin` and `end` for empty Array2 object
    Array2<Ptr, IndexType> empty_array;
    EXPECT_TRUE(empty_array.Empty());
    for (const auto &element : empty_array) {
      ValueType tmp = element;
    }
  }
}

template <typename Ptr, typename IndexType>
void TestArray3(int32_t stride) {
  using ValueType = typename std::iterator_traits<Ptr>::value_type;

  Array2Size<IndexType> size1 = {4, 10};
  std::vector<IndexType> indexes1 = {0, 3, 5, 9, 10};
  std::vector<ValueType> data1(size1.size2);
  std::iota(data1.begin(), data1.end(), 0);
  Array2Storage<Ptr, IndexType> storage1(size1, stride);
  storage1.FillIndexes(indexes1);
  storage1.FillData(data1);
  Array2<Ptr, IndexType> &array1 = storage1.GetArray2();
  EXPECT_EQ(array1.data[array1.indexes[0]], 0);

  Array2Size<IndexType> size2 = {3, 10};
  // note indexes2[0] starts from 3 instead of 0
  std::vector<IndexType> indexes2 = {3, 5, 8, 10};
  std::vector<ValueType> data2(10);  // 10 instead of 7 here on purpose
  std::iota(data2.begin(), data2.end(), 0);
  Array2Storage<Ptr, IndexType> storage2(size2, stride);
  storage2.FillIndexes(indexes2);
  storage2.FillData(data2);
  Array2<Ptr, IndexType> &array2 = storage2.GetArray2();
  array2.size2 = 7;  // change the size to the correct value
  EXPECT_EQ(array2.data[array2.indexes[0]], 3);

  std::vector<Array2<Ptr, IndexType>> arrays;
  arrays.emplace_back(array1);
  arrays.emplace_back(array2);

  Array3<Ptr, IndexType> array3;
  array3.GetSizes(arrays.data(), 2);
  EXPECT_EQ(array3.size1, 2);
  EXPECT_EQ(array3.size2, 7);
  EXPECT_EQ(array3.size3, 17);

  // Test Array3 Creation
  std::vector<IndexType> array3_indexes1(array3.size1 + 1);
  std::vector<IndexType> array3_indexes2(array3.size2 + 1);
  std::unique_ptr<ValueType[]> array3_data(
      new ValueType[array3.size3 * stride]);
  array3.indexes1 = array3_indexes1.data();
  array3.indexes2 = array3_indexes2.data();
  array3.data = DataPtrCreator<Ptr, IndexType>::Create(array3_data, stride);

  array3.Create(arrays.data(), 2);
  EXPECT_THAT(array3_indexes1, ::testing::ElementsAre(0, 4, 7));
  EXPECT_THAT(array3_indexes2,
              ::testing::ElementsAre(0, 3, 5, 9, 10, 12, 15, 17));
  for (auto i = array1.indexes[0]; i != array1.indexes[array1.size1]; ++i) {
    EXPECT_EQ(array3.data[i], array1.data[i]);
  }
  EXPECT_EQ(array2.indexes[0], 3);
  for (auto i = array2.indexes[0]; i != array2.indexes[array2.size1]; ++i) {
    EXPECT_EQ(array3.data[array1.size2 + i - array2.indexes[0]],
              array2.data[i]);
  }

  // Test Array3's operator[]
  Array2<Ptr, IndexType> array1_copy = array3[0];
  EXPECT_EQ(array1_copy.size1, array1.size1);
  EXPECT_EQ(array1_copy.size2, array1.size2);
  for (auto i = 0; i != array1.size1 + 1; ++i) {
    EXPECT_EQ(array1_copy.indexes[i], array1.indexes[i]);
  }
  for (auto i = array1.indexes[0]; i != array1.indexes[array1.size1]; ++i) {
    EXPECT_EQ(array1_copy.data[i], array1.data[i]);
  }

  Array2<Ptr, IndexType> array2_copy = array3[1];
  EXPECT_EQ(array2_copy.size1, array2.size1);
  EXPECT_EQ(array2_copy.size2, array2.size2);
  for (auto i = 0; i != array2.size1 + 1; ++i) {
    // output indexes may start from n > 0
    EXPECT_EQ(array2_copy.indexes[i],
              array2.indexes[i] + array1.size2 - array2.indexes[0]);
  }
  for (auto i = array2.indexes[0]; i != array2.indexes[array2.size1]; ++i) {
    EXPECT_EQ(array1_copy.data[i + array1.size2 - array2.indexes[0]],
              array1.data[i]);
  }
}

TEST(ArrayTest, RawPointer) {
  TestArray2<int32_t *, int32_t>(1);
  TestArray3<int32_t *, int32_t>(1);
}

TEST(ArrayTest, StridedPtr) {
  TestArray2<StridedPtr<int32_t, int32_t>, int32_t>(2);
  TestArray3<StridedPtr<int32_t, int32_t>, int32_t>(2);
}

}  // namespace k2host
