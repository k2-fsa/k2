// k2/csrc/array.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_HOST_ARRAY_H_
#define K2_CSRC_HOST_ARRAY_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "k2/csrc/log.h"
#include "k2/csrc/macros.h"
#include "k2/csrc/nvtx.h"

namespace k2host {

/*
   We will use e.g. StridedPtr<T, I> when the stride is not 1, and
   otherwise just T* (which presumably be faster).
*/
template <typename T, typename I = int32_t>
struct StridedPtr {
  T *data;   // it is NOT owned here
  I stride;  // in number of elements, NOT number of bytes
  explicit StridedPtr(T *data = nullptr, I stride = 0)  // NOLINT
      : data(data), stride(stride) {}

  T &operator[](I i) { return data[i * stride]; }
  const T &operator[](I i) const { return data[i * stride]; }

  T &operator*() { return *data; }
  const T &operator*() const { return *data; }

  // prefix increment
  StridedPtr &operator++() {
    data += stride;
    return *this;
  }
  // postfix increment
  StridedPtr operator++(int32_t) {
    StridedPtr tmp(*this);
    ++(*this);
    return tmp;
  }

  StridedPtr &operator+=(I n) {
    data += n * stride;
    return *this;
  }
  StridedPtr operator+(I n) const {
    StridedPtr tmp(*this);
    tmp += n;
    return tmp;
  }

  bool operator==(const StridedPtr &other) const {
    return data == other.data && stride == other.stride;
  }
  bool operator!=(const StridedPtr &other) const { return !(*this == other); }

  void Swap(StridedPtr &other) {
    std::swap(data, other.data);
    std::swap(stride, other.stride);
  }
};

template <typename Ptr, typename I = int32_t>
struct Array1 {
  // One dimensional array of something, like vector<X>
  // where Ptr is, or behaves like, X*.
  using IndexT = I;
  using PtrT = Ptr;
  using ValueType = typename std::iterator_traits<Ptr>::value_type;

  Array1() : begin(0), end(0), size(0), data(nullptr) {}
  Array1(IndexT begin, IndexT end, PtrT data)
      : begin(begin), end(end), data(data) {
    K2_CHECK_GE(end, begin);
    this->size = end - begin;
  }
  Array1(IndexT size, PtrT data) : begin(0), end(size), size(size), data(data) {
    K2_CHECK_GE(size, 0);
  }
  void Init(IndexT begin, IndexT end, PtrT data) {
    K2_CHECK_GE(end, begin);
    this->begin = begin;
    this->end = end;
    this->size = end - begin;
    this->data = data;
  }
  bool Empty() const { return begin == end; }

  // 'begin' and 'end' are the first and one-past-the-last indexes into `data`
  // that we are allowed to use.
  IndexT begin;
  IndexT end;
  IndexT size;  // the number of elements in `data` that can be accessed, equals
                // to `end - begin`
  PtrT data;
};

/*
  This struct stores the size of an Array2 object; it will generally be used as
  an output argument by functions that work out this size.
 */
template <typename I>
struct Array2Size {
  using IndexT = I;
  // `size1` is the top-level size of the array, equal to the object's .size1
  // element
  I size1;
  // `size2` is the number of elements in the array, equal to the object's
  // .size2 element
  I size2;
};

// Caution: k2host::Array2 is not at all the same as k2::Array2; this is
// ragged, k2::Array2 is regular.
template <typename Ptr, typename I = int32_t>
struct Array2 {
  // Irregular two dimensional array of something, like vector<vector<X> >
  // where Ptr is, or behaves like, X*.
  using IndexT = I;
  using PtrT = Ptr;
  using ValueType = typename std::iterator_traits<Ptr>::value_type;

  Array2() : size1(0), size2(0), indexes(&size1), data(nullptr) {}
  Array2(IndexT size1, IndexT size2, IndexT *indexes, PtrT data)
      : size1(size1), size2(size2), indexes(indexes), data(data) {}
  void Init(IndexT size1, IndexT size2, IndexT *indexes, PtrT data) {
    this->size1 = size1;
    this->size2 = size2;
    this->indexes = indexes;
    this->data = data;
  }

  IndexT size1;
  IndexT size2;     // the number of elements in the array,  equal to
                    // indexes[size1] - indexes[0] (if the object Array2
                    // has been initialized).
  IndexT *indexes;  // indexes[0,1,...size1] should be defined; note,
                    // this means the array must be of at least
                    // size1+1.  We require that indexes[i] <=
                    // indexes[i+1], but it is not required that
                    // indexes[0] == 0, it may be greater than 0.
                    // `indexes` should point to a zero if `size1 == 0`,
                    // i.e. `indexes[0] == 0`
  PtrT data;  // `data` might be an actual pointer, or might be some object
              // supporting operator [].  data[indexes[0]] through
              // data[indexes[size1] - 1] must be accessible through this
              // object.

  /*
     If an Array2 object is initialized and `size1` == 0, it means the object is
     empty. Here we don't care about `indexes`, but it should have at least one
     element and `indexes[0] == 0` according to the definition of `indexes`.
     Users should not access `data` if the object is empty.
  */
  bool Empty() const { return size1 == 0; }

  // as we require `indexes[0] == 0` if Array2 is empty,
  // the implementation of `begin` and `end` would be fine for empty object.
  PtrT begin() { return data + indexes[0]; }              // NOLINT
  const PtrT begin() const { return data + indexes[0]; }  // NOLINT

  PtrT end() { return data + indexes[size1]; }              // NOLINT
  const PtrT end() const { return data + indexes[size1]; }  // NOLINT

  // just to replace `Swap` functions for Fsa and AuxLabels for now,
  // may delete it if we finally find that we don't need to call it.
  void Swap(Array2 &other) {
    std::swap(size1, other.size1);
    std::swap(size2, other.size2);
    std::swap(indexes, other.indexes);
    // it's OK here for Ptr=StridedPtr as we have specialized
    // std::swap for StridedPtr
    std::swap(data, other.data);
  }

  /* initialized definition:
       An Array2 object is initialized if its `size1` member and `size2` member
       are set and its `indexes` and `data` pointer allocated, and the values of
       its `indexes` array are set for indexes[0] and indexes[size1].
 */
};

template <typename Ptr, typename I>
struct Array3 {
  // Irregular three dimensional array of something, like
  // vector<vector<vector<X> > > where Ptr is or behaves like X*.
  using IndexT = I;
  using PtrT = Ptr;

  IndexT size1;  // equal to the number of Array2 object in this Array3 object;
                 // `size1 + 1` will be the number of elements in indexes1.

  IndexT size2;  // equal to indexes1[size1] - indexes1[0];
                 // `size2 + 1` will be the number of elements in indexes2;

  IndexT size3;  // the number of elements in `data`,  equal to
                 // indexes2[indexes1[size1]] - indexes2[indexes1[0]].

  IndexT *indexes1;  // indexes1[0,1,...size1] should be defined; note,
                     // this means the array must be of at least size1+1.
                     // We require that indexes[i] <= indexes[i+1], but it
                     // is not required that indexes[0] == 0, it may be greater
                     // than 0.

  IndexT *indexes2;  // indexes2[indexes1[0]]
                     // .. indexes2[indexes1[size1]] should be defined;
                     // note, this means the array must be of at least size2+1.

  Ptr data;  // `data` might be an actual pointer, or might be some object
             // supporting operator [].  data[indexes2[indexes1[0]]] through
             // data[indexes2[indexes1[size1]] - 1] must be accessible
             // through this object.

  Array2<Ptr, I> operator[](I i) const {
    NVTX_RANGE(K2_FUNC);
    K2_DCHECK_GE(i, 0);
    K2_DCHECK_LT(i, size1);

    Array2<Ptr, I> array;
    array.size1 = indexes1[i + 1] - indexes1[i];
    array.indexes = indexes2 + indexes1[i];
    array.size2 = indexes2[indexes1[i + 1]] - indexes2[indexes1[i]];
    array.data = data;
    return array;
  }

  /*
    Set `size1`, `size2` and `size3` so that we can know how much memory we need
    to allocate for `indexes1`, `indexes2` and `data` to represent the vector
    of Array2 as an Array3.
      @param [in] arrays     A vector of Array2;
      @param [in] array_size The number element of vector `arrays`
  */
  void GetSizes(const Array2<Ptr, I> *arrays, I array_size) {
    size1 = array_size;
    size2 = size3 = 0;
    for (I i = 0; i != array_size; ++i) {
      size2 += arrays[i].size1;
      size3 += arrays[i].size2;
    }
  }

  /*
    Create Array3 from the vector of Array2. `size1`, `size2` and `size3` must
    have been set by calling `GetSizes` above, and the memory of `indexes1`,
    `indexes2`and `data` must have been allocated according to those size.
      @param [in] arrays     A vector of Array2;
      @param [in] array_size The number element of vector `arrays`
   */
  void Create(const Array2<Ptr, I> *arrays, I array_size) {
    NVTX_RANGE(K2_FUNC);
    K2_CHECK_EQ(size1, array_size);
    I size2_tmp = 0, size3_tmp = 0;
    for (I i = 0; i != array_size; ++i) {
      const auto &curr_array = arrays[i];

      indexes1[i] = size2_tmp;

      // copy indexes
      K2_CHECK_LE(size2_tmp + curr_array.size1, size2);
      I begin_index = curr_array.indexes[0];  // indexes[0] is always valid and
                                              // may be greater than 0
      for (I j = 0; j != curr_array.size1; ++j) {
        indexes2[size2_tmp++] = size3_tmp + curr_array.indexes[j] - begin_index;
      }

      // copy data
      K2_CHECK_LE(size3_tmp + curr_array.size2, size3);
      for (I n = 0; n != curr_array.size2; ++n) {
        data[size3_tmp + n] = curr_array.data[n + begin_index];
      }
      size3_tmp += curr_array.size2;
    }
    K2_CHECK_EQ(size2_tmp, size2);
    K2_CHECK_EQ(size3_tmp, size3);

    indexes1[size1] = size2_tmp;
    indexes2[indexes1[size1]] = size3_tmp;
  }
};

// Note: we can create Array4 later if we need it.

/*
  `DataPtrCreator::Create` wraps data storage std::unique_ptr<value_type(Ptr)>
  as a `Ptr` and returns it, this class is just for test purpose for now.
*/
template <typename ValueType, typename I>
struct DataPtrCreator;

// General case where stride is 1 and the underlying data manager is a raw
// pointer, we will just return data_storage.get()
template <typename ValueType, typename I>
struct DataPtrCreator<ValueType *, I> {
  static ValueType *Create(const std::unique_ptr<ValueType[]> &data_storage,
                           int32_t stride) {
    K2_CHECK_EQ(stride, 1);
    return data_storage.get();
  }
};

// Specialized case where stride is bigger than 1 and the underlying data
// manager is StridedPtr, we will create a StridedPtr with data_storage and
// return it.
template <typename ValueType, typename I>
struct DataPtrCreator<StridedPtr<ValueType, I>, I> {
  static StridedPtr<ValueType, I> Create(
      const std::unique_ptr<ValueType[]> &data_storage, int32_t stride) {
    K2_CHECK_GT(stride, 1);
    StridedPtr<ValueType, I> strided_ptr(data_storage.get(), stride);
    return strided_ptr;
  }
};

// Allocate memory for Array2, this class is just for test purpose.
// It will be usually called after those `GetSizes` functions (e.g.
// FstInverter.GetSizes) to allocate memory in test code.
template <typename Ptr, typename I>
struct Array2Storage {
  using ValueType = typename Array2<Ptr, I>::ValueType;
  Array2Storage(const Array2Size<I> &array2_size, I stride)
      : indexes_storage_(new I[array2_size.size1 + 1]),
        data_storage_(new ValueType[array2_size.size2 * stride]) {
    array_.Init(array2_size.size1, array2_size.size2, indexes_storage_.get(),
                DataPtrCreator<Ptr, I>::Create(data_storage_, stride));
    // just for case of empty Array2 object, may be written by the caller
    array_.indexes[0] = 0;
  }

  void FillIndexes(const std::vector<I> &indexes) {
    K2_CHECK_EQ(indexes.size(), array_.size1 + 1);
    std::copy(indexes.begin(), indexes.end(), array_.indexes);
  }

  void FillData(const std::vector<ValueType> &data) {
    K2_CHECK_EQ(data.size(), array_.size2);
    for (auto i = 0; i != array_.size2; ++i) array_.data[i] = data[i];
  }

  Array2<Ptr, I> &GetArray2() { return array_; }

 private:
  Array2<Ptr, I> array_;
  std::unique_ptr<I[]> indexes_storage_;
  std::unique_ptr<ValueType[]> data_storage_;
};

}  // namespace k2host

namespace std {
template <typename T, typename I>
struct iterator_traits<k2host::StridedPtr<T, I>> {
  typedef T value_type;
};

template <typename T, typename I>
void swap(k2host::StridedPtr<T, I> &lhs, k2host::StridedPtr<T, I> &rhs) {
  lhs.Swap(rhs);
}

template <typename Ptr, typename I>
void swap(k2host::Array2<Ptr, I> &lhs, k2host::Array2<Ptr, I> &rhs) {
  lhs.Swap(rhs);
}

}  // namespace std

#endif  // K2_CSRC_HOST_ARRAY_H_
