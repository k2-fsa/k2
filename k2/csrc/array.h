// k2/csrc/array.h

// Copyright (c)  2020  Xiaomi Corporation (authors: Daniel Povey
//                                                   Haowen Qiu)

// See ../../LICENSE for clarification regarding multiple authors

#ifndef K2_CSRC_ARRAY_H_
#define K2_CSRC_ARRAY_H_

#include <algorithm>
#include <functional>
#include <iterator>
#include <limits>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "glog/logging.h"

namespace k2 {

/*
   We will use e.g. StridedPtr<T, I> when the stride is not 1, and
   otherwise just T* (which presumably be faster).
*/
template <typename T, typename I = int32_t>
struct StridedPtr {
  T *data;   // it is NOT owned here
  I stride;  // in number of elements, NOT number of bytes
  StridedPtr(T *data = nullptr, I stride = 0)  // NOLINT
      : data(data), stride(stride) {}
  StridedPtr(const StridedPtr &other)
      : data(other.data), stride(other.stride) {}
  StridedPtr(StridedPtr &&other) { this->Swap(other); }
  StridedPtr &operator=(StridedPtr other) {
    this->Swap(other);
    return *this;
  }

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
  StridedPtr operator++(int) {
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

template <typename Ptr, typename I = int32_t>
struct Array2 {
  // Irregular two dimensional array of something, like vector<vector<X> >
  // where Ptr is, or behaves like, X*.
  using IndexT = I;
  using PtrT = Ptr;
  using ValueType = typename std::iterator_traits<Ptr>::value_type;

  Array2() = default;
  Array2(IndexT size1, IndexT *indexes, IndexT size2, PtrT data)
      : size1(size1), indexes(indexes), size2(size2), data(data) {}

  IndexT size1;
  IndexT *indexes;  // indexes[0,1,...size1] should be defined; note, this
                    // means the array must be of at least size1+1.  We
                    // require that indexes[i] <= indexes[i+1], but it is
                    // not required that indexes[0] == 0, it may be
                    // greater than 0.
  IndexT size2;     // the number of elements in the array,  equal to
                    // indexes[size1] - indexes[0] (if the object Array2
                    // has been initialized).
  PtrT data;  // `data` might be an actual pointer, or might be some object
              // supporting operator [].  data[indexes[0]] through
              // data[indexes[size1] - 1] must be accessible through this
              // object.

  bool Empty() const { return size1 == 0; }

  PtrT begin() const { return data + indexes[0]; }

  PtrT end() const { return data + indexes[size1]; }

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

  IndexT size;
  IndexT *indexes1;  // indexes1[0,1,...size] should be defined; note,
                     // this means the array must be of at least size+1.
                     // We require that indexes[i] <= indexes[i+1], but it
                     // is not required that indexes[0] == 0, it may be
                     // greater than 0.

  IndexT *indexes2;  // indexes2[indexes1[0]]
                     // .. indexes2[indexes1[size]-1] should be defined.

  Ptr data;  // `data` might be an actual pointer, or might be some object
             // supporting operator [].  data[indexes2[indexes1[0]]] through
             // data[indexes2[indexes1[size] - 1]] must be accessible through
             // this object.

  Array2<Ptr, I> operator[](I i) const {
    DCHECK_GE(i, 0);
    DCHECK_LT(i, size);

    Array2<Ptr, I> array;
    array.size1 = indexes1[i + 1] - indexes1[i];
    array.indexes = indexes2 + indexes1[i];
    array.size2 = indexes2[indexes1[i + 1]] - indexes2[indexes1[i]];
    array.data = data;
    return array;
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
  static ValueType *Create(const std::unique_ptr<ValueType> &data_storage,
                           int32_t stride) {
    CHECK_EQ(stride, 1);
    return data_storage.get();
  }
};

// Specialized case where stride is bigger than 1 and the underlying data
// manager is StridedPtr, we will create a StridedPtr with data_storage and
// return it.
template <typename ValueType, typename I>
struct DataPtrCreator<StridedPtr<ValueType, I>, I> {
  static StridedPtr<ValueType, I> Create(
      const std::unique_ptr<ValueType> &data_storage, int32_t stride) {
    CHECK_GT(stride, 1);
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
  explicit Array2Storage(const Array2Size<I> &array2_size, int32_t stride)
      : indexes_storage_(new I[array2_size.size1 + 1]),
        data_storage_(new ValueType[array2_size.size2 * stride]) {
    array_.size1 = array2_size.size1;
    array_.size2 = array2_size.size2;
    array_.indexes = indexes_storage_.get();
    array_.data = DataPtrCreator<Ptr, I>::Create(data_storage_, stride);
  }

  void FillIndexes(const std::vector<I> &indexes) {
    CHECK_EQ(indexes.size(), array_.size1 + 1);
    for (auto i = 0; i != indexes.size(); ++i) array_.indexes[i] = indexes[i];
  }

  void FillData(const std::vector<ValueType> &data) {
    CHECK_EQ(data.size(), array_.size2);
    for (auto i = 0; i != array_.size2; ++i) array_.data[i] = data[i];
  }

  Array2<Ptr, I> &GetArray2() { return array_; }

 private:
  Array2<Ptr, I> array_;
  std::unique_ptr<I> indexes_storage_;
  std::unique_ptr<ValueType> data_storage_;
};

}  // namespace k2

namespace std {
template <typename T, typename I>

struct iterator_traits<k2::StridedPtr<T, I>> {
  typedef T value_type;
};

template <typename T, typename I>
void swap(k2::StridedPtr<T, I> &lhs, k2::StridedPtr<T, I> &rhs) {
  lhs.Swap(rhs);
}

template <typename Ptr, typename I>
void swap(k2::Array2<Ptr, I> &lhs, k2::Array2<Ptr, I> &rhs) {
  lhs.Swap(rhs);
}

}  // namespace std

#endif  // K2_CSRC_ARRAY_H_
