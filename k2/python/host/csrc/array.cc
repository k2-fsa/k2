// k2/python/host/csrc/array.cc
// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/array.h"

#include <memory>
#include <utility>

#include "k2/csrc/host/array.h"
#include "k2/csrc/host/determinize_impl.h"
#include "k2/python/host/csrc/tensor.h"

namespace k2host {

/*
   DLPackArray1 initializes Array1 with `cap_data` which is a DLManagedTensor.

   `cap_data` is usually a one dimensional array with stride >= 1, i.e.,
   `cap_data.ndim == 1 && cap_indexes.strides[0] >= 1`.
*/
template <typename ValueType, typename I>
class DLPackArray1;

template <typename ValueType, typename I>
class DLPackArray1<ValueType *, I> : public Array1<ValueType *, I> {
 public:
  explicit DLPackArray1(py::capsule cap_data)
      : data_tensor_(new Tensor(cap_data)) {
    K2_CHECK_EQ(data_tensor_->NumDim(), 1);
    K2_CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    K2_CHECK_EQ(data_tensor_->Stride(0), 1);

    int32_t size = data_tensor_->Shape(0);
    this->Init(0, size, data_tensor_->Data<ValueType>());
  }

 private:
  std::unique_ptr<Tensor> data_tensor_;
};

template <typename ValueType, typename I>
class DLPackArray1<StridedPtr<ValueType, I>, I>
    : public Array1<StridedPtr<ValueType, I>, I> {
 public:
  using StridedPtrType = StridedPtr<ValueType, I>;
  explicit DLPackArray1(py::capsule cap_data)
      : data_tensor_(new Tensor(cap_data)) {
    K2_CHECK_EQ(data_tensor_->NumDim(), 1);
    K2_CHECK_GE(data_tensor_->Shape(0), 0);   // num-elements
    K2_CHECK_GE(data_tensor_->Stride(0), 1);  // stride > 1

    int32_t size = data_tensor_->Shape(0);
    StridedPtrType strided_ptr(data_tensor_->Data<ValueType>(),
                               data_tensor_->Stride(0));
    this->Init(0, size, strided_ptr);
  }

 private:
  std::unique_ptr<Tensor> data_tensor_;
};

/*
   DLPackArray2 initializes Array2 with `cap_indexes` and `cap_data` which are
   DLManagedTensors.

   `cap_indexes` is usually a one dimensional contiguous array, i.e.,
   `cap_indexes.ndim == 1 && cap_indexes.strides[0] == 1`.

   `cap_data` may have different shapes depending on `ValueType`:
       1. if `ValueType` is a primitive type (e.g. `int32_t`), it will be
          a one dimensional contiguous array, i.e.,
          `cap_data.ndim == 1 && cap_data.strides[0] == 1`.
       2. if `ValueType` is a complex type (e.g. Arc), it will be a two
          dimension array, i.e., it meets the following requirements:
          a) cap_data.ndim == 2.
          b) cap_data.shape[0] == num-elements it stores; note the
             element's type is `ValueType`, which means we view each row of
             `cap_data.data` as one element with type `ValueType`.
          c) cap_data.shape[1] == num-primitive-values in `ValueType`,
             which means we require that `ValueType` can be viewed as a tensor,
             this is true for Arc as it only holds primitive values with same
             type (i.e. `int32_t`), but may need type cast in other cases
             (e.g. ValueType contains both `int32_t` and `float`).
          d) cap_data.strides[0] == num-primitive-values in `ValueType`.
          e) cap_data.strides[1] == 1.

    Note if `data` in Array2 has stride > 1 (i.e. `data`'s type is
    StridedPtr<ValueType>), the requirement of `cap_data` is nearly same with
    case 2 above except cap_data.strides[0] will be greater than
    num-primitive-values in `ValueType`.

*/
template <typename ValueType, bool IsPrimitive, typename I>
class DLPackArray2;

template <typename ValueType, typename I>
class DLPackArray2<ValueType *, true, I> : public Array2<ValueType *, I> {
 public:
  DLPackArray2(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    K2_CHECK_EQ(indexes_tensor_->NumDim(), 1);
    K2_CHECK_GE(indexes_tensor_->Shape(0),
                1);  // must have one element at least
    K2_CHECK_EQ(indexes_tensor_->Stride(0), 1);

    K2_CHECK_EQ(data_tensor_->NumDim(), 1);
    K2_CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    K2_CHECK_EQ(data_tensor_->Stride(0), 1);

    int32_t size1 = indexes_tensor_->Shape(0) - 1;
    int32_t size2 = data_tensor_->Shape(0);
    this->Init(size1, size2, indexes_tensor_->Data<I>(),
               data_tensor_->Data<ValueType>());
  }

 private:
  std::unique_ptr<Tensor> indexes_tensor_;
  std::unique_ptr<Tensor> data_tensor_;
};

template <typename ValueType, typename I>
class DLPackArray2<ValueType *, false, I> : public Array2<ValueType *, I> {
 public:
  DLPackArray2(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    K2_CHECK_EQ(indexes_tensor_->NumDim(), 1);
    K2_CHECK_GE(indexes_tensor_->Shape(0),
                1);  // must have one element at least
    K2_CHECK_EQ(indexes_tensor_->Stride(0), 1);

    K2_CHECK_EQ(data_tensor_->NumDim(), 2);
    K2_CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    K2_CHECK_EQ(data_tensor_->Shape(1) * data_tensor_->BytesPerElement(),
                sizeof(ValueType));
    K2_CHECK_EQ(data_tensor_->Stride(0) * data_tensor_->BytesPerElement(),
                sizeof(ValueType));
    K2_CHECK_EQ(data_tensor_->Stride(1), 1);

    int32_t size1 = indexes_tensor_->Shape(0) - 1;
    int32_t size2 = data_tensor_->Shape(0);
    this->Init(size1, size2, indexes_tensor_->Data<I>(),
               data_tensor_->Data<ValueType>());
  }

 private:
  std::unique_ptr<Tensor> indexes_tensor_;
  std::unique_ptr<Tensor> data_tensor_;
};

template <typename ValueType, typename I>
class DLPackArray2<StridedPtr<ValueType, I>, true, I>
    : public Array2<StridedPtr<ValueType, I>, I> {
 public:
  using StridedPtrType = StridedPtr<ValueType, I>;
  DLPackArray2(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    K2_CHECK_EQ(indexes_tensor_->NumDim(), 1);
    K2_CHECK_GE(indexes_tensor_->Shape(0),
                1);  // must have one element at least
    K2_CHECK_EQ(indexes_tensor_->Stride(0), 1);

    K2_CHECK_EQ(data_tensor_->NumDim(), 1);
    K2_CHECK_GE(data_tensor_->Shape(0), 0);   // num-elements
    K2_CHECK_GE(data_tensor_->Stride(0), 1);  // stride > 1

    int32_t size1 = indexes_tensor_->Shape(0) - 1;
    int32_t size2 = data_tensor_->Shape(0);
    StridedPtrType strided_ptr(data_tensor_->Data<ValueType>(),
                               data_tensor_->Stride(0));
    this->Init(size1, size2, indexes_tensor_->Data<I>(), strided_ptr);
  }

 private:
  std::unique_ptr<Tensor> indexes_tensor_;
  std::unique_ptr<Tensor> data_tensor_;
};

}  // namespace k2host

template <typename Ptr, typename I = int32_t>
void PybindArray1Tpl(py::module &m, const char *name) {
  using PyClass = k2host::DLPackArray1<Ptr, I>;
  using Parent = k2host::Array1<Ptr, I>;
  py::class_<PyClass, Parent>(m, name)
      .def(py::init<py::capsule>(), py::arg("data"))
      .def("empty", &PyClass::Empty)
      .def(
          "get_base", [](PyClass &self) -> Parent * { return &self; },
          py::return_value_policy::reference_internal)
      .def_readonly("size", &PyClass::size)
      .def(
          "get_data",
          [](const PyClass &self, I i) {
            if (i >= self.size) throw py::index_error();
            return self.data[self.begin + i];
          },
          "just for test purpose to check if k2host::Array1 and the "
          "underlying tensor are sharing memory.");
}

template <typename Ptr, bool IsPrimitive, typename I = int32_t>
void PybindArray2Tpl(py::module &m, const char *name) {
  using PyClass = k2host::DLPackArray2<Ptr, IsPrimitive, I>;
  using Parent = k2host::Array2<Ptr, I>;
  py::class_<PyClass, Parent>(m, name)
      .def(py::init<py::capsule, py::capsule>(), py::arg("indexes"),
           py::arg("data"))
      .def("empty", &PyClass::Empty)
      .def(
          "get_base", [](PyClass &self) -> Parent * { return &self; },
          py::return_value_policy::reference_internal)
      .def_readonly("size1", &PyClass::size1)
      .def_readonly("size2", &PyClass::size2)
      .def(
          "get_indexes",
          [](const PyClass &self, I i) {
            if (i > self.size1)  // note indexes.size == size1+1
              throw py::index_error();
            return self.indexes[i];
          },
          "just for test purpose to check if k2host::Array2 and the "
          "underlying tensor are sharing memory.")
      .def(
          "get_data",
          [](const PyClass &self, I i) {
            if (i >= self.size2) throw py::index_error();
            return self.data[self.indexes[0] + i];
          },
          "just for test purpose to check if k2host::Array2 and the "
          "underlying tensor are sharing memory.");
}

template <typename I>
void PybindArray2SizeTpl(py::module &m, const char *name) {
  using PyClass = k2host::Array2Size<I>;
  py::class_<PyClass>(m, name)
      .def(py::init<>())
      .def(py::init<int32_t, int32_t>(), py::arg("size1"), py::arg("size2"))
      .def_readwrite("size1", &PyClass::size1)
      .def_readwrite("size2", &PyClass::size2);
}

void PybindArray(py::module &m) {
  // Note: all the following wrappers whose name starts with `_` are only used
  // by pybind11 internally so that it knows `k2host::DLPackArray1` is a
  // subclass of `k2host::Array1`.
  py::class_<k2host::Array1<int32_t *>>(m, "_IntArray1");
  PybindArray1Tpl<int32_t *>(m, "DLPackIntArray1");

  py::class_<k2host::Array1<k2host::StridedPtr<int32_t>>>(m,
                                                          "_StridedIntArray1");
  PybindArray1Tpl<k2host::StridedPtr<int32_t>>(m, "DLPackStridedIntArray1");

  py::class_<k2host::Array1<float *>>(m, "_FloatArray1");
  PybindArray1Tpl<float *>(m, "DLPackFloatArray1");

  py::class_<k2host::Array1<double *>>(m, "_DoubleArray1");
  PybindArray1Tpl<double *>(m, "DLPackDoubleArray1");

  // Note: all the following wrappers whose name starts with `_` are only used
  // by pybind11 internally so that it knows `k2host::DLPackArray2` is a
  // subclass of `k2host::Array2`.
  py::class_<k2host::Array2<int32_t *>>(m, "_IntArray2");
  PybindArray2Tpl<int32_t *, true>(m, "DLPackIntArray2");

  // note there is a type cast as the underlying Tensor is with type `float`
  using LogSumDerivType = typename k2host::LogSumTracebackState::DerivType;
  py::class_<k2host::Array2<LogSumDerivType *>>(m, "_LogSumArcDerivs");
  PybindArray2Tpl<LogSumDerivType *, false>(m, "DLPackLogSumArcDerivs");
}

void PybindArray2Size(py::module &m) {
  PybindArray2SizeTpl<int32_t>(m, "IntArray2Size");
}
