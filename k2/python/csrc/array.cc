// k2/python/csrc/array.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/array.h"

#include <memory>
#include <utility>

#include "k2/csrc/array.h"
#include "k2/python/csrc/tensor.h"

namespace k2 {
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
    CHECK_EQ(indexes_tensor_->NumDim(), 1);
    CHECK_GE(indexes_tensor_->Shape(0), 1);  // must have one element at least
    CHECK_EQ(indexes_tensor_->Stride(0), 1);

    CHECK_EQ(data_tensor_->NumDim(), 1);
    CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    CHECK_EQ(data_tensor_->Stride(0), 1);

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
    CHECK_EQ(indexes_tensor_->NumDim(), 1);
    CHECK_GE(indexes_tensor_->Shape(0), 1);  // must have one element at least
    CHECK_EQ(indexes_tensor_->Stride(0), 1);

    CHECK_EQ(data_tensor_->NumDim(), 2);
    CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    CHECK_EQ(data_tensor_->Shape(1) * data_tensor_->BytesPerElement(),
             sizeof(ValueType));
    CHECK_EQ(data_tensor_->Stride(0) * data_tensor_->BytesPerElement(),
             sizeof(ValueType));
    CHECK_EQ(data_tensor_->Stride(1), 1);

    int32_t size1 = indexes_tensor_->Shape(0) - 1;
    int32_t size2 = data_tensor_->Shape(0);
    this->Init(size1, size2, indexes_tensor_->Data<I>(),
               data_tensor_->Data<ValueType>());
  }

 private:
  std::unique_ptr<Tensor> indexes_tensor_;
  std::unique_ptr<Tensor> data_tensor_;
};

// Note: we can specialized for `StridedPtr` later if we need it.

}  // namespace k2

template <typename Ptr, bool IsPrimitive, typename I = int32_t>
void PybindArray2Tpl(py::module &m, const char *name) {
  using PyClass = k2::DLPackArray2<Ptr, IsPrimitive, I>;
  using Parent = k2::Array2<Ptr, I>;
  py::class_<PyClass, Parent>(m, name)
      .def(py::init<py::capsule, py::capsule>(), py::arg("indexes"),
           py::arg("data"))
      .def("empty", &PyClass::Empty)
      .def(
          "__iter__",
          [](const PyClass &self) {
            return py::make_iterator(self.begin(), self.end());
          },
          py::keep_alive<0, 1>())
      .def_readonly("size1", &PyClass::size1)
      .def_readonly("size2", &PyClass::size2)
      .def("indexes", [](const PyClass &self, I i) { return self.indexes[i]; })
      .def("data", [](const PyClass &self, I i) { return self.data[i]; });
  // TODO(haowen): expose `indexes` and `data` as an array
  // instead of a function call?
}

void PybindArray(py::module &m) {
  // Note: all the following wrappers whose name starts with `_` are only used
  // by pybind11 internally so that it knows `k2::DLPackArray2` is a subclass of
  // `k2::Array2`.
  py::class_<k2::Array2<int32_t *>>(m, "_IntArray2");
  PybindArray2Tpl<int32_t *, true>(m, "DLPackIntArray2");

  // note there is a type cast as the underlying Tensor is with type `float`
  py::class_<k2::Array2<std::pair<int32_t, float> *>>(m, "_LogSumArcDerivs");
  PybindArray2Tpl<std::pair<int32_t, float> *, false>(m,
                                                      "DLPackLogSumArcDerivs");
}
