// k2/python/csrc/array.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/array.h"

#include <memory>
#include <type_traits>

#include "k2/python/csrc/tensor.h"

namespace k2 {

template <typename Ptr, typename I = int32_t>
class DLPackArray1 : public Array1<Ptr, I> {
 public:
  using Parent = Array1<Ptr, I>;
  using ValueType = typename Parent::ValueType;
  DLPackArray1(py::capsule capsule) : tensor_(new Tensor(capsule)) {
    if (std::is_same<ValueType, int32_t>::value) {
      CHECK_EQ(tensor_->dtype(), kInt32Type);
    } else if (std::is_same<ValueType, float>::value) {
      CHECK_EQ(tensor_->dtype(), kFloatType);
    }

    CHECK_EQ(tensor_->NumDim(), 1);
    CHECK_EQ(tensor_->Stride(0), 1)
        << "Only contiguous arrays are supported at present";
    this->Init(0, tensor_->Shape(0), tensor_->Data<ValueType>());
  }

 private:
  std::shared_ptr<Tensor> tensor_;
};

template <typename Ptr, typename I = int32_t>
void PybindArray1Tpl(py::module &m, const char *name) {
  py::class_<DLPackArray1<Ptr, I>>(m, name)
      .def(py::init<py::capsule>())
      .def("__getitem__",
           [](const DLPackArray1<Ptr, I> &self, int32_t i) { return self[i]; })
      .def("__setitem__",
           [](DLPackArray1<Ptr, I> *self, int32_t i,
              const typename DLPackArray1<Ptr, I>::ValueType &v) {
             (*self)[i] = v;
           })
      .def("__len__",
           [](const DLPackArray1<Ptr, I> &self) { return self.Size(); })
      .def("__iter__",
           [](const DLPackArray1<Ptr, I> &self) {
             return py::make_iterator(self.begin(), self.end());
           },
           py::keep_alive<0, 1>());
}

void PybindArray1(py::module &m) {
  PybindArray1Tpl<int32_t *>(m, "_IntArray1");
  PybindArray1Tpl<float *>(m, "_FloatArray1");
}

}  // namespace k2

void PybindArray(py::module &m) { k2::PybindArray1(m); }
