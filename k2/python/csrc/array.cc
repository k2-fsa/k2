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

  explicit DLPackArray1(py::capsule capsule) : tensor_(new Tensor(capsule)) {
    CHECK_EQ(tensor_->NumDim(), 1);
    CHECK_EQ(tensor_->Stride(0), 1)
        << "Only contiguous arrays are supported at present";

    auto num_elements =
        tensor_->Shape(0) * tensor_->BytesPerElement() / sizeof(ValueType);

    this->Init(0, num_elements, tensor_->Data<ValueType>());
  }

  // this constructor is for Pybind11 only.
  // The `operator []` of `Array2` returns an instance of type
  // `Array1` which is not wrapped to Python.
  DLPackArray1(const Parent &parent) : Parent(parent) {}

 private:
  std::shared_ptr<Tensor> tensor_;
};

template <typename Ptr, typename I = int32_t>
class DLPackArray2 : public Array2<Ptr, I> {
 public:
  using Parent = Array2<Ptr, I>;
  using ValueType = typename Parent::ValueType;

  DLPackArray2(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    CHECK_EQ(indexes_tensor_->dtype(), kInt32Type);
    CHECK_EQ(indexes_tensor_->NumDim(), 1);
    CHECK_GT(indexes_tensor_->Shape(0), 1);
    CHECK_EQ(indexes_tensor_->Stride(0), 1)
        << "Only contiguous index arrays are supported at present";

    CHECK_EQ(data_tensor_->NumDim(), 2);
    CHECK_EQ(data_tensor_->Stride(1), 1)
        << "Only contiguous data arrays at supported at present";
    CHECK_EQ(sizeof(ValueType),
             data_tensor_->Stride(0) * data_tensor_->BytesPerElement());

    this->Init(indexes_tensor_->Shape(0) - 1, indexes_tensor_->Data<int32_t>(),
               data_tensor_->Data<ValueType>());
  }

 private:
  std::shared_ptr<Tensor> indexes_tensor_;
  std::shared_ptr<Tensor> data_tensor_;
};

template <typename Ptr, typename I = int32_t>
void PybindArray1Tpl(py::module &m, const char *name) {
  py::class_<DLPackArray1<Ptr, I>>(m, name)
      .def(py::init<py::capsule>())
      .def("__getitem__",
           [](const DLPackArray1<Ptr, I> &self, int32_t i) { return self[i]; },
           py::return_value_policy::reference)
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

template <typename Ptr, typename I = int32_t>
void PybindArray2Tpl(py::module &m, const char *name) {
  using PyClass = DLPackArray2<Ptr, I>;
  py::class_<PyClass>(m, name)
      .def(py::init<py::capsule, py::capsule>(), py::arg("indexes"),
           py::arg("data"))
      .def("is_empty", &PyClass::Empty)
      .def("__len__", [](const PyClass &self) { return self.size; })
      .def("__getitem__", [](const PyClass &self, int32_t i) {
        return DLPackArray1<Ptr, I>(self[i]);
      });
}

static void PybindArray1(py::module &m) {
  PybindArray1Tpl<int32_t *>(m, "_IntArray1");
  PybindArray1Tpl<float *>(m, "_FloatArray1");
  PybindArray1Tpl<Arc *>(m, "_ArcArray1");
}

static void PybindArray2(py::module &m) {
  PybindArray2Tpl<Arc *>(m, "_ArcArray2");
}

static void PybindFsa_(py::module &m) {
  // __ArcArray1 and __ArcArray2 are for Fsa_ only.
  // We do not wrap their constructors so that
  // users cannot instantiate them in Python.
  //
  // They can be accessed only through `fsa.arcs`.
  py::class_<Array1<Arc *>>(m, "__ArcArray1")
      .def("__getitem__",
           [](const Array1<Arc *> &self, int32_t i) { return self[i]; },
           py::return_value_policy::reference)
      .def("__len__", [](const Array1<Arc *> &self) { return self.Size(); })
      .def("__iter__",
           [](const Array1<Arc *> &self) {
             return py::make_iterator(self.begin(), self.end());
           },
           py::keep_alive<0, 1>());

  py::class_<Array2<Arc *>>(m, "__ArcArray2")
      .def("is_empty", &Array2<Arc *>::Empty)
      .def("__len__", [](const Array2<Arc *> &self) { return self.size; })
      .def("__getitem__",
           [](const Array2<Arc *> &self, int32_t i) { return self[i]; });

  py::class_<Fsa_>(m, "Fsa_")
      .def(py::init<>())
      .def(py::init([](const DLPackArray2<Arc *> &arcs, int final_state) {
             // the memory is owned by Python
             //
             // refer to
             // https://pybind11.readthedocs.io/en/stable/upgrade.html
             return new Fsa_(arcs, final_state);
           }),
           // we should keep `arcs` alive as long as this `fsa` is alive
           // since memory is managed by `arcs`.
           // `1` means the first implicit `this` pointer.
           // `2` designates `arcs`.
           py::arg("arcs"), py::arg("final_state"), py::keep_alive<1, 2>())
      .def("num_states", &Fsa_::NumStates)
      .def("final_state", &Fsa_::FinalState)
      .def_readwrite("arcs", &Fsa_::arcs);
}

}  // namespace k2

void PybindArray(py::module &m) {
  k2::PybindArray1(m);
  k2::PybindArray2(m);
  k2::PybindFsa_(m);
}
