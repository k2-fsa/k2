// k2/python/csrc/fsa.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)
//                      Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa.h"

#include <memory>

#include "k2/csrc/fsa.h"
#include "k2/python/csrc/tensor.h"

namespace k2 {

// it uses external memory passed from DLPack (e.g., by PyTorch)
// to construct an Fsa.
class DLPackFsa : public Fsa {
 public:
  DLPackFsa(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    CHECK_EQ(indexes_tensor_->dtype(), kInt32Type);
    CHECK_EQ(indexes_tensor_->NumDim(), 1);
    CHECK_GE(indexes_tensor_->Shape(0), 1);
    CHECK_EQ(indexes_tensor_->Stride(0), 1);

    CHECK_EQ(data_tensor_->dtype(), kInt32Type);
    CHECK_EQ(data_tensor_->NumDim(), 2);
    CHECK_GE(data_tensor_->Shape(0), 0);  // num-elements
    CHECK_EQ(data_tensor_->Shape(1) * data_tensor_->BytesPerElement(),
             sizeof(Arc));
    CHECK_EQ(data_tensor_->Stride(0) * data_tensor_->BytesPerElement(),
             sizeof(Arc));
    CHECK_EQ(data_tensor_->Stride(1), 1);

    int32_t size1 = indexes_tensor_->Shape(0) - 1;
    int32_t size2 = data_tensor_->Shape(0);
    this->Init(size1, size2, indexes_tensor_->Data<int32_t>(),
               data_tensor_->Data<Arc>());
  }

 private:
  std::unique_ptr<Tensor> indexes_tensor_;
  std::unique_ptr<Tensor> data_tensor_;
};

}  // namespace k2

void PybindArc(py::module &m) {
  using PyClass = k2::Arc;
  py::class_<PyClass>(m, "Arc")
      .def(py::init<>())
      .def(py::init<int32_t, int32_t, int32_t>(), py::arg("src_state"),
           py::arg("dest_state"), py::arg("label"))
      .def_readwrite("src_state", &PyClass::src_state)
      .def_readwrite("dest_state", &PyClass::dest_state)
      .def_readwrite("label", &PyClass::label)
      .def("__str__", [](const PyClass &self) {
        std::ostringstream os;
        os << self;
        return os.str();
      });
}

void PybindFsa(py::module &m) {
  // The following wrapper is only used by pybind11 internally
  // so that it knows `k2::DLPackFsa` is a subclass of `k2::Fsa`.
  py::class_<k2::Fsa>(m, "_Fsa");

  using PyClass = k2::DLPackFsa;
  py::class_<PyClass, k2::Fsa>(m, "DLPackFsa")
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
      .def("indexes",
           [](const PyClass &self, int32_t i) { return self.indexes[i]; })
      .def("data", [](const PyClass &self, int32_t i) { return self.data[i]; })
      .def("num_states", &PyClass::NumStates)
      .def("final_state", &PyClass::FinalState);
}
