// k2/python/csrc/fsa.cc

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/fsa.h"

#include <memory>

#include "k2/csrc/fsa.h"
#include "k2/python/csrc/tensor.h"

namespace k2 {

// it uses external memory passed from DLPack (e.g., by PyTorch)
// to construct an Fsa.
class _Fsa : public Fsa {
 public:
  _Fsa(py::capsule cap_indexes, py::capsule cap_data)
      : indexes_tensor_(new Tensor(cap_indexes)),
        data_tensor_(new Tensor(cap_data)) {
    CHECK_EQ(indexes_tensor_->dtype(), kInt32Type);
    CHECK_EQ(indexes_tensor_->NumDim(), 1);
    CHECK_GT(indexes_tensor_->Shape(0), 1);
    CHECK_EQ(indexes_tensor_->Stride(0), 1)
        << "Only contiguous index arrays are supported at present";

    CHECK_EQ(data_tensor_->dtype(), kInt32Type);
    CHECK_EQ(data_tensor_->NumDim(), 2);
    CHECK_EQ(data_tensor_->Stride(1), 1)
        << "Only contiguous data arrays at supported at present";
    CHECK_EQ(sizeof(Arc),
             data_tensor_->Shape(1) * data_tensor_->BytesPerElement());

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
  // Note(fangjun): Users are not supposed to use `k2::Fsa` directly
  // in Python; the following wrapper is only used by pybind11 internally
  // so that it knows `k2::_Fsa` is a subclass of `k2::Fsa`.
  py::class_<k2::Fsa>(m, "__Fsa");

  using PyClass = k2::_Fsa;
  py::class_<PyClass, k2::Fsa>(m, "Fsa")
      .def(py::init<py::capsule, py::capsule>(), py::arg("indexes"),
           py::arg("data"))
      .def("empty", &PyClass::Empty)
      .def("num_states", &PyClass::NumStates)
      .def("final_state", &PyClass::FinalState);
}
