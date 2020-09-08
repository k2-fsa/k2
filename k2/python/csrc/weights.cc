// k2/python/csrc/weights.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/csrc/weights.h"

#include <memory>

#include "k2/csrc/old/weights.h"

void PybindFbWeightType(py::module &m) {
  using PyEnum = k2::FbWeightType;
  py::enum_<PyEnum>(m, "FbWeightType", py::arithmetic())
      .value("kMaxWeight", PyEnum::kMaxWeight)
      .value("kLogSumWeight", PyEnum::kLogSumWeight);
}

void PybindWfsaWithFbWeights(py::module &m) {
  using PyClass = k2::WfsaWithFbWeights;
  py::class_<PyClass>(m, "_WfsaWithFbWeights")
      .def(py::init(
          [](const k2::Fsa &fsa, const k2::Array1<float *> *arc_weights,
             k2::FbWeightType type, k2::Array1<double *> *forward_state_weights,
             k2::Array1<double *> *backward_state_weights) {
            return std::unique_ptr<PyClass>(new PyClass(
                fsa, arc_weights->data, type, forward_state_weights->data,
                backward_state_weights->data));
          }))
      // We do not expose `self.fsa`, `self.arc_weights`,
      // `self.ForwardStateWeights` `self.BackwardStateWeights` here as they
      // are passed to the constructor of `WfsaWeightFbWeights` from Python
      // code, users can access the Python variables directly as those variables
      // have at least the same lifetime with this WfsaWithFbWeights object.
      // Note those properties and methods will usually not be used in Python
      // code.
      .def_readonly("weight_type", &PyClass::weight_type);
}

void PybindWeights(py::module &m) {
  PybindFbWeightType(m);
  PybindWfsaWithFbWeights(m);
}
