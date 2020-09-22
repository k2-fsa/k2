// k2/python/host/csrc/weights.cc

// Copyright (c)  2020  Xiaomi Corporation (author: Haowen Qiu)

// See ../../../LICENSE for clarification regarding multiple authors

#include "k2/python/host/csrc/weights.h"

#include <memory>

#include "k2/csrc/host/weights.h"

void PybindFbWeightType(py::module &m) {
  using PyEnum = k2host::FbWeightType;
  py::enum_<PyEnum>(m, "FbWeightType", py::arithmetic())
      .value("kMaxWeight", PyEnum::kMaxWeight)
      .value("kLogSumWeight", PyEnum::kLogSumWeight);
}

void PybindWfsaWithFbWeights(py::module &m) {
  using PyClass = k2host::WfsaWithFbWeights;
  py::class_<PyClass>(m, "_WfsaWithFbWeights")
      .def(py::init([](const k2host::Fsa &fsa, k2host::FbWeightType type,
                       k2host::Array1<double *> *forward_state_weights,
                       k2host::Array1<double *> *backward_state_weights) {
        return std::unique_ptr<PyClass>(
            new PyClass(fsa, type, forward_state_weights->data,
                        backward_state_weights->data));
      }))
      // We do not expose `self.fsa`,
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
