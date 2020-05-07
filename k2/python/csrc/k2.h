// k2/python/csrc/k2.h

// Copyright (c)  2020  Fangjun Kuang

// See ../../../LICENSE for clarification regarding multiple authors

#ifndef K2_PYTHON_CSRC_K2_H_
#define K2_PYTHON_CSRC_K2_H_

#include <cstdint>
#include <vector>

#include "pybind11/pybind11.h"
#include "pybind11/stl.h"
#include "pybind11/stl_bind.h"

namespace py = pybind11;

PYBIND11_MAKE_OPAQUE(std::vector<int32_t>);

#endif  // K2_PYTHON_CSRC_K2_H_
