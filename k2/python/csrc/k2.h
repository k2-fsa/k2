// k2/python/csrc/k2.h

// Copyright (c)  2020  Fangjun Kuang (csukuangfj@gmail.com)

// See ../../../LICENSE for clarification regarding multiple authors

#ifndef K2_PYTHON_CSRC_K2_H_
#define K2_PYTHON_CSRC_K2_H_

#include <vector>

#include "k2/csrc/fsa.h"
#include "pybind11/pybind11.h"
#include "pybind11/stl.h"

namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

PYBIND11_MAKE_OPAQUE(std::vector<k2::Arc>);
// PYBIND11_MAKE_OPAQUE(std::vector<k2::Fsa>);
// PYBIND11_MAKE_OPAQUE(std::vector<k2::Cfsa>);

#endif  // K2_PYTHON_CSRC_K2_H_
