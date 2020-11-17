/**
 * @brief python wrappers for k2/csrc/version.h
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#ifndef K2_PYTHON_CSRC_VERSION_H_
#define K2_PYTHON_CSRC_VERSION_H_

#include "k2/python/csrc/k2.h"

void PybindVersion(py::module &m);

#endif  // K2_PYTHON_CSRC_VERSION_H_
