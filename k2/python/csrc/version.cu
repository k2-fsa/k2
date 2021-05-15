/**
 * @brief Python wrappers for k2/csrc/version.h
 *
 * @copyright
 * Copyright (c)  2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 */

#include "k2/csrc/log.h"
#include "k2/csrc/version.h"
#include "k2/python/csrc/version.h"

void PybindVersion(py::module &m) {
  py::module version = m.def_submodule("version", "k2 version information");

  version.attr("__version__") = k2::kVersion;
  version.attr("git_sha1") = k2::kGitSha1;
  version.attr("git_date") = k2::kGitDate;
  version.attr("cuda_version") = k2::kCudaVersion;
  version.attr("cudnn_version") = k2::kCudnnVersion;
  version.attr("python_version") = k2::kPythonVersion;
  version.attr("build_type") = k2::kBuildType;
  version.attr("os_type") = k2::kOS;
  version.attr("cmake_version") = k2::kCMakeVersion;
  version.attr("gcc_version") = k2::kGCCVersion;
  version.attr("cmake_cuda_flags") = k2::kCMakeCudaFlags;
  version.attr("cmake_cxx_flags") = k2::kCMakeCxxFlags;
  version.attr("torch_version") = k2::kTorchVersion;
  version.attr("torch_cuda_version") = k2::kTorchCudaVersion;
  version.attr("enable_nvtx") = k2::kEnableNvtx;
  version.attr("disable_debug") = k2::internal::kDisableDebug;
}
