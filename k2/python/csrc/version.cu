/**
 * @brief Python wrappers for k2/csrc/version.h
 *
 * @copyright
 * Copyright      2020  Mobvoi Inc.        (authors: Fangjun Kuang)
 *
 * @copyright
 * See LICENSE for clarification regarding multiple authors
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <string>

#include "k2/csrc/log.h"
#include "k2/csrc/version.h"
#include "k2/python/csrc/version.h"

void PybindVersion(py::module &m) {
  py::module version = m.def_submodule("version", "k2 version information");

  version.attr("__version__") = std::string(k2::kVersion);
  version.attr("git_sha1") = std::string(k2::kGitSha1);
  version.attr("git_date") = std::string(k2::kGitDate);
  version.attr("cuda_version") = std::string(k2::kCudaVersion);
  version.attr("cudnn_version") = std::string(k2::kCudnnVersion);
  version.attr("python_version") = std::string(k2::kPythonVersion);
  version.attr("build_type") = std::string(k2::kBuildType);
  version.attr("os_type") = std::string(k2::kOS);
  version.attr("cmake_version") = std::string(k2::kCMakeVersion);
  version.attr("gcc_version") = std::string(k2::kGCCVersion);
  version.attr("cmake_cuda_flags") = std::string(k2::kCMakeCudaFlags);
  version.attr("cmake_cxx_flags") = std::string(k2::kCMakeCxxFlags);
  version.attr("torch_version") = std::string(k2::kTorchVersion);
  version.attr("torch_cuda_version") = std::string(k2::kTorchCudaVersion);
  version.attr("enable_nvtx") = k2::kEnableNvtx;
  version.attr("disable_debug") = k2::internal::kDisableDebug;
  version.attr("with_cuda") = k2::kWithCuda;
}
