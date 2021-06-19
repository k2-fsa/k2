#!/usr/bin/env bash
#
# Copyright      2021  Xiaomi Corp.       (author: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

set -ex

CONDA_ENV_DIR=$CONDA_PREFIX

echo "K2_PYTHON_VERSION: $K2_PYTHON_VERSION"
echo "K2_TORCH_VERSION: $K2_TORCH_VERSION"
echo "K2_CUDA_VERSION: $K2_CUDA_VERSION"
echo "K2_BUILD_TYPE: $K2_BUILD_TYPE"
echo "K2_BUILD_VERSION: $K2_BUILD_VERSION"
python3 --version

echo "CC is: $CC"
echo "GCC is: $GCC"
echo "which nvcc: $(which nvcc)"
echo "gcc version: $($CC --version)"
echo "nvcc version: $(nvcc --version)"

export K2_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${K2_BUILD_TYPE}"
export K2_MAKE_ARGS="-j2"

python3 setup.py install --single-version-externally-managed --record=record.txt
