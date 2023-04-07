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

# To use this script, we assume that you have installed cudatoolkit locally.
# That is, `which nvcc` should give the path to nvcc
#
# We also assume that cudnn is installed locally.
#
# The following environment variables are supposed to be set by users
#
# - K2_CUDA_VERSION
#     It represents the cuda version. Example:
#
#       export K2_CUDA_VERSION=10.1
#
#     Defaults to 10.1 if not set.
#
# - K2_TORCH_VERSION
#     The PyTorch version. Example:
#
#       export K2_TORCH_VERSION=1.7.1
#
#     Defaults to 1.7.1 if not set.
#
# - K2_CONDA_TOKEN
#     If not set, auto upload to anaconda.org is disabled.
#
#     Its value is from https://anaconda.org/k2-fsa/settings/access
#      (You need to login as user k2-fsa to see its value)
#
# - K2_BUILD_TYPE
#     If not set, defaults to Release.

set -e
export CONDA_BUILD=1

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

cd $k2_dir

export K2_ROOT_DIR=$k2_dir
echo "K2_ROOT_DIR: $K2_ROOT_DIR"

K2_PYTHON_VERSION=$(python -c "import sys; print('.'.join(sys.version.split('.')[:2]))")

if [ -z $K2_CUDA_VERSION ]; then
  echo "env var K2_CUDA_VERSION is not set, defaults to 10.1"
  K2_CUDA_VERSION=10.1
fi

if [ -z $K2_TORCH_VERSION ]; then
  echo "env var K2_TORCH_VERSION is not set, defaults to 1.7.1"
  K2_TORCH_VERSION=1.7.1
fi

if [ -z $K2_BUILD_TYPE ]; then
  echo "env var K2_BUILD_TYPE is not set, defaults to Release"
  K2_BUILD_TYPE=Release
fi

export K2_IS_FOR_CONDA=1
K2_BUILD_VERSION=$(python3 ./get_version.py)

# Example value: 3.8
export K2_PYTHON_VERSION

# Example value: 10.1
export K2_CUDA_VERSION

# Example value: 1.7.1
export K2_TORCH_VERSION

export K2_BUILD_VERSION

export K2_BUILD_TYPE

if [ ! -z $K2_IS_GITHUB_ACTIONS ]; then
  export K2_IS_GITHUB_ACTIONS
  conda remove -q pytorch
  if [[ $K2_TORCH_VERSION != "1.13.0" && $K2_TORCH_VERSION != "1.13.1" ]]; then
    conda remove -q cudatoolkit
  fi
  conda clean -q -a
else
  export K2_IS_GITHUB_ACTIONS=0
fi

if [ -z $K2_CONDA_TOKEN ]; then
  echo "Auto upload to anaconda.org is disabled since K2_CONDA_TOKEN is not set"
  if [[ $K2_TORCH_VERSION == "1.13.0" || $K2_TORCH_VERSION == "1.13.1" ]]; then
    # From torch 1.13, the command to install torch is
    # conda install pytorch torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    conda build --no-test --no-anaconda-upload -c pytorch -c nvidia ./scripts/conda-torch-ge-1.13/k2
  else
    conda build --no-test --no-anaconda-upload -c pytorch -c conda-forge ./scripts/conda/k2
  fi
else
  if [[ $K2_TORCH_VERSION == "1.13.0" || $K2_TORCH_VERSION == "1.13.1" ]]; then
    # From torch 1.13, the command to install torch is
    # conda install pytorch torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia
    conda build --no-test -c pytorch -c nvidia --token $K2_CONDA_TOKEN ./scripts/conda-torch-ge-1.13/k2
  else
    conda build --no-test -c pytorch -c conda-forge --token $K2_CONDA_TOKEN ./scripts/conda/k2
  fi
fi
