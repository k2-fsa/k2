#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

# The following environment variables are supposed to be set by users
#
# - K2_CUDA_VERSION_STR
#     It represents the cuda version. Example:
#
#       export K2_CUDA_VERSION_STR=10.1
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
# - K2_PYTHON_VERSION
#     The Python version. Example:
#
#       export K2_PYTHON_VERSION=3.8
#
#     Defaults to 3.8 if not set.
#     It is currently used only when creating the conda environment.
#
# - K2_CONDA_TOKEN
#     If not set, auto upload to anaconda.org is disabled.
#
#     Its value is from https://anaconda.org/k2-fsa/settings/access
#      (You need to login as user k2-fsa to see its value)
#

set -e

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)

cd $k2_dir

export K2_ROOT_DIR=$k2_dir
echo "K2_ROOT_DIR: $K2_ROOT_DIR"

if [ -z $K2_PYTHON_VERSION ]; then
  echo "env var K2_PYTHON_VERSION is not set, defaults to 3.8"
  K2_PYTHON_VERSION=3.8
fi

if [ -z $K2_CUDA_VERSION_STR ]; then
  echo "env var K2_CUDA_VERSION_STR is not set, defaults to 10.1"
  K2_CUDA_VERSION_STR=10.1
  K2_CUDA_VERSION=101
else
  K2_CUDA_VERSION=$(echo $K2_CUDA_VERSION_STR | python3 -c "import sys; ver = sys.stdin.read(); major, minor = ver.strip().split('.'); print(int(major)*10 + int(minor))")
fi

if [ -z $K2_TORCH_VERSION ]; then
  echo "env var K2_TORCH_VERSION is not set, defaults to 1.7.1"
  K2_TORCH_VERSION=1.7.1
fi

# Example value: 3.8
export K2_PYTHON_VERSION

# Example value: 10.1
export K2_CUDA_VERSION_STR

# Example value: 101
export K2_CUDA_VERSION

# Example value: 1.7.1
export K2_TORCH_VERSION

if false; then
  #
  # We assume the environment name is k2.
  # If you choose a different environment name, you have to modify
  # the variable `CONDA_ENV_DIR` in ./scripts/conda/k2/build.sh
  #
  conda create -n k2 python=$K2_PYTHON_VERSION
  conda activate k2
  conda install conda-build
  conda install -c conda-forge cudatoolkit-dev=$K2_CUDA_VERSION_STR
  conda install -c nvidia cudatoolkit=$K2_CUDA_VERSION_STR cudnn=8.0.4
  conda install -c pytorch pytorch=$K2_TORCH_VERSION cudatoolkit=$K2_CUDA_VERSION_STR
fi

K2_BUILD_VERSION=$(python3 ./get_version.py)

export K2_BUILD_VERSION

if [ -z $K2_CONDA_TOKEN ]; then
  echo "Auto upload to anaconda.org is disabled since K2_CONDA_TOKEN is not set"
  conda build --no-anaconda-upload -c pytorch ./scripts/conda/k2
else
  conda build --quiet -c pytorch --token $K2_CONDA_TOKEN ./scripts/conda/k2
fi
