#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -ex

CONDA_ENV_DIR=$CONDA_PREFIX

echo "K2_PYTHON_VERSION: $K2_PYTHON_VERSION"
echo "K2_TORCH_VERSION: $K2_TORCH_VERSION"
echo "K2_BUILD_TYPE: $K2_BUILD_TYPE"
echo "K2_BUILD_VERSION: $K2_BUILD_VERSION"
python3 --version

echo "CC is: $CC"
echo "GCC is: $GCC"
echo "gcc version: $($CC --version)"

export K2_CMAKE_ARGS="-DCMAKE_BUILD_TYPE=${K2_BUILD_TYPE} -DK2_WITH_CUDA=OFF"
export K2_MAKE_ARGS="-j"

python3 setup.py install --single-version-externally-managed --record=record.txt
