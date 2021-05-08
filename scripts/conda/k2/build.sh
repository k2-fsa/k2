#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -ex

CONDA_ENV_DIR=$CONDA_PREFIX
if [ $K2_IS_GITHUB_ACTIONS -eq 1 ]; then
  NUM_JOBS="-j3"
else
  NUM_JOBS="-j"
fi

rm -rf build
mkdir build
cd build

echo "K2_PYTHON_VERSION: $K2_PYTHON_VERSION"
echo "K2_TORCH_VERSION: $K2_TORCH_VERSION"
echo "K2_CUDA_VERSION: $K2_CUDA_VERSION"
echo "K2_CUDA_VERSION_STR: $K2_CUDA_VERSION_STR"
echo "K2_BUILD_TYPE: $K2_BUILD_TYPE"
echo "K2_BUILD_VERSION: $K2_BUILD_VERSION"

echo "CC is: $CC"
echo "CXX is: $CXX"
echo "GXX is: $GXX"
echo "which nvcc: $(which nvcc)"
echo "gcc version: $($CC --version)"
echo "nvcc version: $(nvcc --version)"

cmake -DCMAKE_BUILD_TYPE=$K2_BUILD_TYPE ..
cat k2/csrc/version.h

make $NUM_JOBS _k2
cd ..

python3 setup.py install --single-version-externally-managed --record=record.txt
