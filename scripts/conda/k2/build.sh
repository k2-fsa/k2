#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -ex

CONDA_ENV_DIR=$CONDA_PREFIX
NUM_JOBS="-j1"
if [ -z $K2_IS_GITHUB_ACTIONS ]; then
  NUM_JOBS="-j"
fi


rm -rf build
mkdir build
cd build

echo "CC is: $CC"
echo "CXX is: $CXX"
echo "GXX is: $GXX"
echo "which nvcc: $(which nvcc)"

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  ..

# cmake \
#   -DCMAKE_BUILD_TYPE=Release \
#   -DCMAKE_CUDA_COMPILER=$(which nvcc) \
#   -DPYTHON_EXECUTABLE=$(which python3) \
#   -DCUDNN_LIBRARY_PATH=$CONDA_ENV_DIR/lib/libcudnn.so \
#   -DCUDNN_INCLUDE_PATH=$CONDA_ENV_DIR/include \
#   ..

make $NUM_JOBS _k2
cd ..

python3 setup.py install --single-version-externally-managed --record=record.txt
