#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -ex

CONDA_ROOT=$(conda config --show root_prefix | cut -f2 -d' ')
CONDA_ENV_DIR=$CONDA_ROOT/envs/k2
NUM_JOBS="-j"
if [ ! -d $CONDA_ENV_DIR ]; then
  # This is for the miniconda used in GitHub actions.
  # It turns out CONDA_ENV_DIR == CONDA_ROOT in GitHub actions
  CONDA_ENV_DIR=$CONDA_ROOT
  NUM_JOBS="-j1"
fi

find $CONDA_ENV_DIR -name cudnn.h

sed -i '59i#define CUDNN_MAJOR 8' $CONDA_ENV_DIR/include/cudnn.h

rm -rf build
mkdir build
cd build

cmake \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_COMPILER=$(which nvcc) \
  -DCMAKE_CXX_COMPILER=$(which g++) \
  -DCMAKE_C_COMPILER=$(which gcc) \
  -DPYTHON_EXECUTABLE=$(which python3) \
  -DCUDNN_LIBRARY_PATH=$CONDA_ENV_DIR/lib/libcudnn.so \
  -DCUDNN_INCLUDE_PATH=$CONDA_ENV_DIR/include \
  ..



make $NUM_JOBS _k2
cd ..

python3 setup.py install --single-version-externally-managed --record=record.txt
