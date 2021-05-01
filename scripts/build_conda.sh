#!/usr/bin/env bash
#
# Copyright (c)  2021  Xiaomi Corp.       (author: Fangjun Kuang)

set -xe

cur_dir=$(cd $(dirname $BASH_SOURCE) && pwd)
k2_dir=$(cd $cur_dir/.. && pwd)
build_dir=$k2_dir/build

cd $k2_dir

export K2_ROOT_DIR=$k2_dir
echo "K2_ROOT_DIR: $K2_ROOT_DIR"

if false; then
  # This is for demonstration only.
  # You can build an environment to create conda packages
  conda create -n k2 python=3.8
  conda activate k2
  conda install -c conda-forge cudatoolkit-dev=10.1
  conda install -c nvidia cudatoolkit=10.1 cudnn=8.0.4
  conda install -c pytorch pytorch=1.7.1 cudatoolkit=10.1
fi

BUILD_VERSION=$(python3 ./get_version.py)
export BUILD_VERSION

conda build --no-anaconda-upload -c pytorch ./scripts/conda/k2
