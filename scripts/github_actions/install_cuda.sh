#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

echo "cuda version: $cuda"

case "$cuda" in
  10.0)
    url=https://developer.nvidia.com/compute/cuda/10.0/Prod/local_installers/cuda_10.0.130_410.48_linux
    filename=cuda_10.0.130_410.48_linux
    ;;
  10.1)
    url=https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
    filename=cuda_10.1.105_418.39_linux.run
    ;;
  10.2)
    url=http://developer.download.nvidia.com/compute/cuda/10.2/Prod/local_installers/cuda_10.2.89_440.33.01_linux.run
    filename=cuda_10.2.89_440.33.01_linux.run
    ;;
  *)
    echo "Unknown cuda version: $cuda"
    exit 1
    ;;
esac


curl -LSs -O $url
chmod +x ./$filename
sudo ./$filename --toolkit --silent

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
nvcc --version
