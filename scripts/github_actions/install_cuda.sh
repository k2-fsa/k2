#!/bin/bash

ls -l /usr
ls -l /usr/local

if false; then

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-ubuntu1604.pin
sudo mv cuda-ubuntu1604.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/ /"
sudo apt-get -q update
sudo apt-get -y -q install \
  cuda-command-line-tools-10-1 \
  cuda-libraries-dev-10-1 \

wget https://developer.download.nvidia.cn/compute/cuda/repos/ubuntu1604/x86_64/libcublas10_10.1.0.105-1_amd64.deb
sudo dpkg -i libcublas10_10.1.0.105-1_amd64.deb
fi

curl -LSs -O https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run
chmod +x ./cuda_10.1.105_418.39_linux.run
sudo ./cuda_10.1.105_418.39_linux.run --toolkit --samples --silent

ls -l /usr
ls -l /usr/local
ls -l /usr/local/cuda
ls -l /usr/local/cuda-10.1

export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
nvcc --version

echo "searching cublas lib:"
find /usr/local -name "libcublas*" 2>/dev/null
# find /lib -name "*cublas*" 2>/dev/null
