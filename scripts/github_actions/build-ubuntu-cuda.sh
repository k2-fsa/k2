#!/usr/bin/env bash
#
set -ex

if [ -z $PYTHON_VERSION ]; then
  echo "Please set the environment variable PYTHON_VERSION"
  echo "Example: export PYTHON_VERSION=3.8"
  # Valid values: 3.6, 3.7, 3.8, 3.9, 3.10, 3.11
  exit 1
fi

if [ -z $TORCH_VERSION ]; then
  echo "Please set the environment variable TORCH_VERSION"
  echo "Example: export TORCH_VERSION=1.10.0"
  exit 1
fi

if [ -z $CUDA_VERSION ]; then
  echo "Please set the environment variable CUDA_VERSION"
  echo "Example: export CUDA_VERSION=10.2"
  # valid values: 10.2, 11.1, 11.3, 11.6, 11.7, 11.8, 12.1
  exit 1
fi


yum -y install openssl-devel bzip2-devel libffi-devel xz-devel wget redhat-lsb-core


echo "Installing ${PYTHON_VERSION}.3"
curl -O https://www.python.org/ftp/python/${PYTHON_VERSION}.3/Python-${PYTHON_VERSION}.3.tgz
tar xf Python-${PYTHON_VERSION}.3.tgz
pushd Python-${PYTHON_VERSION}.3

PYTHON_INSTALL_DIR=$PWD/py-${PYTHON_VERSION}

if [[ $PYTHON_VERSION =~ 3.1. ]]; then
  yum install -y openssl11-devel
  sed -i 's/PKG_CONFIG openssl /PKG_CONFIG openssl11 /g' configure
fi

./configure --enable-shared --prefix=$PYTHON_INSTALL_DIR >/dev/null 2>&1
make install >/dev/null 2>&1

popd

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH
ls -lh $PYTHON_INSTALL_DIR/lib/

python3 --version
which python3

if [[ $PYTHON_VERSION != 3.6 ]]; then
  curl -O https://bootstrap.pypa.io/get-pip.py
  python3 get-pip.py
fi

python3 -m pip install scikit-build
python3 -m pip install -U pip cmake
python3 -m pip install wheel twine typing_extensions
python3 -m pip install bs4 requests tqdm auditwheel

echo "Installing torch"
./install_torch.sh

rm -rf ~/.cache/pip >/dev/null 2>&1
yum clean all >/dev/null 2>&1

cd /var/www

export CMAKE_CUDA_COMPILER_LAUNCHER=
export K2_CMAKE_ARGS="-DCUDAToolkit_TARGET_DIR=/usr/local/cuda/targets/x86_64-linux -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export K2_MAKE_ARGS=" -j2 "

python3 setup.py bdist_wheel

auditwheel --verbose repair \
  --exclude libc10.so \
  --exclude libc10_cuda.so \
  --exclude libcuda.so.1 \
  --exclude libcudart.so.${CUDA_VERSION} \
  --exclude libnvToolsExt.so.1 \
  --exclude libnvrtc.so.${CUDA_VERSION} \
  --exclude libtorch.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_cuda.so \
  --exclude libtorch_python.so \
  \
  --exclude libcudnn.so.8 \
  --exclude libcublas.so.11 \
  --exclude libcublasLt.so.11 \
  --exclude libcudart.so.11.0 \
  --exclude libcudart.so.12 \
  --exclude libnvrtc.so.11.2 \
  --exclude libtorch_cuda_cu.so \
  --exclude libtorch_cuda_cpp.so \
  --plat manylinux_2_17_x86_64 \
  -w /var/www/wheelhouse \
  dist/*.whl

ls -lh  /var/www
