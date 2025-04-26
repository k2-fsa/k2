#!/usr/bin/env bash
#
set -ex

if [ -z $PYTHON_VERSION ]; then
  echo "Please set the environment variable PYTHON_VERSION"
  echo "Example: export PYTHON_VERSION=3.8"
  # Valid values: 3.8, 3.9, 3.10, 3.11
  exit 1
fi

if [ -z $TORCH_VERSION ]; then
  echo "Please set the environment variable TORCH_VERSION"
  echo "Example: export TORCH_VERSION=1.10.0"
  exit 1
fi

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH
ls -lh $PYTHON_INSTALL_DIR/lib/

# python3 -m pip install scikit-build
python3 -m pip install -U pip cmake "numpy<=1.26.4"
python3 -m pip install wheel twine typing_extensions
python3 -m pip install bs4 requests tqdm auditwheel

echo "Installing torch $TORCH_VERSION"

if [[ $TORCH_VERSION == "2.8.0" ]]; then
  python3 -m pip install -qq torch==2.8.0.dev20250424+cpu -f https://download.pytorch.org/whl/nightly/torch/ -f https://download.pytorch.org/whl/nightly/pytorch-triton
else
  python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html || \
  python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch/ || \
  python3 -m pip install -qq torch==$TORCH_VERSION -f https://download.pytorch.org/whl/torch/ || \
  python3 -m pip install -qq torch==$TORCH_VERSION
fi

python3 -c "import torch; print(torch.__file__)"
python3 -m torch.utils.collect_env

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH

python3 --version
which python3

rm -rf ~/.cache/pip >/dev/null 2>&1
yum clean all >/dev/null 2>&1

nvcc --version || true
rm -rf /usr/local/cuda*
nvcc --version || true

cd /var/www

export CMAKE_CUDA_COMPILER_LAUNCHER=
export K2_CMAKE_ARGS=" -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export K2_MAKE_ARGS=" -j2 "

nvcc --version || true
rm -rf /usr/local/cuda*
nvcc --version || true

python3 setup.py bdist_wheel

if [[ x"$IS_2_28" == x"1" ]]; then
  plat=manylinux_2_28_aarch64
else
  plat=manylinux_2_17_aarch64
fi
export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
python3 --version
which python3

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
  --exclude libnvrtc.so.11.2 \
  --exclude libtorch_cuda_cu.so \
  --exclude libtorch_cuda_cpp.so \
  \
  --plat $plat \
  -w /var/www/wheelhouse \
  dist/*.whl

ls -lh  /var/www/wheelhouse
