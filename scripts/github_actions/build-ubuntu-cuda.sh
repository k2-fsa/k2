#!/usr/bin/env bash
#
set -ex

if [ -z "$PYTHON_VERSION" ]; then
  echo "Please set the environment variable PYTHON_VERSION"
  echo "Example: export PYTHON_VERSION=3.10"
  exit 1
fi

if [ -z "$TORCH_VERSION" ]; then
  echo "Please set the environment variable TORCH_VERSION"
  echo "Example: export TORCH_VERSION=2.12.1"
  exit 1
fi

if [ -z "$CUDA_VERSION" ]; then
  echo "Please set the environment variable CUDA_VERSION"
  echo "Example: export CUDA_VERSION=12.1"
  exit 1
fi

if [ -z "$PYTHON_INSTALL_DIR" ]; then
  echo "Please set the environment variable PYTHON_INSTALL_DIR"
  echo "Example: export PYTHON_INSTALL_DIR=/opt/python/cp310-cp310"
  exit 1
fi

if [[ $TORCH_VERSION =~ 2.2.* && $CUDA_VERSION =~ 12.* ]]; then
  # see https://github.com/pytorch/pytorch/issues/113948
  export TORCH_CUDA_ARCH_LIST="8.0 8.6 8.9 9.0"
fi

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH

python3 -m pip install --no-cache-dir -U pip cmake "numpy<=1.26.4"
python3 -m pip install --no-cache-dir wheel twine typing_extensions
python3 -m pip install --no-cache-dir bs4 requests tqdm auditwheel patchelf
patchelf --version

echo "Installing torch"
./install_torch.sh

python3 -c "import torch; print(torch.__file__)"

sed -i.bak /9.0a/d /Python-*/py-3.*/lib/python3.*/site-packages/torch/share/cmake/Caffe2/Modules_CUDA_fix/upstream/FindCUDA/select_compute_arch.cmake || true

rm -rf ~/.cache/pip >/dev/null 2>&1
yum clean all >/dev/null 2>&1

cd /var/www

export CMAKE_CUDA_COMPILER_LAUNCHER=
# export K2_CMAKE_ARGS="-DCUDAToolkit_TARGET_DIR=/usr/local/cuda/targets/x86_64-linux -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export K2_CMAKE_ARGS="-DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export K2_MAKE_ARGS=" -j2 "

python3 setup.py bdist_wheel
if [[ x"$IS_2_28" == x"1" ]]; then
  plat=manylinux_2_28_x86_64
else
  plat=manylinux_2_17_x86_64
fi

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
  --exclude libcublas.so \
  --exclude libcublas.so.11 \
  --exclude libcublas.so.12 \
  --exclude libcublas.so.13 \
  --exclude libcublasLt.so \
  --exclude libcublasLt.so.11 \
  --exclude libcublasLt.so.12 \
  --exclude libcudart.so.11.0 \
  --exclude libcudart.so.12 \
  --exclude libcudart.so.13 \
  --exclude libcudnn.so.8 \
  --exclude libcufft.so \
  --exclude libcufft.so.11 \
  --exclude libcupti.so \
  --exclude libcupti.so.12 \
  --exclude libcurand.so \
  --exclude libcurand.so.10 \
  --exclude libcusparse.so \
  --exclude libcusparse.so.12 \
  --exclude libnccl.so \
  --exclude libnccl.so.2 \
  --exclude libnvJitLink.so \
  --exclude libnvJitLink.so.12 \
  --exclude libnvrtc.so \
  --exclude libnvrtc.so.11.2 \
  --exclude libnvrtc.so.12 \
  --exclude libnvrtc.so.13 \
  --exclude libshm.so \
  --exclude libtorch_cuda_cpp.so \
  --exclude libtorch_cuda_cu.so \
  --plat $plat \
  -w /var/www/wheelhouse \
  dist/*.whl

ls -lh  /var/www

# Use patchelf to add nvidia rpath entries to the _k2 shared library
pushd /var/www/wheelhouse
whl=$(ls *.whl)
mkdir -p _tmp_whl
pushd _tmp_whl
unzip -o ../$whl
so_file=$(ls _k2.cpython-*.so)
echo "Patching rpath for $so_file"
current_rpath=$(patchelf --print-rpath "$so_file")
echo "Current rpath: $current_rpath"
new_rpath="\$ORIGIN/nvidia/nvtx/lib:\$ORIGIN/nvidia/cuda_runtime/lib:\$ORIGIN/nvidia/cuda_nvrtc/lib:${current_rpath}"
echo "New rpath: $new_rpath"
patchelf --set-rpath "$new_rpath" "$so_file"
echo "Verified rpath:"
patchelf --print-rpath "$so_file"
python3 -c "
import zipfile, os
with zipfile.ZipFile(os.path.join('..', '$whl'), 'w', zipfile.ZIP_DEFLATED) as zf:
    for root, dirs, files in os.walk('.'):
        for f in files:
            path = os.path.join(root, f)
            zf.write(path, path[2:])
"
popd
rm -rf _tmp_whl
popd
