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

if [ -z "$PYTHON_INSTALL_DIR" ]; then
  echo "Please set the environment variable PYTHON_INSTALL_DIR"
  echo "Example: export PYTHON_INSTALL_DIR=/opt/python/cp310-cp310"
  exit 1
fi

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH
ls -lh $PYTHON_INSTALL_DIR/lib/

# python3 -m pip install scikit-build
python3 -m pip install -U pip cmake "numpy<=1.26.4"
python3 -m pip install wheel twine typing_extensions
python3 -m pip install -U bs4 requests tqdm auditwheel patchelf
# torch < 2.0 uses `from pkg_resources import packaging` which was removed in setuptools >= 72
if [[ "${TORCH_VERSION%%.*}" -lt 2 ]]; then
  python3 -m pip install -U "setuptools<72"
else
  python3 -m pip install -U setuptools
fi
patchelf --version

echo "Installing torch $TORCH_VERSION"

# if [[ $TORCH_VERSION == "2.8.0" ]]; then
#   python3 -m pip install -qq torch==2.8.0.dev20250424+cpu -f https://download.pytorch.org/whl/nightly/torch/ -f https://download.pytorch.org/whl/nightly/pytorch-triton
# else
#   python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html || \
#   python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch/ || \
#   python3 -m pip install -qq torch==$TORCH_VERSION -f https://download.pytorch.org/whl/torch/ || \
#   python3 -m pip install -qq torch==$TORCH_VERSION
# fi

python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch_stable.html || \
python3 -m pip install -qq torch==$TORCH_VERSION+cpu -f https://download.pytorch.org/whl/torch/ || \
python3 -m pip install -qq torch==$TORCH_VERSION -f https://download.pytorch.org/whl/torch/ || \
python3 -m pip install -qq torch==$TORCH_VERSION

python3 -c "import torch; print(torch.__file__)"
python3 -m torch.utils.collect_env

rm -rf ~/.cache/pip >/dev/null 2>&1
yum clean all >/dev/null 2>&1

nvcc --version || true
rm -rf /usr/local/cuda*
nvcc --version || true

cd /var/www

export CMAKE_CUDA_COMPILER_LAUNCHER=
export K2_CMAKE_ARGS=" -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3 "
export K2_MAKE_ARGS=" -j2 "

python3 setup.py bdist_wheel

if [[ x"$IS_2_28" == x"1" ]]; then
  plat=manylinux_2_28_aarch64
else
  plat=manylinux_2_17_aarch64
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
