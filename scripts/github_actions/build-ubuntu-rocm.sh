#!/usr/bin/env bash
#
# Build k2 wheel with ROCm/HIP support inside a manylinux Docker container.
# This is the ROCm counterpart of build-ubuntu-cuda.sh.
#
# Required environment variables:
#   PYTHON_VERSION    - e.g. "3.10"
#   TORCH_VERSION     - e.g. "2.12.1"
#   ROCM_VERSION      - e.g. "7.1"
#   PYTHON_INSTALL_DIR - path to the container's Python, e.g. /opt/python/cp310-cp310
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

if [ -z "$ROCM_VERSION" ]; then
  echo "Please set the environment variable ROCM_VERSION"
  echo "Example: export ROCM_VERSION=7.1"
  exit 1
fi

if [ -z "$PYTHON_INSTALL_DIR" ]; then
  echo "Please set the environment variable PYTHON_INSTALL_DIR"
  echo "Example: export PYTHON_INSTALL_DIR=/opt/python/cp310-cp310"
  exit 1
fi

export PATH=$PYTHON_INSTALL_DIR/bin:$PATH
export LD_LIBRARY_PATH=$PYTHON_INSTALL_DIR/lib:$LD_LIBRARY_PATH

python3 -m pip install --no-cache-dir -U pip cmake "numpy<=1.26.4"
python3 -m pip install --no-cache-dir wheel twine typing_extensions
python3 -m pip install --no-cache-dir bs4 requests tqdm auditwheel patchelf
patchelf --version

echo "Installing torch (ROCm ${ROCM_VERSION})"
./install_torch.sh

python3 -c "import torch; print(torch.__file__)"

# Free disk space — the ROCm torch wheel is ~5.8 GB
pip cache purge || true
rm -rf ~/.cache/pip /tmp/torch-*.whl

cd /var/www

# Install libhipcxx (provides <cuda/std/*> headers needed by k2's HIP build)
echo "Installing libhipcxx..."
git clone --depth 1 https://github.com/ROCm/libhipcxx.git /tmp/libhipcxx
LIBHIPCXX_INCLUDE_DIR=/tmp/libhipcxx/include

# ROCm major version for auditwheel library suffixes
ROCM_MAJOR=${ROCM_VERSION%%.*}

# Target GPU architectures.
# Try deriving from the installed torch wheel first; fall back to a
# hardcoded list per ROCm version.  Users can override via HIP_ARCH env var.
if [[ -z "$HIP_ARCH" ]]; then
  HIP_ARCH=$(python3 -c "import torch; print(';'.join(a for a in torch.cuda.get_arch_list() if a.startswith('gfx')))" 2>/dev/null || true)
fi

# Target GPU architectures per ROCm version.
# See https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html
#
# ROCm 6.3/6.4 supports:
#   gfx900  - Vega (RX 580, etc.)
#   gfx906  - Vega VII / MI50
#   gfx908  - MI100
#   gfx90a  - MI200/MI250
#   gfx942  - MI300X
#   gfx1030 - Navi 22 (RX 6700/6800/6900)
#   gfx1033 - Van Gogh (Steam Deck)
#   gfx1100 - Navi 31 (RX 7900)
#   gfx1101 - Navi 32 (RX 7700/7800)
#   gfx1102 - Navi 33 (RX 7600)
#   gfx1103 - Navi 33 APU
#
# ROCm 7.0 adds:
#   gfx1200 - Navi 4x
#   gfx1201 - Navi 4x
#
# ROCm 7.1+ adds:
#   gfx950  - MI350
#   gfx1150 - Navi 4x APU
#   gfx1151 - Navi 4x APU
#
# ROCm 7.2+ drops:
#   gfx900, gfx906 (old Vega, no longer supported)

if [[ -z "$HIP_ARCH" ]]; then
  echo "Warning: could not derive HIP_ARCH from torch, falling back to ROCm version list"
  ROCM_MINOR=${ROCM_VERSION#*.}
  ROCM_MINOR=${ROCM_MINOR%%.*}

  # Base architectures supported by ROCm 6.3+
  HIP_ARCH="gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103"

  # ROCm 6.x still supports old Vega
  if [[ $ROCM_MAJOR -lt 7 ]]; then
    HIP_ARCH="${HIP_ARCH};gfx900;gfx906;gfx1033"
  fi

  # ROCm 7.0+ adds RDNA4
  if [[ $ROCM_MAJOR -ge 7 ]]; then
    HIP_ARCH="${HIP_ARCH};gfx1200;gfx1201"
  fi

  # ROCm 7.1+ adds MI350 and Navi 4x APUs
  if [[ $ROCM_MAJOR -gt 7 || ($ROCM_MAJOR -eq 7 && $ROCM_MINOR -ge 1) ]]; then
    HIP_ARCH="${HIP_ARCH};gfx950;gfx1150;gfx1151"
  fi
fi

echo "HIP_ARCH: $HIP_ARCH"

# Escape semicolons in HIP_ARCH so the shell doesn't treat them as command
# separators when the string is expanded inside setup.py's os.system() call.
HIP_ARCH_ESCAPED="${HIP_ARCH//;/\\;}"
export K2_CMAKE_ARGS="-DK2_WITH_HIP=ON"
export K2_CMAKE_ARGS="$K2_CMAKE_ARGS -DK2_LIBHIPCXX_INCLUDE_DIR=$LIBHIPCXX_INCLUDE_DIR"
export K2_CMAKE_ARGS="$K2_CMAKE_ARGS -DCMAKE_HIP_ARCHITECTURES=$HIP_ARCH_ESCAPED"
export K2_CMAKE_ARGS="$K2_CMAKE_ARGS -DPYTHON_EXECUTABLE=$PYTHON_INSTALL_DIR/bin/python3"
export K2_MAKE_ARGS=" -j2 "

# Free more disk space before the build
yum clean all >/dev/null 2>&1
rm -rf ~/.cache

python3 setup.py bdist_wheel

# libhipcxx no longer needed after build
rm -rf /tmp/libhipcxx

plat=manylinux_2_28_x86_64

auditwheel --verbose repair \
  --exclude libc10.so \
  --exclude libc10_hip.so \
  --exclude libtorch.so \
  --exclude libtorch_cpu.so \
  --exclude libtorch_hip.so \
  --exclude libtorch_python.so \
  \
  --exclude libhiprtc.so \
  --exclude libhiprtc.so.${ROCM_MAJOR} \
  --exclude libhipblas.so \
  --exclude libhipblas.so.${ROCM_MAJOR} \
  --exclude librocblas.so \
  --exclude librocblas.so.${ROCM_MAJOR} \
  --exclude libhipsolver.so \
  --exclude libhipsolver.so.${ROCM_MAJOR} \
  --exclude librocsolver.so \
  --exclude librocsolver.so.${ROCM_MAJOR} \
  --exclude libhipsparse.so \
  --exclude libhipsparse.so.${ROCM_MAJOR} \
  --exclude librocsparse.so \
  --exclude librocsparse.so.${ROCM_MAJOR} \
  --exclude libhipfft.so \
  --exclude libhipfft.so.${ROCM_MAJOR} \
  --exclude libhiprand.so \
  --exclude libhiprand.so.${ROCM_MAJOR} \
  --exclude libamdhip64.so \
  --exclude libamdhip64.so.${ROCM_MAJOR} \
  --exclude librocm_smi64.so \
  --exclude librocm_smi64.so.${ROCM_MAJOR} \
  --exclude librccl.so \
  --exclude librccl.so.${ROCM_MAJOR} \
  --exclude libhipblaslt.so \
  --exclude libhipblaslt.so.${ROCM_MAJOR} \
  --exclude libshm.so \
  --exclude libtorch_cuda_cpp.so \
  --exclude libtorch_cuda_cu.so \
  --plat $plat \
  -w /var/www/wheelhouse \
  dist/*.whl

ls -lh /var/www

# Patch rpath for the _k2 shared library to find ROCm and torch libraries
pushd /var/www/wheelhouse
whl=$(ls *.whl)
mkdir -p _tmp_whl
pushd _tmp_whl
unzip -o ../$whl
so_file=$(ls _k2.cpython-*.so)
echo "Patching rpath for $so_file"
current_rpath=$(patchelf --print-rpath "$so_file")
echo "Current rpath: $current_rpath"
new_rpath="\$ORIGIN/../torch/lib:${current_rpath}"
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
