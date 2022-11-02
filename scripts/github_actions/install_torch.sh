#!/bin/bash
#
# Copyright      2020  Mobvoi Inc. (authors: Fangjun Kuang)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

case ${torch} in
  1.5.*)
    case ${cuda} in
      10.1)
        package="torch==${torch}+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==${torch}"
        # Leave url empty to use PyPI.
        # torch_stable provides cu92 but we want cu102
        url=
        ;;
    esac
    ;;
  1.6.0)
    case ${cuda} in
      10.1)
        package="torch==1.6.0+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==1.6.0"
        # Leave it empty to use PyPI.
        # torch_stable provides cu92 but we want cu102
        url=
        ;;
    esac
    ;;
  1.7.*)
    case ${cuda} in
      10.1)
        package="torch==${torch}+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        # torch_stable provides cu92 but we want cu102
        url=
        ;;
      11.0)
        package="torch==${torch}+cu110"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.8.*)
    case ${cuda} in
      10.1)
        package="torch==${torch}+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
      11.1)
        package="torch==${torch}+cu111"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.9.*)
    case ${cuda} in
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
      11.1)
        package="torch==${torch}+cu111"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.10.*)
    case ${cuda} in
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
      11.1)
        package="torch==${torch}+cu111"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      11.3)
        package="torch==${torch}+cu113"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.11.*)
    case ${cuda} in
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
      11.3)
        package="torch==${torch}+cu113"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      11.5)
        package="torch==${torch}+cu115"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.12.*)
    case ${cuda} in
      10.2)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
      11.3)
        package="torch==${torch}+cu113"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      11.6)
        package="torch==${torch}+cu116"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  1.13.*)
    case ${cuda} in
      11.6)
        package="torch==${torch}+cu116"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      11.7)
        package="torch==${torch}"
        # Leave it empty to use PyPI.
        url=
        ;;
    esac
    ;;
  *)
    echo "Unsupported PyTorch version: ${torch}"
    exit 1
    ;;
esac

function retry() {
  $* || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

if [ x"${url}" == "x" ]; then
  retry python3 -m pip install -q $package
else
  retry python3 -m pip install -q $package -f $url
fi

# To fix the following issue for torch 1.12.x
# https://github.com/pytorch/pytorch/issues/88290
if [[ "${torch}" == "1.12.0" || ${torch} == "1.12.1" ]]; then
  torch_dir=$(python3 -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent)")
  echo "torch_dir: ${torch_dir}"
  cd $torch_dir/include/torch/csrc/jit
  if [ ! -d mobile ]; then
    mkdir mobile
    cd mobile
    files=(
      code.h
      debug_info.h
      function.h
      method.h
      module.h
    )
    for f in ${files[@]}; do
      url=https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/csrc/jit/mobile/$f
      echo "Downloading $url"
      wget $url
    done
  fi
fi
