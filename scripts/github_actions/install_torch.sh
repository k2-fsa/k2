#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

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
  1.8.0)
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
      11.1)
        package="torch==1.8.0+cu111"
        url=https://download.pytorch.org/whl/torch_stable.html
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
