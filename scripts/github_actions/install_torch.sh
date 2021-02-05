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
        url=https://download.pytorch.org/whl/torch_stable.html
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
        url=https://download.pytorch.org/whl/torch_stable.html
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
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      11.0)
        package="torch==1.7.0+cu110"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
    esac
    ;;
  *)
    echo "Unsupported PyTorch version: ${torch}"
    exit 1
    ;;
esac

source ./helper.sh

if [ x"${url}" == "x" ]; then
  retry python3 -m pip install $package
else
  retry python3 -m pip install $package -f $url
fi
