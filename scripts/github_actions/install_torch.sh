#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

case ${torch} in
  1.6.0)
    case ${cuda} in
      10.0)
        package="torch==1.6.0+cu100"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.1)
        package="torch==1.6.0+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==1.6.0"
        url=
        ;;
    esac
    ;;
  1.7.0)
    case ${cuda} in
      10.0)
        package="torch==1.7.0+cu100"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.1)
        package="torch==1.7.0+cu101"
        url=https://download.pytorch.org/whl/torch_stable.html
        ;;
      10.2)
        package="torch==1.7.0"
        url=
        ;;
    esac
    ;;
  *)
    echo "Unsupported Pytorch version: ${torch}"
    exit 1
    ;;
esac

if [ x"${url}" == "x" ]; then
  python3 -m pip install $package
else
  python3 -m pip install $package -f $url
fi
