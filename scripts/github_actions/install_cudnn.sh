#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

case $cuda in
  10.0)
    gdown -q --id "1-Yy-X5oFJKwS-iv3vanHor8SsbYGPmH-" --output cudnn-10.0-linux-x64-v7.6.5.32.tgz
    ;;
  10.1)
    gdown -q --id "10Xx7cQnRo_nLVZ_FOO6wdT_NlWI3I_-p" --output cudnn-10.1-linux-x64-v8.0.2.39.tgz
    ;;
  10.2)
    gdown -q --id "1-qbuQhaZ3115c_BTPHVR7k3SqSaDtatK" --output cudnn-10.2-linux-x64-v8.0.2.39.tgz
    ;;
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

sudo tar xf ./cudnn-10.?-linux-x64-v*.tgz -C /usr/local
