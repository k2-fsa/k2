#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

case $cuda in
  10.0)
    gdown -q --id "1H1MavvYdQosuhoTtjMpFOWUO1MiBMvC8" --output cudnn-10.0-linux-x64-v7.6.5.32.tgz
    ;;
  10.1)
    gdown -q --id "13_IVjoRJmamBVOgEGSp1Tlr5nClZUqwj" --output cudnn-10.1-linux-x64-v8.0.2.39.tgz
    ;;
  10.2)
    gdown -q --id "1beVinj771IPuqUsQovgTh5ZMqujBzb64" --output cudnn-10.2-linux-x64-v8.0.2.39.tgz
    ;;
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

sudo tar xf ./cudnn-10.?-linux-x64-v*.tgz -C /usr/local
