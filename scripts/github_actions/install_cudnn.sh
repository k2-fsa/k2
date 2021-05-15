#!/bin/bash
#
# Copyright (c)  2020  Mobvoi Inc. (authors: Fangjun Kuang)

case $cuda in
  10.0)
    filename=cudnn-10.0-linux-x64-v7.6.5.32.tgz
    url=http://www.mediafire.com/file/1037lb1vmj9qdtq/cudnn-10.0-linux-x64-v7.6.5.32.tgz/file
    ;;
  10.1)
    filename=cudnn-10.1-linux-x64-v8.0.2.39.tgz
    url=http://www.mediafire.com/file/fnl2wg0h757qhd7/cudnn-10.1-linux-x64-v8.0.2.39.tgz/file
    ;;
  10.2)
    filename=cudnn-10.2-linux-x64-v8.0.2.39.tgz
    url=http://www.mediafire.com/file/sc2nvbtyg0f7ien/cudnn-10.2-linux-x64-v8.0.2.39.tgz/file
    ;;
  11.0)
    filename=cudnn-11.0-linux-x64-v8.0.5.39.tgz
    url=https://www.mediafire.com/file/abyhnls106ko9kp/cudnn-11.0-linux-x64-v8.0.5.39.tgz/file
    ;;
  11.1)
    filename=cudnn-11.1-linux-x64-v8.0.5.39.tgz
    url=https://www.mediafire.com/file/qx55zd65773xonv/cudnn-11.1-linux-x64-v8.0.5.39.tgz/file
    ;;
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

function retry() {
  $* || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}

retry wget https://raw.githubusercontent.com/Juvenal-Yescas/mediafire-dl/master/mediafire-dl.py
sed -i 's/quiet=False/quiet=True/' mediafire-dl.py
retry python3 mediafire-dl.py "$url"
sudo tar xf ./$filename -C /usr/local
rm -v ./$filename

sudo sed -i '59i#define CUDNN_MAJOR 8' /usr/local/cuda/include/cudnn.h
