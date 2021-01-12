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
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

wget https://raw.githubusercontent.com/Juvenal-Yescas/mediafire-dl/master/mediafire-dl.py
python3 mediafire-dl.py "$url"
ls -l
sudo tar xf ./$filename -C /usr/local
ls -l
