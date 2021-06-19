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

# It is forked from https://github.com/Juvenal-Yescas/mediafire-dl
# https://github.com/Juvenal-Yescas/mediafire-dl/pull/2 changes the filename and breaks the CI.
# We use a separate fork to keep the link fixed.
retry wget https://raw.githubusercontent.com/csukuangfj/mediafire-dl/master/mediafire_dl.py

sed -i 's/quiet=False/quiet=True/' mediafire_dl.py
retry python3 mediafire_dl.py "$url"
sudo tar xf ./$filename -C /usr/local
rm -v ./$filename

sudo sed -i '59i#define CUDNN_MAJOR 8' /usr/local/cuda/include/cudnn.h
