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
    ;;
  10.1)
    filename=cudnn-10.1-linux-x64-v8.0.2.39.tgz
    ;;
  10.2)
    filename=cudnn-10.2-linux-x64-v8.0.2.39.tgz
    ;;
  11.0)
    filename=cudnn-11.0-linux-x64-v8.0.5.39.tgz
    ;;
  11.1)
    filename=cudnn-11.1-linux-x64-v8.0.4.30.tgz
    ;;
  11.3)
    filename=cudnn-11.3-linux-x64-v8.2.0.53.tgz
    ;;
  11.5)
    filename=cudnn-11.3-linux-x64-v8.2.0.53.tgz
    ;;
  11.6)
    filename=cudnn-11.3-linux-x64-v8.2.0.53.tgz
    ;;
  11.7)
    filename=cudnn-11.3-linux-x64-v8.2.0.53.tgz
    ;;
  *)
    echo "Unsupported cuda version: $cuda"
    exit 1
    ;;
esac

command -v git-lfs >/dev/null 2>&1 || { echo >&2 "\nPlease install 'git-lfs' first."; exit 2; }

GIT_LFS_SKIP_SMUDGE=1 git clone https://huggingface.co/csukuangfj/cudnn
cd cudnn
git lfs pull --include="$filename"

sudo tar xf ./$filename --strip-components=1 -C /usr/local/cuda

# save disk space
git lfs prune && cd .. && rm -rf cudnn

sudo sed -i '59i#define CUDNN_MAJOR 8' /usr/local/cuda/include/cudnn.h
