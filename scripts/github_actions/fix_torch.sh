#!/usr/bin/env bash

# To fix the following issue for torch 1.12.x
# https://github.com/pytorch/pytorch/issues/88290
if [[ "${torch}" == "1.12.0" || ${torch} == "1.12.1" ]]; then
  torch_dir=$(python3 -c "from pathlib import Path; import torch; print(Path(torch.__file__).parent)")
  echo "torch_dir: ${torch_dir}"
  cd $torch_dir/include/torch/csrc/jit
  mkdir -p mobile
  cd mobile
  files=(
    code.h
    debug_info.h
    function.h
    method.h
    module.h
  )
  for f in ${files[@]}; do
    if [ ! -f $f ]; then
      url=https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/csrc/jit/mobile/$f
      echo "Downloading $url"
      wget $url
    fi
  done
fi
