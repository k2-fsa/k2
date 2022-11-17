#!/usr/bin/env python3

import torch
from pathlib import Path
import urllib.request


def get_pytorch_version():
    # if it is 1.7.1+cuda101, then strip +cuda101
    return torch.__version__.split("+")[0]


def fix_pytorch_1_12():
    print("Fix https://github.com/pytorch/pytorch/issues/88290")

    torch_dir = Path(torch.__file__).parent
    print("torch_dir", torch_dir)
    mobile_dir = torch_dir / "include" / "torch" / "csrc" / "jit" / "mobile"
    mobile_dir.mkdir(exist_ok=True)
    files = (
        "code.h",
        "debug_info.h",
        "function.h",
        "method.h",
        "module.h",
    )
    base_url = "https://raw.githubusercontent.com/pytorch/pytorch/v1.12.1/torch/csrc/jit/mobile/"  # noqa
    for f in files:
        path = mobile_dir / f
        if path.is_file():
            print(f"skipping {path}")
            continue
        url = base_url + f
        print(f"Downloading {url} to {path}")
        urllib.request.urlretrieve(url, path)


def main():
    if "1.12" in get_pytorch_version():
        fix_pytorch_1_12()
    else:
        print(f"Skip since version is {get_pytorch_version()}")


if __name__ == "__main__":
    main()
