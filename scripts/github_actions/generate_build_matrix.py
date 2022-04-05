#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-cuda",
        action="store_true",
        default=False,
        help="True to enable CUDA",
    )
    return parser.parse_args()


def generate_build_matrix(enable_cuda: bool):
    matrix = {
        "1.5.0": {
            "python-version": ["3.6", "3.7", "3.8"],
            "cuda": ["10.1", "10.2"],
        },
        "1.5.1": {
            "python-version": ["3.6", "3.7", "3.8"],
            "cuda": ["10.1", "10.2"],
        },
        "1.6.0": {
            "python-version": ["3.6", "3.7", "3.8"],
            "cuda": ["10.1", "10.2"],
        },
        "1.7.0": {
            "python-version": ["3.6", "3.7", "3.8"],
            "cuda": ["10.1", "10.2", "11.0"],
        },
        "1.7.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.1", "10.2", "11.0"],
        },
        "1.8.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.1", "10.2", "11.1"],
        },
        "1.8.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.1", "10.2", "11.1"],
        },
        "1.9.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1"],
        },
        "1.9.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1"],
        },
        "1.10.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1", "11.3"],
        },
        "1.10.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1", "11.3"],
        },
        "1.10.2": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1", "11.3"],
        },
        "1.11.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9", "3.10"],
            "cuda": ["10.2", "11.1", "11.3", "11.5"],
        },
    }

    torch_python_cuda = []
    for torch, python_cuda in matrix.items():
        python_versions = python_cuda["python-version"]
        cuda_versions = python_cuda["cuda"]
        for p in python_versions:
            for c in cuda_versions:
                torch_python_cuda.append(
                    {"torch": torch, "python-version": p, "cuda": c}
                )
    if not enable_cuda:
        for k in torch_python_cuda:
            del k["cuda"]

    m = {"include": torch_python_cuda, "os": ["ubuntu-18.04", "macos-10.15"]}

    print(json.dumps(m))


def main():
    args = get_args()
    generate_build_matrix(enable_cuda=args.enable_cuda)


if __name__ == "__main__":
    main()
