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

    parser.add_argument(
        "--test-only-latest-torch",
        action="store_true",
        default=False,
        help="""If True, we test only the latest PyTorch
        to reduce CI running time.""",
    )
    return parser.parse_args()


def generate_build_matrix(enable_cuda, test_only_latest_torch):
    matrix = {
        # there are issues in serializing ragged tensors in 1.5.0 and 1.5.1
        #  "1.5.0": {
        #      "python-version": ["3.6", "3.7", "3.8"],
        #      "cuda": ["10.1", "10.2"],
        #  },
        #  "1.5.1": {
        #      "python-version": ["3.6", "3.7", "3.8"],
        #      "cuda": ["10.1", "10.2"],
        #  },
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
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": ["10.2", "11.3", "11.5"],
        },
        "1.12.0": {
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": ["10.2", "11.3", "11.6"],
        },
        "1.12.1": {
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": ["10.2", "11.3", "11.6"],
        },
        "1.13.0": {
            "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
            "cuda": ["11.6", "11.7"],  # default 11.7
        },
        "1.13.1": {
            "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
            "cuda": ["11.6", "11.7"],  # default 11.7
        },
    }
    if test_only_latest_torch:
        latest = "1.13.1"
        matrix = {latest: matrix[latest]}

    # We only have limited spaces in anaconda, so we exclude some
    # versions of PyTorch here. If you need them, please consider
    # installing k2 from source
    # Only CUDA build are excluded since it occupies more disk space
    excluded_torch_versions = ["1.6.0", "1.7.0", "1.7.1", "1.8.0", "1.8.1"]
    excluded_torch_versions += ["1.9.0", "1.9.1"]

    ans = []
    for torch, python_cuda in matrix.items():
        if torch in excluded_torch_versions and enable_cuda:
            continue

        python_versions = python_cuda["python-version"]
        cuda_versions = python_cuda["cuda"]
        if enable_cuda:
            for p in python_versions:
                for c in cuda_versions:
                    ans.append({"torch": torch, "python-version": p, "cuda": c})
        else:
            for p in python_versions:
                ans.append({"torch": torch, "python-version": p})

    print(json.dumps({"include": ans}))


def main():
    args = get_args()
    generate_build_matrix(
        enable_cuda=args.enable_cuda,
        test_only_latest_torch=args.test_only_latest_torch,
    )


if __name__ == "__main__":
    main()
