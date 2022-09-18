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
        help="""If True, we test only the latest PyTroch
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
    }

    # k2-fsa always contains the latest version
    channels = {
        "k2-fsa": {"1.11.0", "1.12.0", "1.12.1"},
        "k2-fsa-2": {"1.6.0", "1.7.0", "1.7.1", "1.8.0", "1.8.1"},
        "k2-fsa-3": {"1.9.0", "1.9.1", "1.10.0", "1.10.1", "1.10.2"},
    }

    def get_channel(torch_version):
        for k, v in channels.items():
            if torch_version in v:
                return k
        raise ValueError(f"Unknown torch version {torch_version}")

    if test_only_latest_torch:
        latest = "1.12.1"
        matrix = {latest: matrix[latest]}

    ans = []
    for torch, python_cuda in matrix.items():
        conda_channel = get_channel(torch)
        python_versions = python_cuda["python-version"]
        cuda_versions = python_cuda["cuda"]
        if enable_cuda:
            for p in python_versions:
                for c in cuda_versions:
                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            "cuda": c,
                            "channel": conda_channel,
                        }
                    )
        else:
            for p in python_versions:
                ans.append(
                    {
                        "torch": torch,
                        "python-version": p,
                        "channel": conda_channel,
                    }
                )

    print(json.dumps({"include": ans}))


def main():
    args = get_args()
    generate_build_matrix(
        enable_cuda=args.enable_cuda,
        test_only_latest_torch=args.test_only_latest_torch,
    )


if __name__ == "__main__":
    main()
