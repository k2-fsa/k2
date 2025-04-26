#!/usr/bin/env python3
# Copyright    2022  Xiaomi Corp.        (authors: Fangjun Kuang)

import argparse
import json


def version_ge(a, b):
    a_major, a_minor = list(map(int, a.split(".")))[:2]
    b_major, b_minor = list(map(int, b.split(".")))[:2]
    if a_major > b_major:
        return True

    if a_major == b_major and a_minor >= b_minor:
        return True

    return False


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--enable-cuda",
        action="store_true",
        default=False,
        help="True to enable CUDA",
    )

    parser.add_argument(
        "--for-windows",
        action="store_true",
        default=False,
        help="True for windows",
    )

    parser.add_argument(
        "--for-macos",
        action="store_true",
        default=False,
        help="True for macOS",
    )

    parser.add_argument(
        "--for-macos-m1",
        action="store_true",
        default=False,
        help="True for macOS M1",
    )

    parser.add_argument(
        "--for-arm64",
        action="store_true",
        default=False,
        help="True for arm64",
    )

    parser.add_argument(
        "--test-only-latest-torch",
        action="store_true",
        default=False,
        help="""If True, we test only the latest PyTroch
        to reduce CI running time.""",
    )
    return parser.parse_args()


def generate_build_matrix(
    enable_cuda,
    for_windows,
    for_macos,
    for_macos_m1,
    for_arm64,
    test_only_latest_torch,
):
    matrix = {
        # 1.5.x is removed because there are compilation errors.
        #  See
        #  https://github.com/csukuangfj/k2/runs/2533830771?check_suite_focus=true
        #  and
        #  https://github.com/NVIDIA/apex/issues/805
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
            "cuda": ["10.1", "10.2"] if not for_windows else ["10.1.243", "10.2.89"],
        },
        "1.7.0": {
            "python-version": ["3.6", "3.7", "3.8"],
            "cuda": (
                ["10.1", "10.2", "11.0"]
                if not for_windows
                else ["10.1.243", "10.2.89", "11.0.3"]
            ),
        },
        "1.7.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.1", "10.2", "11.0"]
                if not for_windows
                else ["10.1.243", "10.2.89", "11.0.3"]
            ),
        },
        "1.8.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.1", "10.2", "11.1"]
                if not for_windows
                else ["10.1.243", "10.2.89", "11.1.1"]
            ),
        },
        "1.8.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.1", "10.2", "11.1"]
                if not for_windows
                else ["10.1.243", "10.2.89", "11.1.1"]
            ),
        },
        "1.9.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1"] if not for_windows else ["10.2.89", "11.1.1"],
        },
        "1.9.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": ["10.2", "11.1"] if not for_windows else ["10.2.89", "11.1.1"],
        },
        "1.10.0": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.2", "11.1", "11.3"]
                if not for_windows
                else ["10.2.89", "11.1.1", "11.3.1"]
            ),
        },
        "1.10.1": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.2", "11.1", "11.3"]
                if not for_windows
                else ["10.2.89", "11.1.1", "11.3.1"]
            ),
        },
        "1.10.2": {
            "python-version": ["3.6", "3.7", "3.8", "3.9"],
            "cuda": (
                ["10.2", "11.1", "11.3"]
                if not for_windows
                else ["10.2.89", "11.1.1", "11.3.1"]
            ),
        },
        "1.11.0": {
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": (
                ["10.2", "11.3", "11.5"] if not for_windows else ["11.3.1", "11.5.2"]
            ),
        },
        "1.12.0": {
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": (
                ["10.2", "11.3", "11.6"] if not for_windows else ["11.3.1", "11.6.2"]
            ),
        },
        "1.12.1": {
            "python-version": ["3.7", "3.8", "3.9", "3.10"],
            "cuda": (
                ["10.2", "11.3", "11.6"] if not for_windows else ["11.3.1", "11.6.2"]
            ),
        },
        "1.13.0": {
            "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
            "cuda": ["11.6", "11.7"],  # default 11.7
        },
        "1.13.1": {
            "python-version": ["3.7", "3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.6", "11.7"]  # default 11.7
                if not for_windows
                else ["11.6.2", "11.7.1"]
            ),
        },
        "2.0.0": {
            "python-version": ["3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.7", "11.8"]  # default 11.7
                if not for_windows
                else ["11.7.1", "11.8.0"]
            ),
        },
        "2.0.1": {
            "python-version": ["3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.7", "11.8"]  # default 11.7
                if not for_windows
                else ["11.7.1", "11.8.0"]
            ),
        },
        "2.1.0": {
            "python-version": ["3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.1.1": {
            "python-version": ["3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.1.2": {
            "python-version": ["3.8", "3.9", "3.10", "3.11"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.2.0": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.2.1": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.2.2": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.3.0": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.3.1": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0"]
            ),
        },
        "2.4.0": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1", "12.4"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0", "12.4.0"]
            ),
        },
        "2.4.1": {
            "python-version": ["3.8", "3.9", "3.10", "3.11", "3.12"],
            "cuda": (
                ["11.8", "12.1", "12.4"]  # default 12.1
                if not for_windows
                else ["11.8.0", "12.1.0", "12.4.0"]
            ),
        },
        "2.5.0": {
            # Only Linux supports python 3.13
            "python-version": ["3.9", "3.10", "3.11", "3.12", "3.13"],
            "cuda": (
                ["11.8", "12.1", "12.4"]  # default 12.4
                if not for_windows
                else ["11.8.0", "12.1.0", "12.4.0"]
            ),
        },
        "2.5.1": {
            # Only Linux supports python 3.13
            "python-version": ["3.9", "3.10", "3.11", "3.12", "3.13"],
            "cuda": (
                ["11.8", "12.1", "12.4"]  # default 12.4
                if not for_windows
                else ["11.8.0", "12.1.0", "12.4.0"]
            ),
        },
        "2.6.0": {
            "python-version": ["3.9", "3.10", "3.11", "3.12", "3.13"],
            "cuda": (
                ["11.8", "12.4", "12.6"]  # default 12.4
                if not for_windows
                else ["11.8.0", "12.4.0", "12.6.0"]
            ),
        },
        "2.7.0": {
            "python-version": ["3.9", "3.10", "3.11", "3.12", "3.13"],
            "cuda": (
                ["11.8", "12.6", "12.8"]
                if not for_windows
                else ["11.8.0", "12.6.2", "12.8.1"]
            ),
        },
        # https://github.com/Jimver/cuda-toolkit/blob/master/src/links/windows-links.ts
    }
    if test_only_latest_torch:
        latest = "2.7.0"
        matrix = {latest: matrix[latest]}

    if for_windows or for_macos:
        if "2.5.1" in matrix:
            matrix["2.5.1"]["python-version"].remove("3.13")

        if "2.5.0" in matrix:
            matrix["2.5.0"]["python-version"].remove("3.13")

        if "1.13.0" in matrix:
            matrix["1.13.0"]["python-version"].remove("3.11")

        if "1.13.1" in matrix:
            matrix["1.13.1"]["python-version"].remove("3.11")

    excluded_python_versions = ["3.6", "3.7"]

    enabled_torch_versions = ["1.10.0"]
    enabled_torch_versions += ["1.13.0", "1.13.1"]
    min_torch_version = "2.0.0"

    if for_macos_m1:
        matrix = dict()
        matrix["1.8.0"] = {"python-version": ["3.8"]}
        matrix["1.8.1"] = {"python-version": ["3.8"]}
        matrix["1.9.0"] = {"python-version": ["3.8", "3.9"]}
        matrix["1.9.1"] = {"python-version": ["3.8", "3.9"]}
        matrix["1.10.0"] = {"python-version": ["3.8", "3.9"]}
        matrix["1.10.1"] = {"python-version": ["3.8", "3.9"]}
        matrix["1.10.2"] = {"python-version": ["3.8", "3.9"]}
        matrix["1.11.0"] = {"python-version": ["3.8", "3.9", "3.10"]}
        matrix["1.12.0"] = {"python-version": ["3.7", "3.8", "3.9", "3.10"]}
        matrix["1.12.1"] = {"python-version": ["3.7", "3.8", "3.9", "3.10"]}
        matrix["1.13.0"] = {"python-version": ["3.7", "3.8", "3.9", "3.10"]}
        matrix["1.13.1"] = {"python-version": ["3.7", "3.8", "3.9", "3.10"]}
        matrix["2.0.0"] = {"python-version": ["3.8", "3.9", "3.10", "3.11"]}
        matrix["2.0.1"] = {"python-version": ["3.8", "3.9", "3.10", "3.11"]}
        # TODO(fangjun): we currently don't support macOS M1 build
        # since github actions does not support it.

    ans = []
    for torch, python_cuda in matrix.items():
        if enabled_torch_versions and torch not in enabled_torch_versions:
            if not version_ge(torch, min_torch_version):
                continue

        python_versions = python_cuda["python-version"]
        if enable_cuda:
            cuda_versions = python_cuda["cuda"]
            for p in python_versions:
                if p in excluded_python_versions:
                    continue

                for c in cuda_versions:
                    if c in ["10.1", "11.0"]:
                        # no docker image for cuda 10.1 and 11.0
                        continue

                    if version_ge(torch, "2.7.0") or (
                        version_ge(torch, "2.6.0") and c == "12.6"
                    ):
                        # case 1: torch >= 2.7
                        # case 2: torch == 2.6.0 && cuda == 12.6
                        ans.append(
                            {
                                "torch": torch,
                                "python-version": p,
                                "cuda": c,
                                "image": f"pytorch/manylinux2_28-builder:cuda{c}",
                                "is_2_28": "1",
                            }
                        )
                        continue

                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            "cuda": c,
                            "image": "pytorch/manylinux-builder:cuda" + c,
                            "is_2_28": "0",
                        }
                    )
        else:
            for p in python_versions:
                if p in excluded_python_versions:
                    continue

                if for_windows:
                    #  p = "cp" + "".join(p.split("."))
                    ans.append({"torch": torch, "python-version": p})
                elif for_macos or for_macos_m1:
                    ans.append({"torch": torch, "python-version": p})
                elif version_ge(torch, "2.6.0"):
                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            "image": "pytorch/manylinux2_28-builder:cpu"
                            if not for_arm64
                            else "pytorch/manylinux2_28_aarch64-builder:cpu-aarch64",
                            "is_2_28": "1",
                        }
                    )
                elif version_ge(torch, "2.4.0"):
                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            #  "image": "pytorch/manylinux-builder:cpu-2.4",
                            "image": "pytorch/manylinux-builder:cpu-27677ead7c8293c299a885ae2c474bf445e653a5"
                            if not for_arm64
                            else "pytorch/manylinuxaarch64-builder:cpu-aarch64-195148266541a9789074265141cb7dc19dc14c54",
                            "is_2_28": "0",
                        }
                    )
                elif version_ge(torch, "2.2.0"):
                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            #  "image": "pytorch/manylinux-builder:cpu-2.2",
                            "image": "pytorch/manylinux-builder:cpu-27677ead7c8293c299a885ae2c474bf445e653a5"
                            if not for_arm64
                            else "pytorch/manylinuxaarch64-builder:cpu-aarch64-195148266541a9789074265141cb7dc19dc14c54",
                            "is_2_28": "0",
                        }
                    )
                else:
                    ans.append(
                        {
                            "torch": torch,
                            "python-version": p,
                            #  "image": "pytorch/manylinux-builder:cuda10.2",
                            "image": "pytorch/manylinux-builder:cpu-27677ead7c8293c299a885ae2c474bf445e653a5"
                            if not for_arm64
                            else "pytorch/manylinuxaarch64-builder:cpu-aarch64-195148266541a9789074265141cb7dc19dc14c54",
                            "is_2_28": "0",
                        }
                    )

    print(json.dumps({"include": ans}))


def main():
    args = get_args()
    generate_build_matrix(
        enable_cuda=args.enable_cuda,
        for_windows=args.for_windows,
        for_macos=args.for_macos,
        for_macos_m1=args.for_macos_m1,
        for_arm64=args.for_arm64,
        test_only_latest_torch=args.test_only_latest_torch,
    )


if __name__ == "__main__":
    main()
