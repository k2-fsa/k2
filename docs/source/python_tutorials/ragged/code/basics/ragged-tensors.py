#!/usr/bin/env python3

# Note: If you add/remove lines, please also
# update the line numbers in python_tutorials/ragged/basics.rst

# 2d
import k2

c = k2.RaggedTensor(
    [
        [1, 2, 3, 6, -5],
        [0, 1],
        [],
        [3],
    ]
)


# 3d
d = k2.RaggedTensor(
    [
        [
            [1],
            [],
            [3, 5, 8],
        ],
        [
            [1, 2],
        ],
        [
            [],
            [],
            [5, 9, -1, 10],
            [],
        ],
        [
            [],
        ],
    ]
)

# exercise 1
e = k2.RaggedTensor(
    [
        [1, 10, -1],
        [],
        [-1.5, 2],
    ]
)
