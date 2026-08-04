#!/usr/bin/env python3

# Note: If you add/remove lines, please also
# update the line numbers in python_tutorials/ragged/basics.rst


# 2d
import numpy as np

a = np.array(
    [
        [1, 2, 3, 6],
        [0, 1, 5, 0],
        [3, 6, 8, 10],
    ]
)

print("a.shape:", a.shape)
# It prints
# a.shape: (3, 4)

# 3d

b = np.array(
    [
        [
            [1, 2],
            [0, 1],
            [3, 6],
        ],
        [
            [5, 20],
            [0, -1],
            [-2, 9],
        ],
        [
            [8, 7],
            [-3, 3],
            [2, -2],
        ],
    ]
)
print("b.shape:", b.shape)
# It prints
# b.shape: (3, 3, 2)
