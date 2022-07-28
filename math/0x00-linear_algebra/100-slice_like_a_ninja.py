#!/usr/bin/env python3
"""Slices a matrix along specific axes"""

import numpy as np


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    slices = []
    axes = {0: (2,), 2: (None, None, -2)}
    for i in range(matrix.ndim):
        t = axes.get(i)
        if t is not None:
            slices.append(slice(*t))
        else:
            slices.append(slice(None, None, None))

    return matrix[tuple(slices)]
