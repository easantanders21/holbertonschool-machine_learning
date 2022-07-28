#!/usr/bin/env python3
"""Slices a matrix along specific axes"""


def np_slice(matrix, axes={}):
    """Slices a matrix along specific axes"""
    slices = []
    for i in range(matrix.ndim):
        t = axes.get(i)
        if t is not None:
            slices.append(slice(*t))
        else:
            slices.append(slice(None, None, None))
    return matrix[(slices)]
