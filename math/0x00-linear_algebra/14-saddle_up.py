#!/usr/bin/env python3
"""Performs matrix multiplication"""

import numpy as np


def np_matmul(mat1, mat2):
    """Function that performs matrix multiplication:"""
    new_matrix = np.dot(mat1, mat2)
    return new_matrix
