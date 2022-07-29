#!/usr/bin/env python3
"""function that adds two matrices"""
import numpy as np


def add_matrices(mat1, mat2):
    """function that adds two matrices"""
    add_matrix = []
    mat1, mat2 = np.array(mat1), np.array(mat2)
    if not mat1.shape == mat2.shape:
        return None
    else:
        add_matrix = mat1 + mat2
        return add_matrix
