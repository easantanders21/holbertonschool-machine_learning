#!/usr/bin/env python3
"""Adds two matrices element-wise"""


def add_matrices2D(mat1, mat2):
    """function that adds two matrices element-wise:"""
    sum_mat = []
    sum_by_row = []
    if len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0]):
        for i in range(len(mat1)):
            for j in range(len(mat1[0])):
                sum_by_row.append(mat1[i][j] + mat2[i][j])
            sum_mat.append(sum_by_row)
            sum_by_row = []
        return sum_mat
    else:
        return None
