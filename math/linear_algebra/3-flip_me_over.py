#!/usr/bin/env python3
"""Transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """function that returns the transpose of a 2D matrix"""
    matrix_t = []
    for i in range(len(matrix[0])):
        new_row = []
        for j in matrix:
            new_row.append(j[i])
        matrix_t.append(new_row)
    return matrix_t
