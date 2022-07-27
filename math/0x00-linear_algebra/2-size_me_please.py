#!/usr/bin/env python3
"""Shadow for size the len of matrix"""


def matrix_shape(matrix):
    """Function that calculates the shape of a matrix:"""
    list_shape = []
    while type(matrix) is list:
        list_shape.append(len(matrix))
        matrix = matrix[0]
    return list_shape
