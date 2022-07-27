#!/usr/bin/env python3
"""Transpose of a 2D matrix"""


def matrix_transpose(matrix):
    """function that returns the transpose of a 2D matrix"""
    matrix_transpose = []
    list_by_column = []
    height = len(matrix)
    width = len(matrix[0])
    for j in range(width):
        for i in range(height):
            list_by_column.append(matrix[i][j])
        matrix_transpose.append(list_by_column)
        list_by_column = []
    return(matrix_transpose)
