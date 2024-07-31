#!/usr/bin/env python3

def matrix_transpose(matrix):
    matrix_t = []
    for i in range(len(matrix[0])):
        new_row = []
        for j in matrix:
            new_row.append(j[i])
        matrix_t.append(new_row)
    return matrix_t
