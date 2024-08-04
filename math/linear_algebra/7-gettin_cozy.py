#!/usr/bin/env python3
''' concatenates two matrices along a specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    new_mat = []
    if axis == 0:
        new_mat = [*mat1, *mat2]
        return new_mat
    elif axis != 0:
        new_mat = list(range(len(mat1)))
        for i in range(len(mat1)):
            new_mat[i] = cat_matrices2D(mat1[i], mat2[i], axis - 1)
        return new_mat
    else:
        return None
