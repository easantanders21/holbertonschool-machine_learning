#!/usr/bin/env python3
''' concatenates two matrices along a specific axis '''


def cat_matrices2D(mat1, mat2, axis=0):
    """Function that concatenates two matrices along a specific axis"""
    if not isinstance(mat1, list) or not isinstance(mat2, list):
        return None
    if axis == 0:
        # Check if both matrices have the same number of columns
        if all(len(row) == len(mat1[0]) for row in mat1) and all(len(row) == len(mat2[0]) for row in mat2):
            return [*mat1, *mat2]
        else:
            return None
    elif axis == 1:
        # Check if both matrices have the same number of rows
        if len(mat1) == len(mat2):
            new_mat = []
            for row1, row2 in zip(mat1, mat2):
                if len(row1) == len(row2):
                    new_mat.append(row1 + row2)
                else:
                    return None
            return new_mat
        else:
            return None
    else:
        return None
