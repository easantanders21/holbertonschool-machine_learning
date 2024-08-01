#!/usr/bin/env python3
''' Write a function that adds two matrices element-wise: '''


def add_matrices2D(mat1, mat2):
    ''' Function that adds two matrices element-wise: '''
    files_mat1 = len(mat1)
    files_mat2 = len(mat2)
    columns_mat1 = len(mat1[0])
    columns_mat2 = len(mat2[0])
    add_mat = []
    if files_mat1 == files_mat2 and columns_mat1 == columns_mat2:
        for i in range(files_mat1):
            vector = []
            for j in range(columns_mat1):
                vector.append(mat1[i][j] + mat2[i][j])
            add_mat.append(vector)
        return add_mat
    else:
        return None
