#!/usr/bin/env python3
''' Write a function that performs element-wise addition,
subtraction, multiplication, and division: '''


def np_elementwise(mat1, mat2):
    """function that performs element-wise addition,
    subtraction, multiplication, and division"""
    add = mat1 + mat2
    dif = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    return (add, dif, mul, div)
