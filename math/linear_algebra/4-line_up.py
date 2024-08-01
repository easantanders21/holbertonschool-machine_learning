#!/usr/bin/env python3
''' Write a function that adds two arrays element-wise '''


def add_arrays(arr1, arr2):
    ''' Function that adds two arrays element-wise '''
    if len(arr1) == len(arr2):
        sum_arr = []
        for i in range(len(arr1)):
            sum_arr.append(arr1[i] + arr2[i])
    else:
        return None
