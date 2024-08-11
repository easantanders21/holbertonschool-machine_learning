#!/usr/bin/env python3
''' create a function that calculates the sumaton of n '''


def summation_i_squared(n):
    ''' function that calculates the sumaton of n '''
    if n is None or n <= 0:
        return None
    else:
        sum = 0
        for i in range(n + 1):
            sum += i ** 2
        return sum
