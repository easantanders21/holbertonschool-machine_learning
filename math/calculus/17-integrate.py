#!/usr/bin/env python3
''' Write a function that calculates the integral of a polynomial '''


def poly_integral(poly, C=0):
    ''' function that calculates the integral of a polynomial '''
    if type(poly) is not list or len(poly) == 0 or type(C) is not int:
        return None
    elif len(poly) == 0 or poly == []:
        return None
    elif poly == [0]:
        return [C]
    else:
        if len(poly) == 1:
            return [0]
        else:
            integral = []
            integral.append(C)
            for i in range(1, len(poly) + 1):
                if type(poly[i - 1]) not in [int, float]:
                    return None
                elif poly[i - 1] == 0:
                    integral.append(0)
                elif poly[i - 1] % i == 0:
                    integral.append(poly[i - 1] // i)
                else:
                    integral.append(poly[i - 1] / i)
            return integral
