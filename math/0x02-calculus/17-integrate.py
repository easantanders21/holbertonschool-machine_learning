#!/usr/bin/env python3
""" Calculates the integral of a polynomial """


def poly_integral(poly, C=0):
    """ Calculates the integral of a polynomial """
    if type(poly) is not list or type(C) is not int:
        return None
    if len(poly) == 0 or poly == []:
        return None
    if poly == [0]:
        return [C]
    integral_coefficients = []
    for i in range(len(poly)):
        if type(poly[i]) not in [int, float]:
            return None
        if poly[i] % (i+1) == 0:
            integral_coefficients.append(poly[i]//(i+1))
        else:
            integral_coefficients.append(poly[i]/(i+1))
    integral_coefficients.insert(0, C)
    return integral_coefficients
