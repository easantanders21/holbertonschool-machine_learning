#!/usr/bin/env python3
""" function that calculates the derivative of a polynomial """


def poly_derivative(poly):
    """ function that calculates the derivative """
    if type(poly) is not list or len(poly) == 0:
        return None
    derivate = []
    for i in range(len(poly)):
        if i >= 1:
            derivate.append(poly[i] * i)
    if len(derivate) == 0:
        derivate.append(0)
    return derivate
