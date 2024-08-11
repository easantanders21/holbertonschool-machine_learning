#!/usr/bin/env python3
''' Write a function that calculates the derivative of a polynomial '''


def poly_derivative(poly):
    ''' function that calculates the derivative of a polynomial '''
    if type(poly) is not list or len(poly) == 0:
        return None
    else:
        if len(poly) == 1:
            return [0]
        else:
            der = []
            for i in range(1, len(poly)):
                der.append(i * poly[i])
            return(der)
