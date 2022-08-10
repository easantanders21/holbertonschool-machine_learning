#!/usr/bin/env python3
"""Exponential class module"""


class Exponential:
    """Exponential probability distribution class"""

    e = 2.7182818285

    def __init__(self, data=None, lambtha=1.):
        """Exponential object attributes initialization"""
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                self.lambtha = len(data)/sum(data)

    def pdf(self, x):
        """Calculates the value of the Exponential PDF"""
        if x < 0:
            return 0
        else:
            return ((self.lambtha)*(Exponential.e**(-self.lambtha*x)))

    def cdf(self, x):
        """Calculates the value of the Exponential CDF"""
        if x < 0:
            return 0
        else:
            return (1-Exponential.e**(-self.lambtha*x))
