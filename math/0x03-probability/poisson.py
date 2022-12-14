#!/usr/bin/env python3
"""Poisson class module"""


class Poisson:
    """Poisson probability distribution class"""

    e = 2.7182818285

    @staticmethod
    def factorial(n):
        """
        Factorial
        """
        fact = 1
        for num in range(2, n + 1):
            fact *= num
        return fact

    def __init__(self, data=None, lambtha=1.):
        """Poisson object attributes initialization"""
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
                self.lambtha = sum(data)/len(data)

    def pmf(self, k):
        """Calculates the value of the Poisson PMF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            return (((self.lambtha**k)*(Poisson.e**(-self.lambtha))) /
                    Poisson.factorial(k))

    def cdf(self, k):
        """Calculates the value of the Poisson CDF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            k_sumattion = 0
            for m in range(0, k+1):
                k_sumattion += (self.lambtha**m)/Poisson.factorial(m)
            return (Poisson.e**(-self.lambtha))*k_sumattion
