#!/usr/bin/env python3
"""Binomial class module"""


class Binomial:
    """Binomial probability distribution class"""

    @staticmethod
    def factorial(n):
        """
        Factorial
        """
        fact = 1
        for num in range(2, n + 1):
            fact *= num
        return fact

    def __init__(self, data=None, n=1, p=0.5):
        """Binomial object attributes initialization"""
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            self.n = round(n)
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.p = float(p)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                mean = sum(data)/len(data)
                variance = sum([(mean-xi)**2 for xi in data])/len(data)
                p = 1-variance/mean
                self.n = round(mean/p)
                self.p = mean/self.n

    def pmf(self, k):
        """Calculates the value of the Binomial PMF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            return ((Binomial.factorial(self.n) /
                     (Binomial.factorial(k) *
                     Binomial.factorial(self.n-k))) *
                    self.p ** k * (1-self.p) ** (self.n-k))

    def cdf(self, k):
        """Calculates the value of the Binomial CDF"""
        if type(k) is not int:
            k = int(k)
        if k < 0:
            return 0
        else:
            cdf = 0
            for i in range(k+1):
                cdf += self.pmf(i)
        return cdf
