#!/usr/bin/env python3
"""  class Poisson """
e = 2.7182818285


class Poisson:
    """  class Poisson that represents a poisson distribution """

    def __init__(self, data=None, lambtha=1.):
        """ class Poisson constructor """
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            self.lambtha = float(lambtha)
        elif type(data) is not list:
            raise TypeError("data must be a list")
        elif len(data) <= 1:
            raise ValueError("data must contain multiple values")
        else:
            self.lambtha = float(sum(data)/len(data))


    def pmf(self, k):
        """ instance method pmf """
        if k is not int:
            try:
                k = int(k)
            except:
                return 0
        if k < 0:
            return 0
        pmf = (e ** (-self.lambtha) * self.lambtha ** (k)) / factorial(k)
        return pmf


    def cdf(self, k):
        """ instance method cdf """
        pass
        

def factorial(num):
    """ calculates a factorial """
    factorial = 1
    for i in range(1,num + 1):    
       factorial = factorial * i
    return factorial
