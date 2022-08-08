#!/usr/bin/env python3
"""  class Poisson """


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
