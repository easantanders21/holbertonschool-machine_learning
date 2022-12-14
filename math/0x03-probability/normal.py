#!/usr/bin/env python3
"""Normal class module"""


class Normal:
    """Normal probability distribution class"""

    e = 2.7182818285
    π = 3.1415926536

    def __init__(self, data=None, mean=0., stddev=1.):
        """Normal object attributes initialization"""
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            self.mean = float(mean)
            self.stddev = float(stddev)
        else:
            if type(data) is not list:
                raise TypeError("data must be a list")
            if len(data) < 2:
                raise ValueError("data must contain multiple values")
            else:
                for value in data:
                    if type(value) not in [float, int]:
                        raise ValueError("data must contain multiple values")
                self.mean = sum(data)/len(data)
                self.stddev = (sum((x - self.mean)**2 for x in data) /
                               len(data))**0.5

    def z_score(self, x):
        """Calculates the z-score of a given x-value"""
        return (x - self.mean)/self.stddev

    def x_value(self, z):
        """Calculates the x-value of a given z-score"""
        return (z*self.stddev + self.mean)

    def pdf(self, x):
        """Calculates the value of the Normal PDF for a given x-value"""
        return ((Normal.e**(-0.5*((x-self.mean)/self.stddev)**2)) /
                (self.stddev*(2*Normal.π)**0.5))

    def cdf(self, x):
        """Calculates the value of the Normal CDF for a given x-value"""
        x0 = (x-self.mean)/((2**0.5)*self.stddev)
        erfx = ((2/(Normal.π)**0.5)*(x0-((x0**3)/3) +
                                     ((x0**5)/10)-((x0**7)/42)+((x0**9)/216)))
        return ((1+erfx)/2)
