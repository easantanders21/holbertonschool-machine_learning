#!/usr/bin/env python3
""" class Neuron """
import numpy as np


class Neuron:
    """ class Neuron """
    def __init__(self, nx):
        """ a class Neuron that defines a single neuron
            performing binary classification:
        """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        self.W = np.random.normal(0, 1, (1, nx))
        self.b = 0
        self.A = 0
