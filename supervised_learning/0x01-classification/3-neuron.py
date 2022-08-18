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
        self.__W = np.random.normal(0, 1, (1, nx))
        self.__b = 0
        self.__A = 0

    @property
    def W(self):
        """ W getter method """
        return self.__W

    @property
    def b(self):
        """ b getter method """
        return self.__b

    @property
    def A(self):
        """ A getter method """
        return self.__A

    def forward_prop(self, X):
        """ sigmoid activation function """
        Z = np.matmul(self.W, X) + self.b
        self.__A = 1/(1+np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ cost function """
        J = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return J
