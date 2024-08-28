#!/usr/bin/env python3
"""DeepNeuralNetwork class module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """

    def __init__(self, nx, layers):
        """ DeepNeuralNetwork object attributes initialization """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(self.L):
            if ((type(layers[i]) is not int) or (layers[i] < 1)):
                raise TypeError("layers must be a list of positive integers")
            self.weights["b{}".format(i+1)] = np.zeros((layers[i], 1))
            if i == 0:
                self.weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                   self.nx)*np.sqrt(2/self.nx))
            else:
                self.weights["W{}".format(i+1)] = (np.random.randn(layers[i],
                                                   layers[i-1]) *
                                                   np.sqrt(2/layers[i-1]))
