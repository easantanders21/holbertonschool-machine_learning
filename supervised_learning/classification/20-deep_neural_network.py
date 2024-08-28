#!/usr/bin/env python3
"""DeepNeuralNetwork class module"""
import numpy as np


class DeepNeuralNetwork:
    """DeepNeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """

    def __init__(self, nx, layers):
        """DeepNeuralNetwork object attributes initialization"""
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) is not list or len(layers) < 1:
            raise TypeError("layers must be a list of positive integers")
        self.nx = nx
        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(self.__L):
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

    @property
    def L(self):
        """ L getter method """
        return self.__L

    @property
    def cache(self):
        """ cache getter method """
        return self.__cache

    @property
    def weights(self):
        """ weights getter method """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            Z = (np.matmul(self.__weights["W{}".format(i + 1)],
                 self.__cache["A{}".format(i)]) +
                 self.__weights["b{}".format(i + 1)])
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i + 1)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        J = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return J

    def evaluate(self, X, Y):
        """ Evaluates the neuron’s predictions """
        A = self.forward_prop(X)[0]
        J = self.cost(Y, A)
        A = np.where(A >= 0.5, 1, 0)
        return A, J
