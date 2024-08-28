#!/usr/bin/env python3
"""DeepNeuralNetwork class module"""
import numpy as np
import matplotlib.pyplot as plt


class DeepNeuralNetwork:
    """DeepNeuralNetwork defines a neural network with one hidden layer
        performing binary classification
    """

    def __init__(self, nx, layers):
        """DeepNeuralNetwork object attributes initialization
        """
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
        """
        L getter method
        """
        return self.__L

    @property
    def cache(self):
        """
        cache getter method
        """
        return self.__cache

    @property
    def weights(self):
        """
        weights getter method
        """
        return self.__weights

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
            using sigmoid activation function
        """
        self.__cache['A0'] = X
        for i in range(self.__L):
            Z = (np.matmul(self.__weights["W{}".format(i + 1)],
                 self.__cache["A{}".format(i)]) +
                 self.__weights["b{}".format(i + 1)])
            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Z))

        return self.__cache["A{}".format(i + 1)], self.__cache

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression
        """
        J = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return J

    def evaluate(self, X, Y):
        """Evaluates the neuron’s predictions"""
        m = Y.shape[1]
        for i in range(self.__L, 0, -1):
            Al = cache["A"+str(i)]
            if i == self.__L:
                dAl = (-1*(Y/Al))+(1-Y)/(1-Al)
            Al1 = cache["A"+str(i-1)]
            dZ = dAl*(Al*(1-Al))
            dW = np.matmul(dZ, Al1.T)/m
            db = np.sum(dZ, axis=1, keepdims=True)/m
            dAl = np.matmul((self.__weights["W"+str(i)]).T, dZ)
            self.__weights["W"+str(i)] = self.__weights["W"+str(i)]-alpha*dW
            self.__weights["b"+str(i)] = self.__weights["b"+str(i)]-alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """Trains the neural network"""
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")

        steps = []
        costs = []
        for i in range(iterations):
            A, cache = self.forward_prop(X)
            self.gradient_descent(Y, self.cache, alpha)
            if i % step == 0:
                if graph is True:
                    steps.append(i)
                    costs.append(self.cost(Y,
                                 self.cache["A{}".format(self.L)]))
                if verbose is True:
                    print("Cost after {} iterations: {}"
                          .format(i, self.cost(Y, A)))

        J = self.evaluate(X, Y)[1]
        if verbose is True:
            print("Cost after {} iterations: {}".format(i + 1, J))

        if graph is True:
            steps.append(i+1)
            costs.append(self.cost(Y, self.cache["A{}".format(self.L)]))
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
