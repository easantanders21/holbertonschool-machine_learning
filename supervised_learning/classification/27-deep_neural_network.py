#!/usr/bin/env python3
"""
Module for class DeepNeuralNetwork
"""
import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """
    Defines a deep neural network performing
    binary classification
    """

    def __init__(self, nx, layers):
        """
        Class Constructor

        nx is the number of input features
        layers is a list representing the number of
            nodes in each layer of the network
        """
        if type(nx) != int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(layers) != list or layers == []:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        weights = {}
        prev_layer = nx
        for i in range(self.L):
            if layers[i] < 1:
                raise TypeError("layers must be a list of positive integers")
            weights["W" + str(i + 1)] = np.random.randn(
                layers[i], prev_layer) * np.sqrt(2 / prev_layer)
            weights["b" + str(i + 1)] = np.zeros((layers[i], 1))
            prev_layer = layers[i]
        self.__weights = weights

    @property
    def L(self):
        """ Getter for L """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights """
        return self.__weights

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network """
        A = X
        self.__cache["A0"] = X
        for layer in range(1, self.__L + 1):
            W = self.weights["W" + str(layer)]
            b = self.weights["b" + str(layer)]
            Z = np.matmul(W, A) + b
            if layer == self.__L:
                A_temp = np.exp(Z)
                A = A_temp / np.sum(A_temp, axis=0, keepdims=True)
            else:
                A = A = 1 / (1 + np.exp(-Z))
            self.__cache["A" + str(layer)] = A
        return A, self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression """
        m = Y.shape[1]
        loss = Y * np.log(A)
        cost = (-1 / m) * np.sum(loss)
        return cost

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        A, cache = self.forward_prop(X)
        cost = self.cost(Y, A)
        pred = np.where(A >= 0.5, 1, 0)
        return pred, cost

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron """
        m = Y.shape[1]
        for i in reversed(range(1, self.__L + 1)):
            A = cache["A" + str(i)]
            A_prev = cache["A" + str(i - 1)]
            if i == self.__L:
                dz = A - Y
            else:
                dz = da * (A * (1 - A))
            dw = np.matmul(dz, A_prev.T) / m
            db = np.sum(dz, axis=1, keepdims=True) / m
            W = self.weights["W" + str(i)]
            da = np.matmul(W.T, dz)

            self.__weights["W" + str(i)] -= (alpha * dw)
            self.__weights["b" + str(i)] -= (alpha * db)

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """ Trains the deep neural network """
        if type(iterations) != int:
            raise TypeError("iterations must be an integer")
        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) != float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) != int:
                raise TypeError("step must be an integer")
            if step <= 0 or step >= iterations:
                raise ValueError("step must be positive and <= iterations")
        costs = []
        step_array = list(range(0, iterations + 1, step))
        for i in range(iterations + 1):
            if verbose and i in step_array:
                pred, cost = self.evaluate(X, Y)
                costs.append(cost)
                print("Cost after {} iterations: {}".format(i, cost))
            if i != iterations:
                A, cache = self.forward_prop(X)
                self.gradient_descent(Y, cache, alpha)

        if graph:
            plt.plot(step_array, costs)
            plt.xlabel('iteration')
            plt.ylabel('cost')
            plt.title('Training Cost')
            plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format """
        if type(filename) != str:
            return None
        if filename[-4:] != ".pkl":
            filename = filename + ".pkl"
        with open(filename, 'wb') as f:
            pickle.dump(self, f)

    @staticmethod
    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object """
        try:
            with open(filename, 'rb') as f:
                return pickle.load(f)
        except FileNotFoundError:
            return None
