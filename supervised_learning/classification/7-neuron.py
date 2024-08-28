#!/usr/bin/env python3
""" class Neuron """
import numpy as np
import matplotlib.pyplot as plt


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

    def evaluate(self, X, Y):
        """ evaluate neuron """
        self.forward_prop(X)
        evaluate_array = np.where(self.__A >= 0.5, 1, 0)
        costo = self.cost(Y, self.__A)
        return evaluate_array, costo

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ gradient descent"""
        dZ = A-Y
        m = Y.shape[1]
        dW = np.matmul(dZ, X.T)/m
        db = np.sum(dZ)/m
        self.__W = self.__W-alpha*dW
        self.__b = self.__b-alpha*db

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True, graph=True, step=100):
        """ train the neuron """
        if type(iterations) is not int:
            raise TypeError("iterations must be an integer")
        if iterations < 1:
            raise ValueError("iterations must be a positive integer")
        if type(alpha) is not float:
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose is True or graph is True:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step <= 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")

        steps = []
        costs = []

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A, alpha)
            J_i = self.cost(Y, self.__A)
            if i % step == 0:
                if graph is True:
                    steps.append(i)
                    costs.append(J_i)
                if verbose is True:
                    print("Cost after {} iterations: {}".format(i, J_i))

        J = self.evaluate(X, Y)[1]
        if verbose is True:
            print("Cost after {} iterations: {}".format(i + 1, J))

        if graph is True:
            steps.append(i+1)
            costs.append(J)

        if graph is True:
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
