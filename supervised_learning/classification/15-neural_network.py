#!/usr/bin/env python3
""" class NeuralNetwork """
import numpy as np
import matplotlib as plt


class NeuralNetwork:
    """ class NeuralNetwork """
    def __init__(self, nx, nodes):
        """ Neural Network constructor """
        if type(nx) is not int:
            raise TypeError("nx must be an integer")
        if nx < 1:
            raise ValueError("nx must be a positive integer")
        if type(nodes) is not int:
            raise TypeError("nodes must be an integer")
        if nodes < 1:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(0, 1, (nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(0, 1, (1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """ W1 getter method """
        return self.__W1

    @property
    def b1(self):
        """ b1 getter method """
        return self.__b1

    @property
    def A1(self):
        """ A getter method """
        return self.__A1

    @property
    def W2(self):
        """ W getter method """
        return self.__W2

    @property
    def b2(self):
        """ b getter method """
        return self.__b2

    @property
    def A2(self):
        """ A getter method """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
            using sigmoid activation function
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))
        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))
        return (self.__A1, self.__A2)

    def cost(self, Y, A):
        """Calculates the cost of the model using logistic regression """
        J = -np.sum(Y*np.log(A)+(1-Y)*np.log(1.0000001 - A))/Y.shape[1]
        return J

    def evaluate(self, X, Y):
        """ Evaluates the neural network’s predictions """
        self.forward_prop(X)
        A2 = np.where(self.__A2 >= 0.5, 1, 0)
        J = self.cost(Y, self.__A2)
        return (A2, J)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """Calculates one pass of gradient descent on the neural network
        """
        dZ2 = A2-Y
        m = A1.shape[1]
        dW2 = np.matmul(dZ2, A1.T)/m
        db2 = np.sum(dZ2, axis=1, keepdims=True)/m
        dZ1 = np.matmul(self.__W2.T, dZ2)*(A1*(1-A1))
        dW1 = np.matmul(dZ1, X.T)/m
        db1 = np.sum(dZ1, axis=1, keepdims=True)/m
        self.__W1 = self.__W1-alpha*dW1
        self.__W2 = self.__W2-alpha*dW2
        self.__b1 = self.__b1-alpha*db1
        self.__b2 = self.__b2-alpha*db2

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the neural network """
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
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            J_i = self.cost(Y, self.__A2)
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
            plt.plot(steps, costs)
            plt.xlabel("iteration")
            plt.ylabel("cost")
            plt.title("Training Cost")
            plt.show()

        return self.evaluate(X, Y)
