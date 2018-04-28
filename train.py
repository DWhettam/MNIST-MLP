import numpy as np
from sigmoid import *

class Network(object):
    def __init__(self, layersAndNodes):
        self.layers = len(layersAndNodes)
        self.nodes = layersAndNodes
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layersAndNodes[:-1], layersAndNodes[1:])]

    def feedforward(self, input):
        for weights in self.weights:
            input = sigmoid(np.dot(input, weights))
        return input

    def grad_descent(self, x, y):
        result = self.feedforward(x)
        return result
