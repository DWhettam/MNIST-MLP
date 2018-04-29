import numpy as np
from sigmoid import *

class Network(object):
    def __init__(self, layersAndNodes):
        self.layers = len(layersAndNodes)
        self.nodes = layersAndNodes
        self.weights = [np.random.randn(x, y)
                        for x, y in zip(layersAndNodes[:-1], layersAndNodes[1:])]

    def train(self, x, y):
        for weights in self.weights:
            x = sigmoid(np.dot(x, weights))

        d3 = x - y
