import numpy as np
import load_data
import train
from sigmoid import *

x_train, y_train, x_test, y_test = load_data.load_mnist()

epochs = 1000
weight1 = np.random.randn(x_train.shape[-1], 500)
weight2 = np.random.randn(500, 500)
weight3 = np.random.randn(500, 500)
weight4 = np.random.randn(500 ,10)

for epoch in range(epochs):
    l1 = sigmoid(np.dot(x_train, weight1) + 1)
    l2 = sigmoid(np.dot(l1, weight2) + 1)
    l3 = sigmoid(np.dot(l2, weight3) + 1)
    l4 = sigmoid(np.dot(l3, weight4) + 1)

    d4 = l4 - y_train
    d3 = np.dot(d4, weight4.T) * sigmoid_derivative(l3)
    d2 = np.dot(d3, weight3.T) * sigmoid_derivative(l2)
    d1 = np.dot(d2, weight2.T) * sigmoid_derivative(l1)

    weight4 += np.dot(l3.T, d4)
    weight3 += np.dot(l2.T, d3)
    weight2 += np.dot(l1.T, d2)
    weight1 += np.dot(x_train.T, d1)

    mse = np.average(np.square(y_train - l4))
    print(mse)
