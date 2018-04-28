import numpy as np
import load_data
import train

x_train, y_train, x_test, y_test = load_data.load_mnist()

neuralNet = train.Network([x_train.shape[-1], 32, 10])

result = neuralNet.grad_descent(x_train, y_train)
