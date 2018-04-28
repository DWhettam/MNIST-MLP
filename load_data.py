from mnist import MNIST
import numpy as np

def load_mnist():
    mndata = MNIST('./data')
    mndata.gz = True

    images, labels = mndata.load_training()
    x_train = np.array(images)
    y_train = np.array(labels)
    images, labels = mndata.load_testing()
    x_test = np.array(images)
    y_test = np.array(labels)

    #x_train  = np.array([np.reshape(x, (784, 1)) for x in x_train])
    y_train = np.array([vectorized_result(y) for y in y_train])
    print(x_train.shape)
    #x_test = np.array([np.reshape(x, (784, 1)) for x in x_test])
    y_test = np.array([vectorized_result(y) for y in y_test])
    return x_train, y_train, x_test, y_test

def vectorized_result(j):
    e = np.zeros((10, 1))
    e[j] = 1
    return e
