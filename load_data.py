from mnist import MNIST
import numpy as np
import sklearn.preprocessing
def load_mnist():
    mndata = MNIST('./data')
    mndata.gz = True

    images, labels = mndata.load_training()
    x_train = np.array(images)
    y_train = np.array(labels)
    images, labels = mndata.load_testing()
    x_test = np.array(images)
    y_test = np.array(labels)

    label_binarizer = sklearn.preprocessing.LabelBinarizer()
    label_binarizer.fit(range(max(y_train)+1))
    y_train = label_binarizer.transform(y_train)
    label_binarizer.fit(range(max(y_test)+1))
    y_test = label_binarizer.transform(y_test)

    return x_train, y_train, x_test, y_test
