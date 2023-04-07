import math

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.datasets import mnist


def P(val, a, b, c, i):
    count = 0
    samples = 0
    for j in range(len(b)):
        if b[i] == c:
            samples += 1
            if a[i][j] == val:
                count += 1
    return count / samples


def prepare_data():
    batch_size_train = 64
    batch_size_test = 1000
    data = mnist.load_data()
    x = data[0]
    x_t = x[0]
    y_t = x[1]
    return x_t, y_t


class Bayes:
    def __init__(self):
        self.classes = None
        self.n_classes = 0
        self.priors = []
        self.likelihood = []

    def forward(self, x, y, i):
        y_hat = np.argmax(math.log(self.likelihood[i] + math.log(self.prior[i])))

    def train(self, x, y):
        self.classes, counts = np.unique(y, return_counts=True)
        self.n_classes = len(self.classes)
        self.priors = counts / x.shape[0]

    def compute_log_likelihood(self, xt, yt, i):
        self.likelihood.append(np.mean(yt[yt == i], axis=0))


x, y = prepare_data()
x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25)
b = Bayes()


