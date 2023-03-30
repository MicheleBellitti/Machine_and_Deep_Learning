# {x_i} i = 0, 1, 2,..., N feature vector y  col vector of the labels the rule is 0.5*min(||w||^2) the N constraints
# are y_i*(<w,x_i>+b) - 1 > 0 kernels: we can turn the problem into its dual version, now the goal is to find the
# alpha vector that bets fits the function L = sum(a)-sum_i(sum_j(a_i*a_j*y_i*y_jK(x_i,x_j))) where K is the kernel
# ->linear,poly or gaussian the linear kernel is K = x_i.T*x_j pegasos algorithm -> solves the primal problem through
# the gradient descent method


import numpy as np
from numpy.linalg import norm
from sklearn.svm import SVC

import data_prepare
import datasets

eps = np.finfo(float).eps


class SVM:
    """ Models a Support Vector machine classifier based on the PEGASOS algorithm. """

    def __init__(self, n_epochs, lambDa, use_bias=True):
        """ Constructor method """

        # weights placeholder
        self._w = None
        self._original_labels = None
        self._n_epochs = n_epochs
        self._lambda = lambDa
        self._use_bias = use_bias

    def map_y_to_minus_one_plus_one(self, y):
        """
        Map binary class labels y to -1 and 1
        """
        ynew = np.array(y)
        self._original_labels = np.unique(ynew)
        assert len(self._original_labels) == 2
        ynew[ynew == self._original_labels[0]] = -1.0
        ynew[ynew == self._original_labels[1]] = 1.0
        return ynew

    def map_y_to_original_values(self, y):
        """
        Map binary class labels, in terms of -1 and 1, to the original label set.
        """
        ynew = np.array(y)
        ynew[ynew == -1.0] = self._original_labels[0]
        ynew[ynew == 1.0] = self._original_labels[1]
        return ynew

    def loss(self, y_true, y_pred):
        """
        The PEGASOS loss term

        Parameters
        ----------
        y_true: np.array
            real labels in {0, 1}. shape=(n_examples,)
        y_pred: np.array
            predicted labels in [0, 1]. shape=(n_examples,)

        Returns
        -------
        float
            the value of the pegasos loss.
        """

        """
        Write HERE the code for computing the Pegasos loss function.
        """
        N = y_true.shape[0]
        return 0.5 * self._lambda * norm(self._w) + np.mean(np.maximum(0, 1 - y_true * y_pred))

    def fit_gd(self, X, Y, verbose=False):
        """
        Implements the gradient descent training procedure.

        Parameters
        ----------
        X: np.array
            data. shape=(n_examples, n_features)
        Y: np.array
            labels. shape=(n_examples,)
        verbose: bool
            whether to print the value of cost function.
        """

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        n_samples, n_features = X.shape
        Y = self.map_y_to_minus_one_plus_one(Y)

        # initialize weights
        self._w = np.zeros(shape=(n_features,), dtype=X.dtype)

        t = 0
        # loop over epochs
        for e in range(1, self._n_epochs + 1):
            for j in range(n_samples):
                """
                Write HERE the update step.
                """
                t += 1
                n = 1 / (t + self._lambda)
                if Y[j] * np.sum(self._w.T.dot(X[j])) < 1:
                    self._w = (1 - n * self._lambda) * self._w + n * Y[j] * X[j]
                else:
                    self._w = (1 - n * self._lambda) * self._w

            # predict training data
            cur_prediction = np.dot(X, self._w)

            # compute (and print) cost
            cur_loss = self.loss(y_true=Y, y_pred=cur_prediction)

            if verbose:
                print("Epoch {} Loss {}".format(e, cur_loss))

    def predict(self, X):

        if self._use_bias:
            X = np.concatenate([X, np.ones((X.shape[0], 1), dtype=X.dtype)], axis=-1)

        """
        Write HERE the criterium used during inference. 
        W * X > 0 -> positive class
        X * X < 0 -> negative class
        """

        return np.where(np.dot(X, self._w) > 0.0,
                        self._original_labels[1], self._original_labels[0])

    def score(self, x_test, y_test):
        return np.mean(self.predict(x_test) == y_test)


score = []

m = SVM(10000, 0.01)
s = SVC(kernel='rbf')
x_tr, x_te, y_tr, y_te = datasets.two_moon_dataset()
m.fit_gd(x_tr, y_tr)
s.fit(x_tr,y_tr)
print(str(m.score(x_te, y_te)*100)+'%')

