from typing import Optional, Tuple

import numpy as np
from matplotlib import pyplot as plt


class KMeans:
    def __init__(self, k: int, max_iter: int, initial_centers: Optional[np.ndarray] = None,
                 verbose: Optional[bool] = False):
        self.k = k
        self.max_iter = max_iter
        self.verbose = verbose
        self.initial_centers = initial_centers

    def _init_centers(self, X: np.ndarray, use_sample: Optional[bool] = False):
        n_samples, dim = X.shape
        if self.initial_centers is not None:
            return self.initial_centers
        if use_sample:
            return X[np.random.choice(n_samples, self.k)]
        centers = np.zeros((self.k, dim))
        for i in range(self.k):
            minf, maxf = np.min(X[:, i]), np.max(X[:, i])
            centers[:, i] = np.random.uniform(low=minf, high=maxf, size=dim)
        return centers

    def single_fit_and_predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        n_samples, dim = X.shape
        # init centers and old_assignment
        centers = self._init_centers(X)
        old_ass = np.zeros((n_samples,))
        dists = np.zeros((n_samples, dim))
        # verbose stuff
        new_ass = ...
        if self.verbose:
            _, ax = plt.subplots()
        while True:
            if self.verbose:
                ax.scatter(X[:, 0], X[:, 1], c=old_ass, s=40)
                ax.plot(centers[:, 0], centers[:, 1], 'r*', markersize=20)
                ax.axis('off')
                plt.pause(1)
                plt.cla()

            # compute asssignment through euclidian distance
            for i in range(self.k):
                dists[:, i] = np.sum(np.square(X - centers[i]), axis=-1)

            new_ass = np.argmin(dists, axis=1).reshape((n_samples,))
            # update centers
            for i in range(self.k):
                centers[i] = np.mean(X[new_ass == i])
            if np.all(old_ass == new_ass):
                break
            old_ass = new_ass

        return centers, new_ass

    def cost_function(self, X: np.ndarray, centers: np.ndarray, assignment: np.ndarray):
        cost = 0.0

        for i in range(self.k):
            cost += np.sum(np.square(X[assignment == i] - centers[i]))
        return cost

    def fit_predict(self, X):
        opt = None
        min_loss = 2e+31
        for _ in range(self.max_iter):
            centers, ass = self.single_fit_and_predict(X)
            loss = self.cost_function(X, centers, ass)
            print(loss)
            if loss < min_loss:
                min_loss = loss
                opt = ass
        return opt
