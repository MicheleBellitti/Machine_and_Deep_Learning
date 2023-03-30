from random import uniform

import numpy as np
from datasets import two_moon_dataset, gaussians_dataset
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

plt.ion()


def compute_affinity_matrix(data, sigma):
    """
    Compute affinity matrix.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    sigma: float
        std of radial basis function kernel.

    Returns
    -------
    ndarray
        computed affinity matrix. Has shape (n_samples, n_samples)
    """
    # compute pairwise squared euclidean distances beetwen data points

    pairwise_distances = np.linalg.norm(data[:, None] - data, axis=2) ** 2

    # compute affinity matrix
    affinity_matrix = np.exp(-pairwise_distances / (sigma ** 2))

    return affinity_matrix


# print("Spectral Clustering adj matrix:" + str(compute_affinity_matrix(np.array([[1, 2], [3, 4]]), 1)))


def spectral_clustering(data, n_cl, sigma=1., fiedler_solution=False):
    """
    Spectral clustering.

    Parameters
    ----------
    data: ndarray
        data to partition, has shape (n_samples, dimensionality).
    n_cl: int
        number of clusters.
    sigma: float
        std of radial basis function kernel.
    fiedler_solution: bool
        return fiedler solution instead of kmeans

    Returns
    -------
    ndarray
        computed assignment. Has shape (n_samples,)
    """
    # compute affinity matrix
    affinity_matrix = compute_affinity_matrix(data, sigma)

    # compute degree matrix
    degree_matrix = np.diag(np.sum(affinity_matrix, axis=1))

    # compute laplacian
    laplacian_matrix = degree_matrix - affinity_matrix

    # compute eigenvalues and vectors (suggestion: np.linalg is your friend)
    eigenvalues, eigenvectors = np.linalg.eig(laplacian_matrix)

    # ensure we are not using complex numbers - you shouldn't btw
    if eigenvalues.dtype == 'complex128':
        print(
            "My dude, you got complex eigenvalues. Now I am not gonna break down, but you should totally give me higher sigmas (σ). (;")
        eigenvalues, eigenvectors = eigenvalues.real, eigenvectors.real

    # sort eigenvalues and vectors
    eigenvalues, eigenvectors = np.sort(eigenvalues), eigenvectors[:, np.argsort(eigenvalues)]

    # SOLUTION A: Fiedler-vector solution
    # - consider only the SECOND smallest eigenvector
    # - threshold it at zero
    # - return as labels
    # print("eigenvalues: " + str(eigenvector:s[1]))
    labels = eigenvectors[:, 1]
    labels[labels < 0] = 0
    if fiedler_solution:
        return labels

    # SOLUTION B: K-Means solution
    # - consider eigenvectors up to the n_cl-th 
    # - use them as features instead of data for KMeans
    # - You want to use sklearn's implementation (;
    # - return KMeans' clusters
    new_features = eigenvectors[:, n_cl]
    labels = KMeans(n_clusters=n_cl).fit_predict(new_features.reshape(-1, 1))

    return labels


def main_spectral_clustering():
    """
    Main function for spectral clustering.
    """

    # generate the dataset
    # data, cl = two_moon_dataset(n_samples=300, noise=0.1)
    data, cl = gaussians_dataset(n_gaussian=4, n_points=[100, 100, 70,50], mus=[[1, 1], [-4, 6], [8, 8],[-5,-7]],stds=[[1, 1], [3, 3], [1, 1],[-4,-4]])

    # visualize the dataset
    _, ax = plt.subplots(1, 2)
    ax[0].scatter(data[:, 0], data[:, 1], c=cl, s=40)
    # plt.waitforbuttonpress()
    n_cl = len(np.unique(cl))
    # run spectral clustering - tune n_cl and sigma!!!
    best_acc = 0.3
    best_sigma = 1
    best_labels = None
    process_active = False
    '''for s in np.random.uniform(2, 3, 10):
        process_active = True
        labels = spectral_clustering(data, n_cl=n_cl, sigma=s)
        acc = 1 - len((cl - labels).nonzero()[0]) / len(labels)

        if acc > best_acc:
            best_acc = acc
            best_labels = labels
            best_sigma = s
        if acc == 1:
            break'''
    ideal_sigma = 0.031969285036364904
    best_labels = spectral_clustering(data, n_cl=n_cl, sigma=best_sigma)
    if not process_active:
        print('accuracy:' + str(len(best_labels[best_labels == cl]) / len(best_labels)*100))
    # visualize results

    ax[1].scatter(data[:, 0], data[:, 1], c=best_labels, s=40)
    plt.waitforbuttonpress()
    if process_active:
        print("Ideal sigma: " + str(best_sigma))
        print("Accuracy: %s" % str(100*best_acc)+'%')
        print("Done!")


if __name__ == '__main__':
    main_spectral_clustering()
