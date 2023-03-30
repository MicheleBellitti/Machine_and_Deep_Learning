import time

import math

import numpy as np

"""
Write a function that takes a 1d numpy array and computes its reverse vector
(last element becomes the first).
"""


def rev(ar):
    return a[-1::-1]


a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])

print(f'Reverse vector:{rev(a)}')

"""
Given the following square array, compute the product of the elements on its
diagonal. 
"""


def diag_prod(v):
    return np.prod(v.diagonal())


a = a.reshape((3, 3))
print(f'Product of the diagonal:{diag_prod(a)}')

"""
Create a random vector of size (3, 6) and find its mean value.
"""

v = np.array([np.random.randint(1, 100, (3, 6))])
v = v.reshape((3, 6))
print(f'Random vector mean: {v.mean()}')

"""
Given two arrays a and b, compute how many time an item of a is higher than the
corresponding element of b.
"""

a = np.array([[1, 5, 6, 8], [2, -3, 13, 23], [0, -10, -9, 7]])
b = np.array([[-3, 0, 8, 1], [-20, -9, -1, 32], [7, 7, 7, 7]])
c = a > b

print(f'Volte in cui a_i > b_i :{np.sum([1 for e in c for el in e if el == True])}')

"""
Create and normalize the following matrix (use min-max normalization).

"""

mat = np.array([[0.35, -0.27, 0.56], [0.15, 0.65, 0.42], [0.73, -0.78, -0.08]])
maxval = max(mat.max(0))
minval = min(mat.min(0))
q = maxval - minval
norm = np.array([(x - minval) / q for dx in mat for x in dx]).reshape(mat.shape)

print(f'normalized matrix:\n {norm}')

"""
Let’s run a little benchmark! Given a matrix A ∈ R
NxM and a vector b ∈ R
M,
compute the euclidean distance between b and each row Ai,: of A:
"""

mat = np.random.random((10000,10000))

def raw_distance(m, v):
    for i in range(len(m)):
        for j in range(len(m[0])):
            m[i][j] -= v[i]
            m[i][j] **= 2
    return math.sqrt(sum(m.reshape((m.shape[0]**2,))))


def numpy_distance(m, v):
    return np.sqrt(np.sum([np.square(m[i]-v[i]) for i in range(len(m))]))


def benchmark(f, m, v):
    tim = time.perf_counter()
    f(m, v)

    return tim


# print('time numpy func:'+str(benchmark(numpy_distance, mat, np.ones((10000*10000,)))),', time vanilla func:'+str(
# benchmark(raw_distance, mat, np.ones((10000*10000,)))))
