import numpy as np


def fourier_transform(x):
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    res = np.dot(M, x)
    return res
